#include "poe2/minimax/position_source.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <charconv>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "poe2/minimax/labeling_build.hpp"
#include "poe2/symmetry.hpp"

namespace poe2::minimax::position_source {

namespace {

namespace fs = std::filesystem;

constexpr std::uint64_t kSplitMixIncrement = UINT64_C(0x9e3779b97f4a7c15);
constexpr std::uint64_t kFamilySalt = UINT64_C(0x4e8b2d7d34a2c1f9);
constexpr std::uint64_t kTrajectorySalt = UINT64_C(0xd1b54a32d192ed03);
constexpr std::uint64_t kSplitSalt = UINT64_C(0x94d049bb133111eb);
constexpr std::uint32_t kGeneratorVersion = 1;
constexpr std::string_view kRngName = "splitmix64-rejection-v1";
constexpr std::array<std::pair<int, int>, 8> kPhaseBuckets{{
    {4, 8},
    {9, 14},
    {15, 20},
    {21, 26},
    {27, 32},
    {33, 38},
    {39, 42},
    {43, 46},
}};

[[nodiscard]] constexpr std::uint64_t mix64(std::uint64_t value) noexcept {
  value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31);
}

[[nodiscard]] constexpr std::uint64_t nonzero_id(std::uint64_t value) noexcept {
  return value == 0 ? UINT64_C(1) : value;
}

class StableRng final {
 public:
  explicit constexpr StableRng(std::uint64_t seed) noexcept : state_(seed) {}

  [[nodiscard]] constexpr std::uint64_t next() noexcept {
    state_ += kSplitMixIncrement;
    return mix64(state_);
  }

  [[nodiscard]] std::uint64_t bounded(std::uint64_t bound) {
    if (bound == 0) {
      throw std::invalid_argument{"random bound must be positive"};
    }
    const std::uint64_t threshold = (std::uint64_t{0} - bound) % bound;
    for (;;) {
      const std::uint64_t value = next();
      if (value >= threshold) {
        return value % bound;
      }
    }
  }

 private:
  std::uint64_t state_;
};

struct PositionWords {
  std::uint64_t low = 0;
  std::uint64_t high = 0;

  friend constexpr bool operator==(const PositionWords&, const PositionWords&) = default;
};

struct PositionWordsHash {
  [[nodiscard]] std::size_t operator()(const PositionWords& value) const noexcept {
    return static_cast<std::size_t>(mix64(value.low ^ std::rotl(value.high, 23)));
  }
};

class DisjointSet final {
 public:
  explicit DisjointSet(std::size_t size) : parent_(size) {
    for (std::size_t index = 0; index < size; ++index) {
      parent_[index] = index;
    }
  }

  [[nodiscard]] std::size_t find(std::size_t value) {
    while (parent_[value] != value) {
      parent_[value] = parent_[parent_[value]];
      value = parent_[value];
    }
    return value;
  }

  void unite(std::size_t first, std::size_t second) {
    first = find(first);
    second = find(second);
    if (first == second) {
      return;
    }
    const std::size_t root = std::min(first, second);
    parent_[std::max(first, second)] = root;
  }

 private:
  std::vector<std::size_t> parent_;
};

[[nodiscard]] std::string json_escape(std::string_view text) {
  std::string escaped;
  escaped.reserve(text.size());
  constexpr char kHex[] = "0123456789abcdef";
  for (const unsigned char ch : text) {
    switch (ch) {
      case '"':
        escaped += "\\\"";
        break;
      case '\\':
        escaped += "\\\\";
        break;
      case '\b':
        escaped += "\\b";
        break;
      case '\f':
        escaped += "\\f";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        if (ch < 0x20) {
          escaped += "\\u00";
          escaped += kHex[ch >> 4];
          escaped += kHex[ch & 0x0f];
        } else {
          escaped += static_cast<char>(ch);
        }
        break;
    }
  }
  return escaped;
}

[[nodiscard]] std::string qualified_digest(const labeling::Sha256Digest& digest) {
  return "sha256:" + labeling::sha256_text(digest);
}

[[nodiscard]] std::string hex_bytes(std::string_view bytes) {
  constexpr char kHex[] = "0123456789abcdef";
  std::string encoded;
  encoded.reserve(bytes.size() * 2);
  for (const unsigned char byte : bytes) {
    encoded.push_back(kHex[byte >> 4]);
    encoded.push_back(kHex[byte & 0x0f]);
  }
  return encoded;
}

[[nodiscard]] int hex_digit(char value) noexcept {
  if (value >= '0' && value <= '9') {
    return value - '0';
  }
  if (value >= 'a' && value <= 'f') {
    return value - 'a' + 10;
  }
  return -1;
}

[[nodiscard]] std::string decode_hex_bytes(std::string_view encoded) {
  if (encoded.empty() || encoded.size() % 2 != 0) {
    throw std::runtime_error{"source corpus ID has invalid hexadecimal encoding"};
  }
  std::string decoded;
  decoded.reserve(encoded.size() / 2);
  for (std::size_t index = 0; index < encoded.size(); index += 2) {
    const int high = hex_digit(encoded[index]);
    const int low = hex_digit(encoded[index + 1]);
    if (high < 0 || low < 0) {
      throw std::runtime_error{"source corpus ID has invalid hexadecimal encoding"};
    }
    decoded.push_back(static_cast<char>((high << 4) | low));
  }
  return decoded;
}

void create_parent_directories(const fs::path& path) {
  const fs::path parent = path.parent_path();
  if (!parent.empty()) {
    fs::create_directories(parent);
  }
}

[[nodiscard]] std::string read_text_file(const fs::path& path) {
  std::ifstream input{path, std::ios::binary};
  if (!input) {
    throw std::runtime_error{"failed to read " + path.string()};
  }
  return {std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
}

void write_text_file(const fs::path& path, std::string_view bytes) {
  std::ofstream output{path, std::ios::binary | std::ios::trunc};
  if (!output) {
    throw std::runtime_error{"failed to open " + path.string()};
  }
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  output.close();
  if (!output) {
    throw std::runtime_error{"failed to write " + path.string()};
  }
}

[[nodiscard]] std::uint64_t policy_weight_sum(const PolicyWeights& weights) noexcept {
  return static_cast<std::uint64_t>(weights.random) + weights.immediate_gain +
         weights.opponent_aware + weights.noisy_search;
}

void validate_options(const PositionSourceOptions& options, bool has_progress,
                      std::uint64_t progress_interval) {
  if (options.corpus_id.empty()) {
    throw std::invalid_argument{"position source requires a corpus ID"};
  }
  if (options.trajectory_count == 0 ||
      options.trajectory_count > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument{"position source requires a representable trajectory count"};
  }
  if (options.samples_per_trajectory == 0 ||
      options.samples_per_trajectory > kPhaseBuckets.size()) {
    throw std::invalid_argument{"samples per trajectory must be between one and eight"};
  }
  if (options.shard_count == 0 || options.shard_count > options.trajectory_count) {
    throw std::invalid_argument{"shard count must be positive and no greater than trajectories"};
  }
  if (options.workers == 0) {
    throw std::invalid_argument{"position source requires at least one worker"};
  }
  if (options.noise_percent > 100) {
    throw std::invalid_argument{"noise percent must be between zero and 100"};
  }
  if (policy_weight_sum(options.policy_weights) == 0) {
    throw std::invalid_argument{"at least one source policy weight must be positive"};
  }
  if (options.policy_weights.noisy_search != 0 && options.search_nodes == 0) {
    throw std::invalid_argument{"noisy-search trajectories require a positive node limit"};
  }
  if (options.trajectory_count >
      std::numeric_limits<std::uint32_t>::max() / options.samples_per_trajectory) {
    throw std::invalid_argument{"position source would exceed the source-ordinal range"};
  }
  if (has_progress && progress_interval == 0) {
    throw std::invalid_argument{"position-source progress requires a positive interval"};
  }
}

[[nodiscard]] SourcePolicy choose_policy(const PolicyWeights& weights, StableRng& rng) {
  std::uint64_t selection = rng.bounded(policy_weight_sum(weights));
  if (selection < weights.random) {
    return SourcePolicy::kRandom;
  }
  selection -= weights.random;
  if (selection < weights.immediate_gain) {
    return SourcePolicy::kImmediateGain;
  }
  selection -= weights.immediate_gain;
  if (selection < weights.opponent_aware) {
    return SourcePolicy::kOpponentAware;
  }
  return SourcePolicy::kNoisySearch;
}

[[nodiscard]] int select_nth_move(Bitboard moves, std::uint64_t ordinal) {
  while (ordinal > 0) {
    moves &= moves - Bitboard{1};
    --ordinal;
  }
  return std::countr_zero(moves);
}

[[nodiscard]] int random_move(const Position& position, StableRng& rng) {
  const Bitboard moves = position.legal_moves();
  const int count = std::popcount(moves);
  if (count <= 0) {
    throw std::logic_error{"source policy was asked to move in a terminal position"};
  }
  return select_nth_move(moves, rng.bounded(static_cast<std::uint64_t>(count)));
}

template <typename ScoreFunction>
[[nodiscard]] int select_best_move(const Position& position, StableRng& rng,
                                   ScoreFunction&& score_function) {
  std::array<int, kCellCount> ties{};
  int tie_count = 0;
  Score best = std::numeric_limits<Score>::lowest();
  Bitboard moves = position.legal_moves();
  while (moves != 0) {
    const int move_index = std::countr_zero(moves);
    moves &= moves - Bitboard{1};
    const Score value = score_function(move_index);
    if (value > best) {
      best = value;
      ties[0] = move_index;
      tie_count = 1;
    } else if (value == best) {
      ties[tie_count++] = move_index;
    }
  }
  if (tie_count == 0) {
    throw std::logic_error{"source policy found no legal move"};
  }
  return ties[rng.bounded(static_cast<std::uint64_t>(tie_count))];
}

[[nodiscard]] int immediate_gain_move(const Position& position, StableRng& rng) {
  const Player player = position.side_to_move();
  return select_best_move(position, rng, [&](int move_index) {
    return position.score_gain_unchecked(player, move_index);
  });
}

[[nodiscard]] int opponent_aware_move(const Position& position, StableRng& rng) {
  struct MoveGains {
    int move_index = 0;
    Score own = 0;
    Score opponent = 0;
  };
  std::array<MoveGains, kCellCount> gains{};
  int gain_count = 0;
  const Player player = position.side_to_move();
  Score best_reply = std::numeric_limits<Score>::lowest();
  Score second_reply = std::numeric_limits<Score>::lowest();
  int best_reply_index = -1;

  Bitboard moves = position.legal_moves();
  while (moves != 0) {
    const int move_index = std::countr_zero(moves);
    moves &= moves - Bitboard{1};
    const ScoreByPlayer score_gains = position.score_gains_unchecked(move_index);
    const Score own = player == Player::kOne ? score_gains.player_one : score_gains.player_two;
    const Score reply = player == Player::kOne ? score_gains.player_two : score_gains.player_one;
    gains[gain_count++] = MoveGains{.move_index = move_index, .own = own, .opponent = reply};
    if (reply > best_reply) {
      second_reply = best_reply;
      best_reply = reply;
      best_reply_index = move_index;
    } else if (reply > second_reply) {
      second_reply = reply;
    }
  }

  return select_best_move(position, rng, [&](int move_index) {
    const auto iterator =
        std::find_if(gains.begin(), gains.begin() + gain_count,
                     [=](const MoveGains& value) { return value.move_index == move_index; });
    const Score reply = move_index == best_reply_index ? second_reply : best_reply;
    return iterator->own - reply;
  });
}

[[nodiscard]] int noisy_search_move(const Position& position, StableRng& rng, Search& search,
                                    const PositionSourceOptions& options) {
  if (rng.bounded(100) < options.noise_percent) {
    return random_move(position, rng);
  }
  const engine::EngineResult result = search.run(position,
                                                 engine::EngineLimits{
                                                     .nodes = options.search_nodes,
                                                 },
                                                 {});
  if (result.best_move.has_value() && position.board().can_place(result.best_move->square)) {
    return square_index(result.best_move->square);
  }
  return random_move(position, rng);
}

[[nodiscard]] int choose_move(const Position& position, SourcePolicy policy, StableRng& rng,
                              Search* search, const PositionSourceOptions& options) {
  switch (policy) {
    case SourcePolicy::kRandom:
      return random_move(position, rng);
    case SourcePolicy::kImmediateGain:
      return immediate_gain_move(position, rng);
    case SourcePolicy::kOpponentAware:
      return opponent_aware_move(position, rng);
    case SourcePolicy::kNoisySearch:
      if (search == nullptr) {
        throw std::logic_error{"no search is available for a noisy-search trajectory"};
      }
      return noisy_search_move(position, rng, *search, options);
  }
  throw std::logic_error{"unknown source policy"};
}

[[nodiscard]] std::array<int, kPhaseBuckets.size()> sample_target_plies(
    std::uint64_t trajectory_index, std::uint16_t sample_count, StableRng& rng) {
  std::array<int, kPhaseBuckets.size()> phase_indices{};
  for (std::size_t index = 0; index < phase_indices.size(); ++index) {
    phase_indices[index] = static_cast<int>(index);
  }
  for (std::size_t limit = phase_indices.size(); limit > 1; --limit) {
    const std::size_t selected = static_cast<std::size_t>(rng.bounded(limit));
    std::swap(phase_indices[selected], phase_indices[limit - 1]);
  }

  std::array<int, kPhaseBuckets.size()> targets{};
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    const int phase = phase_indices[sample];
    const auto [minimum, maximum] = kPhaseBuckets[phase];
    const int parity = static_cast<int>((trajectory_index + phase) & 1U);
    const int first = minimum + ((minimum & 1) != parity ? 1 : 0);
    const int choices = (maximum - first) / 2 + 1;
    targets[sample] = first + 2 * static_cast<int>(rng.bounded(choices));
  }
  std::sort(targets.begin(), targets.begin() + sample_count);
  return targets;
}

[[nodiscard]] std::vector<PositionSourceRecord> generate_trajectory(
    std::uint64_t trajectory_index, const PositionSourceOptions& options, Search* search) {
  StableRng rng{mix64(options.seed ^ mix64(trajectory_index + kSplitMixIncrement))};
  const SourcePolicy policy = choose_policy(options.policy_weights, rng);
  if (policy == SourcePolicy::kNoisySearch) {
    if (search == nullptr) {
      throw std::logic_error{"search source policy has no worker search"};
    }
    search->new_game();
  }
  const auto target_plies =
      sample_target_plies(trajectory_index, options.samples_per_trajectory, rng);
  const std::uint64_t family_id =
      nonzero_id(mix64(options.seed ^ mix64(trajectory_index ^ kFamilySalt)));
  const std::uint64_t trajectory_id =
      nonzero_id(mix64(options.seed ^ mix64(trajectory_index ^ kTrajectorySalt)));

  Position position;
  std::vector<PositionSourceRecord> records;
  records.reserve(options.samples_per_trajectory);
  std::size_t next_sample = 0;
  while (next_sample < options.samples_per_trajectory) {
    const int move_index = choose_move(position, policy, rng, search, options);
    if (!position.play(square_from_index(move_index))) {
      throw std::logic_error{"source policy selected an illegal move"};
    }
    if (position.ply() != target_plies[next_sample]) {
      continue;
    }
    records.push_back(PositionSourceRecord{
        .player_one = position.board().bits(Player::kOne),
        .player_two = position.board().bits(Player::kTwo),
        .family_id = family_id,
        .trajectory_id = trajectory_id,
        .trajectory_index = trajectory_index,
        .policy = policy,
        .sample_index = static_cast<std::uint16_t>(next_sample),
        .ply = static_cast<std::uint8_t>(position.ply()),
    });
    ++next_sample;
  }
  return records;
}

[[nodiscard]] labeling::DatasetSplit split_for_component(std::uint64_t seed,
                                                         std::uint64_t family_id) noexcept {
  const std::uint64_t bucket = mix64(seed ^ family_id ^ kSplitSalt) % 100;
  if (bucket < 70) {
    return labeling::DatasetSplit::kTrain;
  }
  if (bucket < 85) {
    return labeling::DatasetSplit::kValidation;
  }
  return labeling::DatasetSplit::kTest;
}

[[nodiscard]] int phase_for_ply(std::uint8_t ply) noexcept {
  for (std::size_t phase = 0; phase < kPhaseBuckets.size(); ++phase) {
    if (ply >= kPhaseBuckets[phase].first && ply <= kPhaseBuckets[phase].second) {
      return static_cast<int>(phase);
    }
  }
  return -1;
}

template <typename Integer>
  requires std::is_unsigned_v<Integer>
void fnv_append(std::uint64_t& hash, Integer value) noexcept {
  const std::uint64_t wide_value = value;
  for (std::size_t index = 0; index < sizeof(Integer); ++index) {
    hash ^= static_cast<std::uint8_t>((wide_value >> (8 * index)) & UINT64_C(0xff));
    hash *= UINT64_C(1099511628211);
  }
}

[[nodiscard]] std::uint64_t source_id(const PositionSourceRecord& record) noexcept {
  std::uint64_t hash = UINT64_C(14695981039346656037);
  fnv_append(hash, record.player_one);
  fnv_append(hash, record.player_two);
  fnv_append(hash, record.family_id);
  fnv_append(hash, record.trajectory_id);
  fnv_append(hash, record.parent_id);
  fnv_append(hash, record.trajectory_index);
  fnv_append(hash, static_cast<std::uint16_t>(record.policy));
  fnv_append(hash, record.sample_index);
  fnv_append(hash, static_cast<std::uint8_t>(record.split));
  fnv_append(hash, record.ply);
  return hash;
}

[[nodiscard]] std::pair<std::uint64_t, std::uint64_t> trajectory_range(
    std::uint64_t trajectory_count, std::uint32_t shard_count, std::uint32_t shard_index) {
  const std::uint64_t base = trajectory_count / shard_count;
  const std::uint64_t remainder = trajectory_count % shard_count;
  const std::uint64_t begin = shard_index * base + std::min<std::uint64_t>(shard_index, remainder);
  const std::uint64_t end = begin + base + (shard_index < remainder ? 1 : 0);
  return {begin, end};
}

void append_hex64(std::ostream& output, std::uint64_t value) {
  output << std::hex << std::setw(16) << std::setfill('0') << value << std::dec;
}

[[nodiscard]] std::string shard_file_name(std::uint32_t index,
                                          const labeling::Sha256Digest& digest) {
  std::ostringstream output;
  output.imbue(std::locale::classic());
  output << "shard-" << std::setw(8) << std::setfill('0') << index << '-'
         << labeling::sha256_text(digest) << ".jsonl";
  return output.str();
}

class LineParser final {
 public:
  explicit LineParser(std::string_view line) : remaining_(line) {}

  void literal(std::string_view expected) {
    if (!remaining_.starts_with(expected)) {
      throw std::runtime_error{"position source line has unexpected syntax"};
    }
    remaining_.remove_prefix(expected.size());
  }

  [[nodiscard]] std::string_view quoted() {
    literal("\"");
    const std::size_t end = remaining_.find('"');
    if (end == std::string_view::npos) {
      throw std::runtime_error{"position source string is unterminated"};
    }
    const std::string_view value = remaining_.substr(0, end);
    remaining_.remove_prefix(end + 1);
    return value;
  }

  template <typename Integer>
    requires std::is_unsigned_v<Integer>
  [[nodiscard]] Integer decimal() {
    Integer value = 0;
    const char* const begin = remaining_.data();
    const char* const end = begin + remaining_.size();
    const auto result = std::from_chars(begin, end, value, 10);
    if (result.ec != std::errc{} || result.ptr == begin) {
      throw std::runtime_error{"position source integer is invalid"};
    }
    remaining_.remove_prefix(static_cast<std::size_t>(result.ptr - begin));
    return value;
  }

  [[nodiscard]] std::uint64_t hex64() {
    const std::string_view encoded = quoted();
    if (encoded.size() != 16) {
      throw std::runtime_error{"position source 64-bit value has the wrong width"};
    }
    std::uint64_t value = 0;
    const auto result = std::from_chars(encoded.data(), encoded.data() + encoded.size(), value, 16);
    if (result.ec != std::errc{} || result.ptr != encoded.data() + encoded.size()) {
      throw std::runtime_error{"position source 64-bit value is not hexadecimal"};
    }
    return value;
  }

  void finish() {
    if (!remaining_.empty()) {
      throw std::runtime_error{"position source line has trailing data"};
    }
  }

 private:
  std::string_view remaining_;
};

struct ParsedHeader {
  std::string corpus_id;
  std::uint64_t seed = 0;
  std::uint64_t trajectory_count = 0;
  std::uint16_t samples_per_trajectory = 0;
  std::uint32_t shard_index = 0;
  std::uint32_t shard_count = 0;
  std::uint64_t trajectory_begin = 0;
  std::uint64_t trajectory_end = 0;
  std::uint64_t search_nodes = 0;
  std::size_t search_hash_bytes = 0;
  std::uint8_t noise_percent = 0;
  PolicyWeights policy_weights;
};

[[nodiscard]] ParsedHeader parse_header(std::string_view line) {
  LineParser parser{line};
  parser.literal("{\"type\":\"poe2-position-source\",\"schema\":");
  const std::uint32_t schema = parser.decimal<std::uint32_t>();
  parser.literal(",\"generator\":");
  const std::uint32_t generator = parser.decimal<std::uint32_t>();
  parser.literal(",\"rng\":\"");
  parser.literal(kRngName);
  parser.literal("\",\"corpus_id_hex\":");
  const std::string corpus_id = decode_hex_bytes(parser.quoted());
  parser.literal(",\"seed\":");
  const std::uint64_t seed = parser.hex64();
  parser.literal(",\"trajectory_count\":");
  const std::uint64_t trajectory_count = parser.decimal<std::uint64_t>();
  parser.literal(",\"samples_per_trajectory\":");
  const std::uint16_t samples = parser.decimal<std::uint16_t>();
  parser.literal(",\"shard_index\":");
  const std::uint32_t shard_index = parser.decimal<std::uint32_t>();
  parser.literal(",\"shard_count\":");
  const std::uint32_t shard_count = parser.decimal<std::uint32_t>();
  parser.literal(",\"trajectory_begin\":");
  const std::uint64_t trajectory_begin = parser.decimal<std::uint64_t>();
  parser.literal(",\"trajectory_end\":");
  const std::uint64_t trajectory_end = parser.decimal<std::uint64_t>();
  parser.literal(",\"search_nodes\":");
  const std::uint64_t search_nodes = parser.decimal<std::uint64_t>();
  parser.literal(",\"search_hash_bytes\":");
  const std::uint64_t search_hash_bytes = parser.decimal<std::uint64_t>();
  if (search_hash_bytes > std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error{"source search hash allocation is not representable"};
  }
  parser.literal(",\"noise_percent\":");
  const std::uint8_t noise_percent = parser.decimal<std::uint8_t>();
  parser.literal(",\"policy_weights\":[");
  const std::uint32_t random = parser.decimal<std::uint32_t>();
  parser.literal(",");
  const std::uint32_t immediate = parser.decimal<std::uint32_t>();
  parser.literal(",");
  const std::uint32_t opponent = parser.decimal<std::uint32_t>();
  parser.literal(",");
  const std::uint32_t search = parser.decimal<std::uint32_t>();
  parser.literal("]}");
  parser.finish();

  if (schema != kPositionSourceSchemaVersion || generator != kGeneratorVersion) {
    throw std::runtime_error{"position source schema or generator version is unsupported"};
  }
  return ParsedHeader{
      .corpus_id = corpus_id,
      .seed = seed,
      .trajectory_count = trajectory_count,
      .samples_per_trajectory = samples,
      .shard_index = shard_index,
      .shard_count = shard_count,
      .trajectory_begin = trajectory_begin,
      .trajectory_end = trajectory_end,
      .search_nodes = search_nodes,
      .search_hash_bytes = static_cast<std::size_t>(search_hash_bytes),
      .noise_percent = noise_percent,
      .policy_weights =
          PolicyWeights{
              .random = random,
              .immediate_gain = immediate,
              .opponent_aware = opponent,
              .noisy_search = search,
          },
  };
}

[[nodiscard]] PositionSourceRecord parse_record(std::string_view line) {
  LineParser parser{line};
  parser.literal("{\"p1\":");
  const Bitboard player_one = parser.hex64();
  parser.literal(",\"p2\":");
  const Bitboard player_two = parser.hex64();
  parser.literal(",\"source_id\":");
  const std::uint64_t parsed_source_id = parser.hex64();
  parser.literal(",\"family_id\":");
  const std::uint64_t family_id = parser.hex64();
  parser.literal(",\"trajectory_id\":");
  const std::uint64_t trajectory_id = parser.hex64();
  parser.literal(",\"parent_id\":");
  const std::uint64_t parent_id = parser.hex64();
  parser.literal(",\"trajectory_index\":");
  const std::uint64_t trajectory_index = parser.decimal<std::uint64_t>();
  parser.literal(",\"policy_id\":");
  const std::uint16_t policy = parser.decimal<std::uint16_t>();
  parser.literal(",\"sample_index\":");
  const std::uint16_t sample_index = parser.decimal<std::uint16_t>();
  parser.literal(",\"split\":");
  const std::uint8_t split = parser.decimal<std::uint8_t>();
  parser.literal(",\"ply\":");
  const std::uint8_t ply = parser.decimal<std::uint8_t>();
  parser.literal("}");
  parser.finish();
  return PositionSourceRecord{
      .player_one = player_one,
      .player_two = player_two,
      .source_id = parsed_source_id,
      .family_id = family_id,
      .trajectory_id = trajectory_id,
      .parent_id = parent_id,
      .trajectory_index = trajectory_index,
      .policy = static_cast<SourcePolicy>(policy),
      .sample_index = sample_index,
      .split = static_cast<labeling::DatasetSplit>(split),
      .ply = ply,
  };
}

[[nodiscard]] Position reconstruct_position(const PositionSourceRecord& record) {
  if (((record.player_one | record.player_two) & ~kBoardMask) != 0 ||
      (record.player_one & record.player_two) != 0) {
    throw std::runtime_error{"position source record has invalid bitboards"};
  }
  if (std::popcount(record.player_one | record.player_two) != record.ply ||
      std::popcount(record.player_one) != (record.ply + 1) / 2 ||
      std::popcount(record.player_two) != record.ply / 2) {
    throw std::runtime_error{"position source record has inconsistent piece counts"};
  }

  Position position;
  Bitboard remaining_one = record.player_one;
  Bitboard remaining_two = record.player_two;
  for (int ply = 0; ply < record.ply; ++ply) {
    Bitboard& remaining = ply % 2 == 0 ? remaining_one : remaining_two;
    if (remaining == 0) {
      throw std::runtime_error{"position source record cannot be reconstructed"};
    }
    const int move_index = std::countr_zero(remaining);
    remaining &= remaining - Bitboard{1};
    if (!position.play(square_from_index(move_index))) {
      throw std::runtime_error{"position source record contains an illegal position"};
    }
  }
  if (remaining_one != 0 || remaining_two != 0 ||
      position.board().bits(Player::kOne) != record.player_one ||
      position.board().bits(Player::kTwo) != record.player_two) {
    throw std::runtime_error{"position source record reconstruction failed"};
  }
  return position;
}

[[nodiscard]] bool valid_policy(SourcePolicy policy) noexcept {
  return policy == SourcePolicy::kRandom || policy == SourcePolicy::kImmediateGain ||
         policy == SourcePolicy::kOpponentAware || policy == SourcePolicy::kNoisySearch;
}

[[nodiscard]] bool valid_split(labeling::DatasetSplit split) noexcept {
  return split == labeling::DatasetSplit::kTrain || split == labeling::DatasetSplit::kValidation ||
         split == labeling::DatasetSplit::kTest;
}

[[nodiscard]] std::optional<labeling::Sha256Digest> parse_complete_marker(std::string_view marker) {
  constexpr std::string_view kPrefix = "poe2-position-source\nmanifest_sha256=";
  if (!marker.starts_with(kPrefix) || marker.size() != kPrefix.size() + 64 + 1 ||
      marker.back() != '\n') {
    return std::nullopt;
  }
  const std::string_view encoded = marker.substr(kPrefix.size(), 64);
  labeling::Sha256Digest digest{};
  for (std::size_t index = 0; index < digest.size(); ++index) {
    const int high = hex_digit(encoded[index * 2]);
    const int low = hex_digit(encoded[index * 2 + 1]);
    if (high < 0 || low < 0) {
      return std::nullopt;
    }
    digest[index] = static_cast<std::uint8_t>((high << 4) | low);
  }
  return digest;
}

[[nodiscard]] bool is_requested_shard_name(std::string_view name, std::uint32_t shard_index) {
  std::ostringstream prefix;
  prefix.imbue(std::locale::classic());
  prefix << "shard-" << std::setw(8) << std::setfill('0') << shard_index << '-';
  constexpr std::size_t kSuffixSize = sizeof(".jsonl") - 1;
  return name.starts_with(prefix.str()) && name.ends_with(".jsonl") &&
         name.size() == prefix.str().size() + 64 + kSuffixSize;
}

[[nodiscard]] labeling::Sha256Digest digest_from_shard_name(std::string_view name) {
  const std::size_t first_dash = name.find('-', 6);
  if (first_dash == std::string_view::npos || first_dash + 65 > name.size()) {
    throw std::runtime_error{"position source shard name is malformed"};
  }
  const std::string_view encoded = name.substr(first_dash + 1, 64);
  labeling::Sha256Digest digest{};
  for (std::size_t index = 0; index < digest.size(); ++index) {
    const int high = hex_digit(encoded[index * 2]);
    const int low = hex_digit(encoded[index * 2 + 1]);
    if (high < 0 || low < 0) {
      throw std::runtime_error{"position source shard digest is not hexadecimal"};
    }
    digest[index] = static_cast<std::uint8_t>((high << 4) | low);
  }
  return digest;
}

}  // namespace

SourceOutput::SourceOutput(fs::path directory) : directory_(std::move(directory)) {}

const fs::path& SourceOutput::directory() const noexcept { return directory_; }

bool SourceOutput::committed() const noexcept { return committed_; }

std::string_view source_policy_name(SourcePolicy policy) noexcept {
  switch (policy) {
    case SourcePolicy::kRandom:
      return "random";
    case SourcePolicy::kImmediateGain:
      return "immediate-gain";
    case SourcePolicy::kOpponentAware:
      return "opponent-aware";
    case SourcePolicy::kNoisySearch:
      return "noisy-search";
  }
  return "unknown";
}

PositionSourceCorpus generate_position_source(const PositionSourceOptions& options,
                                              const PositionSourceProgressSink& progress,
                                              std::uint64_t progress_interval) {
  validate_options(options, static_cast<bool>(progress), progress_interval);
  const std::size_t trajectory_count = static_cast<std::size_t>(options.trajectory_count);
  const std::size_t workers_used = std::min(options.workers, trajectory_count);
  std::vector<std::vector<PositionSourceRecord>> trajectories(trajectory_count);
  std::atomic<std::size_t> next_trajectory = 0;
  std::atomic<bool> stop = false;
  std::mutex failure_mutex;
  std::mutex progress_mutex;
  std::exception_ptr failure;
  std::uint64_t completed = 0;
  std::uint64_t next_progress = progress_interval;

  const auto record_progress = [&] {
    if (!progress) {
      return;
    }
    const std::scoped_lock lock{progress_mutex};
    ++completed;
    if (completed < next_progress && completed != options.trajectory_count) {
      return;
    }
    if (completed >= next_progress) {
      const std::uint64_t remaining = std::numeric_limits<std::uint64_t>::max() - completed;
      next_progress = remaining < progress_interval ? std::numeric_limits<std::uint64_t>::max()
                                                    : completed + progress_interval;
    }
    progress(PositionSourceProgress{
        .completed_trajectories = completed,
        .total_trajectories = options.trajectory_count,
    });
  };

  const auto run_worker = [&] {
    try {
      std::optional<Search> search;
      if (options.policy_weights.noisy_search != 0) {
        search.emplace(SearchOptions{
            .hash_bytes = options.search_hash_bytes,
            .use_symmetry = true,
            .evaluator = Evaluator::kTwoPlyClosure,
        });
      }
      while (!stop.load(std::memory_order_relaxed)) {
        const std::size_t trajectory = next_trajectory.fetch_add(1, std::memory_order_relaxed);
        if (trajectory >= trajectory_count) {
          break;
        }
        trajectories[trajectory] =
            generate_trajectory(trajectory, options, search.has_value() ? &*search : nullptr);
        record_progress();
      }
    } catch (...) {
      {
        const std::scoped_lock lock{failure_mutex};
        if (failure == nullptr) {
          failure = std::current_exception();
        }
      }
      stop.store(true, std::memory_order_relaxed);
    }
  };

  std::vector<std::jthread> workers;
  workers.reserve(workers_used);
  try {
    for (std::size_t worker = 0; worker < workers_used; ++worker) {
      workers.emplace_back(run_worker);
    }
  } catch (...) {
    stop.store(true, std::memory_order_relaxed);
    throw;
  }
  for (std::jthread& worker : workers) {
    worker.join();
  }
  if (failure != nullptr) {
    std::rethrow_exception(failure);
  }

  DisjointSet components{trajectory_count};
  std::unordered_map<PositionWords, std::size_t, PositionWordsHash> first_trajectory;
  std::size_t duplicate_positions = 0;
  for (std::size_t trajectory = 0; trajectory < trajectories.size(); ++trajectory) {
    for (const PositionSourceRecord& record : trajectories[trajectory]) {
      const Player side_to_move = record.ply % 2 == 0 ? Player::kOne : Player::kTwo;
      const PositionKey canonical =
          canonicalize_position_key(
              make_position_key(record.player_one, record.player_two, side_to_move))
              .key;
      const auto [iterator, inserted] = first_trajectory.emplace(
          PositionWords{.low = canonical.low, .high = canonical.high}, trajectory);
      if (!inserted) {
        ++duplicate_positions;
        components.unite(trajectory, iterator->second);
      }
    }
  }

  std::vector<labeling::DatasetSplit> splits(trajectory_count);
  for (std::size_t trajectory = 0; trajectory < trajectory_count; ++trajectory) {
    const std::size_t root = components.find(trajectory);
    splits[trajectory] = split_for_component(options.seed, trajectories[root].front().family_id);
  }

  PositionSourceCorpus corpus{
      .options = options,
      .workers_used = workers_used,
      .duplicate_positions = duplicate_positions,
  };
  corpus.records.reserve(trajectory_count * options.samples_per_trajectory);
  for (std::size_t trajectory = 0; trajectory < trajectories.size(); ++trajectory) {
    for (PositionSourceRecord& record : trajectories[trajectory]) {
      record.split = splits[trajectory];
      record.source_id = source_id(record);
      corpus.records.push_back(record);
    }
  }
  return corpus;
}

std::vector<SerializedSourceShard> serialize_source_shards(const PositionSourceCorpus& corpus) {
  validate_options(corpus.options, false, 0);
  const std::size_t expected_records = static_cast<std::size_t>(corpus.options.trajectory_count) *
                                       corpus.options.samples_per_trajectory;
  if (corpus.records.size() != expected_records) {
    throw std::invalid_argument{"position source record count does not match its configuration"};
  }

  std::vector<SerializedSourceShard> shards;
  shards.reserve(corpus.options.shard_count);
  std::size_t next_record = 0;
  for (std::uint32_t shard_index = 0; shard_index < corpus.options.shard_count; ++shard_index) {
    const auto [begin, end] =
        trajectory_range(corpus.options.trajectory_count, corpus.options.shard_count, shard_index);
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << "{\"type\":\"poe2-position-source\",\"schema\":" << kPositionSourceSchemaVersion
           << ",\"generator\":" << kGeneratorVersion << ",\"rng\":\"" << kRngName
           << "\",\"corpus_id_hex\":\"" << hex_bytes(corpus.options.corpus_id) << "\",\"seed\":\"";
    append_hex64(output, corpus.options.seed);
    output << "\",\"trajectory_count\":" << corpus.options.trajectory_count
           << ",\"samples_per_trajectory\":" << corpus.options.samples_per_trajectory
           << ",\"shard_index\":" << shard_index
           << ",\"shard_count\":" << corpus.options.shard_count << ",\"trajectory_begin\":" << begin
           << ",\"trajectory_end\":" << end << ",\"search_nodes\":" << corpus.options.search_nodes
           << ",\"search_hash_bytes\":" << corpus.options.search_hash_bytes
           << ",\"noise_percent\":" << static_cast<unsigned>(corpus.options.noise_percent)
           << ",\"policy_weights\":[" << corpus.options.policy_weights.random << ','
           << corpus.options.policy_weights.immediate_gain << ','
           << corpus.options.policy_weights.opponent_aware << ','
           << corpus.options.policy_weights.noisy_search << "]}\n";

    const std::size_t record_begin = next_record;
    while (next_record < corpus.records.size() &&
           corpus.records[next_record].trajectory_index < end) {
      const PositionSourceRecord& record = corpus.records[next_record];
      if (record.trajectory_index < begin) {
        throw std::invalid_argument{"position source records are not in trajectory order"};
      }
      output << "{\"p1\":\"";
      append_hex64(output, record.player_one);
      output << "\",\"p2\":\"";
      append_hex64(output, record.player_two);
      output << "\",\"source_id\":\"";
      append_hex64(output, record.source_id);
      output << "\",\"family_id\":\"";
      append_hex64(output, record.family_id);
      output << "\",\"trajectory_id\":\"";
      append_hex64(output, record.trajectory_id);
      output << "\",\"parent_id\":\"";
      append_hex64(output, record.parent_id);
      output << "\",\"trajectory_index\":" << record.trajectory_index
             << ",\"policy_id\":" << static_cast<std::uint16_t>(record.policy)
             << ",\"sample_index\":" << record.sample_index
             << ",\"split\":" << static_cast<unsigned>(record.split)
             << ",\"ply\":" << static_cast<unsigned>(record.ply) << "}\n";
      ++next_record;
    }
    SerializedSourceShard shard{
        .shard_index = shard_index,
        .trajectory_begin = begin,
        .trajectory_end = end,
        .record_count = next_record - record_begin,
        .bytes = output.str(),
    };
    shard.digest = labeling::sha256(shard.bytes);
    shards.push_back(std::move(shard));
  }
  if (next_record != corpus.records.size()) {
    throw std::invalid_argument{"position source contains out-of-range trajectory records"};
  }
  return shards;
}

std::string serialize_source_manifest(const PositionSourceCorpus& corpus,
                                      std::span<const SerializedSourceShard> shards) {
  if (shards.size() != corpus.options.shard_count) {
    throw std::invalid_argument{"source manifest shard count does not match its configuration"};
  }
  std::array<std::size_t, 4> policy_counts{};
  std::array<std::size_t, 4> split_counts{};
  for (const PositionSourceRecord& record : corpus.records) {
    if (!valid_policy(record.policy) || !valid_split(record.split)) {
      throw std::invalid_argument{"source manifest encountered invalid record metadata"};
    }
    ++policy_counts[static_cast<std::uint16_t>(record.policy) - 1];
    ++split_counts[static_cast<std::uint8_t>(record.split)];
  }

  std::ostringstream output;
  output.imbue(std::locale::classic());
  output << "{\n"
         << "  \"schema\": \"poe2-position-source\",\n"
         << "  \"schema_version\": " << kPositionSourceSchemaVersion << ",\n"
         << "  \"corpus\": {\"id\": \"" << json_escape(corpus.options.corpus_id)
         << "\", \"digest\": \"" << qualified_digest(labeling::sha256(corpus.options.corpus_id))
         << "\"},\n"
         << "  \"generator\": {\n"
         << "    \"version\": " << kGeneratorVersion << ",\n"
         << "    \"rng\": \"" << kRngName << "\",\n"
         << "    \"seed\": \"";
  append_hex64(output, corpus.options.seed);
  output << "\",\n"
         << "    \"trajectory_count\": " << corpus.options.trajectory_count << ",\n"
         << "    \"samples_per_trajectory\": " << corpus.options.samples_per_trajectory << ",\n"
         << "    \"shard_count\": " << corpus.options.shard_count << ",\n"
         << "    \"workers_requested\": " << corpus.options.workers << ",\n"
         << "    \"workers_used\": " << corpus.workers_used << ",\n"
         << "    \"search_nodes\": " << corpus.options.search_nodes << ",\n"
         << "    \"search_hash_bytes\": " << corpus.options.search_hash_bytes << ",\n"
         << "    \"noise_percent\": " << static_cast<unsigned>(corpus.options.noise_percent)
         << ",\n"
         << "    \"policy_weights\": {\"random\": " << corpus.options.policy_weights.random
         << ", \"immediate_gain\": " << corpus.options.policy_weights.immediate_gain
         << ", \"opponent_aware\": " << corpus.options.policy_weights.opponent_aware
         << ", \"noisy_search\": " << corpus.options.policy_weights.noisy_search << "}\n"
         << "  },\n"
         << "  \"build\": {\n"
         << "    \"git_commit\": \"" << json_escape(labeling::build::kGitCommit) << "\",\n"
         << "    \"git_dirty\": " << (labeling::build::kGitDirty ? "true" : "false") << ",\n"
         << "    \"project_version\": \"" << json_escape(labeling::build::kProjectVersion)
         << "\",\n"
         << "    \"compiler_id\": \"" << json_escape(labeling::build::kCompilerId) << "\",\n"
         << "    \"compiler_version\": \"" << json_escape(labeling::build::kCompilerVersion)
         << "\",\n"
         << "    \"build_type\": \"" << json_escape(labeling::build::kBuildType) << "\",\n"
         << "    \"target_processor\": \"" << json_escape(labeling::build::kTargetProcessor)
         << "\",\n"
         << "    \"native_architecture\": "
         << (labeling::build::kNativeArchitecture ? "true" : "false") << "\n"
         << "  },\n"
         << "  \"results\": {\n"
         << "    \"records\": " << corpus.records.size() << ",\n"
         << "    \"duplicate_positions\": " << corpus.duplicate_positions << ",\n"
         << "    \"policy_counts\": {\"random\": " << policy_counts[0]
         << ", \"immediate_gain\": " << policy_counts[1]
         << ", \"opponent_aware\": " << policy_counts[2]
         << ", \"noisy_search\": " << policy_counts[3] << "},\n"
         << "    \"split_counts\": {\"train\": " << split_counts[1]
         << ", \"validation\": " << split_counts[2] << ", \"test\": " << split_counts[3] << "}\n"
         << "  },\n"
         << "  \"shards\": [\n";
  for (std::size_t index = 0; index < shards.size(); ++index) {
    const SerializedSourceShard& shard = shards[index];
    output << "    {\"index\": " << shard.shard_index << ", \"name\": \""
           << shard_file_name(shard.shard_index, shard.digest) << "\", \"digest\": \""
           << qualified_digest(shard.digest) << "\", \"records\": " << shard.record_count
           << ", \"trajectory_begin\": " << shard.trajectory_begin
           << ", \"trajectory_end\": " << shard.trajectory_end << '}';
    output << (index + 1 == shards.size() ? "\n" : ",\n");
  }
  output << "  ]\n"
         << "}\n";
  return output.str();
}

SourceOutput reserve_source_output(const fs::path& directory) {
  if (directory.empty()) {
    throw std::invalid_argument{"position source output directory must not be empty"};
  }
  create_parent_directories(directory);
  std::error_code error;
  const bool created = fs::create_directory(directory, error);
  if (!created) {
    if (error) {
      throw fs::filesystem_error{"failed to reserve position source output", directory, error};
    }
    throw std::invalid_argument{"position source output directory already exists"};
  }
  SourceOutput output{directory};
  write_text_file(directory / kSourceIncompleteMarkerName, "poe2-position-source\n");
  fs::create_directory(directory / kSourceShardDirectoryName);
  return output;
}

void write_position_source(SourceOutput& output, const PositionSourceCorpus& corpus) {
  if (output.committed_) {
    throw std::logic_error{"position source output is already committed"};
  }
  const fs::path incomplete = output.directory_ / kSourceIncompleteMarkerName;
  const fs::path complete = output.directory_ / kSourceCompleteMarkerName;
  const fs::path manifest = output.directory_ / kSourceManifestFileName;
  const fs::path manifest_temporary = output.directory_ / "manifest.json.tmp";
  const fs::path shard_directory = output.directory_ / kSourceShardDirectoryName;
  if (!fs::is_regular_file(incomplete) || fs::exists(complete) || fs::exists(manifest) ||
      fs::exists(manifest_temporary) || !fs::is_directory(shard_directory) ||
      !fs::is_empty(shard_directory)) {
    throw std::runtime_error{"position source reservation is not empty and incomplete"};
  }

  const std::vector<SerializedSourceShard> shards = serialize_source_shards(corpus);
  const std::string manifest_bytes = serialize_source_manifest(corpus, shards);
  const labeling::Sha256Digest manifest_digest = labeling::sha256(manifest_bytes);
  for (const SerializedSourceShard& shard : shards) {
    const fs::path final_path = shard_directory / shard_file_name(shard.shard_index, shard.digest);
    const fs::path temporary_path = final_path.string() + ".tmp";
    write_text_file(temporary_path, shard.bytes);
    fs::rename(temporary_path, final_path);
  }
  write_text_file(manifest_temporary, manifest_bytes);
  fs::rename(manifest_temporary, manifest);
  write_text_file(incomplete, "poe2-position-source\nmanifest_sha256=" +
                                  labeling::sha256_text(manifest_digest) + "\n");
  fs::rename(incomplete, complete);
  output.committed_ = true;
}

ReadSourceShard read_position_source_shard(const fs::path& directory, std::uint32_t shard_index) {
  const fs::path complete = directory / kSourceCompleteMarkerName;
  const fs::path incomplete = directory / kSourceIncompleteMarkerName;
  const fs::path manifest = directory / kSourceManifestFileName;
  const fs::path shard_directory = directory / kSourceShardDirectoryName;
  if (!fs::is_directory(directory) || !fs::is_regular_file(complete) || fs::exists(incomplete) ||
      !fs::is_regular_file(manifest) || !fs::is_directory(shard_directory)) {
    throw std::runtime_error{"position source is missing a complete artifact layout"};
  }
  const std::optional<labeling::Sha256Digest> expected_manifest_digest =
      parse_complete_marker(read_text_file(complete));
  if (!expected_manifest_digest.has_value()) {
    throw std::runtime_error{"position source COMPLETE marker is malformed"};
  }
  const std::string manifest_bytes = read_text_file(manifest);
  if (labeling::sha256(manifest_bytes) != *expected_manifest_digest ||
      manifest_bytes.find("\"schema\": \"poe2-position-source\"") == std::string::npos ||
      manifest_bytes.find("\"schema_version\": 1") == std::string::npos) {
    throw std::runtime_error{"position source manifest is invalid or has the wrong digest"};
  }

  std::optional<fs::path> shard_path;
  for (const fs::directory_entry& entry : fs::directory_iterator(shard_directory)) {
    if (entry.is_regular_file() &&
        is_requested_shard_name(entry.path().filename().string(), shard_index)) {
      if (shard_path.has_value()) {
        throw std::runtime_error{"position source contains multiple files for one shard"};
      }
      shard_path = entry.path();
    }
  }
  if (!shard_path.has_value()) {
    throw std::runtime_error{"requested position source shard does not exist"};
  }

  const std::string shard_bytes = read_text_file(*shard_path);
  const labeling::Sha256Digest expected_shard_digest =
      digest_from_shard_name(shard_path->filename().string());
  if (labeling::sha256(shard_bytes) != expected_shard_digest) {
    throw std::runtime_error{"position source shard digest does not match its file name"};
  }
  const std::string manifest_name = "\"name\": \"" + shard_path->filename().string() + "\"";
  const std::string manifest_digest =
      "\"digest\": \"" + qualified_digest(expected_shard_digest) + "\"";
  if (manifest_bytes.find(manifest_name) == std::string::npos ||
      manifest_bytes.find(manifest_digest) == std::string::npos) {
    throw std::runtime_error{"position source shard is not authenticated by its manifest"};
  }

  std::istringstream lines{shard_bytes};
  lines.imbue(std::locale::classic());
  std::string line;
  if (!std::getline(lines, line)) {
    throw std::runtime_error{"position source shard is empty"};
  }
  const ParsedHeader header = parse_header(line);
  if (header.shard_count == 0 || header.shard_index >= header.shard_count) {
    throw std::runtime_error{"position source shard header is inconsistent"};
  }
  const auto [expected_begin, expected_end] =
      trajectory_range(header.trajectory_count, header.shard_count, header.shard_index);
  if (header.shard_index != shard_index || header.trajectory_begin >= header.trajectory_end ||
      header.trajectory_end > header.trajectory_count || header.samples_per_trajectory == 0 ||
      header.samples_per_trajectory > kPhaseBuckets.size() || header.noise_percent > 100 ||
      policy_weight_sum(header.policy_weights) == 0 ||
      (header.policy_weights.noisy_search != 0 && header.search_nodes == 0) ||
      header.trajectory_begin != expected_begin || header.trajectory_end != expected_end) {
    throw std::runtime_error{"position source shard header is inconsistent"};
  }

  ReadSourceShard result{
      .source =
          labeling::LabelSource{
              .corpus_id = header.corpus_id,
              .source_name = shard_path->filename().string(),
              .source_digest = expected_shard_digest,
              .shard_index = header.shard_index,
              .shard_count = header.shard_count,
          },
  };
  const std::uint64_t trajectory_span = header.trajectory_end - header.trajectory_begin;
  if (trajectory_span > std::numeric_limits<std::size_t>::max() / header.samples_per_trajectory) {
    throw std::runtime_error{"position source shard record count is not representable"};
  }
  result.inputs.reserve(static_cast<std::size_t>(trajectory_span) * header.samples_per_trajectory);
  std::uint64_t previous_trajectory = header.trajectory_begin;
  std::uint16_t previous_sample = 0;
  std::uint64_t previous_family = 0;
  std::uint64_t previous_trajectory_id = 0;
  SourcePolicy previous_policy = SourcePolicy::kRandom;
  labeling::DatasetSplit previous_split = labeling::DatasetSplit::kUnspecified;
  std::uint8_t previous_ply = 0;
  std::uint16_t phase_mask = 0;
  bool first_record = true;
  std::size_t source_ordinal = 0;
  while (std::getline(lines, line)) {
    if (line.empty()) {
      throw std::runtime_error{"position source shard contains an empty record line"};
    }
    PositionSourceRecord record = parse_record(line);
    const std::uint32_t policy_weight = [&] {
      switch (record.policy) {
        case SourcePolicy::kRandom:
          return header.policy_weights.random;
        case SourcePolicy::kImmediateGain:
          return header.policy_weights.immediate_gain;
        case SourcePolicy::kOpponentAware:
          return header.policy_weights.opponent_aware;
        case SourcePolicy::kNoisySearch:
          return header.policy_weights.noisy_search;
      }
      return std::uint32_t{0};
    }();
    const int phase = phase_for_ply(record.ply);
    if (record.trajectory_index < header.trajectory_begin ||
        record.trajectory_index >= header.trajectory_end || !valid_policy(record.policy) ||
        !valid_split(record.split) || record.sample_index >= header.samples_per_trajectory ||
        record.source_id != source_id(record) || policy_weight == 0 || phase < 0 ||
        record.family_id !=
            nonzero_id(mix64(header.seed ^ mix64(record.trajectory_index ^ kFamilySalt))) ||
        record.trajectory_id !=
            nonzero_id(mix64(header.seed ^ mix64(record.trajectory_index ^ kTrajectorySalt))) ||
        record.parent_id != 0) {
      throw std::runtime_error{"position source record metadata is invalid"};
    }
    if (!first_record) {
      if (record.trajectory_index < previous_trajectory ||
          (record.trajectory_index == previous_trajectory &&
           record.sample_index != previous_sample + 1) ||
          (record.trajectory_index != previous_trajectory && record.sample_index != 0)) {
        throw std::runtime_error{"position source records are not in trajectory/sample order"};
      }
      if (record.trajectory_index == previous_trajectory &&
          (record.family_id != previous_family || record.trajectory_id != previous_trajectory_id ||
           record.policy != previous_policy || record.split != previous_split)) {
        throw std::runtime_error{"position source trajectory metadata changes between samples"};
      }
      if (record.trajectory_index == previous_trajectory && record.ply <= previous_ply) {
        throw std::runtime_error{"position source trajectory samples are not in phase order"};
      }
    } else if (record.trajectory_index != header.trajectory_begin || record.sample_index != 0) {
      throw std::runtime_error{"position source shard does not begin at its declared trajectory"};
    }
    if (first_record || record.trajectory_index != previous_trajectory) {
      phase_mask = 0;
    }
    const std::uint16_t phase_bit = std::uint16_t{1} << phase;
    if ((phase_mask & phase_bit) != 0) {
      throw std::runtime_error{"position source trajectory repeats a phase bucket"};
    }
    phase_mask |= phase_bit;
    if (source_ordinal > std::numeric_limits<std::uint32_t>::max() - 1) {
      throw std::runtime_error{"position source shard contains too many records"};
    }
    Position position = reconstruct_position(record);
    result.inputs.push_back(labeling::LabelInput{
        .position = position,
        .source_id = record.source_id,
        .family_id = record.family_id,
        .trajectory_id = record.trajectory_id,
        .parent_id = record.parent_id,
        .trajectory_index = record.trajectory_index,
        .source_line = static_cast<std::uint32_t>(source_ordinal + 2),
        .source_ordinal = static_cast<std::uint32_t>(source_ordinal),
        .policy_id = static_cast<std::uint16_t>(record.policy),
        .sample_index = record.sample_index,
        .split = record.split,
    });
    previous_trajectory = record.trajectory_index;
    previous_sample = record.sample_index;
    previous_family = record.family_id;
    previous_trajectory_id = record.trajectory_id;
    previous_policy = record.policy;
    previous_split = record.split;
    previous_ply = record.ply;
    first_record = false;
    ++source_ordinal;
  }
  const std::size_t expected_records =
      static_cast<std::size_t>(trajectory_span) * header.samples_per_trajectory;
  if (result.inputs.size() != expected_records ||
      previous_trajectory + 1 != header.trajectory_end ||
      previous_sample + 1 != header.samples_per_trajectory) {
    throw std::runtime_error{"position source shard is missing declared trajectory samples"};
  }
  return result;
}

}  // namespace poe2::minimax::position_source
