#include "poe2/minimax/feature_data.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "poe2/minimax/evaluation.hpp"
#include "poe2/minimax/labeling_build.hpp"
#include "poe2/minimax/position_source.hpp"
#include "poe2/symmetry.hpp"

namespace poe2::minimax::feature_data {

namespace {

namespace fs = std::filesystem;

constexpr std::array<std::uint8_t, 8> kFeatureMagic{{'P', 'O', 'E', '2', 'F', 'T', 'R', 0}};
constexpr std::array<std::uint8_t, 8> kLabelMagic{{'P', 'O', 'E', '2', 'L', 'B', 'L', 0}};
constexpr std::uint32_t kEndianMarker = UINT32_C(0x01020304);

struct ScoringLine {
  std::array<std::uint8_t, kBoardSize> cells{};
  std::uint8_t length = 0;
};

[[nodiscard]] consteval std::array<ScoringLine, kScoringLineCount> make_scoring_lines() {
  std::array<ScoringLine, kScoringLineCount> lines{};
  std::size_t next = 0;
  const auto add = [&lines, &next](int row, int column, int row_step, int column_step) {
    ScoringLine line;
    while (row >= 0 && row < kBoardSize && column >= 0 && column < kBoardSize) {
      line.cells[line.length++] = static_cast<std::uint8_t>(square_index(Square{row, column}));
      row += row_step;
      column += column_step;
    }
    lines[next++] = line;
  };

  for (int row = 0; row < kBoardSize; ++row) {
    add(row, 0, 0, 1);
  }
  for (int column = 0; column < kBoardSize; ++column) {
    add(0, column, 1, 0);
  }
  for (int column = 0; column < kBoardSize - 1; ++column) {
    add(0, column, 1, 1);
  }
  for (int row = 1; row < kBoardSize - 1; ++row) {
    add(row, 0, 1, 1);
  }
  for (int column = 1; column < kBoardSize; ++column) {
    add(0, column, 1, -1);
  }
  for (int row = 1; row < kBoardSize - 1; ++row) {
    add(row, kBoardSize - 1, 1, -1);
  }
  return lines;
}

inline constexpr std::array<ScoringLine, kScoringLineCount> kScoringLines = make_scoring_lines();

[[nodiscard]] consteval std::array<std::uint16_t, kBoardSize + 1> make_pattern_offsets() {
  std::array<std::uint16_t, kBoardSize + 1> offsets{};
  std::uint16_t offset = 0;
  std::uint16_t patterns = 1;
  for (int length = 0; length <= kBoardSize; ++length) {
    if (length >= 2) {
      offsets[length] = offset;
      offset = static_cast<std::uint16_t>(offset + patterns);
    }
    patterns = static_cast<std::uint16_t>(patterns * 3);
  }
  return offsets;
}

inline constexpr std::array<std::uint16_t, kBoardSize + 1> kPatternOffsets = make_pattern_offsets();

static_assert(kScoringLines.size() == kScoringLineCount);
static_assert(kPatternOffsets[7] + 2187 == 3276);
static_assert(sizeof(Score) <= sizeof(std::int32_t));

[[nodiscard]] std::vector<std::uint8_t> read_binary_file(const fs::path& path) {
  std::ifstream input{path, std::ios::binary};
  if (!input) {
    throw std::runtime_error{"failed to read " + path.string()};
  }
  return {std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
}

[[nodiscard]] std::string read_text_file(const fs::path& path) {
  std::ifstream input{path, std::ios::binary};
  if (!input) {
    throw std::runtime_error{"failed to read " + path.string()};
  }
  return {std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
}

void write_binary_file(const fs::path& path, std::span<const std::uint8_t> bytes) {
  std::ofstream output{path, std::ios::binary | std::ios::trunc};
  if (!output) {
    throw std::runtime_error{"failed to create " + path.string()};
  }
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  if (!output) {
    throw std::runtime_error{"failed to write " + path.string()};
  }
}

void write_text_file(const fs::path& path, std::string_view text) {
  write_binary_file(path,
                    std::span{reinterpret_cast<const std::uint8_t*>(text.data()), text.size()});
}

void create_parent_directories(const fs::path& path) {
  const fs::path parent = path.parent_path();
  if (parent.empty()) {
    return;
  }
  std::error_code error;
  fs::create_directories(parent, error);
  if (error) {
    throw fs::filesystem_error{"failed to create feature output parent", parent, error};
  }
}

[[nodiscard]] int hex_digit(char value) noexcept {
  if (value >= '0' && value <= '9') {
    return value - '0';
  }
  if (value >= 'a' && value <= 'f') {
    return value - 'a' + 10;
  }
  if (value >= 'A' && value <= 'F') {
    return value - 'A' + 10;
  }
  return -1;
}

[[nodiscard]] labeling::Sha256Digest parse_digest(std::string_view encoded,
                                                  std::string_view field) {
  if (encoded.size() != 64) {
    throw std::runtime_error{std::string{field} + " has the wrong digest width"};
  }
  labeling::Sha256Digest digest{};
  for (std::size_t index = 0; index < digest.size(); ++index) {
    const int high = hex_digit(encoded[index * 2]);
    const int low = hex_digit(encoded[index * 2 + 1]);
    if (high < 0 || low < 0) {
      throw std::runtime_error{std::string{field} + " is not hexadecimal"};
    }
    digest[index] = static_cast<std::uint8_t>((high << 4) | low);
  }
  return digest;
}

struct ArtifactDigests {
  labeling::Sha256Digest binary{};
  labeling::Sha256Digest manifest{};
};

[[nodiscard]] ArtifactDigests parse_label_complete_marker(std::string_view marker) {
  constexpr std::string_view kHeader = "poe2-minimax-labels\n";
  constexpr std::string_view kBinary = "binary_sha256=";
  constexpr std::string_view kManifest = "manifest_sha256=";
  if (!marker.starts_with(kHeader)) {
    throw std::runtime_error{"label COMPLETE marker has the wrong header"};
  }
  marker.remove_prefix(kHeader.size());
  const std::size_t first_end = marker.find('\n');
  if (first_end == std::string_view::npos || !marker.substr(0, first_end).starts_with(kBinary)) {
    throw std::runtime_error{"label COMPLETE marker lacks its binary digest"};
  }
  const labeling::Sha256Digest binary =
      parse_digest(marker.substr(kBinary.size(), first_end - kBinary.size()), "binary digest");
  marker.remove_prefix(first_end + 1);
  const std::size_t second_end = marker.find('\n');
  if (second_end == std::string_view::npos ||
      !marker.substr(0, second_end).starts_with(kManifest) || second_end + 1 != marker.size()) {
    throw std::runtime_error{"label COMPLETE marker lacks its manifest digest"};
  }
  return ArtifactDigests{
      .binary = binary,
      .manifest = parse_digest(marker.substr(kManifest.size(), second_end - kManifest.size()),
                               "manifest digest"),
  };
}

[[nodiscard]] std::size_t unique_field_offset(std::string_view text, std::string_view prefix) {
  const std::size_t offset = text.find(prefix);
  if (offset == std::string_view::npos ||
      text.find(prefix, offset + prefix.size()) != std::string_view::npos) {
    throw std::runtime_error{"label manifest field is missing or ambiguous: " +
                             std::string{prefix}};
  }
  return offset + prefix.size();
}

[[nodiscard]] std::string manifest_string(std::string_view text, std::string_view prefix) {
  const std::size_t begin = unique_field_offset(text, prefix);
  const std::size_t end = text.find('"', begin);
  if (end == std::string_view::npos) {
    throw std::runtime_error{"label manifest string is unterminated"};
  }
  return std::string{text.substr(begin, end - begin)};
}

template <typename Integer>
  requires std::is_unsigned_v<Integer>
[[nodiscard]] Integer manifest_unsigned(std::string_view text, std::string_view prefix) {
  const std::size_t begin = unique_field_offset(text, prefix);
  Integer value = 0;
  const char* const first = text.data() + begin;
  const char* const last = text.data() + text.size();
  const auto result = std::from_chars(first, last, value);
  if (result.ec != std::errc{} || result.ptr == first) {
    throw std::runtime_error{"label manifest integer is invalid"};
  }
  return value;
}

[[nodiscard]] bool manifest_bool(std::string_view text, std::string_view prefix) {
  const std::size_t begin = unique_field_offset(text, prefix);
  if (text.substr(begin).starts_with("true")) {
    return true;
  }
  if (text.substr(begin).starts_with("false")) {
    return false;
  }
  throw std::runtime_error{"label manifest boolean is invalid"};
}

class ByteReader final {
 public:
  explicit ByteReader(std::span<const std::uint8_t> bytes) : bytes_(bytes) {}

  template <typename Integer>
    requires std::is_unsigned_v<Integer>
  [[nodiscard]] Integer unsigned_value() {
    require(sizeof(Integer));
    Integer value = 0;
    for (std::size_t index = 0; index < sizeof(Integer); ++index) {
      value |= static_cast<Integer>(bytes_[offset_ + index]) << (index * 8);
    }
    offset_ += sizeof(Integer);
    return value;
  }

  [[nodiscard]] std::int32_t signed32() {
    return std::bit_cast<std::int32_t>(unsigned_value<std::uint32_t>());
  }

  [[nodiscard]] std::uint8_t byte() { return unsigned_value<std::uint8_t>(); }

  [[nodiscard]] labeling::Sha256Digest digest() {
    require(32);
    labeling::Sha256Digest result{};
    std::copy_n(bytes_.begin() + static_cast<std::ptrdiff_t>(offset_), result.size(),
                result.begin());
    offset_ += result.size();
    return result;
  }

  void expect(std::span<const std::uint8_t> expected) {
    require(expected.size());
    if (!std::equal(expected.begin(), expected.end(),
                    bytes_.begin() + static_cast<std::ptrdiff_t>(offset_))) {
      throw std::runtime_error{"binary artifact magic is wrong"};
    }
    offset_ += expected.size();
  }

  [[nodiscard]] std::size_t offset() const noexcept { return offset_; }
  [[nodiscard]] bool finished() const noexcept { return offset_ == bytes_.size(); }

 private:
  void require(std::size_t count) const {
    if (count > bytes_.size() - offset_) {
      throw std::runtime_error{"binary artifact is truncated"};
    }
  }

  std::span<const std::uint8_t> bytes_;
  std::size_t offset_ = 0;
};

struct LoadedLabelShard {
  labeling::LabelSource source;
  labeling::Sha256Digest corpus_digest{};
  ArtifactDigests digests;
  LabelProvenance provenance;
  std::size_t input_count = 0;
  std::vector<labeling::LabelRecord> records;
};

[[nodiscard]] LabelProvenance parse_label_provenance(std::string_view manifest) {
  if (manifest.find("\"schema\": \"poe2-minimax-labels\"") == std::string_view::npos ||
      manifest.find("\"schema_version\": 2") == std::string_view::npos ||
      manifest.find("\"evaluator\": \"b\"") == std::string_view::npos ||
      manifest.find("\"symmetry\": true") == std::string_view::npos ||
      manifest.find("\"two_ply_closure\": true") == std::string_view::npos) {
    throw std::runtime_error{"label manifest has unsupported semantics"};
  }
  const std::uint64_t hash_bytes =
      manifest_unsigned<std::uint64_t>(manifest, "\"hash_bytes_effective\": ");
  const std::uint64_t workers =
      manifest_unsigned<std::uint64_t>(manifest, "\"workers_requested\": ");
  if (hash_bytes > std::numeric_limits<std::size_t>::max() ||
      workers > std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error{"label manifest sizes are not representable"};
  }
  return LabelProvenance{
      .git_commit = manifest_string(manifest, "\"git_commit\": \""),
      .git_dirty = manifest_bool(manifest, "\"git_dirty\": "),
      .mode = manifest_string(manifest, "\"mode\": \""),
      .node_limit = manifest_unsigned<std::uint64_t>(manifest, "\"node_limit\": "),
      .hash_bytes_effective = static_cast<std::size_t>(hash_bytes),
      .workers_requested = static_cast<std::size_t>(workers),
      .target_selection = manifest_string(manifest, "\"target_selection\": \""),
  };
}

[[nodiscard]] LoadedLabelShard load_label_shard(const fs::path& directory) {
  const fs::path complete = directory / labeling::kCompleteMarkerName;
  const fs::path incomplete = directory / labeling::kIncompleteMarkerName;
  const fs::path binary_path = directory / labeling::kBinaryFileName;
  const fs::path manifest_path = directory / labeling::kManifestFileName;
  if (!fs::is_directory(directory) || !fs::is_regular_file(complete) || fs::exists(incomplete) ||
      !fs::is_regular_file(binary_path) || !fs::is_regular_file(manifest_path)) {
    throw std::runtime_error{"label shard is not a completed artifact: " + directory.string()};
  }

  const ArtifactDigests digests = parse_label_complete_marker(read_text_file(complete));
  const std::vector<std::uint8_t> binary = read_binary_file(binary_path);
  const std::string manifest = read_text_file(manifest_path);
  if (labeling::sha256(std::span<const std::uint8_t>{binary}) != digests.binary ||
      labeling::sha256(manifest) != digests.manifest) {
    throw std::runtime_error{"label shard digest verification failed: " + directory.string()};
  }

  ByteReader reader{binary};
  reader.expect(kLabelMagic);
  const std::uint32_t schema = reader.unsigned_value<std::uint32_t>();
  const std::uint32_t header_size = reader.unsigned_value<std::uint32_t>();
  const std::uint32_t record_size = reader.unsigned_value<std::uint32_t>();
  const std::uint32_t endian = reader.unsigned_value<std::uint32_t>();
  const std::uint64_t record_count = reader.unsigned_value<std::uint64_t>();
  const std::uint64_t input_count = reader.unsigned_value<std::uint64_t>();
  const labeling::Sha256Digest source_digest = reader.digest();
  const labeling::Sha256Digest corpus_digest = reader.digest();
  const std::uint32_t shard_index = reader.unsigned_value<std::uint32_t>();
  const std::uint32_t shard_count = reader.unsigned_value<std::uint32_t>();
  if (schema != labeling::kLabelDatasetSchemaVersion ||
      header_size != labeling::kLabelDatasetHeaderSize ||
      record_size != labeling::kLabelDatasetRecordSize || endian != kEndianMarker ||
      shard_count == 0 || shard_index >= shard_count || record_count > input_count ||
      input_count > std::numeric_limits<std::size_t>::max() ||
      record_count > std::numeric_limits<std::size_t>::max() ||
      binary.size() != header_size + record_count * record_size) {
    throw std::runtime_error{"label binary header is invalid: " + directory.string()};
  }

  const std::string corpus_id = manifest_string(manifest, "\"id\": \"");
  if (labeling::sha256(corpus_id) != corpus_digest) {
    throw std::runtime_error{"label corpus ID does not match its binary digest"};
  }
  LoadedLabelShard result{
      .source =
          labeling::LabelSource{
              .corpus_id = corpus_id,
              .source_digest = source_digest,
              .shard_index = shard_index,
              .shard_count = shard_count,
          },
      .corpus_digest = corpus_digest,
      .digests = digests,
      .provenance = parse_label_provenance(manifest),
      .input_count = static_cast<std::size_t>(input_count),
  };
  result.records.reserve(static_cast<std::size_t>(record_count));
  for (std::uint64_t index = 0; index < record_count; ++index) {
    const std::size_t start = reader.offset();
    labeling::LabelRecord record;
    record.player_one = reader.unsigned_value<std::uint64_t>();
    record.player_two = reader.unsigned_value<std::uint64_t>();
    record.canonical_key.low = reader.unsigned_value<std::uint64_t>();
    record.canonical_key.high = reader.unsigned_value<std::uint64_t>();
    record.source_id = reader.unsigned_value<std::uint64_t>();
    record.family_id = reader.unsigned_value<std::uint64_t>();
    record.trajectory_id = reader.unsigned_value<std::uint64_t>();
    record.parent_id = reader.unsigned_value<std::uint64_t>();
    record.trajectory_index = reader.unsigned_value<std::uint64_t>();
    record.nodes = reader.unsigned_value<std::uint64_t>();
    record.completed_nodes = reader.unsigned_value<std::uint64_t>();
    record.source_line = reader.unsigned_value<std::uint32_t>();
    record.source_ordinal = reader.unsigned_value<std::uint32_t>();
    record.value = reader.signed32();
    record.ply = reader.byte();
    record.side_to_move = static_cast<Player>(reader.byte());
    record.mode = static_cast<labeling::LabelMode>(reader.byte());
    record.completed_depth = reader.byte();
    record.attempted_depth = reader.byte();
    record.terminal_depth = reader.byte();
    record.best_move_index = reader.byte();
    record.split = static_cast<labeling::DatasetSplit>(reader.byte());
    record.policy_id = reader.unsigned_value<std::uint16_t>();
    record.sample_index = reader.unsigned_value<std::uint16_t>();
    record.deepest_completed_nodes = reader.unsigned_value<std::uint64_t>();
    record.deepest_value = reader.signed32();
    record.deepest_completed_depth = reader.byte();
    record.deepest_best_move_index = reader.byte();
    record.previous_completed_nodes = reader.unsigned_value<std::uint64_t>();
    record.previous_value = reader.signed32();
    record.previous_completed_depth = reader.byte();
    record.previous_best_move_index = reader.byte();
    if (reader.offset() - start != labeling::kLabelDatasetRecordSize) {
      throw std::logic_error{"label record reader has the wrong width"};
    }
    result.records.push_back(record);
  }
  if (!reader.finished()) {
    throw std::runtime_error{"label binary contains trailing bytes"};
  }
  return result;
}

[[nodiscard]] bool valid_split(labeling::DatasetSplit split) noexcept {
  return split == labeling::DatasetSplit::kTrain || split == labeling::DatasetSplit::kValidation ||
         split == labeling::DatasetSplit::kTest;
}

void validate_label(const labeling::LabelRecord& label, const labeling::LabelInput& source,
                    std::uint32_t shard_index, const LabelProvenance& provenance) {
  const Position& position = source.position;
  const PositionKey canonical = canonicalize_position_key(position.key()).key;
  const labeling::LabelMode manifest_mode =
      provenance.mode == "teacher" ? labeling::LabelMode::kTeacher
      : provenance.mode == "exact" ? labeling::LabelMode::kExact
                                   : static_cast<labeling::LabelMode>(0);
  if (manifest_mode == static_cast<labeling::LabelMode>(0) ||
      label.player_one != position.board().bits(Player::kOne) ||
      label.player_two != position.board().bits(Player::kTwo) || label.canonical_key != canonical ||
      label.source_id != source.source_id || label.family_id != source.family_id ||
      label.trajectory_id != source.trajectory_id || label.parent_id != source.parent_id ||
      label.trajectory_index != source.trajectory_index ||
      label.source_ordinal != source.source_ordinal || label.source_line != source.source_line ||
      label.ply != position.ply() || label.side_to_move != position.side_to_move() ||
      label.mode != manifest_mode || label.policy_id != source.policy_id ||
      label.sample_index != source.sample_index || label.split != source.split ||
      !valid_split(label.split) || label.completed_depth == 0 ||
      label.completed_depth > label.deepest_completed_depth ||
      label.deepest_completed_depth > label.attempted_depth ||
      label.attempted_depth > label.terminal_depth ||
      label.completed_depth % 2 != label.terminal_depth % 2 || label.completed_nodes == 0 ||
      label.completed_nodes > label.nodes || label.nodes > provenance.node_limit ||
      label.best_move_index >= kCellCount ||
      (position.board().occupied() & (Bitboard{1} << label.best_move_index)) != 0) {
    throw std::runtime_error{"label/source mismatch in shard " + std::to_string(shard_index) +
                             " record " + std::to_string(source.source_ordinal)};
  }
}

struct PositionKeyHash {
  [[nodiscard]] std::size_t operator()(PositionKey key) const noexcept {
    return static_cast<std::size_t>(position_key_hash(key));
  }
};

struct SelectedCandidate {
  labeling::LabelRecord label;
  Position position;
  std::uint32_t source_shard = 0;
  std::uint32_t duplicate_count = 1;
  bool varying_label = false;
};

[[nodiscard]] auto candidate_identity(const labeling::LabelRecord& label,
                                      std::uint32_t shard_index) noexcept {
  return std::tuple{label.trajectory_index, label.sample_index, label.source_id, shard_index,
                    label.source_ordinal};
}

[[nodiscard]] bool prefer_candidate(const labeling::LabelRecord& candidate,
                                    std::uint32_t candidate_shard,
                                    const SelectedCandidate& selected) noexcept {
  if (candidate.completed_depth != selected.label.completed_depth) {
    return candidate.completed_depth > selected.label.completed_depth;
  }
  return candidate_identity(candidate, candidate_shard) <
         candidate_identity(selected.label, selected.source_shard);
}

[[nodiscard]] FeatureRecord make_feature_record(const SelectedCandidate& candidate) {
  const PositionFeatures features = extract_position_features(candidate.position);
  const labeling::LabelRecord& label = candidate.label;
  std::uint8_t flags = 0;
  if (label.completed_depth == label.terminal_depth) {
    flags |= kTerminalLabelFlag;
  }
  if (label.completed_depth < label.deepest_completed_depth) {
    flags |= kParityBackoffFlag;
  }
  return FeatureRecord{
      .player_one = label.player_one,
      .player_two = label.player_two,
      .canonical_key = label.canonical_key,
      .source_id = label.source_id,
      .family_id = label.family_id,
      .trajectory_id = label.trajectory_id,
      .parent_id = label.parent_id,
      .trajectory_index = label.trajectory_index,
      .nodes = label.nodes,
      .completed_nodes = label.completed_nodes,
      .deepest_completed_nodes = label.deepest_completed_nodes,
      .previous_completed_nodes = label.previous_completed_nodes,
      .source_shard = candidate.source_shard,
      .source_ordinal = label.source_ordinal,
      .teacher_value = label.value,
      .deepest_value = label.deepest_value,
      .previous_value = label.previous_value,
      .normalized_value = features.normalized_value,
      .b_value = features.b_value,
      .residual = label.value - features.b_value,
      .duplicate_count = candidate.duplicate_count,
      .policy_id = label.policy_id,
      .sample_index = label.sample_index,
      .ply = label.ply,
      .side_to_move = label.side_to_move,
      .split = label.split,
      .mode = label.mode,
      .completed_depth = label.completed_depth,
      .deepest_completed_depth = label.deepest_completed_depth,
      .previous_completed_depth = label.previous_completed_depth,
      .attempted_depth = label.attempted_depth,
      .terminal_depth = label.terminal_depth,
      .best_move_index = label.best_move_index,
      .deepest_best_move_index = label.deepest_best_move_index,
      .previous_best_move_index = label.previous_best_move_index,
      .flags = flags,
      .line_patterns = features.line_patterns,
      .own_gains = features.own_gains,
      .opponent_gains = features.opponent_gains,
  };
}

template <typename Integer>
  requires std::is_unsigned_v<Integer>
void append_little_endian(std::vector<std::uint8_t>& output, Integer value) {
  for (std::size_t index = 0; index < sizeof(Integer); ++index) {
    output.push_back(static_cast<std::uint8_t>(value >> (index * 8)));
  }
}

void append_signed32(std::vector<std::uint8_t>& output, Score value) {
  if (value < std::numeric_limits<std::int32_t>::min() ||
      value > std::numeric_limits<std::int32_t>::max()) {
    throw std::overflow_error{"feature score does not fit int32"};
  }
  append_little_endian(output, static_cast<std::uint32_t>(static_cast<std::int32_t>(value)));
}

void append_signed16(std::vector<std::uint8_t>& output, std::int16_t value) {
  append_little_endian(output, static_cast<std::uint16_t>(value));
}

void append_digest(std::vector<std::uint8_t>& output, const labeling::Sha256Digest& digest) {
  output.insert(output.end(), digest.begin(), digest.end());
}

[[nodiscard]] std::string json_escape(std::string_view text) {
  std::string escaped;
  for (const char value : text) {
    switch (value) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
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
        if (static_cast<unsigned char>(value) < 0x20) {
          throw std::invalid_argument{"feature manifest text contains a control character"};
        }
        escaped.push_back(value);
        break;
    }
  }
  return escaped;
}

[[nodiscard]] std::string qualified_digest(const labeling::Sha256Digest& digest) {
  return "sha256:" + labeling::sha256_text(digest);
}

}  // namespace

const std::array<std::uint8_t, kScoringLineCount>& scoring_line_lengths() noexcept {
  static constexpr std::array<std::uint8_t, kScoringLineCount> kLengths = [] {
    std::array<std::uint8_t, kScoringLineCount> lengths{};
    for (std::size_t index = 0; index < lengths.size(); ++index) {
      lengths[index] = kScoringLines[index].length;
    }
    return lengths;
  }();
  return kLengths;
}

PositionFeatures extract_position_features(const Position& position) {
  PositionFeatures result{
      .normalized_value = evaluate(position),
      .b_value = evaluate_two_ply_closure(position),
  };
  const Bitboard player_one = position.board().bits(Player::kOne);
  const Bitboard player_two = position.board().bits(Player::kTwo);
  const Player side_to_move = position.side_to_move();
  for (std::size_t line_index = 0; line_index < kScoringLines.size(); ++line_index) {
    const ScoringLine& line = kScoringLines[line_index];
    std::uint16_t code = 0;
    std::uint16_t place = 1;
    for (std::uint8_t offset = 0; offset < line.length; ++offset) {
      const Bitboard bit = Bitboard{1} << line.cells[offset];
      std::uint16_t digit = 0;
      if ((player_one & bit) != 0) {
        digit = side_to_move == Player::kOne ? 1 : 2;
      } else if ((player_two & bit) != 0) {
        digit = side_to_move == Player::kTwo ? 1 : 2;
      }
      code = static_cast<std::uint16_t>(code + digit * place);
      place = static_cast<std::uint16_t>(place * 3);
    }
    result.line_patterns[line_index] =
        static_cast<std::uint16_t>(kPatternOffsets[line.length] + code);
  }

  const Bitboard occupied = position.board().occupied();
  for (int move_index = 0; move_index < kCellCount; ++move_index) {
    const Bitboard bit = Bitboard{1} << move_index;
    if ((occupied & bit) != 0) {
      result.own_gains[move_index] = kOccupiedGain;
      result.opponent_gains[move_index] = kOccupiedGain;
      continue;
    }
    const ScoreByPlayer gains = position.score_gains_unchecked(move_index);
    const Score own = side_to_move == Player::kOne ? gains.player_one : gains.player_two;
    const Score reply = side_to_move == Player::kOne ? gains.player_two : gains.player_one;
    if (own < std::numeric_limits<std::int16_t>::min() ||
        own > std::numeric_limits<std::int16_t>::max() ||
        reply < std::numeric_limits<std::int16_t>::min() ||
        reply > std::numeric_limits<std::int16_t>::max()) {
      throw std::overflow_error{"marginal score gain does not fit int16"};
    }
    result.own_gains[move_index] = static_cast<std::int16_t>(own);
    result.opponent_gains[move_index] = static_cast<std::int16_t>(reply);
  }
  return result;
}

FeatureDataset build_feature_dataset(const fs::path& position_source_directory,
                                     const fs::path& label_corpus_directory) {
  position_source::ReadSourceShard first_source =
      position_source::read_position_source_shard(position_source_directory, 0);
  const std::uint32_t shard_count = first_source.source.shard_count;
  if (shard_count == 0) {
    throw std::runtime_error{"position source contains no shards"};
  }

  const fs::path label_shard_directory = label_corpus_directory / "shards";
  if (!fs::is_directory(label_shard_directory)) {
    throw std::runtime_error{"label corpus has no shards directory"};
  }
  std::vector<LoadedLabelShard> labels(shard_count);
  std::vector<bool> label_seen(shard_count, false);
  for (const fs::directory_entry& entry : fs::directory_iterator(label_shard_directory)) {
    if (!entry.is_directory()) {
      throw std::runtime_error{"label shards directory contains a non-directory entry"};
    }
    LoadedLabelShard shard = load_label_shard(entry.path());
    if (shard.source.shard_count != shard_count || shard.source.shard_index >= shard_count ||
        label_seen[shard.source.shard_index]) {
      throw std::runtime_error{"label corpus has duplicate or inconsistent shard coordinates"};
    }
    const std::uint32_t index = shard.source.shard_index;
    labels[index] = std::move(shard);
    label_seen[index] = true;
  }
  if (std::ranges::any_of(label_seen, [](bool seen) { return !seen; })) {
    throw std::runtime_error{"label corpus is missing one or more shards"};
  }

  FeatureDataset dataset{
      .corpus_id = first_source.source.corpus_id,
      .corpus_digest = labeling::sha256(first_source.source.corpus_id),
      .labels = labels.front().provenance,
      .shard_count = shard_count,
  };
  if (dataset.labels.target_selection != "deepest_terminal_parity") {
    throw std::runtime_error{"label corpus does not use terminal-parity teacher targets"};
  }

  std::vector<std::uint8_t> label_digest_material;
  label_digest_material.reserve(static_cast<std::size_t>(shard_count) * 64);
  std::vector<SelectedCandidate> selected;
  std::unordered_map<PositionKey, std::size_t, PositionKeyHash> by_position;
  std::size_t source_record_count = 0;

  for (std::uint32_t shard_index = 0; shard_index < shard_count; ++shard_index) {
    position_source::ReadSourceShard source =
        position_source::read_position_source_shard(position_source_directory, shard_index);
    const LoadedLabelShard& label_shard = labels[shard_index];
    if (source.source.corpus_id != dataset.corpus_id ||
        source.source.source_digest != label_shard.source.source_digest ||
        label_shard.corpus_digest != dataset.corpus_digest ||
        label_shard.provenance != dataset.labels ||
        label_shard.input_count != source.inputs.size() ||
        label_shard.records.size() != source.inputs.size()) {
      throw std::runtime_error{"label shard provenance differs from its position source"};
    }
    label_digest_material.insert(label_digest_material.end(), label_shard.digests.binary.begin(),
                                 label_shard.digests.binary.end());
    label_digest_material.insert(label_digest_material.end(), label_shard.digests.manifest.begin(),
                                 label_shard.digests.manifest.end());
    source_record_count += source.inputs.size();

    for (std::size_t index = 0; index < source.inputs.size(); ++index) {
      const labeling::LabelInput& input = source.inputs[index];
      const labeling::LabelRecord& label = label_shard.records[index];
      validate_label(label, input, shard_index, dataset.labels);
      const auto [iterator, inserted] =
          by_position.try_emplace(label.canonical_key, selected.size());
      if (inserted) {
        selected.push_back(SelectedCandidate{
            .label = label,
            .position = input.position,
            .source_shard = shard_index,
        });
        continue;
      }

      SelectedCandidate& current = selected[iterator->second];
      if (current.label.split != label.split ||
          current.label.terminal_depth != label.terminal_depth) {
        throw std::runtime_error{"duplicate canonical position crosses splits or game phases"};
      }
      if (current.label.completed_depth == label.completed_depth &&
          current.label.value != label.value) {
        throw std::runtime_error{"duplicate labels disagree at the same completed depth"};
      }
      current.varying_label = current.varying_label || current.label.value != label.value;
      ++current.duplicate_count;
      if (prefer_candidate(label, shard_index, current)) {
        current.label = label;
        current.position = input.position;
        current.source_shard = shard_index;
        ++dataset.representative_upgrade_count;
      }
    }
  }

  dataset.source_record_count = source_record_count;
  dataset.label_set_digest = labeling::sha256(std::span<const std::uint8_t>{label_digest_material});
  dataset.records.reserve(selected.size());
  for (const SelectedCandidate& candidate : selected) {
    if (candidate.duplicate_count > 1) {
      ++dataset.duplicate_group_count;
    }
    dataset.duplicate_record_count += candidate.duplicate_count - 1;
    dataset.maximum_duplicate_count = std::max(dataset.maximum_duplicate_count,
                                               static_cast<std::size_t>(candidate.duplicate_count));
    dataset.varying_label_group_count += candidate.varying_label ? 1 : 0;
    dataset.records.push_back(make_feature_record(candidate));
  }
  std::ranges::sort(dataset.records, {}, [](const FeatureRecord& record) {
    return std::pair{record.canonical_key.low, record.canonical_key.high};
  });
  if (dataset.records.size() + dataset.duplicate_record_count != dataset.source_record_count) {
    throw std::logic_error{"feature deduplication accounting is inconsistent"};
  }
  return dataset;
}

std::vector<std::uint8_t> serialize_feature_binary(const FeatureDataset& dataset) {
  if (dataset.shard_count == 0 || dataset.source_record_count < dataset.records.size() ||
      dataset.source_record_count - dataset.records.size() != dataset.duplicate_record_count) {
    throw std::invalid_argument{"feature dataset metadata is inconsistent"};
  }
  std::vector<std::uint8_t> output;
  output.reserve(kFeatureDatasetHeaderSize + dataset.records.size() * kFeatureDatasetRecordSize);
  output.insert(output.end(), kFeatureMagic.begin(), kFeatureMagic.end());
  append_little_endian(output, kFeatureDatasetSchemaVersion);
  append_little_endian(output, kFeatureDatasetHeaderSize);
  append_little_endian(output, kFeatureDatasetRecordSize);
  append_little_endian(output, kEndianMarker);
  append_little_endian(output, static_cast<std::uint64_t>(dataset.records.size()));
  append_little_endian(output, static_cast<std::uint64_t>(dataset.source_record_count));
  append_little_endian(output, static_cast<std::uint64_t>(dataset.duplicate_record_count));
  append_digest(output, dataset.corpus_digest);
  append_digest(output, dataset.label_set_digest);
  append_little_endian(output, static_cast<std::uint32_t>(kScoringLineCount));
  append_little_endian(output, static_cast<std::uint32_t>(kCellCount));
  append_little_endian(output, dataset.shard_count);
  append_little_endian(output, std::uint32_t{0});
  assert(output.size() == kFeatureDatasetHeaderSize);

  for (const FeatureRecord& record : dataset.records) {
    const std::size_t start = output.size();
    append_little_endian(output, record.player_one);
    append_little_endian(output, record.player_two);
    append_little_endian(output, record.canonical_key.low);
    append_little_endian(output, record.canonical_key.high);
    append_little_endian(output, record.source_id);
    append_little_endian(output, record.family_id);
    append_little_endian(output, record.trajectory_id);
    append_little_endian(output, record.parent_id);
    append_little_endian(output, record.trajectory_index);
    append_little_endian(output, record.nodes);
    append_little_endian(output, record.completed_nodes);
    append_little_endian(output, record.deepest_completed_nodes);
    append_little_endian(output, record.previous_completed_nodes);
    append_little_endian(output, record.source_shard);
    append_little_endian(output, record.source_ordinal);
    append_signed32(output, record.teacher_value);
    append_signed32(output, record.deepest_value);
    append_signed32(output, record.previous_value);
    append_signed32(output, record.normalized_value);
    append_signed32(output, record.b_value);
    append_signed32(output, record.residual);
    append_little_endian(output, record.duplicate_count);
    append_little_endian(output, record.policy_id);
    append_little_endian(output, record.sample_index);
    output.push_back(record.ply);
    output.push_back(static_cast<std::uint8_t>(record.side_to_move));
    output.push_back(static_cast<std::uint8_t>(record.split));
    output.push_back(static_cast<std::uint8_t>(record.mode));
    output.push_back(record.completed_depth);
    output.push_back(record.deepest_completed_depth);
    output.push_back(record.previous_completed_depth);
    output.push_back(record.attempted_depth);
    output.push_back(record.terminal_depth);
    output.push_back(record.best_move_index);
    output.push_back(record.deepest_best_move_index);
    output.push_back(record.previous_best_move_index);
    output.push_back(record.flags);
    output.push_back(0);
    output.push_back(0);
    output.push_back(0);
    for (const std::uint16_t pattern : record.line_patterns) {
      append_little_endian(output, pattern);
    }
    for (const std::int16_t gain : record.own_gains) {
      append_signed16(output, gain);
    }
    for (const std::int16_t gain : record.opponent_gains) {
      append_signed16(output, gain);
    }
    append_little_endian(output, std::uint32_t{0});
    if (output.size() - start != kFeatureDatasetRecordSize) {
      throw std::logic_error{"serialized feature record has the wrong width"};
    }
  }
  return output;
}

std::string serialize_feature_manifest(const FeatureDataset& dataset,
                                       const labeling::Sha256Digest& binary_digest) {
  std::array<std::size_t, 4> split_counts{};
  std::size_t terminal_records = 0;
  std::size_t parity_backoffs = 0;
  for (const FeatureRecord& record : dataset.records) {
    const std::size_t split = static_cast<std::size_t>(record.split);
    if (split >= split_counts.size()) {
      throw std::invalid_argument{"feature record has an invalid split"};
    }
    ++split_counts[split];
    terminal_records += (record.flags & kTerminalLabelFlag) != 0 ? 1 : 0;
    parity_backoffs += (record.flags & kParityBackoffFlag) != 0 ? 1 : 0;
  }

  std::ostringstream output;
  output << "{\n"
         << "  \"schema\": \"poe2-minimax-features\",\n"
         << "  \"schema_version\": " << kFeatureDatasetSchemaVersion << ",\n"
         << "  \"format\": {\"header_bytes\": " << kFeatureDatasetHeaderSize
         << ", \"record_bytes\": " << kFeatureDatasetRecordSize << "},\n"
         << "  \"binary_digest\": \"" << qualified_digest(binary_digest) << "\",\n"
         << "  \"corpus\": {\"id\": \"" << json_escape(dataset.corpus_id) << "\", \"digest\": \""
         << qualified_digest(dataset.corpus_digest) << "\"},\n"
         << "  \"inputs\": {\n"
         << "    \"label_set_digest\": \"" << qualified_digest(dataset.label_set_digest) << "\",\n"
         << "    \"shards\": " << dataset.shard_count << ",\n"
         << "    \"source_records\": " << dataset.source_record_count << ",\n"
         << "    \"label_build\": {\"git_commit\": \"" << json_escape(dataset.labels.git_commit)
         << "\", \"git_dirty\": " << (dataset.labels.git_dirty ? "true" : "false") << "},\n"
         << "    \"label_search\": {\"mode\": \"" << json_escape(dataset.labels.mode)
         << "\", \"node_limit\": " << dataset.labels.node_limit
         << ", \"hash_bytes_effective\": " << dataset.labels.hash_bytes_effective
         << ", \"workers_requested\": " << dataset.labels.workers_requested
         << ", \"target_selection\": \"" << json_escape(dataset.labels.target_selection) << "\"}\n"
         << "  },\n"
         << "  \"exporter_build\": {\n"
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
         << "  \"features\": {\n"
         << "    \"definition\": \"b-residual-line-pattern-gains-v1\",\n"
         << "    \"line_order\": \"rows-columns-down-diagonals-up-diagonals-v1\",\n"
         << "    \"line_pattern_encoding\": "
            "\"length-offset-plus-base3-empty0-stm1-opponent2-v1\",\n"
         << "    \"gain_encoding\": \"raw-score-gain-stm-first-int16-min-occupied-v1\",\n"
         << "    \"line_count\": " << kScoringLineCount << ",\n"
         << "    \"cell_count\": " << kCellCount << ",\n"
         << "    \"line_lengths\": [";
  const auto& lengths = scoring_line_lengths();
  for (std::size_t index = 0; index < lengths.size(); ++index) {
    output << (index == 0 ? "" : ", ") << static_cast<int>(lengths[index]);
  }
  output << "]\n"
         << "  },\n"
         << "  \"results\": {\n"
         << "    \"records\": " << dataset.records.size() << ",\n"
         << "    \"duplicates_removed\": " << dataset.duplicate_record_count << ",\n"
         << "    \"duplicate_groups\": " << dataset.duplicate_group_count << ",\n"
         << "    \"maximum_duplicate_count\": " << dataset.maximum_duplicate_count << ",\n"
         << "    \"representative_upgrades\": " << dataset.representative_upgrade_count << ",\n"
         << "    \"varying_label_groups\": " << dataset.varying_label_group_count << ",\n"
         << "    \"terminal_records\": " << terminal_records << ",\n"
         << "    \"parity_backoffs\": " << parity_backoffs << ",\n"
         << "    \"split_counts\": {\"train\": "
         << split_counts[static_cast<std::size_t>(labeling::DatasetSplit::kTrain)]
         << ", \"validation\": "
         << split_counts[static_cast<std::size_t>(labeling::DatasetSplit::kValidation)]
         << ", \"test\": " << split_counts[static_cast<std::size_t>(labeling::DatasetSplit::kTest)]
         << "}\n"
         << "  }\n"
         << "}\n";
  return output.str();
}

void write_feature_dataset(const fs::path& directory, const FeatureDataset& dataset) {
  if (directory.empty()) {
    throw std::invalid_argument{"feature output directory must not be empty"};
  }
  create_parent_directories(directory);
  std::error_code error;
  const bool created = fs::create_directory(directory, error);
  if (!created) {
    if (error) {
      throw fs::filesystem_error{"failed to reserve feature output directory", directory, error};
    }
    throw std::invalid_argument{"feature output directory already exists"};
  }

  const fs::path incomplete = directory / kFeatureIncompleteMarkerName;
  const fs::path complete = directory / kFeatureCompleteMarkerName;
  const fs::path binary = directory / kFeatureBinaryFileName;
  const fs::path manifest = directory / kFeatureManifestFileName;
  const fs::path binary_temporary = directory / "features.bin.tmp";
  const fs::path manifest_temporary = directory / "manifest.json.tmp";
  write_text_file(incomplete, "poe2-minimax-features\n");

  const std::vector<std::uint8_t> binary_bytes = serialize_feature_binary(dataset);
  const labeling::Sha256Digest binary_digest =
      labeling::sha256(std::span<const std::uint8_t>{binary_bytes});
  const std::string manifest_text = serialize_feature_manifest(dataset, binary_digest);
  const labeling::Sha256Digest manifest_digest = labeling::sha256(manifest_text);
  write_binary_file(binary_temporary, binary_bytes);
  write_text_file(manifest_temporary, manifest_text);
  fs::rename(binary_temporary, binary);
  fs::rename(manifest_temporary, manifest);
  write_text_file(incomplete,
                  "poe2-minimax-features\nbinary_sha256=" + labeling::sha256_text(binary_digest) +
                      "\nmanifest_sha256=" + labeling::sha256_text(manifest_digest) + "\n");
  fs::rename(incomplete, complete);
}

}  // namespace poe2::minimax::feature_data
