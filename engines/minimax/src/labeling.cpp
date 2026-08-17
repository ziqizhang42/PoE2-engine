#include "poe2/minimax/labeling.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <type_traits>

#include "poe2/symmetry.hpp"

namespace poe2::minimax::labeling {

namespace {

namespace fs = std::filesystem;

constexpr std::array<std::uint8_t, 8> kDatasetMagic{{'P', 'O', 'E', '2', 'L', 'B', 'L', 0}};
constexpr std::uint32_t kEndianMarker = 0x01020304;

template <typename Integer>
  requires std::is_unsigned_v<Integer>
void append_little_endian(std::vector<std::uint8_t>& output, Integer value) {
  for (std::size_t index = 0; index < sizeof(Integer); ++index) {
    output.push_back(static_cast<std::uint8_t>(value & Integer{0xff}));
    value >>= 8;
  }
}

[[nodiscard]] std::string digest_text(std::uint64_t digest) {
  std::ostringstream output;
  output << "fnv1a64:" << std::hex << std::setw(16) << std::setfill('0') << digest;
  return output.str();
}

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

[[nodiscard]] std::uint64_t digest_binary(std::span<const std::uint8_t> bytes) noexcept {
  std::uint64_t hash = UINT64_C(14695981039346656037);
  for (const std::uint8_t byte : bytes) {
    hash ^= byte;
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

void create_parent_directories(const fs::path& path) {
  const fs::path parent = path.parent_path();
  if (!parent.empty()) {
    fs::create_directories(parent);
  }
}

void write_binary_file(const fs::path& path, std::span<const std::uint8_t> bytes) {
  std::ofstream output{path, std::ios::binary};
  if (!output) {
    throw std::runtime_error{"failed to open " + path.string()};
  }
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  if (!output) {
    throw std::runtime_error{"failed to write " + path.string()};
  }
}

void write_text_file(const fs::path& path, std::string_view text) {
  std::ofstream output{path, std::ios::binary};
  if (!output) {
    throw std::runtime_error{"failed to open " + path.string()};
  }
  output.write(text.data(), static_cast<std::streamsize>(text.size()));
  if (!output) {
    throw std::runtime_error{"failed to write " + path.string()};
  }
}

[[nodiscard]] std::optional<LabelRecord> generate_label(Search& search, const LabelInput& input,
                                                        const LabelingOptions& options) {
  search.new_game();
  const int terminal_depth = search.terminal_depth(input.position);
  engine::EngineLimits limits{
      .nodes = options.node_limit,
  };
  if (options.mode == LabelMode::kExact) {
    limits.depth = terminal_depth;
  }

  const engine::EngineResult result = search.run(input.position, limits, {});
  const bool has_label = terminal_depth > 0 && result.score.has_value() &&
                         result.best_move.has_value() && result.depth > 0;
  const bool is_exact = has_label && result.depth == terminal_depth;
  if (!has_label || (options.mode == LabelMode::kExact && !is_exact)) {
    return std::nullopt;
  }
  if (result.nodes > options.node_limit) {
    throw std::logic_error{"search exceeded the configured node limit"};
  }

  const PositionKey canonical_key = canonicalize_position_key(input.position.key()).key;
  const int move_index = square_index(result.best_move->square);
  assert(input.position.ply() >= 0 && input.position.ply() <= kCellCount);
  assert(result.depth >= 0 && result.depth <= kCellCount);
  assert(terminal_depth >= 0 && terminal_depth <= kCellCount);
  assert(move_index >= 0 && move_index < kCellCount);
  return LabelRecord{
      .player_one = input.position.board().bits(Player::kOne),
      .player_two = input.position.board().bits(Player::kTwo),
      .canonical_key = canonical_key,
      .source_id = input.source_id,
      .nodes = result.nodes,
      .source_line = input.source_line,
      .value = *result.score,
      .ply = static_cast<std::uint8_t>(input.position.ply()),
      .side_to_move = input.position.side_to_move(),
      .mode = options.mode,
      .completed_depth = static_cast<std::uint8_t>(result.depth),
      .terminal_depth = static_cast<std::uint8_t>(terminal_depth),
      .best_move_index = static_cast<std::uint8_t>(move_index),
  };
}

}  // namespace

std::string_view label_mode_name(LabelMode mode) noexcept {
  switch (mode) {
    case LabelMode::kExact:
      return "exact";
    case LabelMode::kTeacher:
      return "teacher";
  }
  return "unknown";
}

std::uint64_t stable_digest(std::string_view bytes) noexcept {
  return digest_binary(
      std::span{reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()});
}

LabelDataset generate_labels(std::span<const LabelInput> inputs, const LabelingOptions& options,
                             std::string_view source_name, std::uint64_t source_digest) {
  if (options.node_limit == 0) {
    throw std::invalid_argument{"label generation requires a positive node limit"};
  }
  if (options.mode != LabelMode::kExact && options.mode != LabelMode::kTeacher) {
    throw std::invalid_argument{"unknown label mode"};
  }
  if (options.workers == 0) {
    throw std::invalid_argument{"label generation requires at least one worker"};
  }

  const std::size_t workers_used = std::min(options.workers, inputs.size());

  LabelDataset dataset{
      .source_name = std::string{source_name},
      .source_digest = source_digest,
      .options = options,
      .input_count = inputs.size(),
      .workers_used = workers_used,
  };
  dataset.records.reserve(inputs.size());

  std::vector<std::optional<LabelRecord>> records_by_input(inputs.size());
  std::atomic<std::size_t> next_input = 0;
  std::atomic<bool> stop = false;
  std::mutex failure_mutex;
  std::exception_ptr failure;

  const auto run_worker = [&] {
    try {
      Search search{SearchOptions{
          .hash_bytes = options.hash_bytes,
          .use_symmetry = true,
          .use_two_ply_closure = true,
      }};
      while (!stop.load(std::memory_order_relaxed)) {
        const std::size_t input_index = next_input.fetch_add(1, std::memory_order_relaxed);
        if (input_index >= inputs.size()) {
          break;
        }
        records_by_input[input_index] = generate_label(search, inputs[input_index], options);
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
  for (std::size_t worker = 0; worker < workers_used; ++worker) {
    workers.emplace_back(run_worker);
  }
  for (std::jthread& worker : workers) {
    worker.join();
  }
  if (failure != nullptr) {
    std::rethrow_exception(failure);
  }

  for (std::size_t input_index = 0; input_index < inputs.size(); ++input_index) {
    std::optional<LabelRecord>& record = records_by_input[input_index];
    if (record.has_value()) {
      dataset.records.push_back(*record);
    } else {
      dataset.unsolved_source_lines.push_back(inputs[input_index].source_line);
    }
  }

  if (options.require_all && !dataset.unsolved_source_lines.empty()) {
    throw std::runtime_error{"failed to solve " +
                             std::to_string(dataset.unsolved_source_lines.size()) + " of " +
                             std::to_string(dataset.input_count) + " input positions"};
  }
  return dataset;
}

std::vector<std::uint8_t> serialize_binary(const LabelDataset& dataset) {
  if (dataset.records.size() > std::numeric_limits<std::uint64_t>::max() ||
      dataset.input_count > std::numeric_limits<std::uint64_t>::max()) {
    throw std::overflow_error{"label dataset is too large"};
  }

  std::vector<std::uint8_t> output;
  output.reserve(kLabelDatasetHeaderSize + dataset.records.size() * kLabelDatasetRecordSize);
  output.insert(output.end(), kDatasetMagic.begin(), kDatasetMagic.end());
  append_little_endian(output, kLabelDatasetSchemaVersion);
  append_little_endian(output, kLabelDatasetHeaderSize);
  append_little_endian(output, kLabelDatasetRecordSize);
  append_little_endian(output, kEndianMarker);
  append_little_endian(output, static_cast<std::uint64_t>(dataset.records.size()));
  append_little_endian(output, static_cast<std::uint64_t>(dataset.input_count));
  append_little_endian(output, dataset.source_digest);
  assert(output.size() == kLabelDatasetHeaderSize);

  for (const LabelRecord& record : dataset.records) {
    const std::size_t record_start = output.size();
    append_little_endian(output, record.player_one);
    append_little_endian(output, record.player_two);
    append_little_endian(output, record.canonical_key.low);
    append_little_endian(output, record.canonical_key.high);
    append_little_endian(output, record.source_id);
    append_little_endian(output, record.nodes);
    append_little_endian(output, record.source_line);
    append_little_endian(output, static_cast<std::uint32_t>(record.value));
    output.push_back(record.ply);
    output.push_back(static_cast<std::uint8_t>(record.side_to_move));
    output.push_back(static_cast<std::uint8_t>(record.mode));
    output.push_back(record.completed_depth);
    output.push_back(record.terminal_depth);
    output.push_back(record.best_move_index);
    append_little_endian(output, std::uint16_t{0});
    assert(output.size() - record_start == kLabelDatasetRecordSize);
  }
  return output;
}

std::string serialize_manifest(const LabelDataset& dataset, std::uint64_t binary_digest) {
  std::size_t terminal_records = 0;
  for (const LabelRecord& record : dataset.records) {
    terminal_records += record.completed_depth == record.terminal_depth ? 1 : 0;
  }

  std::ostringstream output;
  output << "{\n"
         << "  \"schema\": \"poe2-minimax-labels\",\n"
         << "  \"schema_version\": " << kLabelDatasetSchemaVersion << ",\n"
         << "  \"binary_digest\": \"" << digest_text(binary_digest) << "\",\n"
         << "  \"source\": {\n"
         << "    \"name\": \"" << json_escape(dataset.source_name) << "\",\n"
         << "    \"digest\": \"" << digest_text(dataset.source_digest) << "\",\n"
         << "    \"positions\": " << dataset.input_count << "\n"
         << "  },\n"
         << "  \"search\": {\n"
         << "    \"evaluator\": \"b\",\n"
         << "    \"mode\": \"" << label_mode_name(dataset.options.mode) << "\",\n"
         << "    \"node_limit\": " << dataset.options.node_limit << ",\n"
         << "    \"hash_bytes\": " << dataset.options.hash_bytes << ",\n"
         << "    \"workers_requested\": " << dataset.options.workers << ",\n"
         << "    \"workers_used\": " << dataset.workers_used << ",\n"
         << "    \"symmetry\": true,\n"
         << "    \"two_ply_closure\": true\n"
         << "  },\n"
         << "  \"results\": {\n"
         << "    \"records\": " << dataset.records.size() << ",\n"
         << "    \"terminal_records\": " << terminal_records << ",\n"
         << "    \"unsolved\": " << dataset.unsolved_source_lines.size() << ",\n"
         << "    \"unsolved_source_lines\": [";
  for (std::size_t index = 0; index < dataset.unsolved_source_lines.size(); ++index) {
    if (index != 0) {
      output << ", ";
    }
    output << dataset.unsolved_source_lines[index];
  }
  output << "]\n"
         << "  }\n"
         << "}\n";
  return output.str();
}

void write_dataset(const LabelDataset& dataset, const fs::path& binary_path,
                   const fs::path& manifest_path) {
  if (binary_path.empty() || manifest_path.empty()) {
    throw std::invalid_argument{"label output paths must not be empty"};
  }
  if (binary_path.lexically_normal() == manifest_path.lexically_normal()) {
    throw std::invalid_argument{"binary and manifest output paths must be different"};
  }
  if (fs::exists(binary_path) || fs::exists(manifest_path)) {
    throw std::invalid_argument{"label output already exists"};
  }

  create_parent_directories(binary_path);
  create_parent_directories(manifest_path);
  fs::path binary_temporary = binary_path;
  fs::path manifest_temporary = manifest_path;
  binary_temporary += ".tmp";
  manifest_temporary += ".tmp";
  if (fs::exists(binary_temporary) || fs::exists(manifest_temporary)) {
    throw std::invalid_argument{"temporary label output already exists"};
  }

  const std::vector<std::uint8_t> binary = serialize_binary(dataset);
  const std::uint64_t binary_digest = digest_binary(binary);
  const std::string manifest = serialize_manifest(dataset, binary_digest);
  bool binary_committed = false;
  try {
    write_binary_file(binary_temporary, binary);
    write_text_file(manifest_temporary, manifest);
    fs::rename(binary_temporary, binary_path);
    binary_committed = true;
    fs::rename(manifest_temporary, manifest_path);
  } catch (...) {
    std::error_code ignored;
    fs::remove(binary_temporary, ignored);
    fs::remove(manifest_temporary, ignored);
    if (binary_committed) {
      fs::remove(binary_path, ignored);
    }
    throw;
  }
}

}  // namespace poe2::minimax::labeling
