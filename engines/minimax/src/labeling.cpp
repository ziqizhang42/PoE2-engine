#include "poe2/minimax/labeling.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <exception>
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>

#include "poe2/minimax/labeling_build.hpp"
#include "poe2/symmetry.hpp"

namespace poe2::minimax::labeling {

namespace {

namespace fs = std::filesystem;

constexpr std::array<std::uint8_t, 8> kDatasetMagic{{'P', 'O', 'E', '2', 'L', 'B', 'L', 0}};
constexpr std::uint32_t kEndianMarker = 0x01020304;

static_assert(sizeof(Score) == sizeof(std::int32_t));
static_assert(std::is_signed_v<Score>);

template <typename Integer>
  requires std::is_unsigned_v<Integer>
void append_little_endian(std::vector<std::uint8_t>& output, Integer value) {
  for (std::size_t index = 0; index < sizeof(Integer); ++index) {
    output.push_back(static_cast<std::uint8_t>(value & Integer{0xff}));
    value >>= 8;
  }
}

void append_digest(std::vector<std::uint8_t>& output, const Sha256Digest& digest) {
  output.insert(output.end(), digest.begin(), digest.end());
}

[[nodiscard]] std::string qualified_digest(const Sha256Digest& digest) {
  return "sha256:" + sha256_text(digest);
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

void create_parent_directories(const fs::path& path) {
  const fs::path parent = path.parent_path();
  if (!parent.empty()) {
    fs::create_directories(parent);
  }
}

void write_binary_file(const fs::path& path, std::span<const std::uint8_t> bytes) {
  std::ofstream output{path, std::ios::binary | std::ios::trunc};
  if (!output) {
    throw std::runtime_error{"failed to open " + path.string()};
  }
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  output.close();
  if (!output) {
    throw std::runtime_error{"failed to write " + path.string()};
  }
}

void write_text_file(const fs::path& path, std::string_view text) {
  std::ofstream output{path, std::ios::binary | std::ios::trunc};
  if (!output) {
    throw std::runtime_error{"failed to open " + path.string()};
  }
  output.write(text.data(), static_cast<std::streamsize>(text.size()));
  output.close();
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
  if (result.nodes > options.node_limit || result.completed_nodes > result.nodes) {
    throw std::logic_error{"search exceeded or misreported the configured node limit"};
  }

  const PositionKey canonical_key = canonicalize_position_key(input.position.key()).key;
  const int move_index = square_index(result.best_move->square);
  const int attempted_depth =
      result.depth < terminal_depth ? std::min(result.depth + 1, terminal_depth) : result.depth;
  assert(input.position.ply() >= 0 && input.position.ply() <= kCellCount);
  assert(result.depth >= 0 && result.depth <= kCellCount);
  assert(attempted_depth >= result.depth && attempted_depth <= kCellCount);
  assert(terminal_depth >= 0 && terminal_depth <= kCellCount);
  assert(move_index >= 0 && move_index < kCellCount);
  return LabelRecord{
      .player_one = input.position.board().bits(Player::kOne),
      .player_two = input.position.board().bits(Player::kTwo),
      .canonical_key = canonical_key,
      .source_id = input.source_id,
      .family_id = input.family_id,
      .trajectory_id = input.trajectory_id,
      .parent_id = input.parent_id,
      .trajectory_index = input.trajectory_index,
      .nodes = result.nodes,
      .completed_nodes = result.completed_nodes,
      .source_line = input.source_line,
      .source_ordinal = input.source_ordinal,
      .value = *result.score,
      .ply = static_cast<std::uint8_t>(input.position.ply()),
      .side_to_move = input.position.side_to_move(),
      .mode = options.mode,
      .completed_depth = static_cast<std::uint8_t>(result.depth),
      .attempted_depth = static_cast<std::uint8_t>(attempted_depth),
      .terminal_depth = static_cast<std::uint8_t>(terminal_depth),
      .best_move_index = static_cast<std::uint8_t>(move_index),
      .policy_id = input.policy_id,
      .sample_index = input.sample_index,
      .split = input.split,
  };
}

}  // namespace

DatasetOutput::DatasetOutput(fs::path directory) : directory_(std::move(directory)) {}

const fs::path& DatasetOutput::directory() const noexcept { return directory_; }

bool DatasetOutput::committed() const noexcept { return committed_; }

std::string_view label_mode_name(LabelMode mode) noexcept {
  switch (mode) {
    case LabelMode::kExact:
      return "exact";
    case LabelMode::kTeacher:
      return "teacher";
  }
  return "unknown";
}

std::string_view dataset_split_name(DatasetSplit split) noexcept {
  switch (split) {
    case DatasetSplit::kUnspecified:
      return "unspecified";
    case DatasetSplit::kTrain:
      return "train";
    case DatasetSplit::kValidation:
      return "validation";
    case DatasetSplit::kTest:
      return "test";
  }
  return "unknown";
}

std::uint64_t stable_digest(std::string_view bytes) noexcept {
  std::uint64_t hash = UINT64_C(14695981039346656037);
  for (const unsigned char byte : bytes) {
    hash ^= byte;
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

LabelDataset generate_labels(std::span<const LabelInput> inputs, const LabelingOptions& options,
                             LabelSource source, const LabelProgressSink& progress,
                             std::size_t progress_interval) {
  if (options.node_limit == 0) {
    throw std::invalid_argument{"label generation requires a positive node limit"};
  }
  if (options.mode != LabelMode::kExact && options.mode != LabelMode::kTeacher) {
    throw std::invalid_argument{"unknown label mode"};
  }
  if (options.workers == 0) {
    throw std::invalid_argument{"label generation requires at least one worker"};
  }
  if (inputs.empty()) {
    throw std::invalid_argument{"label generation requires at least one input position"};
  }
  if (source.corpus_id.empty()) {
    throw std::invalid_argument{"label generation requires a corpus ID"};
  }
  if (source.source_name.empty()) {
    throw std::invalid_argument{"label generation requires a source name"};
  }
  if (source.shard_count == 0 || source.shard_index >= source.shard_count) {
    throw std::invalid_argument{"invalid label shard index or count"};
  }
  if (progress && progress_interval == 0) {
    throw std::invalid_argument{"label progress requires a positive interval"};
  }
  for (const LabelInput& input : inputs) {
    if (input.split != DatasetSplit::kTrain && input.split != DatasetSplit::kValidation &&
        input.split != DatasetSplit::kTest) {
      throw std::invalid_argument{"label input has no valid dataset split"};
    }
  }

  const std::size_t workers_used = std::min(options.workers, inputs.size());
  LabelDataset dataset{
      .source = std::move(source),
      .options = options,
      .input_count = inputs.size(),
      .workers_used = workers_used,
  };
  dataset.corpus_digest = sha256(dataset.source.corpus_id);
  {
    Search sized_search{SearchOptions{
        .hash_bytes = options.hash_bytes,
        .use_symmetry = true,
        .use_two_ply_closure = true,
    }};
    dataset.hash_capacity = sized_search.transposition_table().capacity();
    dataset.hash_storage_bytes = sized_search.transposition_table().storage_bytes();
  }
  dataset.records.reserve(inputs.size());

  std::vector<std::optional<LabelRecord>> records_by_input(inputs.size());
  std::atomic<std::size_t> next_input = 0;
  std::atomic<bool> stop = false;
  std::mutex failure_mutex;
  std::mutex progress_mutex;
  std::exception_ptr failure;
  std::size_t completed = 0;
  std::size_t emitted = 0;
  std::size_t unsolved = 0;
  std::size_t next_progress = progress_interval;

  const auto record_progress = [&](bool has_record) {
    if (!progress) {
      return;
    }
    const std::scoped_lock lock{progress_mutex};
    ++completed;
    if (has_record) {
      ++emitted;
    } else {
      ++unsolved;
    }
    if (completed < next_progress && completed != inputs.size()) {
      return;
    }
    if (completed >= next_progress) {
      const std::size_t remaining = std::numeric_limits<std::size_t>::max() - completed;
      next_progress = remaining < progress_interval ? std::numeric_limits<std::size_t>::max()
                                                    : completed + progress_interval;
    }
    progress(LabelProgress{
        .completed = completed,
        .total = inputs.size(),
        .records = emitted,
        .unsolved = unsolved,
    });
  };

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
        record_progress(records_by_input[input_index].has_value());
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
  if (dataset.source.shard_count == 0 || dataset.source.shard_index >= dataset.source.shard_count) {
    throw std::invalid_argument{"invalid label shard index or count"};
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
  append_digest(output, dataset.source.source_digest);
  append_digest(output, dataset.corpus_digest);
  append_little_endian(output, dataset.source.shard_index);
  append_little_endian(output, dataset.source.shard_count);
  assert(output.size() == kLabelDatasetHeaderSize);

  for (const LabelRecord& record : dataset.records) {
    const std::size_t record_start = output.size();
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
    append_little_endian(output, record.source_line);
    append_little_endian(output, record.source_ordinal);
    append_little_endian(output,
                         static_cast<std::uint32_t>(static_cast<std::int32_t>(record.value)));
    output.push_back(record.ply);
    output.push_back(static_cast<std::uint8_t>(record.side_to_move));
    output.push_back(static_cast<std::uint8_t>(record.mode));
    output.push_back(record.completed_depth);
    output.push_back(record.attempted_depth);
    output.push_back(record.terminal_depth);
    output.push_back(record.best_move_index);
    output.push_back(static_cast<std::uint8_t>(record.split));
    append_little_endian(output, record.policy_id);
    append_little_endian(output, record.sample_index);
    assert(output.size() - record_start == kLabelDatasetRecordSize);
  }
  return output;
}

std::string serialize_manifest(const LabelDataset& dataset, const Sha256Digest& binary_digest) {
  std::size_t terminal_records = 0;
  for (const LabelRecord& record : dataset.records) {
    terminal_records += record.completed_depth == record.terminal_depth ? 1 : 0;
  }

  std::ostringstream output;
  output << "{\n"
         << "  \"schema\": \"poe2-minimax-labels\",\n"
         << "  \"schema_version\": " << kLabelDatasetSchemaVersion << ",\n"
         << "  \"format\": {\"header_bytes\": " << kLabelDatasetHeaderSize
         << ", \"record_bytes\": " << kLabelDatasetRecordSize << "},\n"
         << "  \"binary_digest\": \"" << qualified_digest(binary_digest) << "\",\n"
         << "  \"corpus\": {\n"
         << "    \"id\": \"" << json_escape(dataset.source.corpus_id) << "\",\n"
         << "    \"digest\": \"" << qualified_digest(dataset.corpus_digest) << "\",\n"
         << "    \"shard_index\": " << dataset.source.shard_index << ",\n"
         << "    \"shard_count\": " << dataset.source.shard_count << "\n"
         << "  },\n"
         << "  \"source\": {\n"
         << "    \"name\": \"" << json_escape(dataset.source.source_name) << "\",\n"
         << "    \"digest\": \"" << qualified_digest(dataset.source.source_digest) << "\",\n"
         << "    \"positions\": " << dataset.input_count << "\n"
         << "  },\n"
         << "  \"build\": {\n"
         << "    \"git_commit\": \"" << json_escape(build::kGitCommit) << "\",\n"
         << "    \"git_dirty\": " << (build::kGitDirty ? "true" : "false") << ",\n"
         << "    \"project_version\": \"" << json_escape(build::kProjectVersion) << "\",\n"
         << "    \"compiler_id\": \"" << json_escape(build::kCompilerId) << "\",\n"
         << "    \"compiler_version\": \"" << json_escape(build::kCompilerVersion) << "\",\n"
         << "    \"build_type\": \"" << json_escape(build::kBuildType) << "\",\n"
         << "    \"target_processor\": \"" << json_escape(build::kTargetProcessor) << "\",\n"
         << "    \"native_architecture\": " << (build::kNativeArchitecture ? "true" : "false")
         << "\n"
         << "  },\n"
         << "  \"search\": {\n"
         << "    \"evaluator\": \"b\",\n"
         << "    \"mode\": \"" << label_mode_name(dataset.options.mode) << "\",\n"
         << "    \"node_limit\": " << dataset.options.node_limit << ",\n"
         << "    \"hash_bytes_requested\": " << dataset.options.hash_bytes << ",\n"
         << "    \"hash_bytes_effective\": " << dataset.hash_storage_bytes << ",\n"
         << "    \"hash_capacity\": " << dataset.hash_capacity << ",\n"
         << "    \"workers_requested\": " << dataset.options.workers << ",\n"
         << "    \"workers_used\": " << dataset.workers_used << ",\n"
         << "    \"require_all\": " << (dataset.options.require_all ? "true" : "false") << ",\n"
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

DatasetOutput reserve_dataset_output(const fs::path& directory) {
  if (directory.empty()) {
    throw std::invalid_argument{"label output directory must not be empty"};
  }
  create_parent_directories(directory);

  std::error_code error;
  const bool created = fs::create_directory(directory, error);
  if (!created) {
    if (error) {
      throw fs::filesystem_error{"failed to reserve label output directory", directory, error};
    }
    throw std::invalid_argument{"label output directory already exists"};
  }

  DatasetOutput output{directory};
  write_text_file(directory / kIncompleteMarkerName, "poe2-minimax-labels\n");
  return output;
}

void write_dataset(DatasetOutput& output, const LabelDataset& dataset) {
  if (output.committed_) {
    throw std::logic_error{"label dataset output is already committed"};
  }

  const fs::path incomplete = output.directory_ / kIncompleteMarkerName;
  const fs::path complete = output.directory_ / kCompleteMarkerName;
  const fs::path binary = output.directory_ / kBinaryFileName;
  const fs::path manifest = output.directory_ / kManifestFileName;
  const fs::path binary_temporary = output.directory_ / "labels.bin.tmp";
  const fs::path manifest_temporary = output.directory_ / "manifest.json.tmp";
  if (!fs::is_regular_file(incomplete) || fs::exists(complete) || fs::exists(binary) ||
      fs::exists(manifest) || fs::exists(binary_temporary) || fs::exists(manifest_temporary)) {
    throw std::runtime_error{"label output reservation is not empty and incomplete"};
  }

  const std::vector<std::uint8_t> binary_bytes = serialize_binary(dataset);
  const Sha256Digest binary_digest = sha256(std::span<const std::uint8_t>{binary_bytes});
  const std::string manifest_text = serialize_manifest(dataset, binary_digest);
  const Sha256Digest manifest_digest = sha256(manifest_text);

  write_binary_file(binary_temporary, binary_bytes);
  write_text_file(manifest_temporary, manifest_text);
  fs::rename(binary_temporary, binary);
  fs::rename(manifest_temporary, manifest);
  write_text_file(incomplete, "poe2-minimax-labels\nbinary_sha256=" + sha256_text(binary_digest) +
                                  "\nmanifest_sha256=" + sha256_text(manifest_digest) + "\n");
  fs::rename(incomplete, complete);
  output.committed_ = true;
}

}  // namespace poe2::minimax::labeling
