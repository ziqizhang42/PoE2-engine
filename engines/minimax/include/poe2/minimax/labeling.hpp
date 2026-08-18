#ifndef POE2_MINIMAX_LABELING_HPP
#define POE2_MINIMAX_LABELING_HPP

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "poe2/minimax/search.hpp"
#include "poe2/minimax/sha256.hpp"

namespace poe2::minimax::labeling {

inline constexpr std::uint32_t kLabelDatasetSchemaVersion = 2;
inline constexpr std::uint32_t kLabelDatasetHeaderSize = 112;
inline constexpr std::uint32_t kLabelDatasetRecordSize = 140;
inline constexpr std::string_view kBinaryFileName = "labels.bin";
inline constexpr std::string_view kManifestFileName = "manifest.json";
inline constexpr std::string_view kIncompleteMarkerName = "INCOMPLETE";
inline constexpr std::string_view kCompleteMarkerName = "COMPLETE";

enum class LabelMode : std::uint8_t {
  kExact = 1,
  kTeacher = 2,
};

enum class DatasetSplit : std::uint8_t {
  kUnspecified = 0,
  kTrain = 1,
  kValidation = 2,
  kTest = 3,
};

struct LabelSource {
  std::string corpus_id;
  std::string source_name;
  Sha256Digest source_digest{};
  std::uint32_t shard_index = 0;
  std::uint32_t shard_count = 1;
};

struct LabelInput {
  Position position;
  std::uint64_t source_id = 0;
  std::uint64_t family_id = 0;
  std::uint64_t trajectory_id = 0;
  std::uint64_t parent_id = 0;
  std::uint64_t trajectory_index = 0;
  std::uint32_t source_line = 0;
  std::uint32_t source_ordinal = 0;
  std::uint16_t policy_id = 0;
  std::uint16_t sample_index = 0;
  DatasetSplit split = DatasetSplit::kTrain;
};

struct LabelingOptions {
  LabelMode mode = LabelMode::kExact;
  std::uint64_t node_limit = 0;
  std::size_t hash_bytes = kDefaultHashBytes;
  std::size_t workers = 1;
  bool require_all = false;
};

struct LabelRecord {
  Bitboard player_one = 0;
  Bitboard player_two = 0;
  PositionKey canonical_key;
  std::uint64_t source_id = 0;
  std::uint64_t family_id = 0;
  std::uint64_t trajectory_id = 0;
  std::uint64_t parent_id = 0;
  std::uint64_t trajectory_index = 0;
  std::uint64_t nodes = 0;
  std::uint64_t completed_nodes = 0;
  std::uint32_t source_line = 0;
  std::uint32_t source_ordinal = 0;
  Score value = 0;
  std::uint8_t ply = 0;
  Player side_to_move = Player::kOne;
  LabelMode mode = LabelMode::kExact;
  std::uint8_t completed_depth = 0;
  std::uint8_t attempted_depth = 0;
  std::uint8_t terminal_depth = 0;
  std::uint8_t best_move_index = 0xff;
  std::uint16_t policy_id = 0;
  std::uint16_t sample_index = 0;
  DatasetSplit split = DatasetSplit::kTrain;
  std::uint64_t deepest_completed_nodes = 0;
  Score deepest_value = 0;
  std::uint8_t deepest_completed_depth = 0;
  std::uint8_t deepest_best_move_index = 0xff;
  std::uint64_t previous_completed_nodes = 0;
  Score previous_value = 0;
  std::uint8_t previous_completed_depth = 0;
  std::uint8_t previous_best_move_index = 0xff;

  friend constexpr bool operator==(const LabelRecord&, const LabelRecord&) = default;
};

struct LabelDataset {
  LabelSource source;
  Sha256Digest corpus_digest{};
  LabelingOptions options;
  std::size_t input_count = 0;
  std::size_t workers_used = 0;
  std::size_t hash_capacity = 0;
  std::size_t hash_storage_bytes = 0;
  std::vector<LabelRecord> records;
  std::vector<std::uint32_t> unsolved_source_lines;
};

struct LabelProgress {
  std::size_t completed = 0;
  std::size_t total = 0;
  std::size_t records = 0;
  std::size_t unsolved = 0;
};

using LabelProgressSink = std::function<void(const LabelProgress&)>;

class DatasetOutput final {
 public:
  DatasetOutput(const DatasetOutput&) = delete;
  DatasetOutput& operator=(const DatasetOutput&) = delete;
  DatasetOutput(DatasetOutput&&) noexcept = default;
  DatasetOutput& operator=(DatasetOutput&&) noexcept = default;

  [[nodiscard]] const std::filesystem::path& directory() const noexcept;
  [[nodiscard]] bool committed() const noexcept;

 private:
  explicit DatasetOutput(std::filesystem::path directory);

  std::filesystem::path directory_;
  bool committed_ = false;

  friend DatasetOutput reserve_dataset_output(const std::filesystem::path& directory);
  friend void write_dataset(DatasetOutput& output, const LabelDataset& dataset);
};

[[nodiscard]] std::string_view label_mode_name(LabelMode mode) noexcept;
[[nodiscard]] std::string_view dataset_split_name(DatasetSplit split) noexcept;
[[nodiscard]] std::uint64_t stable_digest(std::string_view bytes) noexcept;
[[nodiscard]] LabelDataset generate_labels(std::span<const LabelInput> inputs,
                                           const LabelingOptions& options, LabelSource source,
                                           const LabelProgressSink& progress = {},
                                           std::size_t progress_interval = 0);
[[nodiscard]] std::vector<std::uint8_t> serialize_binary(const LabelDataset& dataset);
[[nodiscard]] std::string serialize_manifest(const LabelDataset& dataset,
                                             const Sha256Digest& binary_digest);
[[nodiscard]] DatasetOutput reserve_dataset_output(const std::filesystem::path& directory);
void write_dataset(DatasetOutput& output, const LabelDataset& dataset);

}  // namespace poe2::minimax::labeling

#endif  // POE2_MINIMAX_LABELING_HPP
