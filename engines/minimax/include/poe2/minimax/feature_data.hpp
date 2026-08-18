#ifndef POE2_MINIMAX_FEATURE_DATA_HPP
#define POE2_MINIMAX_FEATURE_DATA_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "poe2/minimax/labeling.hpp"

namespace poe2::minimax::feature_data {

inline constexpr std::uint32_t kFeatureDatasetSchemaVersion = 1;
inline constexpr std::uint32_t kFeatureDatasetHeaderSize = 128;
inline constexpr std::uint32_t kFeatureDatasetRecordSize = 432;
inline constexpr std::size_t kScoringLineCount = 36;
inline constexpr std::int16_t kOccupiedGain = std::numeric_limits<std::int16_t>::min();
inline constexpr std::string_view kFeatureBinaryFileName = "features.bin";
inline constexpr std::string_view kFeatureManifestFileName = "manifest.json";
inline constexpr std::string_view kFeatureIncompleteMarkerName = "INCOMPLETE";
inline constexpr std::string_view kFeatureCompleteMarkerName = "COMPLETE";

struct PositionFeatures {
  Score normalized_value = 0;
  Score b_value = 0;
  std::array<std::uint16_t, kScoringLineCount> line_patterns{};
  std::array<std::int16_t, kCellCount> own_gains{};
  std::array<std::int16_t, kCellCount> opponent_gains{};

  friend constexpr bool operator==(const PositionFeatures&, const PositionFeatures&) = default;
};

struct FeatureRecord {
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
  std::uint64_t deepest_completed_nodes = 0;
  std::uint64_t previous_completed_nodes = 0;
  std::uint32_t source_shard = 0;
  std::uint32_t source_ordinal = 0;
  Score teacher_value = 0;
  Score deepest_value = 0;
  Score previous_value = 0;
  Score normalized_value = 0;
  Score b_value = 0;
  Score residual = 0;
  std::uint32_t duplicate_count = 1;
  std::uint16_t policy_id = 0;
  std::uint16_t sample_index = 0;
  std::uint8_t ply = 0;
  Player side_to_move = Player::kOne;
  labeling::DatasetSplit split = labeling::DatasetSplit::kUnspecified;
  labeling::LabelMode mode = labeling::LabelMode::kTeacher;
  std::uint8_t completed_depth = 0;
  std::uint8_t deepest_completed_depth = 0;
  std::uint8_t previous_completed_depth = 0;
  std::uint8_t attempted_depth = 0;
  std::uint8_t terminal_depth = 0;
  std::uint8_t best_move_index = 0xff;
  std::uint8_t deepest_best_move_index = 0xff;
  std::uint8_t previous_best_move_index = 0xff;
  std::uint8_t flags = 0;
  std::array<std::uint16_t, kScoringLineCount> line_patterns{};
  std::array<std::int16_t, kCellCount> own_gains{};
  std::array<std::int16_t, kCellCount> opponent_gains{};

  friend constexpr bool operator==(const FeatureRecord&, const FeatureRecord&) = default;
};

inline constexpr std::uint8_t kTerminalLabelFlag = UINT8_C(1) << 0;
inline constexpr std::uint8_t kParityBackoffFlag = UINT8_C(1) << 1;

struct LabelProvenance {
  std::string git_commit;
  bool git_dirty = false;
  std::string mode;
  std::uint64_t node_limit = 0;
  std::size_t hash_bytes_effective = 0;
  std::size_t workers_requested = 0;
  std::string target_selection;

  friend bool operator==(const LabelProvenance&, const LabelProvenance&) = default;
};

struct FeatureDataset {
  std::string corpus_id;
  labeling::Sha256Digest corpus_digest{};
  labeling::Sha256Digest label_set_digest{};
  LabelProvenance labels;
  std::uint32_t shard_count = 0;
  std::size_t source_record_count = 0;
  std::size_t duplicate_group_count = 0;
  std::size_t duplicate_record_count = 0;
  std::size_t maximum_duplicate_count = 1;
  std::size_t representative_upgrade_count = 0;
  std::size_t varying_label_group_count = 0;
  std::vector<FeatureRecord> records;
};

[[nodiscard]] const std::array<std::uint8_t, kScoringLineCount>& scoring_line_lengths() noexcept;
[[nodiscard]] PositionFeatures extract_position_features(const Position& position);
[[nodiscard]] FeatureDataset build_feature_dataset(
    const std::filesystem::path& position_source_directory,
    const std::filesystem::path& label_corpus_directory);
[[nodiscard]] std::vector<std::uint8_t> serialize_feature_binary(const FeatureDataset& dataset);
[[nodiscard]] std::string serialize_feature_manifest(const FeatureDataset& dataset,
                                                     const labeling::Sha256Digest& binary_digest);
void write_feature_dataset(const std::filesystem::path& directory, const FeatureDataset& dataset);

}  // namespace poe2::minimax::feature_data

#endif  // POE2_MINIMAX_FEATURE_DATA_HPP
