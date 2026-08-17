#ifndef POE2_MINIMAX_POSITION_SOURCE_HPP
#define POE2_MINIMAX_POSITION_SOURCE_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "poe2/minimax/labeling.hpp"

namespace poe2::minimax::position_source {

inline constexpr std::uint32_t kPositionSourceSchemaVersion = 1;
inline constexpr std::string_view kSourceManifestFileName = "manifest.json";
inline constexpr std::string_view kSourceIncompleteMarkerName = "INCOMPLETE";
inline constexpr std::string_view kSourceCompleteMarkerName = "COMPLETE";
inline constexpr std::string_view kSourceShardDirectoryName = "shards";

enum class SourcePolicy : std::uint8_t {
  kRandom = 1,
  kImmediateGain = 2,
  kOpponentAware = 3,
  kNoisySearch = 4,
};

struct PolicyWeights {
  std::uint32_t random = 30;
  std::uint32_t immediate_gain = 20;
  std::uint32_t opponent_aware = 20;
  std::uint32_t noisy_search = 30;

  friend constexpr bool operator==(const PolicyWeights&, const PolicyWeights&) = default;
};

struct PositionSourceOptions {
  std::string corpus_id;
  std::uint64_t seed = 0;
  std::uint64_t trajectory_count = 0;
  std::uint16_t samples_per_trajectory = 8;
  std::uint32_t shard_count = 1;
  std::size_t workers = 1;
  std::uint64_t search_nodes = 10'000;
  std::size_t search_hash_bytes = 8 * kMebibyte;
  std::uint8_t noise_percent = 15;
  PolicyWeights policy_weights;
};

struct PositionSourceRecord {
  Bitboard player_one = 0;
  Bitboard player_two = 0;
  std::uint64_t source_id = 0;
  std::uint64_t family_id = 0;
  std::uint64_t trajectory_id = 0;
  std::uint64_t parent_id = 0;
  std::uint64_t trajectory_index = 0;
  SourcePolicy policy = SourcePolicy::kRandom;
  std::uint16_t sample_index = 0;
  labeling::DatasetSplit split = labeling::DatasetSplit::kUnspecified;
  std::uint8_t ply = 0;

  friend constexpr bool operator==(const PositionSourceRecord&,
                                   const PositionSourceRecord&) = default;
};

struct PositionSourceCorpus {
  PositionSourceOptions options;
  std::size_t workers_used = 0;
  std::size_t duplicate_positions = 0;
  std::vector<PositionSourceRecord> records;
};

struct PositionSourceProgress {
  std::uint64_t completed_trajectories = 0;
  std::uint64_t total_trajectories = 0;
};

using PositionSourceProgressSink = std::function<void(const PositionSourceProgress&)>;

struct SerializedSourceShard {
  std::uint32_t shard_index = 0;
  std::uint64_t trajectory_begin = 0;
  std::uint64_t trajectory_end = 0;
  std::size_t record_count = 0;
  std::string bytes;
  labeling::Sha256Digest digest{};
};

struct ReadSourceShard {
  labeling::LabelSource source;
  std::vector<labeling::LabelInput> inputs;
};

class SourceOutput final {
 public:
  SourceOutput(const SourceOutput&) = delete;
  SourceOutput& operator=(const SourceOutput&) = delete;
  SourceOutput(SourceOutput&&) noexcept = default;
  SourceOutput& operator=(SourceOutput&&) noexcept = default;

  [[nodiscard]] const std::filesystem::path& directory() const noexcept;
  [[nodiscard]] bool committed() const noexcept;

 private:
  explicit SourceOutput(std::filesystem::path directory);

  std::filesystem::path directory_;
  bool committed_ = false;

  friend SourceOutput reserve_source_output(const std::filesystem::path& directory);
  friend void write_position_source(SourceOutput& output, const PositionSourceCorpus& corpus);
};

[[nodiscard]] std::string_view source_policy_name(SourcePolicy policy) noexcept;
[[nodiscard]] PositionSourceCorpus generate_position_source(
    const PositionSourceOptions& options, const PositionSourceProgressSink& progress = {},
    std::uint64_t progress_interval = 0);
[[nodiscard]] std::vector<SerializedSourceShard> serialize_source_shards(
    const PositionSourceCorpus& corpus);
[[nodiscard]] std::string serialize_source_manifest(const PositionSourceCorpus& corpus,
                                                    std::span<const SerializedSourceShard> shards);
[[nodiscard]] SourceOutput reserve_source_output(const std::filesystem::path& directory);
void write_position_source(SourceOutput& output, const PositionSourceCorpus& corpus);
[[nodiscard]] ReadSourceShard read_position_source_shard(const std::filesystem::path& directory,
                                                         std::uint32_t shard_index);

}  // namespace poe2::minimax::position_source

#endif  // POE2_MINIMAX_POSITION_SOURCE_HPP
