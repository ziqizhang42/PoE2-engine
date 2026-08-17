#ifndef POE2_MINIMAX_LABELING_HPP
#define POE2_MINIMAX_LABELING_HPP

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "poe2/minimax/search.hpp"

namespace poe2::minimax::labeling {

inline constexpr std::uint32_t kLabelDatasetSchemaVersion = 1;
inline constexpr std::uint32_t kLabelDatasetHeaderSize = 48;
inline constexpr std::uint32_t kLabelDatasetRecordSize = 64;

enum class LabelMode : std::uint8_t {
  kExact = 1,
  kTeacher = 2,
};

struct LabelInput {
  Position position;
  std::uint64_t source_id = 0;
  std::uint32_t source_line = 0;
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
  std::uint64_t nodes = 0;
  std::uint32_t source_line = 0;
  Score value = 0;
  std::uint8_t ply = 0;
  Player side_to_move = Player::kOne;
  LabelMode mode = LabelMode::kExact;
  std::uint8_t completed_depth = 0;
  std::uint8_t terminal_depth = 0;
  std::uint8_t best_move_index = 0xff;

  friend constexpr bool operator==(const LabelRecord&, const LabelRecord&) = default;
};

struct LabelDataset {
  std::string source_name;
  std::uint64_t source_digest = 0;
  LabelingOptions options;
  std::size_t input_count = 0;
  std::size_t workers_used = 0;
  std::vector<LabelRecord> records;
  std::vector<std::uint32_t> unsolved_source_lines;
};

[[nodiscard]] std::string_view label_mode_name(LabelMode mode) noexcept;
[[nodiscard]] std::uint64_t stable_digest(std::string_view bytes) noexcept;
[[nodiscard]] LabelDataset generate_labels(std::span<const LabelInput> inputs,
                                           const LabelingOptions& options,
                                           std::string_view source_name,
                                           std::uint64_t source_digest);
[[nodiscard]] std::vector<std::uint8_t> serialize_binary(const LabelDataset& dataset);
[[nodiscard]] std::string serialize_manifest(const LabelDataset& dataset,
                                             std::uint64_t binary_digest);
void write_dataset(const LabelDataset& dataset, const std::filesystem::path& binary_path,
                   const std::filesystem::path& manifest_path);

}  // namespace poe2::minimax::labeling

#endif  // POE2_MINIMAX_LABELING_HPP
