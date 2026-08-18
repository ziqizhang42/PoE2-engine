#include <algorithm>
#include <array>
#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "poe2/minimax/evaluation.hpp"
#include "poe2/minimax/feature_data.hpp"
#include "poe2/symmetry.hpp"

namespace {

[[nodiscard]] std::uint32_t read_u32(std::span<const std::uint8_t> bytes, std::size_t offset) {
  REQUIRE(offset + sizeof(std::uint32_t) <= bytes.size());
  std::uint32_t value = 0;
  for (std::size_t index = 0; index < sizeof(value); ++index) {
    value |= static_cast<std::uint32_t>(bytes[offset + index]) << (index * 8);
  }
  return value;
}

[[nodiscard]] std::uint64_t read_u64(std::span<const std::uint8_t> bytes, std::size_t offset) {
  REQUIRE(offset + sizeof(std::uint64_t) <= bytes.size());
  std::uint64_t value = 0;
  for (std::size_t index = 0; index < sizeof(value); ++index) {
    value |= static_cast<std::uint64_t>(bytes[offset + index]) << (index * 8);
  }
  return value;
}

}  // namespace

TEST_CASE("offline feature extraction reuses exact B values and marginal gains",
          "[minimax][features][extraction]") {
  poe2::Position position;
  constexpr std::array<poe2::Square, 9> kMoves{{
      {0, 0},
      {6, 6},
      {0, 1},
      {5, 5},
      {2, 2},
      {4, 4},
      {3, 2},
      {1, 5},
      {3, 3},
  }};
  for (const poe2::Square move : kMoves) {
    REQUIRE(position.play(move));
  }

  const poe2::minimax::feature_data::PositionFeatures features =
      poe2::minimax::feature_data::extract_position_features(position);
  REQUIRE(features.normalized_value == poe2::minimax::evaluate(position));
  REQUIRE(features.b_value == poe2::minimax::evaluate_two_ply_closure(position));

  const auto& lengths = poe2::minimax::feature_data::scoring_line_lengths();
  REQUIRE(lengths.size() == poe2::minimax::feature_data::kScoringLineCount);
  REQUIRE(std::ranges::count(lengths, std::uint8_t{7}) == 16);
  for (std::uint8_t length = 2; length <= 6; ++length) {
    REQUIRE(std::ranges::count(lengths, length) == 4);
  }
  for (const std::uint16_t pattern : features.line_patterns) {
    REQUIRE(pattern < 3276);
  }

  const poe2::Bitboard occupied = position.board().occupied();
  for (int move_index = 0; move_index < poe2::kCellCount; ++move_index) {
    if ((occupied & (poe2::Bitboard{1} << move_index)) != 0) {
      REQUIRE(features.own_gains[move_index] == poe2::minimax::feature_data::kOccupiedGain);
      REQUIRE(features.opponent_gains[move_index] == poe2::minimax::feature_data::kOccupiedGain);
      continue;
    }
    const poe2::ScoreByPlayer gains = position.score_gains_unchecked(move_index);
    const poe2::Score own =
        position.side_to_move() == poe2::Player::kOne ? gains.player_one : gains.player_two;
    const poe2::Score opponent =
        position.side_to_move() == poe2::Player::kOne ? gains.player_two : gains.player_one;
    REQUIRE(features.own_gains[move_index] == own);
    REQUIRE(features.opponent_gains[move_index] == opponent);
  }
}

TEST_CASE("offline feature datasets have fixed deterministic framing",
          "[minimax][features][serialization]") {
  poe2::Position position;
  REQUIRE(position.play({0, 0}));
  REQUIRE(position.play({6, 6}));
  const poe2::minimax::feature_data::PositionFeatures values =
      poe2::minimax::feature_data::extract_position_features(position);
  const poe2::PositionKey canonical = poe2::canonicalize_position_key(position.key()).key;

  poe2::minimax::feature_data::FeatureDataset dataset{
      .corpus_id = "feature-test-corpus",
      .corpus_digest = poe2::minimax::labeling::sha256("feature-test-corpus"),
      .label_set_digest = poe2::minimax::labeling::sha256("feature-test-labels"),
      .labels =
          poe2::minimax::feature_data::LabelProvenance{
              .git_commit = "abc123",
              .mode = "teacher",
              .node_limit = 1'000,
              .hash_bytes_effective = 1'024,
              .workers_requested = 2,
              .target_selection = "deepest_terminal_parity",
          },
      .shard_count = 1,
      .source_record_count = 1,
  };
  dataset.records.push_back(poe2::minimax::feature_data::FeatureRecord{
      .player_one = position.board().bits(poe2::Player::kOne),
      .player_two = position.board().bits(poe2::Player::kTwo),
      .canonical_key = canonical,
      .source_id = 11,
      .family_id = 12,
      .trajectory_id = 13,
      .trajectory_index = 14,
      .completed_nodes = 500,
      .teacher_value = values.b_value + 4,
      .normalized_value = values.normalized_value,
      .b_value = values.b_value,
      .residual = 4,
      .policy_id = 1,
      .ply = static_cast<std::uint8_t>(position.ply()),
      .side_to_move = position.side_to_move(),
      .split = poe2::minimax::labeling::DatasetSplit::kTrain,
      .completed_depth = 1,
      .terminal_depth = 47,
      .best_move_index = 1,
      .line_patterns = values.line_patterns,
      .own_gains = values.own_gains,
      .opponent_gains = values.opponent_gains,
  });

  const std::vector<std::uint8_t> first =
      poe2::minimax::feature_data::serialize_feature_binary(dataset);
  const std::vector<std::uint8_t> second =
      poe2::minimax::feature_data::serialize_feature_binary(dataset);
  REQUIRE(first == second);
  REQUIRE(first.size() == poe2::minimax::feature_data::kFeatureDatasetHeaderSize +
                              poe2::minimax::feature_data::kFeatureDatasetRecordSize);
  REQUIRE(std::string_view{reinterpret_cast<const char*>(first.data()), 7} == "POE2FTR");
  REQUIRE(read_u32(first, 8) == poe2::minimax::feature_data::kFeatureDatasetSchemaVersion);
  REQUIRE(read_u32(first, 12) == poe2::minimax::feature_data::kFeatureDatasetHeaderSize);
  REQUIRE(read_u32(first, 16) == poe2::minimax::feature_data::kFeatureDatasetRecordSize);
  REQUIRE(read_u64(first, 24) == 1);
  REQUIRE(read_u64(first, 32) == 1);
  REQUIRE(read_u64(first, 40) == 0);

  const poe2::minimax::labeling::Sha256Digest digest =
      poe2::minimax::labeling::sha256(std::span<const std::uint8_t>{first});
  const std::string manifest =
      poe2::minimax::feature_data::serialize_feature_manifest(dataset, digest);
  REQUIRE(manifest.find("b-residual-line-pattern-gains-v1") != std::string::npos);
  REQUIRE(manifest.find("\"records\": 1") != std::string::npos);
}
