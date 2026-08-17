#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "poe2/minimax/evaluation.hpp"
#include "poe2/minimax/labeling.hpp"
#include "poe2/symmetry.hpp"

namespace {

[[nodiscard]] poe2::Position position_with_empty_squares(poe2::Bitboard empty_squares) {
  poe2::Position position;
  for (int index = 0; index < poe2::kCellCount; ++index) {
    if ((empty_squares & (poe2::Bitboard{1} << index)) == 0) {
      REQUIRE(position.play(poe2::square_from_index(index)));
    }
  }
  return position;
}

[[nodiscard]] poe2::Score terminal_oracle(poe2::Position& position) {
  poe2::Bitboard moves = position.legal_moves();
  if (moves == 0) {
    return poe2::minimax::evaluate(position);
  }

  poe2::Score best = std::numeric_limits<poe2::Score>::lowest();
  while (moves != 0) {
    const int move_index = std::countr_zero(moves);
    moves &= moves - poe2::Bitboard{1};
    poe2::MoveUndo undo;
    REQUIRE(position.make_move(poe2::square_from_index(move_index), undo));
    best = std::max(best, -terminal_oracle(position));
    position.unmake_move(undo);
  }
  return best;
}

[[nodiscard]] std::uint32_t read_u32(std::span<const std::uint8_t> bytes, std::size_t offset) {
  REQUIRE(offset + sizeof(std::uint32_t) <= bytes.size());
  std::uint32_t value = 0;
  for (std::size_t index = 0; index < sizeof(value); ++index) {
    value |= static_cast<std::uint32_t>(bytes[offset + index]) << (8 * index);
  }
  return value;
}

[[nodiscard]] std::uint64_t read_u64(std::span<const std::uint8_t> bytes, std::size_t offset) {
  REQUIRE(offset + sizeof(std::uint64_t) <= bytes.size());
  std::uint64_t value = 0;
  for (std::size_t index = 0; index < sizeof(value); ++index) {
    value |= static_cast<std::uint64_t>(bytes[offset + index]) << (8 * index);
  }
  return value;
}

[[nodiscard]] std::string_view byte_string(const std::vector<std::uint8_t>& bytes) {
  return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}

}  // namespace

TEST_CASE("exact label generation reaches terminal values and records canonical provenance",
          "[minimax][labeling][exact]") {
  const std::vector<poe2::minimax::labeling::LabelInput> inputs{
      {
          .position = position_with_empty_squares(
              poe2::square_bit({0, 0}) | poe2::square_bit({2, 3}) | poe2::square_bit({6, 6})),
          .source_id = UINT64_C(0x1111222233334444),
          .source_line = 7,
      },
      {
          .position =
              position_with_empty_squares(poe2::square_bit({0, 1}) | poe2::square_bit({1, 4}) |
                                          poe2::square_bit({4, 2}) | poe2::square_bit({6, 5})),
          .source_id = UINT64_C(0x5555666677778888),
          .source_line = 11,
      },
  };

  const poe2::minimax::labeling::LabelDataset dataset = poe2::minimax::labeling::generate_labels(
      inputs,
      poe2::minimax::labeling::LabelingOptions{
          .mode = poe2::minimax::labeling::LabelMode::kExact,
          .node_limit = 100'000,
          .hash_bytes = 0,
          .require_all = true,
      },
      "fixture positions", UINT64_C(0xabcdef));

  REQUIRE(dataset.records.size() == inputs.size());
  REQUIRE(dataset.unsolved_source_lines.empty());
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    poe2::Position oracle_position = inputs[index].position;
    const poe2::Score expected = terminal_oracle(oracle_position);
    const poe2::minimax::labeling::LabelRecord& record = dataset.records[index];
    CAPTURE(index, record.source_line);
    REQUIRE(record.value == expected);
    REQUIRE(record.mode == poe2::minimax::labeling::LabelMode::kExact);
    REQUIRE(record.completed_depth == record.terminal_depth);
    REQUIRE(record.nodes <= dataset.options.node_limit);
    REQUIRE(record.source_id == inputs[index].source_id);
    REQUIRE(record.source_line == inputs[index].source_line);
    REQUIRE(record.player_one == inputs[index].position.board().bits(poe2::Player::kOne));
    REQUIRE(record.player_two == inputs[index].position.board().bits(poe2::Player::kTwo));
    REQUIRE(record.canonical_key ==
            poe2::canonicalize_position_key(inputs[index].position.key()).key);
    REQUIRE((inputs[index].position.legal_moves() &
             (poe2::Bitboard{1} << record.best_move_index)) != 0);
  }
}

TEST_CASE("label generation is byte deterministic and has a fixed versioned header",
          "[minimax][labeling][determinism]") {
  const std::vector<poe2::minimax::labeling::LabelInput> inputs{{
      .position = position_with_empty_squares(poe2::square_bit({0, 0}) | poe2::square_bit({3, 4}) |
                                              poe2::square_bit({6, 6})),
      .source_id = 42,
      .source_line = 3,
  }};
  const poe2::minimax::labeling::LabelingOptions options{
      .mode = poe2::minimax::labeling::LabelMode::kExact,
      .node_limit = 10'000,
      .hash_bytes = poe2::minimax::kMebibyte,
      .require_all = true,
  };

  const poe2::minimax::labeling::LabelDataset first =
      poe2::minimax::labeling::generate_labels(inputs, options, "fixture\nname", 99);
  const poe2::minimax::labeling::LabelDataset second =
      poe2::minimax::labeling::generate_labels(inputs, options, "fixture\nname", 99);
  const std::vector<std::uint8_t> first_binary = poe2::minimax::labeling::serialize_binary(first);
  const std::vector<std::uint8_t> second_binary = poe2::minimax::labeling::serialize_binary(second);

  REQUIRE(first.records == second.records);
  REQUIRE(first_binary == second_binary);
  REQUIRE(first_binary.size() == poe2::minimax::labeling::kLabelDatasetHeaderSize +
                                     poe2::minimax::labeling::kLabelDatasetRecordSize);
  REQUIRE(std::string_view{reinterpret_cast<const char*>(first_binary.data()), 7} == "POE2LBL");
  REQUIRE(read_u32(first_binary, 8) == poe2::minimax::labeling::kLabelDatasetSchemaVersion);
  REQUIRE(read_u32(first_binary, 12) == poe2::minimax::labeling::kLabelDatasetHeaderSize);
  REQUIRE(read_u32(first_binary, 16) == poe2::minimax::labeling::kLabelDatasetRecordSize);
  REQUIRE(read_u32(first_binary, 20) == UINT32_C(0x01020304));
  REQUIRE(read_u64(first_binary, 24) == 1);
  REQUIRE(read_u64(first_binary, 32) == 1);
  REQUIRE(read_u64(first_binary, 40) == 99);

  const std::uint64_t digest = poe2::minimax::labeling::stable_digest(byte_string(first_binary));
  REQUIRE(poe2::minimax::labeling::serialize_manifest(first, digest) ==
          poe2::minimax::labeling::serialize_manifest(second, digest));
  REQUIRE(poe2::minimax::labeling::serialize_manifest(first, digest).find("fixture\\nname") !=
          std::string::npos);
}

TEST_CASE("parallel label generation is identical to serial generation and preserves input order",
          "[minimax][labeling][determinism][workers]") {
  std::vector<poe2::minimax::labeling::LabelInput> inputs;
  constexpr std::size_t kInputCount = 12;
  inputs.reserve(kInputCount);
  for (std::size_t index = 0; index < kInputCount; ++index) {
    const poe2::Bitboard empty_squares = (poe2::Bitboard{1} << index) |
                                         (poe2::Bitboard{1} << (index + kInputCount)) |
                                         (poe2::Bitboard{1} << (index + 2 * kInputCount)) |
                                         (poe2::Bitboard{1} << (index + 3 * kInputCount));
    inputs.push_back(poe2::minimax::labeling::LabelInput{
        .position = position_with_empty_squares(empty_squares),
        .source_id = 1'000 + index,
        .source_line = static_cast<std::uint32_t>(200 + index),
    });
  }

  const poe2::minimax::labeling::LabelingOptions serial_options{
      .mode = poe2::minimax::labeling::LabelMode::kExact,
      .node_limit = 100'000,
      .hash_bytes = 0,
      .workers = 1,
      .require_all = true,
  };
  poe2::minimax::labeling::LabelingOptions parallel_options = serial_options;
  parallel_options.workers = 6;

  const poe2::minimax::labeling::LabelDataset serial =
      poe2::minimax::labeling::generate_labels(inputs, serial_options, "parallel fixture", 123);
  const poe2::minimax::labeling::LabelDataset parallel =
      poe2::minimax::labeling::generate_labels(inputs, parallel_options, "parallel fixture", 123);
  const poe2::minimax::labeling::LabelDataset repeated_parallel =
      poe2::minimax::labeling::generate_labels(inputs, parallel_options, "parallel fixture", 123);

  REQUIRE(serial.workers_used == 1);
  REQUIRE(parallel.workers_used == 6);
  REQUIRE(repeated_parallel.workers_used == 6);
  REQUIRE(parallel.records == serial.records);
  REQUIRE(repeated_parallel.records == parallel.records);
  REQUIRE(parallel.unsolved_source_lines.empty());
  REQUIRE(poe2::minimax::labeling::serialize_binary(parallel) ==
          poe2::minimax::labeling::serialize_binary(serial));
  REQUIRE(poe2::minimax::labeling::serialize_binary(repeated_parallel) ==
          poe2::minimax::labeling::serialize_binary(parallel));
  for (std::size_t index = 0; index < parallel.records.size(); ++index) {
    REQUIRE(parallel.records[index].source_line == inputs[index].source_line);
  }

  const std::string manifest = poe2::minimax::labeling::serialize_manifest(parallel, 456);
  REQUIRE(manifest.find("\"workers_requested\": 6") != std::string::npos);
  REQUIRE(manifest.find("\"workers_used\": 6") != std::string::npos);

  poe2::minimax::labeling::LabelingOptions invalid_options = serial_options;
  invalid_options.workers = 0;
  REQUIRE_THROWS_AS(
      poe2::minimax::labeling::generate_labels(inputs, invalid_options, "parallel fixture", 123),
      std::invalid_argument);
}

TEST_CASE("exact mode rejects incomplete searches while teacher mode records completed work",
          "[minimax][labeling][teacher][nodes]") {
  const std::vector<poe2::minimax::labeling::LabelInput> inputs{{
      .position = poe2::Position{},
      .source_id = 1,
      .source_line = 5,
  }};
  const poe2::minimax::labeling::LabelDataset incomplete = poe2::minimax::labeling::generate_labels(
      inputs,
      poe2::minimax::labeling::LabelingOptions{
          .mode = poe2::minimax::labeling::LabelMode::kExact,
          .node_limit = 100,
          .hash_bytes = 0,
      },
      "fixture", 1);
  REQUIRE(incomplete.records.empty());
  REQUIRE(incomplete.unsolved_source_lines == std::vector<std::uint32_t>{5});

  REQUIRE_THROWS_AS(poe2::minimax::labeling::generate_labels(
                        inputs,
                        poe2::minimax::labeling::LabelingOptions{
                            .mode = poe2::minimax::labeling::LabelMode::kExact,
                            .node_limit = 100,
                            .hash_bytes = 0,
                            .require_all = true,
                        },
                        "fixture", 1),
                    std::runtime_error);

  const poe2::minimax::labeling::LabelDataset teacher = poe2::minimax::labeling::generate_labels(
      inputs,
      poe2::minimax::labeling::LabelingOptions{
          .mode = poe2::minimax::labeling::LabelMode::kTeacher,
          .node_limit = 100,
          .hash_bytes = 0,
          .require_all = true,
      },
      "fixture", 1);
  REQUIRE(teacher.records.size() == 1);
  REQUIRE(teacher.records.front().mode == poe2::minimax::labeling::LabelMode::kTeacher);
  REQUIRE(teacher.records.front().completed_depth < teacher.records.front().terminal_depth);
  REQUIRE(teacher.records.front().nodes == 100);
}
