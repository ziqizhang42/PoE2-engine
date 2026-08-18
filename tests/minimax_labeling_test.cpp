#include <algorithm>
#include <atomic>
#include <barrier>
#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
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

[[nodiscard]] poe2::minimax::labeling::LabelSource label_source(std::string_view name = "fixture") {
  return poe2::minimax::labeling::LabelSource{
      .corpus_id = "labeling-test-corpus",
      .source_name = std::string{name},
      .source_digest = poe2::minimax::labeling::sha256("labeling test source"),
      .shard_index = 1,
      .shard_count = 3,
  };
}

}  // namespace

TEST_CASE("exact label generation reaches terminal values and records canonical provenance",
          "[minimax][labeling][exact]") {
  const std::vector<poe2::minimax::labeling::LabelInput> inputs{
      {
          .position = position_with_empty_squares(
              poe2::square_bit({0, 0}) | poe2::square_bit({2, 3}) | poe2::square_bit({6, 6})),
          .source_id = UINT64_C(0x1111222233334444),
          .family_id = 101,
          .trajectory_id = 201,
          .trajectory_index = 301,
          .source_line = 7,
          .source_ordinal = 0,
          .policy_id = 3,
          .sample_index = 4,
          .split = poe2::minimax::labeling::DatasetSplit::kValidation,
      },
      {
          .position =
              position_with_empty_squares(poe2::square_bit({0, 1}) | poe2::square_bit({1, 4}) |
                                          poe2::square_bit({4, 2}) | poe2::square_bit({6, 5})),
          .source_id = UINT64_C(0x5555666677778888),
          .family_id = 102,
          .trajectory_id = 202,
          .trajectory_index = 302,
          .source_line = 11,
          .source_ordinal = 1,
          .policy_id = 5,
          .sample_index = 6,
          .split = poe2::minimax::labeling::DatasetSplit::kTest,
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
      label_source("fixture positions"));

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
    REQUIRE(record.attempted_depth == record.terminal_depth);
    REQUIRE(record.deepest_completed_depth == record.completed_depth);
    REQUIRE(record.deepest_completed_nodes == record.completed_nodes);
    REQUIRE(record.deepest_value == record.value);
    REQUIRE(record.deepest_best_move_index == record.best_move_index);
    REQUIRE(record.nodes <= dataset.options.node_limit);
    REQUIRE(record.completed_nodes <= record.nodes);
    REQUIRE(record.source_id == inputs[index].source_id);
    REQUIRE(record.family_id == inputs[index].family_id);
    REQUIRE(record.trajectory_id == inputs[index].trajectory_id);
    REQUIRE(record.trajectory_index == inputs[index].trajectory_index);
    REQUIRE(record.source_line == inputs[index].source_line);
    REQUIRE(record.source_ordinal == inputs[index].source_ordinal);
    REQUIRE(record.policy_id == inputs[index].policy_id);
    REQUIRE(record.sample_index == inputs[index].sample_index);
    REQUIRE(record.split == inputs[index].split);
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
      poe2::minimax::labeling::generate_labels(inputs, options, label_source("fixture\nname"));
  const poe2::minimax::labeling::LabelDataset second =
      poe2::minimax::labeling::generate_labels(inputs, options, label_source("fixture\nname"));
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
  REQUIRE(std::equal(first.source.source_digest.begin(), first.source.source_digest.end(),
                     first_binary.begin() + 40));
  REQUIRE(std::equal(first.corpus_digest.begin(), first.corpus_digest.end(),
                     first_binary.begin() + 72));
  REQUIRE(read_u32(first_binary, 104) == 1);
  REQUIRE(read_u32(first_binary, 108) == 3);

  const poe2::minimax::labeling::Sha256Digest digest =
      poe2::minimax::labeling::sha256(byte_string(first_binary));
  REQUIRE(poe2::minimax::labeling::serialize_manifest(first, digest) ==
          poe2::minimax::labeling::serialize_manifest(second, digest));
  REQUIRE(poe2::minimax::labeling::serialize_manifest(first, digest).find("fixture\\nname") !=
          std::string::npos);
  REQUIRE(poe2::minimax::labeling::serialize_manifest(first, digest)
              .find("\"target_selection\": \"deepest_terminal_parity\"") != std::string::npos);
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
        .family_id = 2'000 + index,
        .trajectory_id = 3'000 + index,
        .source_line = static_cast<std::uint32_t>(200 + index),
        .source_ordinal = static_cast<std::uint32_t>(index),
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

  const poe2::minimax::labeling::LabelDataset serial = poe2::minimax::labeling::generate_labels(
      inputs, serial_options, label_source("parallel fixture"));
  const poe2::minimax::labeling::LabelDataset parallel = poe2::minimax::labeling::generate_labels(
      inputs, parallel_options, label_source("parallel fixture"));
  std::vector<poe2::minimax::labeling::LabelProgress> progress_updates;
  const poe2::minimax::labeling::LabelDataset repeated_parallel =
      poe2::minimax::labeling::generate_labels(
          inputs, parallel_options, label_source("parallel fixture"),
          [&progress_updates](const auto& progress) { progress_updates.push_back(progress); }, 5);

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
  REQUIRE(progress_updates.size() == 3);
  REQUIRE(progress_updates[0].completed == 5);
  REQUIRE(progress_updates[1].completed == 10);
  REQUIRE(progress_updates[2].completed == kInputCount);
  for (const auto& progress : progress_updates) {
    REQUIRE(progress.records + progress.unsolved == progress.completed);
  }
  for (std::size_t index = 0; index < parallel.records.size(); ++index) {
    REQUIRE(parallel.records[index].source_line == inputs[index].source_line);
  }

  const std::string manifest = poe2::minimax::labeling::serialize_manifest(
      parallel, poe2::minimax::labeling::sha256("parallel binary"));
  REQUIRE(manifest.find("\"workers_requested\": 6") != std::string::npos);
  REQUIRE(manifest.find("\"workers_used\": 6") != std::string::npos);

  poe2::minimax::labeling::LabelingOptions invalid_options = serial_options;
  invalid_options.workers = 0;
  REQUIRE_THROWS_AS(poe2::minimax::labeling::generate_labels(inputs, invalid_options,
                                                             label_source("parallel fixture")),
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
      label_source());
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
                        label_source()),
                    std::runtime_error);

  const poe2::minimax::labeling::LabelDataset teacher = poe2::minimax::labeling::generate_labels(
      inputs,
      poe2::minimax::labeling::LabelingOptions{
          .mode = poe2::minimax::labeling::LabelMode::kTeacher,
          .node_limit = 100,
          .hash_bytes = 0,
          .require_all = true,
      },
      label_source());
  REQUIRE(teacher.records.size() == 1);
  REQUIRE(teacher.records.front().mode == poe2::minimax::labeling::LabelMode::kTeacher);
  REQUIRE(teacher.records.front().completed_depth < teacher.records.front().terminal_depth);
  REQUIRE(teacher.records.front().completed_depth % 2 ==
          teacher.records.front().terminal_depth % 2);
  REQUIRE(teacher.records.front().attempted_depth ==
          teacher.records.front().deepest_completed_depth + 1);
  REQUIRE(teacher.records.front().completed_nodes < teacher.records.front().nodes);
  REQUIRE(teacher.records.front().nodes == 100);
}

TEST_CASE("teacher labels select terminal parity and retain the two deepest iterations",
          "[minimax][labeling][teacher][parity]") {
  const poe2::Position position = position_with_empty_squares(
      poe2::square_bit({0, 0}) | poe2::square_bit({0, 1}) | poe2::square_bit({0, 2}) |
      poe2::square_bit({0, 3}) | poe2::square_bit({0, 4}));
  poe2::minimax::Search control{poe2::minimax::SearchOptions{
      .hash_bytes = 0,
      .use_symmetry = true,
      .evaluator = poe2::minimax::Evaluator::kTwoPlyClosure,
  }};
  const poe2::engine::EngineResult depth_two =
      control.run(position, poe2::engine::EngineLimits{.depth = 2}, {});
  REQUIRE(depth_two.depth == 2);
  if (!depth_two.previous_iteration.has_value() || !depth_two.score.has_value() ||
      !depth_two.best_move.has_value()) {
    FAIL("depth-two search should return current and previous complete iterations");
    return;
  }
  const poe2::engine::CompletedSearchIteration& previous = *depth_two.previous_iteration;
  REQUIRE(previous.depth == 1);

  const std::vector<poe2::minimax::labeling::LabelInput> inputs{{
      .position = position,
      .source_id = 17,
      .source_line = 9,
  }};
  const poe2::minimax::labeling::LabelDataset dataset = poe2::minimax::labeling::generate_labels(
      inputs,
      poe2::minimax::labeling::LabelingOptions{
          .mode = poe2::minimax::labeling::LabelMode::kTeacher,
          .node_limit = depth_two.completed_nodes,
          .hash_bytes = 0,
          .require_all = true,
      },
      label_source("parity fixture"));

  REQUIRE(dataset.records.size() == 1);
  const auto& record = dataset.records.front();
  REQUIRE(record.terminal_depth == 3);
  REQUIRE(record.deepest_completed_depth == 2);
  REQUIRE(record.previous_completed_depth == 1);
  REQUIRE(record.completed_depth == 1);
  REQUIRE(record.attempted_depth == 3);
  REQUIRE(record.value == previous.score);
  REQUIRE(record.completed_nodes == previous.nodes);
  REQUIRE(record.best_move_index == poe2::square_index(previous.best_move.square));
  REQUIRE(record.deepest_value == *depth_two.score);
  REQUIRE(record.deepest_completed_nodes == depth_two.completed_nodes);
  REQUIRE(record.deepest_best_move_index == poe2::square_index(depth_two.best_move->square));
  REQUIRE(record.previous_value == record.value);
  REQUIRE(record.previous_completed_nodes == record.completed_nodes);
  REQUIRE(record.previous_best_move_index == record.best_move_index);
}

TEST_CASE("SHA-256 uses the standard byte representation", "[minimax][labeling][digest]") {
  REQUIRE(poe2::minimax::labeling::sha256_text(poe2::minimax::labeling::sha256("")) ==
          "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  REQUIRE(poe2::minimax::labeling::sha256_text(poe2::minimax::labeling::sha256("abc")) ==
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

TEST_CASE("only one concurrent writer can reserve a dataset directory",
          "[minimax][labeling][output][concurrency]") {
  namespace fs = std::filesystem;
  const std::string suffix =
      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  const fs::path root = fs::temp_directory_path() / ("poe2-label-reservation-" + suffix);
  const fs::path output_path = root / "dataset";
  fs::create_directories(root);

  std::barrier start{2};
  std::atomic<int> successes = 0;
  std::atomic<int> failures = 0;
  const auto reserve = [&] {
    start.arrive_and_wait();
    try {
      auto output = poe2::minimax::labeling::reserve_dataset_output(output_path);
      ++successes;
    } catch (const std::exception&) {
      ++failures;
    }
  };
  std::jthread first{reserve};
  std::jthread second{reserve};
  first.join();
  second.join();

  REQUIRE(successes == 1);
  REQUIRE(failures == 1);
  REQUIRE(fs::is_regular_file(output_path / poe2::minimax::labeling::kIncompleteMarkerName));
  REQUIRE_FALSE(fs::exists(output_path / poe2::minimax::labeling::kCompleteMarkerName));
  fs::remove_all(root);
}
