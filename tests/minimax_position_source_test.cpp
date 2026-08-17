#include <array>
#include <atomic>
#include <barrier>
#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include "poe2/minimax/position_source.hpp"

namespace {

namespace labeling = poe2::minimax::labeling;
namespace position_source = poe2::minimax::position_source;

[[nodiscard]] position_source::PositionSourceOptions random_options(std::size_t workers) {
  return position_source::PositionSourceOptions{
      .corpus_id = "position-source-test",
      .seed = UINT64_C(0x0123456789abcdef),
      .trajectory_count = 16,
      .samples_per_trajectory = 4,
      .shard_count = 3,
      .workers = workers,
      .search_nodes = 100,
      .search_hash_bytes = 0,
      .noise_percent = 15,
      .policy_weights =
          position_source::PolicyWeights{
              .random = 1,
              .immediate_gain = 0,
              .opponent_aware = 0,
              .noisy_search = 0,
          },
  };
}

[[nodiscard]] std::filesystem::path unique_temporary_root(std::string_view prefix) {
  return std::filesystem::temp_directory_path() /
         (std::string{prefix} + '-' +
          std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
}

}  // namespace

TEST_CASE("position source is byte deterministic across worker counts",
          "[minimax][position-source][determinism][workers]") {
  const position_source::PositionSourceCorpus serial =
      position_source::generate_position_source(random_options(1));
  const position_source::PositionSourceCorpus parallel =
      position_source::generate_position_source(random_options(6));

  REQUIRE(serial.workers_used == 1);
  REQUIRE(parallel.workers_used == 6);
  REQUIRE(serial.records == parallel.records);
  REQUIRE(serial.duplicate_positions == parallel.duplicate_positions);
  REQUIRE(serial.records.size() == 64);

  const std::vector<position_source::SerializedSourceShard> serial_shards =
      position_source::serialize_source_shards(serial);
  const std::vector<position_source::SerializedSourceShard> parallel_shards =
      position_source::serialize_source_shards(parallel);
  REQUIRE(serial_shards.size() == 3);
  REQUIRE(parallel_shards.size() == serial_shards.size());
  for (std::size_t index = 0; index < serial_shards.size(); ++index) {
    REQUIRE(parallel_shards[index].bytes == serial_shards[index].bytes);
    REQUIRE(parallel_shards[index].digest == serial_shards[index].digest);
    REQUIRE(parallel_shards[index].record_count == serial_shards[index].record_count);
  }

  std::uint64_t trajectory_index = 0;
  std::uint16_t sample_index = 0;
  int previous_ply = 0;
  for (const position_source::PositionSourceRecord& record : serial.records) {
    CAPTURE(trajectory_index, sample_index);
    REQUIRE(record.trajectory_index == trajectory_index);
    REQUIRE(record.sample_index == sample_index);
    REQUIRE(record.policy == position_source::SourcePolicy::kRandom);
    REQUIRE(record.source_id != 0);
    REQUIRE(record.family_id != 0);
    REQUIRE(record.trajectory_id != 0);
    REQUIRE(record.parent_id == 0);
    REQUIRE((record.player_one & record.player_two) == 0);
    REQUIRE(std::popcount(record.player_one | record.player_two) == record.ply);
    REQUIRE(record.ply > previous_ply);
    REQUIRE((record.split == labeling::DatasetSplit::kTrain ||
             record.split == labeling::DatasetSplit::kValidation ||
             record.split == labeling::DatasetSplit::kTest));
    ++sample_index;
    previous_ply = record.ply;
    if (sample_index == serial.options.samples_per_trajectory) {
      ++trajectory_index;
      sample_index = 0;
      previous_ply = 0;
    }
  }
  REQUIRE(trajectory_index == serial.options.trajectory_count);

  position_source::PositionSourceOptions mixed_serial_options = random_options(1);
  mixed_serial_options.trajectory_count = 8;
  mixed_serial_options.samples_per_trajectory = 2;
  mixed_serial_options.shard_count = 2;
  mixed_serial_options.search_nodes = 25;
  mixed_serial_options.policy_weights = {};
  position_source::PositionSourceOptions mixed_parallel_options = mixed_serial_options;
  mixed_parallel_options.workers = 6;
  const position_source::PositionSourceCorpus mixed_serial =
      position_source::generate_position_source(mixed_serial_options);
  const position_source::PositionSourceCorpus mixed_parallel =
      position_source::generate_position_source(mixed_parallel_options);
  REQUIRE(mixed_parallel.records == mixed_serial.records);
  const auto mixed_serial_shards = position_source::serialize_source_shards(mixed_serial);
  const auto mixed_parallel_shards = position_source::serialize_source_shards(mixed_parallel);
  REQUIRE(mixed_parallel_shards.size() == mixed_serial_shards.size());
  for (std::size_t index = 0; index < mixed_serial_shards.size(); ++index) {
    REQUIRE(mixed_parallel_shards[index].bytes == mixed_serial_shards[index].bytes);
  }
}

TEST_CASE("every position source policy produces legal sampled trajectories",
          "[minimax][position-source][policies]") {
  constexpr std::array<position_source::SourcePolicy, 4> kPolicies{{
      position_source::SourcePolicy::kRandom,
      position_source::SourcePolicy::kImmediateGain,
      position_source::SourcePolicy::kOpponentAware,
      position_source::SourcePolicy::kNoisySearch,
  }};
  for (const position_source::SourcePolicy policy : kPolicies) {
    position_source::PolicyWeights weights{
        .random = 0,
        .immediate_gain = 0,
        .opponent_aware = 0,
        .noisy_search = 0,
    };
    switch (policy) {
      case position_source::SourcePolicy::kRandom:
        weights.random = 1;
        break;
      case position_source::SourcePolicy::kImmediateGain:
        weights.immediate_gain = 1;
        break;
      case position_source::SourcePolicy::kOpponentAware:
        weights.opponent_aware = 1;
        break;
      case position_source::SourcePolicy::kNoisySearch:
        weights.noisy_search = 1;
        break;
    }
    const position_source::PositionSourceCorpus corpus =
        position_source::generate_position_source(position_source::PositionSourceOptions{
            .corpus_id = "policy-test",
            .seed = static_cast<std::uint64_t>(policy),
            .trajectory_count = 1,
            .samples_per_trajectory = 2,
            .shard_count = 1,
            .workers = 1,
            .search_nodes = 50,
            .search_hash_bytes = 0,
            .noise_percent = 10,
            .policy_weights = weights,
        });
    REQUIRE(corpus.records.size() == 2);
    for (const position_source::PositionSourceRecord& record : corpus.records) {
      CAPTURE(static_cast<int>(policy), record.ply);
      REQUIRE(record.policy == policy);
      REQUIRE(std::popcount(record.player_one) == (record.ply + 1) / 2);
      REQUIRE(std::popcount(record.player_two) == record.ply / 2);
    }
  }
}

TEST_CASE("completed position source shards round trip into label inputs",
          "[minimax][position-source][output][reader]") {
  namespace fs = std::filesystem;
  position_source::PositionSourceOptions options = random_options(3);
  options.trajectory_count = 6;
  options.samples_per_trajectory = 3;
  options.shard_count = 2;
  const position_source::PositionSourceCorpus corpus =
      position_source::generate_position_source(options);

  const fs::path root = unique_temporary_root("poe2-position-source");
  const fs::path output_path = root / "source";
  fs::create_directories(root);
  position_source::SourceOutput output = position_source::reserve_source_output(output_path);
  position_source::write_position_source(output, corpus);
  REQUIRE(output.committed());
  REQUIRE(fs::is_regular_file(output_path / position_source::kSourceCompleteMarkerName));
  REQUIRE_FALSE(fs::exists(output_path / position_source::kSourceIncompleteMarkerName));

  std::size_t record_offset = 0;
  for (std::uint32_t shard_index = 0; shard_index < options.shard_count; ++shard_index) {
    const position_source::ReadSourceShard shard =
        position_source::read_position_source_shard(output_path, shard_index);
    REQUIRE(shard.source.corpus_id == options.corpus_id);
    REQUIRE(shard.source.shard_index == shard_index);
    REQUIRE(shard.source.shard_count == options.shard_count);
    for (const labeling::LabelInput& input : shard.inputs) {
      const position_source::PositionSourceRecord& record = corpus.records[record_offset++];
      REQUIRE(input.position.board().bits(poe2::Player::kOne) == record.player_one);
      REQUIRE(input.position.board().bits(poe2::Player::kTwo) == record.player_two);
      REQUIRE(input.source_id == record.source_id);
      REQUIRE(input.family_id == record.family_id);
      REQUIRE(input.trajectory_id == record.trajectory_id);
      REQUIRE(input.trajectory_index == record.trajectory_index);
      REQUIRE(input.policy_id == static_cast<std::uint16_t>(record.policy));
      REQUIRE(input.sample_index == record.sample_index);
      REQUIRE(input.split == record.split);
    }
  }
  REQUIRE(record_offset == corpus.records.size());
  REQUIRE_THROWS_AS(position_source::reserve_source_output(output_path), std::invalid_argument);
  fs::remove_all(root);
}

TEST_CASE("position source rejects incomplete deterministic configuration",
          "[minimax][position-source][validation]") {
  position_source::PositionSourceOptions options = random_options(1);
  options.trajectory_count = 0;
  REQUIRE_THROWS_AS(position_source::generate_position_source(options), std::invalid_argument);

  options = random_options(1);
  options.samples_per_trajectory = 9;
  REQUIRE_THROWS_AS(position_source::generate_position_source(options), std::invalid_argument);

  options = random_options(1);
  options.policy_weights = {};
  options.policy_weights.random = 0;
  options.policy_weights.immediate_gain = 0;
  options.policy_weights.opponent_aware = 0;
  options.policy_weights.noisy_search = 0;
  REQUIRE_THROWS_AS(position_source::generate_position_source(options), std::invalid_argument);
}

TEST_CASE("only one concurrent writer can reserve a position source",
          "[minimax][position-source][output][concurrency]") {
  namespace fs = std::filesystem;
  const fs::path root = unique_temporary_root("poe2-position-source-reservation");
  const fs::path output_path = root / "source";
  fs::create_directories(root);

  std::barrier start{2};
  std::atomic<int> successes = 0;
  std::atomic<int> failures = 0;
  const auto reserve = [&] {
    start.arrive_and_wait();
    try {
      auto output = position_source::reserve_source_output(output_path);
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
  REQUIRE(fs::is_regular_file(output_path / position_source::kSourceIncompleteMarkerName));
  REQUIRE_FALSE(fs::exists(output_path / position_source::kSourceCompleteMarkerName));
  fs::remove_all(root);
}
