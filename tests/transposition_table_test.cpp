#include "poe2/transposition_table.hpp"

#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <optional>

namespace {

[[nodiscard]] poe2::PositionKey key_for(int first_index, int second_index,
                                        poe2::Player side_to_move = poe2::Player::kOne) {
  return poe2::make_position_key(poe2::square_bit(poe2::square_from_index(first_index)),
                                 poe2::square_bit(poe2::square_from_index(second_index)),
                                 side_to_move);
}

[[nodiscard]] poe2::TranspositionEntry require_entry(const poe2::TranspositionTable& table,
                                                     poe2::PositionKey key) {
  const std::optional<poe2::TranspositionEntry> entry = table.probe(key);
  if (!entry.has_value()) {
    FAIL("expected transposition-table entry");
    return {};
  }
  return *entry;
}

}  // namespace

TEST_CASE("transposition-table byte budgets choose the largest fitting power of two", "[tt]") {
  poe2::TranspositionTable table;
  const std::size_t budgets[] = {
      0, 31, 63, 64, 65, 127, 128, 191, 255, 256,
  };
  for (const std::size_t budget : budgets) {
    table.resize_bytes(budget);
    const std::size_t fitting_buckets = budget / 64;
    const std::size_t expected_buckets = fitting_buckets == 0 ? 0 : std::bit_floor(fitting_buckets);
    const std::size_t expected_capacity = expected_buckets * 2;

    REQUIRE(table.capacity() == expected_capacity);
    REQUIRE(table.storage_bytes() == expected_buckets * 64);
    REQUIRE(table.storage_bytes() <= budget);
    REQUIRE(table.empty());
  }
}

TEST_CASE("transposition-table capacity counts packed entry slots", "[tt][storage]") {
  poe2::TranspositionTable one_slot(1);
  REQUIRE(one_slot.capacity() == 1);
  REQUIRE(one_slot.storage_bytes() == 64);

  poe2::TranspositionTable rounded_slots(3);
  REQUIRE(rounded_slots.capacity() == 4);
  REQUIRE(rounded_slots.storage_bytes() == 128);
  REQUIRE(rounded_slots.storage_bytes() / rounded_slots.capacity() == 32);

  rounded_slots.resize_bytes(4096);
  REQUIRE(rounded_slots.capacity() == 128);
  REQUIRE(rounded_slots.storage_bytes() == 4096);
}

TEST_CASE("transposition-table entry-count resize remains compatible", "[tt][storage]") {
  poe2::TranspositionTable empty;
  REQUIRE(empty.capacity() == 0);
  REQUIRE(empty.empty());

  poe2::TranspositionTable table(3);
  REQUIRE(table.capacity() == 4);

  const poe2::PositionKey key = key_for(0, poe2::kCellCount - 1);
  REQUIRE(table.store(key, {.score = 10, .depth = 4}));
  REQUIRE(table.size() == 1);
  REQUIRE(table.probe(key).has_value());

  table.resize(17);
  REQUIRE(table.capacity() == 32);
  REQUIRE(table.empty());
  REQUIRE_FALSE(table.probe(key).has_value());

  table.resize(1);
  REQUIRE(table.capacity() == 1);
}

TEST_CASE("position overloads use exact bitboards side and incremental hash", "[tt][position]") {
  poe2::Position position;
  REQUIRE(position.play({0, 0}));
  REQUIRE(position.play({6, 6}));
  REQUIRE(position.play({0, 1}));

  poe2::TranspositionTable table(16);
  const poe2::TranspositionValue value{
      .score = 42,
      .depth = 6,
      .bound = poe2::TranspositionBound::kExact,
      .best_move = poe2::Square{4, 4},
  };
  REQUIRE(table.store(position, value));

  const std::optional<poe2::TranspositionEntry> hit = table.probe(position);
  if (!hit.has_value() || !hit->value.best_move.has_value()) {
    FAIL("stored position entry and best move should be available");
    return;
  }

  REQUIRE(hit->key == position.key());
  REQUIRE(hit->hash == position.hash());
  REQUIRE(hit->hash == poe2::position_key_hash(position.key()));
  REQUIRE(hit->value.score == value.score);
  REQUIRE(hit->value.depth == value.depth);
  REQUIRE(hit->value.bound == value.bound);
  REQUIRE(hit->value.best_move == value.best_move);

  const poe2::PositionKey key = position.key();
  const poe2::PositionKey other_side = poe2::make_position_key(
      poe2::position_key_bits(key, poe2::Player::kOne),
      poe2::position_key_bits(key, poe2::Player::kTwo), poe2::opponent(position.side_to_move()));
  const poe2::PositionKey swapped_players = poe2::make_position_key(
      poe2::position_key_bits(key, poe2::Player::kTwo),
      poe2::position_key_bits(key, poe2::Player::kOne), position.side_to_move());
  REQUIRE_FALSE(table.probe(other_side).has_value());
  REQUIRE_FALSE(table.probe(swapped_players).has_value());
}

TEST_CASE("a zero-sized transposition table rejects stores", "[tt]") {
  poe2::TranspositionTable table;
  const poe2::PositionKey key = key_for(0, poe2::kCellCount - 1);

  REQUIRE_FALSE(table.store(key, {.score = 10, .depth = 2}));
  REQUIRE_FALSE(table.probe(key).has_value());
  REQUIRE(table.capacity() == 0);
  REQUIRE(table.storage_bytes() == 0);
}

TEST_CASE("transposition-table stores report acceptance and protect same-key depth", "[tt]") {
  poe2::TranspositionTable table(1);
  const poe2::PositionKey key = key_for(0, poe2::kCellCount - 1);

  REQUIRE(table.store(key, {.score = 50, .depth = 5}));
  REQUIRE_FALSE(table.store(key, {.score = 40, .depth = 4}));
  REQUIRE(require_entry(table, key).value.score == 50);
  REQUIRE(require_entry(table, key).value.depth == 5);

  REQUIRE(table.store(key, {.score = 51, .depth = 5}));
  REQUIRE(require_entry(table, key).value.score == 51);
  REQUIRE(table.size() == 1);
}

TEST_CASE("transposition-table bucket collisions retain deeper entries", "[tt]") {
  poe2::TranspositionTable table(1);
  const poe2::PositionKey deeper = key_for(0, poe2::kCellCount - 1);
  const poe2::PositionKey shallower = key_for(1, poe2::kCellCount - 2, poe2::Player::kTwo);

  REQUIRE(table.store(deeper, {.score = 20, .depth = 4}));
  REQUIRE_FALSE(table.store(shallower, {.score = 10, .depth = 3}));
  REQUIRE(table.probe(deeper).has_value());
  REQUIRE_FALSE(table.probe(shallower).has_value());

  REQUIRE(table.store(shallower, {.score = 30, .depth = 5}));
  REQUIRE_FALSE(table.probe(deeper).has_value());
  REQUIRE(require_entry(table, shallower).value.score == 30);
}

TEST_CASE("transposition-table equal-depth replacement prefers exact entries", "[tt]") {
  poe2::TranspositionTable table(1);
  const poe2::PositionKey first = key_for(0, poe2::kCellCount - 1);
  const poe2::PositionKey second = key_for(2, poe2::kCellCount - 3, poe2::Player::kTwo);

  REQUIRE(table.store(first, {.score = 10, .depth = 4, .bound = poe2::TranspositionBound::kLower}));
  REQUIRE(
      table.store(second, {.score = 20, .depth = 4, .bound = poe2::TranspositionBound::kExact}));
  REQUIRE_FALSE(
      table.store(first, {.score = 30, .depth = 4, .bound = poe2::TranspositionBound::kUpper}));
  REQUIRE(require_entry(table, second).value.score == 20);

  REQUIRE(table.store(first, {.score = 40, .depth = 4, .bound = poe2::TranspositionBound::kExact}));
  REQUIRE(require_entry(table, first).value.score == 40);
}

TEST_CASE("transposition-table probes verify the full key and clear removes entries", "[tt]") {
  poe2::TranspositionTable table(1);
  const poe2::PositionKey stored = key_for(0, poe2::kCellCount - 1);
  const poe2::PositionKey collision = key_for(1, poe2::kCellCount - 2, poe2::Player::kTwo);

  REQUIRE(table.store(stored, {.score = 7, .depth = 1}));
  REQUIRE(table.probe(stored).has_value());
  REQUIRE_FALSE(table.probe(collision).has_value());

  table.clear();
  REQUIRE(table.empty());
  REQUIRE(table.capacity() == 1);
  REQUIRE_FALSE(table.probe(stored).has_value());
}

TEST_CASE("two-way buckets retain two exact keys selected by a supplied hash", "[tt][collision]") {
  poe2::TranspositionTable table(2);
  const poe2::PositionHash forced_hash = 0x1234;
  const poe2::PositionKey first = key_for(0, poe2::kCellCount - 1);
  const poe2::PositionKey second = key_for(1, poe2::kCellCount - 2, poe2::Player::kTwo);
  const poe2::PositionKey missing = key_for(2, poe2::kCellCount - 3);

  REQUIRE(table.store(first, forced_hash, {.score = 10, .depth = 2}));
  REQUIRE(table.store(second, forced_hash, {.score = 20, .depth = 3}));
  REQUIRE(table.size() == 2);

  const std::optional<poe2::TranspositionEntry> first_entry = table.probe(first, forced_hash);
  const std::optional<poe2::TranspositionEntry> second_entry = table.probe(second, forced_hash);
  if (!first_entry.has_value() || !second_entry.has_value()) {
    FAIL("both two-way entries should remain available");
    return;
  }
  REQUIRE(first_entry->hash == forced_hash);
  REQUIRE(second_entry->hash == forced_hash);
  REQUIRE_FALSE(table.probe(missing, forced_hash).has_value());
}

TEST_CASE("two-way replacement chooses the shallowest eligible entry", "[tt][replacement]") {
  poe2::TranspositionTable table(2);
  const poe2::PositionHash forced_hash = 7;
  const poe2::PositionKey deep = key_for(0, 48);
  const poe2::PositionKey shallow = key_for(1, 47);
  const poe2::PositionKey middle = key_for(2, 46);
  const poe2::PositionKey too_shallow = key_for(3, 45);

  REQUIRE(table.store(deep, forced_hash, {.score = 50, .depth = 5}));
  REQUIRE(table.store(shallow, forced_hash, {.score = 20, .depth = 2}));
  REQUIRE(table.store(middle, forced_hash, {.score = 30, .depth = 3}));
  REQUIRE(table.probe(deep, forced_hash).has_value());
  REQUIRE_FALSE(table.probe(shallow, forced_hash).has_value());
  REQUIRE(table.probe(middle, forced_hash).has_value());

  REQUIRE_FALSE(table.store(too_shallow, forced_hash, {.score = 10, .depth = 1}));
  REQUIRE_FALSE(table.probe(too_shallow, forced_hash).has_value());
  REQUIRE(table.size() == 2);
}

TEST_CASE("two-way replacement preserves exact ties and alternates equivalent victims",
          "[tt][replacement]") {
  const poe2::PositionHash forced_hash = 11;
  const poe2::PositionKey exact = key_for(0, 48);
  const poe2::PositionKey lower = key_for(1, 47);
  const poe2::PositionKey incoming_exact = key_for(2, 46);
  const poe2::PositionKey incoming_upper = key_for(3, 45);

  poe2::TranspositionTable exact_table(2);
  REQUIRE(exact_table.store(exact, forced_hash,
                            {.score = 1, .depth = 4, .bound = poe2::TranspositionBound::kExact}));
  REQUIRE(exact_table.store(lower, forced_hash,
                            {.score = 2, .depth = 4, .bound = poe2::TranspositionBound::kLower}));
  REQUIRE(exact_table.store(incoming_exact, forced_hash,
                            {.score = 3, .depth = 4, .bound = poe2::TranspositionBound::kExact}));
  REQUIRE(exact_table.probe(exact, forced_hash).has_value());
  REQUIRE_FALSE(exact_table.probe(lower, forced_hash).has_value());
  REQUIRE(exact_table.probe(incoming_exact, forced_hash).has_value());
  REQUIRE_FALSE(
      exact_table.store(incoming_upper, forced_hash,
                        {.score = 4, .depth = 4, .bound = poe2::TranspositionBound::kUpper}));

  const poe2::PositionKey first = key_for(4, 44);
  const poe2::PositionKey second = key_for(5, 43);
  const poe2::PositionKey third = key_for(6, 42);
  const poe2::PositionKey fourth = key_for(7, 41);
  poe2::TranspositionTable alternating_table(2);
  REQUIRE(alternating_table.store(
      first, forced_hash, {.score = 1, .depth = 3, .bound = poe2::TranspositionBound::kLower}));
  REQUIRE(alternating_table.store(
      second, forced_hash, {.score = 2, .depth = 3, .bound = poe2::TranspositionBound::kUpper}));
  REQUIRE(alternating_table.store(
      third, forced_hash, {.score = 3, .depth = 3, .bound = poe2::TranspositionBound::kLower}));
  REQUIRE_FALSE(alternating_table.probe(first, forced_hash).has_value());
  REQUIRE(alternating_table.probe(second, forced_hash).has_value());
  REQUIRE(alternating_table.probe(third, forced_hash).has_value());

  REQUIRE(alternating_table.store(
      fourth, forced_hash, {.score = 4, .depth = 3, .bound = poe2::TranspositionBound::kUpper}));
  REQUIRE_FALSE(alternating_table.probe(second, forced_hash).has_value());
  REQUIRE(alternating_table.probe(third, forced_hash).has_value());
  REQUIRE(alternating_table.probe(fourth, forced_hash).has_value());
}

TEST_CASE("prehashed probes use the supplied hash only for bucket selection", "[tt][hash]") {
  poe2::TranspositionTable table(4);
  const poe2::PositionKey key = key_for(0, 48);
  const poe2::PositionHash stored_hash = 0;

  REQUIRE(table.store(key, stored_hash, {.score = 12, .depth = 2}));
  REQUIRE(table.probe(key, stored_hash).has_value());
  REQUIRE_FALSE(table.probe(key, 1).has_value());
  REQUIRE_FALSE(table.probe(key_for(1, 47), stored_hash).has_value());

  table.clear();
  REQUIRE(table.empty());
  REQUIRE(table.capacity() == 4);
  REQUIRE_FALSE(table.probe(key, stored_hash).has_value());
}
