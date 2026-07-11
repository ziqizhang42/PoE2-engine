#include "poe2/symmetry.hpp"

#include <array>
#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace {

void require_square(poe2::Symmetry symmetry, poe2::Square square, poe2::Square expected) {
  const poe2::Square actual = poe2::transform_square(symmetry, square);

  REQUIRE(actual.row == expected.row);
  REQUIRE(actual.col == expected.col);
}

poe2::Bitboard squares(std::initializer_list<poe2::Square> squares) {
  poe2::Bitboard bits = 0;
  for (const poe2::Square square : squares) {
    bits |= poe2::square_bit(square);
  }
  return bits;
}

bool key_less(poe2::PositionKey lhs, poe2::PositionKey rhs) {
  return lhs.low < rhs.low || (lhs.low == rhs.low && lhs.high < rhs.high);
}

[[nodiscard]] std::uint64_t next_random(std::uint64_t& state) {
  state = state * 6364136223846793005ULL + 1442695040888963407ULL;
  return state;
}

[[nodiscard]] poe2::Square select_move(poe2::Bitboard legal_moves, std::uint64_t random) {
  int offset = static_cast<int>(random % static_cast<std::uint64_t>(std::popcount(legal_moves)));
  while (offset-- > 0) {
    legal_moves &= legal_moves - poe2::Bitboard{1};
  }
  return poe2::square_from_index(std::countr_zero(legal_moves));
}

void require_tracker_matches(const poe2::Position& position,
                             const poe2::PositionSymmetryTracker& tracker) {
  REQUIRE(tracker.side_to_move() == position.side_to_move());

  for (const poe2::Symmetry symmetry : poe2::kAllSymmetries) {
    const poe2::PositionKey expected = poe2::transform_position_key(symmetry, position.key());
    REQUIRE(tracker.key(symmetry) == expected);
    REQUIRE(tracker.hash(symmetry) == poe2::position_key_hash(expected));
  }

  const poe2::CanonicalPositionKey brute = poe2::canonicalize_position_key(position.key());
  const poe2::CanonicalPositionView incremental = tracker.canonical_view();
  REQUIRE(incremental.key == brute.key);
  REQUIRE(incremental.hash == poe2::position_key_hash(brute.key));
  REQUIRE(incremental.transform == brute.transform);
  REQUIRE(incremental.inverse_transform == brute.inverse_transform);
}

[[nodiscard]] bool contains_key(const std::vector<poe2::PositionKey>& keys,
                                poe2::PositionKey candidate) {
  for (const poe2::PositionKey key : keys) {
    if (key == candidate) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] poe2::Position position_from_history(const std::vector<poe2::Square>& history,
                                                   poe2::Symmetry symmetry) {
  poe2::Position position;
  for (const poe2::Square square : history) {
    REQUIRE(position.play(poe2::transform_square(symmetry, square)));
  }
  return position;
}

}  // namespace

TEST_CASE("symmetry transforms representative squares", "[symmetry]") {
  require_square(poe2::Symmetry::kIdentity, {1, 4}, {1, 4});
  require_square(poe2::Symmetry::kRotate90, {1, 4}, {4, 5});
  require_square(poe2::Symmetry::kRotate180, {1, 4}, {5, 2});
  require_square(poe2::Symmetry::kRotate270, {1, 4}, {2, 1});
  require_square(poe2::Symmetry::kReflectHorizontal, {1, 4}, {5, 4});
  require_square(poe2::Symmetry::kReflectVertical, {1, 4}, {1, 2});
  require_square(poe2::Symmetry::kReflectMainDiagonal, {1, 4}, {4, 1});
  require_square(poe2::Symmetry::kReflectAntiDiagonal, {1, 4}, {2, 5});

  require_square(poe2::Symmetry::kRotate90, {0, 0}, {0, 6});
  require_square(poe2::Symmetry::kRotate180, {0, 0}, {6, 6});
  require_square(poe2::Symmetry::kRotate270, {0, 0}, {6, 0});
  require_square(poe2::Symmetry::kReflectHorizontal, {0, 3}, {6, 3});
  require_square(poe2::Symmetry::kReflectVertical, {3, 0}, {3, 6});

  for (const poe2::Symmetry symmetry : poe2::kAllSymmetries) {
    require_square(symmetry, {3, 3}, {3, 3});
  }
}

TEST_CASE("every symmetry has a square inverse", "[symmetry]") {
  for (const poe2::Symmetry symmetry : poe2::kAllSymmetries) {
    const poe2::Symmetry inverse = poe2::inverse_symmetry(symmetry);
    for (int index = 0; index < poe2::kCellCount; ++index) {
      const poe2::Square square = poe2::square_from_index(index);
      const poe2::Square restored =
          poe2::transform_square(inverse, poe2::transform_square(symmetry, square));

      REQUIRE(restored == square);
    }
  }
}

TEST_CASE("bitboard transforms match transformed individual squares", "[symmetry]") {
  const std::array<poe2::Square, 5> occupied{{{0, 0}, {0, 5}, {1, 4}, {3, 3}, {6, 2}}};
  poe2::Bitboard bits = 0;
  for (const poe2::Square square : occupied) {
    bits |= poe2::square_bit(square);
  }

  for (const poe2::Symmetry symmetry : poe2::kAllSymmetries) {
    poe2::Bitboard expected = 0;
    for (const poe2::Square square : occupied) {
      expected |= poe2::square_bit(poe2::transform_square(symmetry, square));
    }

    REQUIRE(poe2::transform_bitboard(symmetry, bits) == expected);
  }
}

TEST_CASE("position-key transforms preserve player assignment and side to move",
          "[symmetry][key]") {
  const poe2::Bitboard player_one = squares({{0, 0}, {1, 4}, {3, 2}});
  const poe2::Bitboard player_two = squares({{2, 0}, {6, 6}});
  const poe2::PositionKey key = poe2::make_position_key(player_one, player_two, poe2::Player::kTwo);

  const poe2::PositionKey transformed =
      poe2::transform_position_key(poe2::Symmetry::kRotate90, key);

  REQUIRE(poe2::position_key_bits(transformed, poe2::Player::kOne) ==
          poe2::transform_bitboard(poe2::Symmetry::kRotate90, player_one));
  REQUIRE(poe2::position_key_bits(transformed, poe2::Player::kTwo) ==
          poe2::transform_bitboard(poe2::Symmetry::kRotate90, player_two));
  REQUIRE(poe2::position_key_side_to_move(transformed) == poe2::Player::kTwo);
}

TEST_CASE("canonical keys match across equivalent orientations", "[symmetry][key]") {
  const poe2::PositionKey key = poe2::make_position_key(
      squares({{0, 1}, {1, 4}, {3, 2}}), squares({{2, 6}, {5, 0}}), poe2::Player::kTwo);
  const poe2::CanonicalPositionKey canonical = poe2::canonicalize_position_key(key);

  for (const poe2::Symmetry symmetry : poe2::kAllSymmetries) {
    const poe2::PositionKey equivalent = poe2::transform_position_key(symmetry, key);
    const poe2::CanonicalPositionKey equivalent_canonical =
        poe2::canonicalize_position_key(equivalent);

    REQUIRE(equivalent_canonical.key == canonical.key);
    REQUIRE(poe2::position_key_side_to_move(equivalent_canonical.key) == poe2::Player::kTwo);
  }
}

TEST_CASE("canonicalization metadata maps moves to canonical orientation and back",
          "[symmetry][key]") {
  const poe2::PositionKey key = poe2::make_position_key(
      squares({{0, 1}, {2, 5}, {4, 3}}), squares({{1, 6}, {5, 0}}), poe2::Player::kOne);
  const poe2::CanonicalPositionKey canonical = poe2::canonicalize_position_key(key);
  const poe2::Square original_move{1, 5};

  REQUIRE(poe2::transform_position_key(canonical.transform, key) == canonical.key);

  const poe2::Square canonical_move = poe2::transform_square(canonical.transform, original_move);
  const poe2::Square restored_move =
      poe2::transform_square(canonical.inverse_transform, canonical_move);

  REQUIRE(restored_move == original_move);
}

TEST_CASE("asymmetric position-key canonicalization is deterministic", "[symmetry][key]") {
  const poe2::PositionKey key =
      poe2::make_position_key(squares({{0, 1}, {1, 5}, {2, 2}, {4, 6}}),
                              squares({{0, 4}, {3, 1}, {5, 5}}), poe2::Player::kTwo);

  const poe2::CanonicalPositionKey first = poe2::canonicalize_position_key(key);
  const poe2::CanonicalPositionKey second = poe2::canonicalize_position_key(key);

  REQUIRE(second.key == first.key);
  REQUIRE(second.transform == first.transform);
  REQUIRE(second.inverse_transform == first.inverse_transform);

  for (const poe2::Symmetry symmetry : poe2::kAllSymmetries) {
    const poe2::PositionKey transformed = poe2::transform_position_key(symmetry, key);
    REQUIRE_FALSE(key_less(transformed, first.key));
  }
}

TEST_CASE("transformed move table matches square transforms", "[symmetry][table]") {
  for (std::size_t symmetry = 0; symmetry < poe2::kAllSymmetries.size(); ++symmetry) {
    for (int index = 0; index < poe2::kCellCount; ++index) {
      const poe2::Square square = poe2::square_from_index(index);
      REQUIRE(poe2::transformed_move_bits[symmetry][index] ==
              poe2::square_bit(poe2::transform_square(poe2::kAllSymmetries[symmetry], square)));
    }
  }
}

TEST_CASE("incremental symmetry keys previews and hashes match brute force through unmake",
          "[symmetry][tracker]") {
  std::uint64_t random = 0x6a09e667f3bcc909ULL;

  for (int game = 0; game < 8; ++game) {
    poe2::Position position;
    poe2::PositionSymmetryTracker tracker{position.key()};
    std::vector<poe2::MoveUndo> undos;
    std::vector<poe2::Square> moves;
    require_tracker_matches(position, tracker);

    while (!position.is_full()) {
      const poe2::Square move = select_move(position.legal_moves(), next_random(random));
      const poe2::PositionKey tracker_key_before = tracker.key();
      const poe2::PositionHash tracker_hash_before = tracker.hash();
      const poe2::CanonicalPositionView preview = tracker.preview_move(move);

      poe2::Position child = position;
      REQUIRE(child.play(move));
      const poe2::CanonicalPositionKey brute_child = poe2::canonicalize_position_key(child.key());
      REQUIRE(preview.key == brute_child.key);
      REQUIRE(preview.hash == poe2::position_key_hash(brute_child.key));
      REQUIRE(preview.transform == brute_child.transform);
      REQUIRE(preview.inverse_transform == brute_child.inverse_transform);
      REQUIRE(tracker.key() == tracker_key_before);
      REQUIRE(tracker.hash() == tracker_hash_before);

      poe2::MoveUndo undo;
      REQUIRE(position.make_move(move, undo));
      REQUIRE(tracker.make_move(move));
      undos.push_back(undo);
      moves.push_back(move);
      require_tracker_matches(position, tracker);
      REQUIRE(tracker.canonical_view() == preview);
    }

    while (!moves.empty()) {
      tracker.unmake_move(moves.back());
      position.unmake_move(undos.back());
      moves.pop_back();
      undos.pop_back();
      require_tracker_matches(position, tracker);
    }
  }
}

TEST_CASE("symmetry tracker rejects invalid and occupied moves without changing state",
          "[symmetry][tracker]") {
  poe2::PositionSymmetryTracker tracker{poe2::Position{}.key()};
  const poe2::CanonicalPositionView empty = tracker.canonical_view();

  REQUIRE_FALSE(tracker.make_move({-1, 0}));
  REQUIRE(tracker.canonical_view() == empty);

  const poe2::Square move{3, 3};
  REQUIRE(tracker.make_move(move));
  const poe2::CanonicalPositionView after_move = tracker.canonical_view();
  REQUIRE_FALSE(tracker.make_move(move));
  REQUIRE(tracker.canonical_view() == after_move);

  tracker.unmake_move(move);
  REQUIRE(tracker.canonical_view() == empty);
}

TEST_CASE("stabilizer move orbits produce the same successors as brute-force deduplication",
          "[symmetry][tracker][orbit]") {
  std::vector<std::vector<poe2::Square>> histories{
      {},
      {{3, 3}},
      {{0, 0}, {1, 0}, {0, 6}, {1, 6}},
      {{0, 1}, {6, 5}, {2, 4}, {4, 2}, {1, 6}, {5, 0}, {3, 3}},
  };

  for (const std::vector<poe2::Square>& history : histories) {
    const poe2::Position position = position_from_history(history, poe2::Symmetry::kIdentity);
    const poe2::PositionSymmetryTracker tracker{position.key()};

    std::uint8_t expected_stabilizer = 0;
    for (std::size_t index = 0; index < poe2::kAllSymmetries.size(); ++index) {
      if (poe2::transform_position_key(poe2::kAllSymmetries[index], position.key()) ==
          position.key()) {
        expected_stabilizer |= static_cast<std::uint8_t>(std::uint8_t{1} << index);
      }
    }
    REQUIRE(tracker.stabilizer_mask() == expected_stabilizer);

    std::vector<poe2::PositionKey> brute_successors;
    poe2::Bitboard brute_moves = position.legal_moves();
    while (brute_moves != 0) {
      const int index = std::countr_zero(brute_moves);
      brute_moves &= brute_moves - poe2::Bitboard{1};
      poe2::Position child = position;
      REQUIRE(child.play(poe2::square_from_index(index)));
      const poe2::PositionKey key = poe2::canonicalize_position_key(child.key()).key;
      if (!contains_key(brute_successors, key)) {
        brute_successors.push_back(key);
      }
    }

    std::vector<poe2::PositionKey> orbit_successors;
    poe2::Bitboard remaining_moves = position.legal_moves();
    while (remaining_moves != 0) {
      const int index = std::countr_zero(remaining_moves);
      const poe2::Square move = poe2::square_from_index(index);
      const poe2::Bitboard orbit = tracker.transformed_move_orbit(move);
      REQUIRE((orbit & poe2::square_bit(move)) != 0);

      poe2::Bitboard equivalent_moves = remaining_moves & orbit;
      const poe2::PositionKey representative = tracker.preview_move(move).key;
      while (equivalent_moves != 0) {
        const int equivalent_index = std::countr_zero(equivalent_moves);
        equivalent_moves &= equivalent_moves - poe2::Bitboard{1};
        REQUIRE(tracker.preview_move(poe2::square_from_index(equivalent_index)).key ==
                representative);
      }

      remaining_moves &= ~orbit;
      if (!contains_key(orbit_successors, representative)) {
        orbit_successors.push_back(representative);
      }
    }

    REQUIRE(orbit_successors.size() == brute_successors.size());
    for (const poe2::PositionKey key : brute_successors) {
      REQUIRE(contains_key(orbit_successors, key));
    }
  }
}

TEST_CASE("equivalent orientations share incremental canonical views and remap moves",
          "[symmetry][tracker][move]") {
  const std::vector<poe2::Square> history{
      {0, 1}, {6, 5}, {2, 4}, {4, 2}, {1, 6}, {5, 0}, {3, 3}, {0, 4}, {6, 2},
  };
  const poe2::Square live_move{2, 5};
  const poe2::Position base = position_from_history(history, poe2::Symmetry::kIdentity);
  const poe2::PositionSymmetryTracker base_tracker{base.key()};
  const poe2::PositionKey expected_child = base_tracker.preview_move(live_move).key;

  for (const poe2::Symmetry symmetry : poe2::kAllSymmetries) {
    const poe2::Position equivalent = position_from_history(history, symmetry);
    const poe2::PositionSymmetryTracker tracker{equivalent.key()};
    const poe2::Square equivalent_move = poe2::transform_square(symmetry, live_move);
    const poe2::CanonicalPositionView preview = tracker.preview_move(equivalent_move);
    const poe2::Square canonical_move = poe2::transform_square(preview.transform, equivalent_move);

    REQUIRE(tracker.canonical_view().key == base_tracker.canonical_view().key);
    REQUIRE(preview.key == expected_child);
    REQUIRE(poe2::transform_square(preview.inverse_transform, canonical_move) == equivalent_move);
    REQUIRE((poe2::position_key_bits(preview.key, equivalent.side_to_move()) &
             poe2::square_bit(canonical_move)) != 0);
  }
}
