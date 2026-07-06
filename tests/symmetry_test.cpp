#include "poe2/symmetry.hpp"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <initializer_list>

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
