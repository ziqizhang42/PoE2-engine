#include <catch2/catch_test_macros.hpp>
#include <vector>

#include "poe2/minimax/evaluation.hpp"
#include "poe2/symmetry.hpp"

TEST_CASE("minimax evaluation uses exact half-point handicap", "[minimax][evaluation]") {
  poe2::Position position;

  REQUIRE(position.side_to_move() == poe2::Player::kOne);
  REQUIRE(poe2::minimax::evaluate(position) == -poe2::kPlayerTwoHandicapHalfPoints);

  REQUIRE(position.play({0, 0}));
  REQUIRE(position.side_to_move() == poe2::Player::kTwo);
  REQUIRE(poe2::minimax::evaluate(position) == poe2::kPlayerTwoHandicapHalfPoints - 2);

  REQUIRE(position.play({6, 6}));
  REQUIRE(position.side_to_move() == poe2::Player::kOne);
  REQUIRE(poe2::minimax::evaluate(position) == -poe2::kPlayerTwoHandicapHalfPoints);
}

TEST_CASE("minimax evaluation is invariant under every board symmetry",
          "[minimax][evaluation][symmetry]") {
  const std::vector<poe2::Square> history{
      {0, 1}, {6, 5}, {2, 4}, {4, 2}, {1, 6}, {5, 0}, {3, 3}, {0, 4}, {6, 2},
  };
  poe2::Position base;
  for (const poe2::Square square : history) {
    REQUIRE(base.play(square));
  }
  const poe2::Score expected = poe2::minimax::evaluate(base);

  for (const poe2::Symmetry symmetry : poe2::kAllSymmetries) {
    poe2::Position equivalent;
    for (const poe2::Square square : history) {
      REQUIRE(equivalent.play(poe2::transform_square(symmetry, square)));
    }
    REQUIRE(poe2::minimax::evaluate(equivalent) == expected);
  }
}
