#include <catch2/catch_test_macros.hpp>

#include "poe2/minimax/evaluation.hpp"

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
