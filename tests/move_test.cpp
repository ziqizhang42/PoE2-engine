#include "poe2/move.hpp"

#include <catch2/catch_test_macros.hpp>
#include <optional>
#include <string_view>

namespace {

poe2::Move require_parse(std::string_view text) {
  const std::optional<poe2::Move> move = poe2::parse_move(text);
  if (!move.has_value()) {
    FAIL("move should parse");
    return poe2::Move{.square = {}};
  }

  return *move;
}

void require_error(const poe2::MoveResult& result, poe2::MoveError expected) {
  if (!result.error.has_value()) {
    FAIL("move error is missing");
    return;
  }

  REQUIRE(*result.error == expected);
}

}  // namespace

TEST_CASE("moves parse and format board coordinates", "[move]") {
  const poe2::Move lower_corner = require_parse("a1");
  REQUIRE(lower_corner.square.row == 0);
  REQUIRE(lower_corner.square.col == 0);
  REQUIRE(poe2::format_move(lower_corner) == "a1");

  const poe2::Move upper_corner = require_parse("g7");
  REQUIRE(upper_corner.square.row == 6);
  REQUIRE(upper_corner.square.col == 6);
  REQUIRE(poe2::format_move(upper_corner) == "g7");

  const poe2::Move upper_case = require_parse("C4");
  REQUIRE(upper_case.square.row == 3);
  REQUIRE(upper_case.square.col == 2);
  REQUIRE(poe2::format_move(upper_case) == "c4");
}

TEST_CASE("move parser rejects malformed coordinates", "[move]") {
  REQUIRE_FALSE(poe2::parse_move("").has_value());
  REQUIRE_FALSE(poe2::parse_move("a").has_value());
  REQUIRE_FALSE(poe2::parse_move("a10").has_value());
  REQUIRE_FALSE(poe2::parse_move("h1").has_value());
  REQUIRE_FALSE(poe2::parse_move("a0").has_value());
  REQUIRE_FALSE(poe2::parse_move("11").has_value());
  REQUIRE_FALSE(poe2::parse_move("aa").has_value());

  REQUIRE(poe2::format_move(poe2::Move{.square = {-1, 0}}).empty());
}

TEST_CASE("apply_move accepts legal moves and rejects invalid moves", "[move][position]") {
  poe2::Position position;

  const poe2::Move first{.square = {0, 0}};
  const poe2::Move occupied{.square = {0, 0}};
  const poe2::Move out_of_bounds{.square = {-1, 0}};

  const poe2::MoveResult accepted = poe2::apply_move(position, first);
  REQUIRE(accepted.accepted);
  REQUIRE_FALSE(accepted.error.has_value());
  REQUIRE_FALSE(accepted.game_result.has_value());
  REQUIRE(position.ply() == 1);
  REQUIRE(position.side_to_move() == poe2::Player::kTwo);
  REQUIRE(position.board().cell_at(first.square) == poe2::Cell::kPlayerOne);

  const poe2::MoveResult occupied_result = poe2::apply_move(position, occupied);
  REQUIRE_FALSE(occupied_result.accepted);
  require_error(occupied_result, poe2::MoveError::kOccupied);
  REQUIRE_FALSE(occupied_result.game_result.has_value());
  REQUIRE(position.ply() == 1);
  REQUIRE(position.side_to_move() == poe2::Player::kTwo);

  const poe2::MoveResult out_of_bounds_result = poe2::apply_move(position, out_of_bounds);
  REQUIRE_FALSE(out_of_bounds_result.accepted);
  require_error(out_of_bounds_result, poe2::MoveError::kOutOfBounds);
  REQUIRE_FALSE(out_of_bounds_result.game_result.has_value());
  REQUIRE(position.ply() == 1);
  REQUIRE(position.side_to_move() == poe2::Player::kTwo);
}

TEST_CASE("apply_move reports terminal results and rejects moves after game over", "[move][game]") {
  poe2::Position position;
  poe2::MoveResult result;

  for (int row = 0; row < poe2::kBoardSize; ++row) {
    for (int col = 0; col < poe2::kBoardSize; ++col) {
      result = poe2::apply_move(position, poe2::Move{.square = {row, col}});
      REQUIRE(result.accepted);
    }
  }

  REQUIRE(position.is_full());
  REQUIRE(result.game_result.has_value());

  const poe2::MoveResult after_game = poe2::apply_move(position, poe2::Move{.square = {0, 0}});
  REQUIRE_FALSE(after_game.accepted);
  require_error(after_game, poe2::MoveError::kGameOver);
  REQUIRE(after_game.game_result.has_value());
}

TEST_CASE("move errors have stable names", "[move]") {
  REQUIRE(poe2::move_error_name(poe2::MoveError::kOutOfBounds) == "out_of_bounds");
  REQUIRE(poe2::move_error_name(poe2::MoveError::kGameOver) == "game_over");
  REQUIRE(poe2::move_error_name(poe2::MoveError::kOccupied) == "occupied");
}
