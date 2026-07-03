#include "poe2/board.hpp"

#include <catch2/catch_test_macros.hpp>
#include <initializer_list>

namespace {

void require_place(poe2::Board& board, poe2::Player player, poe2::Square square) {
  REQUIRE(board.place(player, square));
}

void require_places(poe2::Board& board, poe2::Player player,
                    std::initializer_list<poe2::Square> squares) {
  for (const poe2::Square square : squares) {
    require_place(board, player, square);
  }
}

}  // namespace

TEST_CASE("board starts empty", "[board]") {
  const poe2::Board board;

  REQUIRE(board.piece_count() == 0);
  REQUIRE_FALSE(board.is_full());
  REQUIRE(board.cell_at({0, 0}) == poe2::Cell::kEmpty);
  REQUIRE(board.can_place({0, 0}));
}

TEST_CASE("board rejects invalid and occupied squares", "[board]") {
  poe2::Board board;

  REQUIRE_FALSE(board.can_place({-1, 0}));
  REQUIRE_FALSE(board.can_place({0, -1}));
  REQUIRE_FALSE(board.can_place({poe2::kBoardSize, 0}));
  REQUIRE_FALSE(board.can_place({0, poe2::kBoardSize}));
  REQUIRE_FALSE(board.is_empty({-1, 0}));

  REQUIRE(board.place(poe2::Player::kOne, {3, 4}));
  REQUIRE(board.cell_at({3, 4}) == poe2::Cell::kPlayerOne);
  REQUIRE_FALSE(board.can_place({3, 4}));
  REQUIRE_FALSE(board.place(poe2::Player::kTwo, {3, 4}));
}

TEST_CASE("side to play alternates turns after legal moves", "[position]") {
  poe2::Position position;

  REQUIRE(position.side_to_move() == poe2::Player::kOne);
  REQUIRE(position.ply() == 0);

  REQUIRE(position.play({0, 0}));
  REQUIRE(position.side_to_move() == poe2::Player::kTwo);
  REQUIRE(position.ply() == 1);
  REQUIRE(position.board().cell_at({0, 0}) == poe2::Cell::kPlayerOne);

  REQUIRE(position.play({0, 1}));
  REQUIRE(position.side_to_move() == poe2::Player::kOne);
  REQUIRE(position.ply() == 2);
  REQUIRE(position.board().cell_at({0, 1}) == poe2::Cell::kPlayerTwo);

  REQUIRE_FALSE(position.play({0, 1}));
  REQUIRE(position.side_to_move() == poe2::Player::kOne);
  REQUIRE(position.ply() == 2);
}

TEST_CASE("isolated pieces count as single-point lines", "[score]") {
  poe2::Board board;
  require_place(board, poe2::Player::kOne, {0, 0});
  require_place(board, poe2::Player::kOne, {2, 3});

  REQUIRE(poe2::score(board, poe2::Player::kOne) == 2);
  REQUIRE(poe2::score(board, poe2::Player::kTwo) == 0);
}

TEST_CASE("diagonal neighbors form a line instead of isolated singles", "[score]") {
  poe2::Board board;
  require_place(board, poe2::Player::kOne, {0, 0});
  require_place(board, poe2::Player::kOne, {1, 1});

  REQUIRE(poe2::score(board, poe2::Player::kOne) == 2);
}

TEST_CASE("maximal horizontal lines score without counting subsets", "[score]") {
  poe2::Board board;
  require_place(board, poe2::Player::kOne, {1, 1});
  require_place(board, poe2::Player::kOne, {1, 2});
  require_place(board, poe2::Player::kOne, {1, 3});
  require_place(board, poe2::Player::kOne, {1, 4});

  REQUIRE(poe2::score(board, poe2::Player::kOne) == 8);
}

TEST_CASE("separated runs in the same row score independently", "[score]") {
  poe2::Board board;
  require_place(board, poe2::Player::kOne, {2, 0});
  require_place(board, poe2::Player::kOne, {2, 1});
  require_place(board, poe2::Player::kOne, {2, 3});
  require_place(board, poe2::Player::kOne, {2, 4});
  require_place(board, poe2::Player::kOne, {2, 5});

  REQUIRE(poe2::score(board, poe2::Player::kOne) == 6);
}

TEST_CASE("opponent pieces split lines", "[score]") {
  poe2::Board board;
  require_place(board, poe2::Player::kOne, {0, 0});
  require_place(board, poe2::Player::kOne, {0, 1});
  require_place(board, poe2::Player::kTwo, {0, 2});
  require_place(board, poe2::Player::kOne, {0, 3});
  require_place(board, poe2::Player::kOne, {0, 4});

  REQUIRE(poe2::score(board, poe2::Player::kOne) == 4);
  REQUIRE(poe2::score(board, poe2::Player::kTwo) == 1);
}

TEST_CASE("overlapping lines score across every direction they create", "[score]") {
  poe2::Board board;
  require_place(board, poe2::Player::kOne, {3, 1});
  require_place(board, poe2::Player::kOne, {3, 2});
  require_place(board, poe2::Player::kOne, {3, 3});
  require_place(board, poe2::Player::kOne, {1, 3});
  require_place(board, poe2::Player::kOne, {2, 3});

  REQUIRE(poe2::score(board, poe2::Player::kOne) == 10);
}

TEST_CASE("corner triangle counts every length-two axis it creates", "[score]") {
  poe2::Board board;
  require_place(board, poe2::Player::kOne, {0, 0});
  require_place(board, poe2::Player::kOne, {0, 1});
  require_place(board, poe2::Player::kOne, {1, 0});

  REQUIRE(poe2::score(board, poe2::Player::kOne) == 6);
}

TEST_CASE("both diagonal directions are scored", "[score]") {
  poe2::Board board;
  require_place(board, poe2::Player::kOne, {1, 1});
  require_place(board, poe2::Player::kOne, {2, 2});
  require_place(board, poe2::Player::kOne, {3, 3});
  require_place(board, poe2::Player::kOne, {1, 5});
  require_place(board, poe2::Player::kOne, {2, 4});

  REQUIRE(poe2::score(board, poe2::Player::kOne) == 8);
}

TEST_CASE("edge to edge diagonal scores as one maximal line", "[score]") {
  poe2::Board board;
  require_place(board, poe2::Player::kTwo, {0, 6});
  require_place(board, poe2::Player::kTwo, {1, 5});
  require_place(board, poe2::Player::kTwo, {2, 4});
  require_place(board, poe2::Player::kTwo, {3, 3});
  require_place(board, poe2::Player::kTwo, {4, 2});
  require_place(board, poe2::Player::kTwo, {5, 1});
  require_place(board, poe2::Player::kTwo, {6, 0});

  REQUIRE(poe2::score(board, poe2::Player::kOne) == 0);
  REQUIRE(poe2::score(board, poe2::Player::kTwo) == 64);
}

TEST_CASE("messy mixed board scores only maximal lines and true singles", "[score]") {
  poe2::Board board;
  require_places(board, poe2::Player::kOne,
                 {{0, 0},
                  {0, 1},
                  {0, 2},
                  {1, 1},
                  {2, 2},
                  {3, 3},
                  {4, 4},
                  {5, 5},
                  {6, 6},
                  {3, 0},
                  {4, 0},
                  {6, 2}});
  require_places(board, poe2::Player::kTwo,
                 {{0, 5}, {1, 5}, {2, 5}, {3, 5}, {2, 0}, {2, 1}, {4, 2}, {5, 3}, {5, 0}});

  // P1: 4 + 2 + 2 + 64 + 2 + 1. P2: 8 + 2 + 2 + 1.
  const poe2::ScoreByPlayer scores = poe2::score(board);
  REQUIRE(scores.player_one == 75);
  REQUIRE(scores.player_two == 13);
}

TEST_CASE("dense checkerboard scores all diagonal maximal lines", "[score]") {
  poe2::Board board;
  for (int row = 0; row < poe2::kBoardSize; ++row) {
    for (int col = 0; col < poe2::kBoardSize; ++col) {
      const poe2::Player player = (row + col) % 2 == 0 ? poe2::Player::kOne : poe2::Player::kTwo;
      require_place(board, player, {row, col});
    }
  }

  // P1 owns diagonal lengths 3, 5, 7, 5, 3 on each diagonal axis.
  // P2 owns diagonal lengths 2, 4, 6, 6, 4, 2 on each diagonal axis.
  const poe2::ScoreByPlayer scores = poe2::score(board);
  REQUIRE(scores.player_one == 208);
  REQUIRE(scores.player_two == 168);
}

TEST_CASE("score board reports both players", "[score]") {
  poe2::Board board;
  require_place(board, poe2::Player::kOne, {0, 0});
  require_place(board, poe2::Player::kOne, {0, 1});
  require_place(board, poe2::Player::kTwo, {2, 0});
  require_place(board, poe2::Player::kTwo, {2, 1});
  require_place(board, poe2::Player::kTwo, {2, 2});

  const poe2::ScoreByPlayer scores = poe2::score(board);

  REQUIRE(scores.player_one == 2);
  REQUIRE(scores.player_two == 4);
}

TEST_CASE("player two handicap is represented exactly in half-points", "[score]") {
  REQUIRE(poe2::leader_after_handicap({.player_one = 10, .player_two = 4}) == poe2::Player::kOne);
  REQUIRE(poe2::leader_after_handicap({.player_one = 10, .player_two = 5}) == poe2::Player::kTwo);
}

TEST_CASE("winner is only available for full boards", "[game]") {
  poe2::Board board;

  REQUIRE_FALSE(poe2::winner_if_full(board).has_value());

  for (int row = 0; row < poe2::kBoardSize; ++row) {
    for (int col = 0; col < poe2::kBoardSize; ++col) {
      require_place(board, poe2::Player::kOne, {row, col});
    }
  }

  REQUIRE(board.is_full());
  REQUIRE(poe2::winner_if_full(board) == poe2::Player::kOne);
}
