#include "poe2/board.hpp"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <initializer_list>
#include <optional>
#include <vector>

#include "poe2/transposition_table.hpp"

namespace {

struct PositionSnapshot {
  poe2::PositionKey key;
  poe2::PositionHash hash = 0;
  poe2::Player side_to_move = poe2::Player::kOne;
  int ply = 0;
  poe2::Bitboard legal_moves = poe2::kBoardMask;
  poe2::ScoreByPlayer scores;
};

void require_place(poe2::Board& board, poe2::Player player, poe2::Square square) {
  REQUIRE(board.place(player, square));
}

void require_places(poe2::Board& board, poe2::Player player,
                    std::initializer_list<poe2::Square> squares) {
  for (const poe2::Square square : squares) {
    require_place(board, player, square);
  }
}

void require_cached_scores_match(const poe2::Position& position) {
  const poe2::ScoreByPlayer cached = position.scores();
  const poe2::ScoreByPlayer full = poe2::score(position.board());
  const poe2::PositionKey key = position.key();

  REQUIRE(cached.player_one == full.player_one);
  REQUIRE(cached.player_two == full.player_two);
  REQUIRE(position.score(poe2::Player::kOne) == full.player_one);
  REQUIRE(position.score(poe2::Player::kTwo) == full.player_two);
  REQUIRE(poe2::position_key_bits(key, poe2::Player::kOne) ==
          position.board().bits(poe2::Player::kOne));
  REQUIRE(poe2::position_key_bits(key, poe2::Player::kTwo) ==
          position.board().bits(poe2::Player::kTwo));
  REQUIRE(poe2::position_key_side_to_move(key) == position.side_to_move());
  REQUIRE(position.hash() == poe2::position_key_hash(key));
}

void require_play_and_match(poe2::Position& position, poe2::Square square) {
  REQUIRE(position.play(square));
  require_cached_scores_match(position);
}

PositionSnapshot snapshot(const poe2::Position& position) {
  return PositionSnapshot{
      .key = position.key(),
      .hash = position.hash(),
      .side_to_move = position.side_to_move(),
      .ply = position.ply(),
      .legal_moves = position.legal_moves(),
      .scores = position.scores(),
  };
}

void require_snapshot(const poe2::Position& position, const PositionSnapshot& expected) {
  const poe2::PositionKey key = position.key();

  REQUIRE(key == expected.key);
  REQUIRE(position.hash() == expected.hash);
  REQUIRE(position.side_to_move() == expected.side_to_move);
  REQUIRE(position.ply() == expected.ply);
  REQUIRE(position.legal_moves() == expected.legal_moves);
  REQUIRE(position.scores().player_one == expected.scores.player_one);
  REQUIRE(position.scores().player_two == expected.scores.player_two);
}

}  // namespace

TEST_CASE("board starts empty", "[board]") {
  const poe2::Board board;

  REQUIRE(board.piece_count() == 0);
  REQUIRE(board.empty_count() == poe2::kCellCount);
  REQUIRE(board.empty_squares() == poe2::kBoardMask);
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
  REQUIRE(board.empty_count() == poe2::kCellCount - 1);
  REQUIRE((board.empty_squares() & poe2::square_bit({3, 4})) == 0);
  REQUIRE_FALSE(board.can_place({3, 4}));
  REQUIRE_FALSE(board.place(poe2::Player::kTwo, {3, 4}));
}

TEST_CASE("board removes only matching occupied squares", "[board]") {
  poe2::Board board;

  REQUIRE_FALSE(board.remove(poe2::Player::kOne, {0, 0}));
  REQUIRE_FALSE(board.remove(poe2::Player::kOne, {-1, 0}));

  REQUIRE(board.place(poe2::Player::kOne, {3, 4}));
  REQUIRE_FALSE(board.remove(poe2::Player::kTwo, {3, 4}));
  REQUIRE(board.remove(poe2::Player::kOne, {3, 4}));
  REQUIRE(board.cell_at({3, 4}) == poe2::Cell::kEmpty);
  REQUIRE(board.empty_count() == poe2::kCellCount);
  REQUIRE(board.can_place({3, 4}));
}

TEST_CASE("side to play alternates turns after legal moves", "[position]") {
  poe2::Position position;

  REQUIRE(position.side_to_move() == poe2::Player::kOne);
  REQUIRE(position.ply() == 0);
  REQUIRE(position.legal_moves() == poe2::kBoardMask);
  require_cached_scores_match(position);

  REQUIRE(position.play({0, 0}));
  REQUIRE(position.side_to_move() == poe2::Player::kTwo);
  REQUIRE(position.ply() == 1);
  REQUIRE(position.board().cell_at({0, 0}) == poe2::Cell::kPlayerOne);
  REQUIRE((position.legal_moves() & poe2::square_bit({0, 0})) == 0);
  require_cached_scores_match(position);

  REQUIRE(position.play({0, 1}));
  REQUIRE(position.side_to_move() == poe2::Player::kOne);
  REQUIRE(position.ply() == 2);
  REQUIRE(position.board().cell_at({0, 1}) == poe2::Cell::kPlayerTwo);
  require_cached_scores_match(position);

  REQUIRE_FALSE(position.play({0, 1}));
  REQUIRE(position.side_to_move() == poe2::Player::kOne);
  REQUIRE(position.ply() == 2);
  require_cached_scores_match(position);
}

TEST_CASE("position cached scores handle singleton removal and run merging", "[position][score]") {
  poe2::Position position;

  require_cached_scores_match(position);
  require_play_and_match(position, {0, 0});
  require_play_and_match(position, {6, 6});
  require_play_and_match(position, {0, 1});
  require_play_and_match(position, {5, 5});
  require_play_and_match(position, {0, 3});
  require_play_and_match(position, {4, 4});
  require_play_and_match(position, {0, 2});

  REQUIRE(position.score(poe2::Player::kOne) == 8);
  REQUIRE(position.score(poe2::Player::kTwo) == 4);

  const poe2::ScoreByPlayer scores = position.scores();
  REQUIRE_FALSE(position.play({0, 2}));
  REQUIRE(position.ply() == 7);
  REQUIRE(position.side_to_move() == poe2::Player::kTwo);
  REQUIRE(position.scores().player_one == scores.player_one);
  REQUIRE(position.scores().player_two == scores.player_two);
}

TEST_CASE("position make and unmake restore board cache and key", "[position][score][key]") {
  poe2::Position position;
  std::vector<poe2::MoveUndo> undos;
  std::vector<PositionSnapshot> snapshots;
  const std::array<poe2::Square, 11> sequence{{
      {3, 1},
      {0, 0},
      {3, 2},
      {0, 1},
      {2, 3},
      {0, 2},
      {4, 3},
      {0, 3},
      {3, 3},
      {1, 4},
      {5, 3},
  }};

  undos.reserve(sequence.size());
  snapshots.reserve(sequence.size() + 1);
  snapshots.push_back(snapshot(position));

  for (const poe2::Square square : sequence) {
    poe2::MoveUndo undo;
    REQUIRE(position.make_move(square, undo));
    undos.push_back(undo);
    require_cached_scores_match(position);
    snapshots.push_back(snapshot(position));
  }

  const PositionSnapshot before_rejected_move = snapshot(position);
  poe2::MoveUndo rejected_undo;
  REQUIRE_FALSE(position.make_move(sequence.front(), rejected_undo));
  require_snapshot(position, before_rejected_move);
  require_cached_scores_match(position);

  for (std::size_t index = sequence.size(); index > 0; --index) {
    const poe2::Square square = sequence[index - 1];
    position.unmake_move(undos[index - 1]);

    REQUIRE(position.board().is_empty(square));
    require_snapshot(position, snapshots[index - 1]);
    require_cached_scores_match(position);
  }
}

TEST_CASE("position cached scores handle crossing lines", "[position][score]") {
  poe2::Position position;

  require_play_and_match(position, {3, 1});
  require_play_and_match(position, {0, 0});
  require_play_and_match(position, {3, 2});
  require_play_and_match(position, {0, 1});
  require_play_and_match(position, {2, 3});
  require_play_and_match(position, {0, 2});
  require_play_and_match(position, {4, 3});
  require_play_and_match(position, {0, 3});
  require_play_and_match(position, {3, 3});

  REQUIRE(position.score(poe2::Player::kOne) == 12);
  REQUIRE(position.score(poe2::Player::kTwo) == 8);
}

TEST_CASE("position key packs both bitboards and side to move", "[key]") {
  const poe2::Bitboard player_one = poe2::square_bit({0, 0}) | poe2::square_bit({1, 1});
  const poe2::Bitboard player_two = poe2::square_bit({6, 6});
  const poe2::PositionKey key = poe2::make_position_key(player_one, player_two, poe2::Player::kOne);
  const poe2::PositionKey other_side =
      poe2::make_position_key(player_one, player_two, poe2::Player::kTwo);
  const poe2::PositionKey swapped_players =
      poe2::make_position_key(player_two, player_one, poe2::Player::kOne);

  REQUIRE(poe2::position_key_bits(key, poe2::Player::kOne) == player_one);
  REQUIRE(poe2::position_key_bits(key, poe2::Player::kTwo) == player_two);
  REQUIRE(poe2::position_key_side_to_move(key) == poe2::Player::kOne);
  REQUIRE(poe2::position_key_side_to_move(other_side) == poe2::Player::kTwo);
  REQUIRE(key != other_side);
  REQUIRE(key != swapped_players);
  REQUIRE(poe2::Position{}.hash() == poe2::position_key_hash(poe2::Position{}.key()));
  REQUIRE(poe2::position_key_hash(key) != poe2::position_key_hash(other_side));
  REQUIRE(poe2::position_key_hash(key) != poe2::position_key_hash(swapped_players));
}

TEST_CASE("transposition table probes by exact bitboards and side to move", "[key][tt]") {
  poe2::Position position;
  require_play_and_match(position, {0, 0});
  require_play_and_match(position, {6, 6});
  require_play_and_match(position, {0, 1});

  poe2::TranspositionTable table(16);
  const poe2::TranspositionValue value{
      .score = 42,
      .depth = 6,
      .bound = poe2::TranspositionBound::kExact,
      .best_move = poe2::Square{4, 4},
  };

  table.store(position, value);
  const std::optional<poe2::TranspositionEntry> hit = table.probe(position);

  if (!hit.has_value()) {
    FAIL("stored transposition entry is missing");
    return;
  }

  const poe2::TranspositionEntry entry = *hit;
  REQUIRE(entry.key == position.key());
  REQUIRE(poe2::position_key_bits(entry.key, poe2::Player::kOne) ==
          position.board().bits(poe2::Player::kOne));
  REQUIRE(poe2::position_key_bits(entry.key, poe2::Player::kTwo) ==
          position.board().bits(poe2::Player::kTwo));
  REQUIRE(poe2::position_key_side_to_move(entry.key) == position.side_to_move());
  REQUIRE(entry.hash == position.hash());
  REQUIRE(entry.hash == poe2::position_key_hash(position.key()));
  REQUIRE(entry.value.score == value.score);
  REQUIRE(entry.value.depth == value.depth);
  REQUIRE(entry.value.bound == value.bound);
  const std::optional<poe2::Square> best_move = entry.value.best_move;
  if (!best_move.has_value()) {
    FAIL("stored best move is missing");
    return;
  }
  const poe2::Square best_move_value = *best_move;
  REQUIRE(best_move_value.row == 4);
  REQUIRE(best_move_value.col == 4);

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

TEST_CASE("transposition table rounds capacity to power of two", "[tt]") {
  poe2::TranspositionTable empty;
  REQUIRE(empty.capacity() == 0);
  REQUIRE(empty.empty());

  poe2::TranspositionTable table(3);
  REQUIRE(table.capacity() == 4);

  const poe2::PositionKey key = poe2::make_position_key(
      poe2::square_bit({0, 0}), poe2::square_bit({6, 6}), poe2::Player::kOne);
  table.store(key, {.score = 10, .depth = 4});
  REQUIRE(table.size() == 1);
  REQUIRE(table.probe(key).has_value());

  table.resize(17);
  REQUIRE(table.capacity() == 32);
  REQUIRE(table.empty());
  REQUIRE_FALSE(table.probe(key).has_value());

  table.resize(1);
  REQUIRE(table.capacity() == 1);
}

TEST_CASE("transposition table keeps deeper entries on bucket collisions", "[tt]") {
  poe2::TranspositionTable table(1);
  const poe2::PositionKey first = poe2::make_position_key(
      poe2::square_bit({0, 0}), poe2::square_bit({6, 6}), poe2::Player::kOne);
  const poe2::PositionKey second = poe2::make_position_key(
      poe2::square_bit({0, 1}), poe2::square_bit({6, 5}), poe2::Player::kTwo);

  table.store(first, {.score = 10, .depth = 4});
  table.store(second, {.score = 20, .depth = 2});

  REQUIRE(table.size() == 1);
  REQUIRE(table.probe(first).has_value());
  REQUIRE_FALSE(table.probe(second).has_value());

  table.store(second, {.score = 20, .depth = 5});

  const std::optional<poe2::TranspositionEntry> hit = table.probe(second);
  REQUIRE_FALSE(table.probe(first).has_value());
  if (!hit.has_value()) {
    FAIL("deeper transposition entry is missing");
    return;
  }

  const poe2::TranspositionEntry entry = *hit;
  REQUIRE(entry.value.score == 20);
  REQUIRE(entry.value.depth == 5);
}

TEST_CASE("position terminal result uses cached scores", "[position][game]") {
  poe2::Position position;

  for (int row = 0; row < poe2::kBoardSize; ++row) {
    for (int col = 0; col < poe2::kBoardSize; ++col) {
      require_play_and_match(position, {row, col});
    }
  }

  const std::optional<poe2::GameResult> board_result = poe2::result_if_full(position.board());
  const std::optional<poe2::GameResult> position_result = poe2::result_if_full(position);

  if (!board_result.has_value()) {
    FAIL("full-board terminal result is missing");
    return;
  }
  if (!position_result.has_value()) {
    FAIL("cached terminal result is missing");
    return;
  }

  const poe2::GameResult board_value = *board_result;
  const poe2::GameResult position_value = *position_result;
  REQUIRE(position_value.scores.player_one == board_value.scores.player_one);
  REQUIRE(position_value.scores.player_two == board_value.scores.player_two);
  REQUIRE(position_value.winner == board_value.winner);
  REQUIRE(poe2::winner_if_full(position) == board_value.winner);
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
