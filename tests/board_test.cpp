#include "poe2/board.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <initializer_list>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

#include "poe2/symmetry.hpp"

namespace {

struct PositionSnapshot {
  poe2::PositionKey key;
  poe2::PositionHash hash = 0;
  poe2::Bitboard player_one = 0;
  poe2::Bitboard player_two = 0;
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
      .player_one = position.board().bits(poe2::Player::kOne),
      .player_two = position.board().bits(poe2::Player::kTwo),
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
  REQUIRE(position.board().bits(poe2::Player::kOne) == expected.player_one);
  REQUIRE(position.board().bits(poe2::Player::kTwo) == expected.player_two);
  REQUIRE(position.side_to_move() == expected.side_to_move);
  REQUIRE(position.ply() == expected.ply);
  REQUIRE(position.legal_moves() == expected.legal_moves);
  REQUIRE(position.scores().player_one == expected.scores.player_one);
  REQUIRE(position.scores().player_two == expected.scores.player_two);
}

[[nodiscard]] std::optional<poe2::Score> full_recalculation_gain(const poe2::Position& position,
                                                                 poe2::Player player,
                                                                 poe2::Square square) {
  if (!position.board().can_place(square)) {
    return std::nullopt;
  }

  poe2::Board board = position.board();
  const poe2::Score before = poe2::score(board, player);
  REQUIRE(board.place(player, square));
  return poe2::score(board, player) - before;
}

void require_all_score_gains_match_full_recalculation(const poe2::Position& position) {
  const PositionSnapshot before = snapshot(position);

  for (const poe2::Player player : {poe2::Player::kOne, poe2::Player::kTwo}) {
    for (int index = 0; index < poe2::kCellCount; ++index) {
      const poe2::Square square = poe2::square_from_index(index);
      CAPTURE(position.ply(), player, square.row, square.col);
      REQUIRE(position.score_gain(player, square) ==
              full_recalculation_gain(position, player, square));
    }
  }

  poe2::Bitboard legal_moves = position.legal_moves();
  while (legal_moves != 0) {
    const int move_index = std::countr_zero(legal_moves);
    legal_moves &= legal_moves - poe2::Bitboard{1};
    const poe2::ScoreByPlayer gains = position.score_gains_unchecked(move_index);
    REQUIRE(position.score_gain_unchecked(poe2::Player::kOne, move_index) == gains.player_one);
    REQUIRE(position.score_gain_unchecked(poe2::Player::kTwo, move_index) == gains.player_two);
  }

  require_snapshot(position, before);
}

[[nodiscard]] poe2::Position position_from_history(
    const std::vector<poe2::Square>& history, poe2::Symmetry symmetry = poe2::Symmetry::kIdentity) {
  poe2::Position position;
  for (const poe2::Square square : history) {
    REQUIRE(position.play(poe2::transform_square(symmetry, square)));
  }
  return position;
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

  for (std::size_t index = 0; index < sequence.size(); ++index) {
    const poe2::Square square = sequence[index];
    poe2::MoveUndo undo;
    if (index % 2 == 0) {
      REQUIRE(position.make_move(square, undo));
    } else {
      position.make_move_unchecked(poe2::square_index(square), undo);
    }
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

TEST_CASE("position score gain matches full scoring for runs bridges and crossings",
          "[position][score][gain]") {
  for (int run_length = 1; run_length <= 6; ++run_length) {
    poe2::Position position;
    for (int col = 0; col < run_length; ++col) {
      REQUIRE(position.play({3, col}));
      REQUIRE(position.play({0, col}));
    }

    const poe2::Score expected_extension_gain = poe2::Score{1} << (run_length - 1);
    REQUIRE(position.score_gain(poe2::Player::kOne, {3, run_length}) == expected_extension_gain);
    REQUIRE(position.score_gain(poe2::Player::kOne, {6, 6}) == 1);
    require_all_score_gains_match_full_recalculation(position);
  }

  const std::vector<std::vector<poe2::Square>> motif_histories{
      {
          {3, 0},
          {0, 0},
          {3, 1},
          {0, 2},
          {3, 3},
          {0, 4},
          {3, 4},
          {0, 6},
          {3, 5},
      },
      {
          {0, 0},
          {0, 6},
          {1, 1},
          {1, 5},
          {2, 2},
          {2, 4},
          {4, 4},
          {4, 2},
          {5, 5},
          {5, 1},
          {6, 6},
          {6, 0},
      },
      {
          {3, 2},
          {0, 0},
          {3, 4},
          {0, 1},
          {2, 3},
          {0, 5},
          {4, 3},
          {0, 6},
          {2, 2},
          {1, 0},
          {4, 4},
          {1, 6},
          {2, 4},
          {5, 0},
          {4, 2},
          {5, 6},
      },
  };

  const poe2::Position bridge = position_from_history(motif_histories.front());
  REQUIRE(bridge.score_gain(poe2::Player::kOne, {3, 2}) == 26);

  for (const std::vector<poe2::Square>& history : motif_histories) {
    for (const poe2::Symmetry symmetry : poe2::kAllSymmetries) {
      require_all_score_gains_match_full_recalculation(position_from_history(history, symmetry));
    }
  }
}

TEST_CASE("position score gain matches a deterministic all-ply D4 oracle corpus",
          "[position][score][gain][symmetry]") {
  std::array<int, poe2::kCellCount> move_order{};
  std::iota(move_order.begin(), move_order.end(), 0);
  // A fixed seed makes this oracle corpus exactly reproducible.
  std::mt19937_64 generator{UINT64_C(0x8b5ad4cef1972360)};  // NOLINT
  std::shuffle(move_order.begin(), move_order.end(), generator);

  for (const poe2::Symmetry symmetry : poe2::kAllSymmetries) {
    poe2::Position position;
    require_all_score_gains_match_full_recalculation(position);

    for (const int move_index : move_order) {
      const poe2::Square square =
          poe2::transform_square(symmetry, poe2::square_from_index(move_index));
      REQUIRE(position.play(square));
      require_all_score_gains_match_full_recalculation(position);
    }
  }
}

TEST_CASE("position score gain rejects nonempty and invalid squares without mutation",
          "[position][score][gain]") {
  const std::vector<poe2::Square> history{
      {3, 1}, {0, 0}, {3, 2}, {0, 1}, {2, 3}, {0, 2}, {4, 3}, {0, 3}, {3, 3},
  };
  poe2::Position position = position_from_history(history);
  const PositionSnapshot before = snapshot(position);

  for (int repetition = 0; repetition < 8; ++repetition) {
    for (const poe2::Player player : {poe2::Player::kOne, poe2::Player::kTwo}) {
      REQUIRE_FALSE(position.score_gain(player, history.front()).has_value());
      REQUIRE_FALSE(position.score_gain(player, {-1, 0}).has_value());
      REQUIRE_FALSE(position.score_gain(player, {0, -1}).has_value());
      REQUIRE_FALSE(position.score_gain(player, {poe2::kBoardSize, 0}).has_value());
      REQUIRE_FALSE(position.score_gain(player, {0, poe2::kBoardSize}).has_value());
      REQUIRE(position.score_gain(player, {6, 6}) ==
              full_recalculation_gain(position, player, {6, 6}));
    }
  }
  require_snapshot(position, before);

  poe2::Bitboard legal_moves = position.legal_moves();
  while (legal_moves != 0) {
    const int move_index = std::countr_zero(legal_moves);
    legal_moves &= legal_moves - poe2::Bitboard{1};
    const poe2::Square square = poe2::square_from_index(move_index);
    const PositionSnapshot child_before = snapshot(position);
    poe2::MoveUndo undo;
    REQUIRE(position.make_move(square, undo));
    require_cached_scores_match(position);
    position.unmake_move(undo);
    require_snapshot(position, child_before);
  }
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

TEST_CASE("position hash move updates are reversible and match full recomputation", "[key][hash]") {
  poe2::Position position;
  const poe2::PositionHash initial_hash = position.hash();
  const poe2::Square move{2, 4};
  const poe2::PositionHash child_hash =
      poe2::update_position_hash(initial_hash, position.side_to_move(), move);
  REQUIRE(child_hash == poe2::update_position_hash(initial_hash, position.side_to_move(),
                                                   poe2::square_index(move)));

  REQUIRE(position.play(move));
  REQUIRE(position.hash() == child_hash);
  REQUIRE(position.hash() == poe2::position_key_hash(position.key()));
  REQUIRE(poe2::update_position_hash(child_hash, poe2::Player::kOne, move) == initial_hash);
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
