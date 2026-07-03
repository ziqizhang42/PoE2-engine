#include "poe2/board.hpp"

#include <array>
#include <bit>

namespace poe2 {

namespace {

struct Direction {
  int row_delta = 0;
  int col_delta = 0;
};

constexpr std::array<Direction, 4> kLineDirections{{
    {0, 1},
    {1, 0},
    {1, 1},
    {1, -1},
}};

[[nodiscard]] constexpr Square step(Square square, Direction direction) noexcept {
  return Square{square.row + direction.row_delta, square.col + direction.col_delta};
}

[[nodiscard]] constexpr Square previous(Square square, Direction direction) noexcept {
  return Square{square.row - direction.row_delta, square.col - direction.col_delta};
}

[[nodiscard]] bool contains(Bitboard bits, Square square) noexcept {
  return is_valid(square) && (bits & square_bit(square)) != 0;
}

[[nodiscard]] constexpr Score line_score(int length) noexcept { return Score{1} << (length - 1); }

}  // namespace

Bitboard Board::bits(Player player) const noexcept { return pieces_[player_index(player)]; }

Bitboard Board::occupied() const noexcept { return pieces_[0] | pieces_[1]; }

Cell Board::cell_at(Square square) const noexcept {
  if (!is_valid(square)) {
    return Cell::kEmpty;
  }

  const Bitboard bit = square_bit(square);
  if ((pieces_[0] & bit) != 0) {
    return Cell::kPlayerOne;
  }
  if ((pieces_[1] & bit) != 0) {
    return Cell::kPlayerTwo;
  }
  return Cell::kEmpty;
}

bool Board::can_place(Square square) const noexcept {
  return is_valid(square) && (occupied() & square_bit(square)) == 0;
}

bool Board::is_empty(Square square) const noexcept {
  return is_valid(square) && cell_at(square) == Cell::kEmpty;
}

bool Board::is_full() const noexcept { return piece_count() == kCellCount; }

int Board::piece_count() const noexcept { return std::popcount(occupied()); }

bool Board::place(Player player, Square square) noexcept {
  if (!can_place(square)) {
    return false;
  }

  pieces_[player_index(player)] |= square_bit(square);
  return true;
}

const Board& Position::board() const noexcept { return board_; }

Player Position::side_to_move() const noexcept { return side_to_move_; }

int Position::ply() const noexcept { return ply_; }

bool Position::is_full() const noexcept { return board_.is_full(); }

bool Position::play(Square square) noexcept {
  if (!board_.place(side_to_move_, square)) {
    return false;
  }

  side_to_move_ = opponent(side_to_move_);
  ++ply_;
  return true;
}

Score score(const Board& board, Player player) noexcept {
  const Bitboard pieces = board.bits(player);
  Bitboard pieces_in_lines = 0;
  Score total = 0;

  for (int row = 0; row < kBoardSize; ++row) {
    for (int col = 0; col < kBoardSize; ++col) {
      const Square start{row, col};
      if (!contains(pieces, start)) {
        continue;
      }

      for (const Direction direction : kLineDirections) {
        if (contains(pieces, previous(start, direction))) {
          continue;
        }

        int length = 0;
        Bitboard run_bits = 0;
        for (Square current = start; contains(pieces, current);
             current = step(current, direction)) {
          ++length;
          run_bits |= square_bit(current);
        }

        if (length >= 2) {
          total += line_score(length);
          pieces_in_lines |= run_bits;
        }
      }
    }
  }

  total += std::popcount(pieces & ~pieces_in_lines);
  return total;
}

ScoreByPlayer score(const Board& board) noexcept {
  return ScoreByPlayer{
      .player_one = score(board, Player::kOne),
      .player_two = score(board, Player::kTwo),
  };
}

Player leader_after_handicap(ScoreByPlayer scores) noexcept {
  const int player_one_half_points = scores.player_one * 2;
  const int player_two_half_points = scores.player_two * 2 + kPlayerTwoHandicapHalfPoints;

  return player_one_half_points > player_two_half_points ? Player::kOne : Player::kTwo;
}

std::optional<Player> winner_if_full(const Board& board) noexcept {
  if (!board.is_full()) {
    return std::nullopt;
  }

  return leader_after_handicap(score(board));
}

}  // namespace poe2
