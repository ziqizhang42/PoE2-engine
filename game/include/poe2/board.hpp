#pragma once

#include <array>
#include <cstdint>
#include <optional>

namespace poe2 {

constexpr int kBoardSize = 7;
constexpr int kCellCount = kBoardSize * kBoardSize;
constexpr int kPlayerTwoHandicapHalfPoints = 11;

using Bitboard = std::uint64_t;
using Score = int;

enum class Player : std::uint8_t {
  kOne = 0,
  kTwo = 1,
};

enum class Cell : std::uint8_t {
  kEmpty = 0,
  kPlayerOne,
  kPlayerTwo,
};

struct Square {
  int row = 0;
  int col = 0;

  friend constexpr bool operator==(const Square&, const Square&) = default;
};

struct ScoreByPlayer {
  Score player_one = 0;
  Score player_two = 0;
};

[[nodiscard]] constexpr Player opponent(Player player) noexcept {
  return player == Player::kOne ? Player::kTwo : Player::kOne;
}

[[nodiscard]] constexpr int player_index(Player player) noexcept {
  return player == Player::kOne ? 0 : 1;
}

[[nodiscard]] constexpr bool is_valid(Square square) noexcept {
  return square.row >= 0 && square.row < kBoardSize && square.col >= 0 && square.col < kBoardSize;
}

[[nodiscard]] constexpr int square_index(Square square) noexcept {
  return square.row * kBoardSize + square.col;
}

[[nodiscard]] constexpr Bitboard square_bit(Square square) noexcept {
  return is_valid(square) ? Bitboard{1} << square_index(square) : Bitboard{0};
}

class Board final {
 public:
  constexpr Board() noexcept = default;

  [[nodiscard]] Bitboard bits(Player player) const noexcept;
  [[nodiscard]] Bitboard occupied() const noexcept;
  [[nodiscard]] Cell cell_at(Square square) const noexcept;
  [[nodiscard]] bool can_place(Square square) const noexcept;
  [[nodiscard]] bool is_empty(Square square) const noexcept;
  [[nodiscard]] bool is_full() const noexcept;
  [[nodiscard]] int piece_count() const noexcept;

  [[nodiscard]] bool place(Player player, Square square) noexcept;

 private:
  std::array<Bitboard, 2> pieces_{};
};

class Position final {
 public:
  constexpr Position() noexcept = default;

  [[nodiscard]] const Board& board() const noexcept;
  [[nodiscard]] Player side_to_move() const noexcept;
  [[nodiscard]] int ply() const noexcept;
  [[nodiscard]] bool is_full() const noexcept;

  [[nodiscard]] bool play(Square square) noexcept;

 private:
  Board board_;
  Player side_to_move_ = Player::kOne;
  int ply_ = 0;
};

[[nodiscard]] Score score(const Board& board, Player player) noexcept;
[[nodiscard]] ScoreByPlayer score(const Board& board) noexcept;
[[nodiscard]] Player leader_after_handicap(ScoreByPlayer scores) noexcept;
[[nodiscard]] std::optional<Player> winner_if_full(const Board& board) noexcept;

}  // namespace poe2
