#ifndef POE2_MOVE_HPP
#define POE2_MOVE_HPP

#include <optional>
#include <string>
#include <string_view>

#include "poe2/board.hpp"

namespace poe2 {

struct Move {
  Square square;

  friend constexpr bool operator==(const Move&, const Move&) = default;
};

enum class MoveError : std::uint8_t {
  kOutOfBounds = 0,
  kGameOver,
  kOccupied,
};

struct MoveResult {
  bool accepted = false;
  std::optional<MoveError> error;
  std::optional<GameResult> game_result;
};

[[nodiscard]] std::optional<MoveError> validate_move(const Position& position, Move move) noexcept;
[[nodiscard]] MoveResult apply_move(Position& position, Move move) noexcept;

[[nodiscard]] std::optional<Move> parse_move(std::string_view text) noexcept;
[[nodiscard]] std::string format_move(Move move);
[[nodiscard]] std::string_view move_error_name(MoveError error) noexcept;

}  // namespace poe2

#endif  // POE2_MOVE_HPP
