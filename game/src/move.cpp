#include "poe2/move.hpp"

namespace poe2 {

namespace {

constexpr char kFirstFile = 'a';
constexpr char kLastFile = kFirstFile + kBoardSize - 1;
constexpr char kFirstRank = '1';
constexpr char kLastRank = kFirstRank + kBoardSize - 1;

[[nodiscard]] constexpr char normalize_file(char file) noexcept {
  return file >= 'A' && file <= 'Z' ? static_cast<char>(file - 'A' + 'a') : file;
}

}  // namespace

std::optional<MoveError> validate_move(const Position& position, Move move) noexcept {
  if (!is_valid(move.square)) {
    return MoveError::kOutOfBounds;
  }
  if (position.is_full()) {
    return MoveError::kGameOver;
  }
  if (!position.board().is_empty(move.square)) {
    return MoveError::kOccupied;
  }

  return std::nullopt;
}

MoveResult apply_move(Position& position, Move move) noexcept {
  const std::optional<MoveError> error = validate_move(position, move);
  if (error.has_value()) {
    return MoveResult{
        .accepted = false,
        .error = error,
        .game_result = result_if_full(position),
    };
  }

  const bool accepted = position.play(move.square);
  if (!accepted) {
    return MoveResult{
        .accepted = false,
        .error = MoveError::kOccupied,
        .game_result = result_if_full(position),
    };
  }

  return MoveResult{
      .accepted = true,
      .error = std::nullopt,
      .game_result = result_if_full(position),
  };
}

std::optional<Move> parse_move(std::string_view text) noexcept {
  if (text.size() != 2) {
    return std::nullopt;
  }

  const char file = normalize_file(text[0]);
  const char rank = text[1];
  if (file < kFirstFile || file > kLastFile || rank < kFirstRank || rank > kLastRank) {
    return std::nullopt;
  }

  return Move{
      .square =
          Square{
              .row = rank - kFirstRank,
              .col = file - kFirstFile,
          },
  };
}

std::string format_move(Move move) {
  if (!is_valid(move.square)) {
    return {};
  }

  std::string text;
  text.resize(2);
  text[0] = static_cast<char>(kFirstFile + move.square.col);
  text[1] = static_cast<char>(kFirstRank + move.square.row);
  return text;
}

std::string_view move_error_name(MoveError error) noexcept {
  switch (error) {
    case MoveError::kOutOfBounds:
      return "out_of_bounds";
    case MoveError::kGameOver:
      return "game_over";
    case MoveError::kOccupied:
      return "occupied";
  }

  return "unknown";
}

}  // namespace poe2
