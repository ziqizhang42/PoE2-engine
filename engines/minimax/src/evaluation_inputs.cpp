#include "evaluation_inputs.hpp"

#include <bit>
#include <limits>

namespace poe2::minimax {

namespace {

constexpr Bitboard kFirstColumn = []() consteval {
  Bitboard column = 0;
  for (int row = 0; row < kBoardSize; ++row) {
    column |= Bitboard{1} << (row * kBoardSize);
  }
  return column;
}();
constexpr Bitboard kLastColumn = kFirstColumn << (kBoardSize - 1);

[[nodiscard]] constexpr Bitboard with_neighbors(Bitboard squares) noexcept {
  const Bitboard rows =
      squares | ((squares << 1) & ~kFirstColumn) | ((squares >> 1) & ~kLastColumn);
  return (rows | (rows << kBoardSize) | (rows >> kBoardSize)) & kBoardMask;
}

// For each square, the row, column, and two diagonals through it.
constexpr std::array<Bitboard, kCellCount> kLinesThrough = []() consteval {
  std::array<Bitboard, kCellCount> lines{};
  for (int index = 0; index < kCellCount; ++index) {
    const Square square = square_from_index(index);
    for (int other_index = 0; other_index < kCellCount; ++other_index) {
      const Square other = square_from_index(other_index);
      const int rows = other.row - square.row;
      const int columns = other.col - square.col;
      if (rows == 0 || columns == 0 || rows == columns || rows == -columns) {
        lines[static_cast<std::size_t>(index)] |= Bitboard{1} << other_index;
      }
    }
  }
  return lines;
}();

template <std::size_t Size>
[[nodiscard]] constexpr std::array<Bitboard, Size> without(std::array<Bitboard, Size> boards,
                                                           Bitboard squares) noexcept {
  for (Bitboard& board : boards) {
    board &= ~squares;
  }
  return boards;
}

}  // namespace

void MoveGains::set(Player player, int move_index, Score gain) noexcept {
  assert(gain > 0 && gain <= std::numeric_limits<std::uint8_t>::max());
  const std::size_t index = static_cast<std::size_t>(player_index(player));
  const Bitboard square = Bitboard{1} << move_index;
  gains_[index][static_cast<std::size_t>(move_index)] = static_cast<std::uint8_t>(gain);
  for (std::size_t threshold = 0; threshold < kGainThresholds.size(); ++threshold) {
    Bitboard& reaching = reaching_[index][threshold];
    reaching = gain >= kGainThresholds[threshold] ? reaching | square : reaching & ~square;
  }
}

void MoveGains::reset(const Position& position) noexcept {
  *this = MoveGains{};
  Bitboard moves = position.legal_moves();
  while (moves != 0) {
    const int move_index = std::countr_zero(moves);
    moves &= moves - Bitboard{1};
    const ScoreByPlayer gains = position.score_gains_unchecked(move_index);
    set(Player::kOne, move_index, gains.player_one);
    set(Player::kTwo, move_index, gains.player_two);
  }
}

Bitboard MoveGains::changed_by(const MoveUndo& move) noexcept {
  // A gain depends on the player's runs that end next to the square and on which of their
  // pieces already score. The move lengthens only the runs through it, so run lengths change
  // only next to those runs on the move's own lines. Only pieces of those runs start to score,
  // and each of them was a run of one in every other direction, so its neighbors change too.
  const Bitboard square = Bitboard{1} << move.move_index;
  const Bitboard runs = move.lines_through_move | square;
  const Bitboard started_scoring = runs & ~move.previous_pieces_in_lines & ~square;
  return (with_neighbors(runs) & kLinesThrough[move.move_index]) | with_neighbors(started_scoring);
}

void MoveGains::advance(const MoveGains& parent, const Position& position, int move_index,
                        Bitboard changed) noexcept {
  const Player mover = opponent(position.side_to_move());
  const std::size_t mover_index = static_cast<std::size_t>(player_index(mover));
  const Bitboard move = Bitboard{1} << move_index;
  gains_ = parent.gains_;
  for (std::size_t index = 0; index < gains_.size(); ++index) {
    gains_[index][static_cast<std::size_t>(move_index)] = 0;
  }
  reaching_[1 - mover_index] = without(parent.reaching_[1 - mover_index], move);

  std::array<Bitboard, kGainThresholds.size()> reaching =
      without(parent.reaching_[mover_index], move);
  changed &= position.legal_moves();
  while (changed != 0) {
    const int changed_index = std::countr_zero(changed);
    const Bitboard square = Bitboard{1} << changed_index;
    changed ^= square;
    const Score gain = position.score_gain_unchecked(mover, changed_index);
    assert(gain > 0 && gain <= std::numeric_limits<std::uint8_t>::max());
    gains_[mover_index][static_cast<std::size_t>(changed_index)] = static_cast<std::uint8_t>(gain);
    for (std::size_t threshold = 0; threshold < kGainThresholds.size(); ++threshold) {
      reaching[threshold] = gain >= kGainThresholds[threshold] ? reaching[threshold] | square
                                                               : reaching[threshold] & ~square;
    }
  }
  reaching_[mover_index] = reaching;
}

void LinePatterns::place(Player mover, int move_index) noexcept {
  Patterns& own = patterns_[static_cast<std::size_t>(player_index(mover))];
  Patterns& other = patterns_[static_cast<std::size_t>(player_index(opponent(mover)))];
  for (const pattern_gain::LineContribution contribution :
       pattern_gain::kCellLineContributions[static_cast<std::size_t>(move_index)].values) {
    own[contribution.line] =
        static_cast<std::uint16_t>(own[contribution.line] + contribution.place);
    other[contribution.line] =
        static_cast<std::uint16_t>(other[contribution.line] + 2 * contribution.place);
  }
}

void LinePatterns::reset(const Position& position) noexcept {
  for (Patterns& patterns : patterns_) {
    for (std::size_t line = 0; line < patterns.size(); ++line) {
      patterns[line] = pattern_gain::kPatternOffsets[pattern_gain::kScoringLines[line].length];
    }
  }
  for (const Player player : {Player::kOne, Player::kTwo}) {
    Bitboard pieces = position.board().bits(player);
    while (pieces != 0) {
      place(player, std::countr_zero(pieces));
      pieces &= pieces - Bitboard{1};
    }
  }
}

void LinePatterns::advance(const LinePatterns& parent, Player mover, int move_index) noexcept {
  *this = parent;
  place(mover, move_index);
}

void EvaluationInputs::reset(const Position& position) noexcept {
  gains.reset(position);
  patterns.reset(position);
}

void EvaluationInputs::advance(const EvaluationInputs& parent, const Position& position,
                               int move_index, Bitboard changed_gains) noexcept {
  gains.advance(parent.gains, position, move_index, changed_gains);
  patterns.advance(parent.patterns, opponent(position.side_to_move()), move_index);
}

}  // namespace poe2::minimax
