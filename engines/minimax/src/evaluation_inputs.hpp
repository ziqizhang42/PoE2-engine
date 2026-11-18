#ifndef POE2_MINIMAX_EVALUATION_INPUTS_HPP
#define POE2_MINIMAX_EVALUATION_INPUTS_HPP

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>

#include "pattern_gain_layout.hpp"
#include "poe2/board.hpp"
#include "poe2/minimax/evaluation.hpp"

namespace poe2::minimax {

inline constexpr std::array<Score, 4> kGainThresholds{{2, 4, 8, 16}};

// Both players' score gains on every empty square, and the squares whose gain reaches each
// threshold. Occupied squares hold zero, so equal positions have equal tables.
class MoveGains final {
 public:
  void reset(const Position& position) noexcept;
  // The squares where a move may have changed its player's gain. The other gains keep their
  // values, and so do all of the opponent's.
  [[nodiscard]] static Bitboard changed_by(const MoveUndo& move) noexcept;
  // Becomes the table of `position`, which is the parent's position after `move_index`.
  // `changed` is changed_by() of that move.
  void advance(const MoveGains& parent, const Position& position, int move_index,
               Bitboard changed) noexcept;

  [[nodiscard]] Score gain(Player player, int move_index) const noexcept {
    assert(move_index >= 0 && move_index < kCellCount);
    return gains_[static_cast<std::size_t>(player_index(player))]
                 [static_cast<std::size_t>(move_index)];
  }
  [[nodiscard]] Bitboard reaching(Player player, std::size_t threshold) const noexcept {
    return reaching_[static_cast<std::size_t>(player_index(player))][threshold];
  }

  friend bool operator==(const MoveGains&, const MoveGains&) = default;

 private:
  void set(Player player, int move_index, Score gain) noexcept;

  std::array<std::array<std::uint8_t, kCellCount>, 2> gains_{};
  std::array<std::array<Bitboard, kGainThresholds.size()>, 2> reaching_{};
};

// Every scoring line's pattern index as each player sees it: own pieces are digit one, the
// opponent's digit two.
class LinePatterns final {
 public:
  using Patterns = std::array<std::uint16_t, pattern_gain::kScoringLineCount>;

  void reset(const Position& position) noexcept;
  void advance(const LinePatterns& parent, Player mover, int move_index) noexcept;

  [[nodiscard]] const Patterns& seen_by(Player player) const noexcept {
    return patterns_[static_cast<std::size_t>(player_index(player))];
  }

  friend bool operator==(const LinePatterns&, const LinePatterns&) = default;

 private:
  void place(Player mover, int move_index) noexcept;

  std::array<Patterns, 2> patterns_{};
};

// What move ordering and a leaf evaluation read from a position. A search keeps one per ply
// and advances it move by move, which costs a few squares and lines instead of the whole board.
struct EvaluationInputs {
  MoveGains gains;
  LinePatterns patterns;

  void reset(const Position& position) noexcept;
  void advance(const EvaluationInputs& parent, const Position& position, int move_index,
               Bitboard changed_gains) noexcept;

  friend bool operator==(const EvaluationInputs&, const EvaluationInputs&) = default;
};

// Requires more than two empty squares and the inputs of `position`.
[[nodiscard]] FixedEvaluation evaluate_pattern_gain_fixed(const Position& position,
                                                          Bitboard legal_moves, int empty_count,
                                                          const EvaluationInputs& inputs,
                                                          const PatternGainTables& tables) noexcept;

}  // namespace poe2::minimax

#endif  // POE2_MINIMAX_EVALUATION_INPUTS_HPP
