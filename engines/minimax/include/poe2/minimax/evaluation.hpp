#ifndef POE2_MINIMAX_EVALUATION_HPP
#define POE2_MINIMAX_EVALUATION_HPP

#include <cstdint>

#include "poe2/board.hpp"

namespace poe2::minimax {

// Returns the final-piece-count-normalized score difference from the side-to-move perspective.
// Values are measured in half-points so Player Two's 5.5-point handicap remains exact.
[[nodiscard]] Score evaluate(const Position& position) noexcept;

// Returns the exact minimax value after the next two placements, using evaluate() at the resulting
// leaf. With fewer than two legal moves, the horizon is truncated at the terminal position.
[[nodiscard]] Score evaluate_two_ply_closure(const Position& position) noexcept;
// Hot-path overload for callers that have already computed the position's legal moves and count.
[[nodiscard]] Score evaluate_two_ply_closure(const Position& position, Bitboard legal_moves,
                                             int empty_count) noexcept;

// The frozen pattern/gain evaluator uses a scale-32 integer accumulator. Search keeps this value
// scaled so fractional residuals participate in alpha-beta comparisons and rounds only at API
// boundaries.
using FixedEvaluation = std::int32_t;
inline constexpr int kPatternGainFractionalBits = 5;
inline constexpr int kPatternGainScale = 1 << kPatternGainFractionalBits;

// Returns two-ply closure plus the frozen line-pattern/gain-summary residual in scale-32 units.
// Terminal positions and positions with at most two empty squares retain exact two-ply closure.
[[nodiscard]] FixedEvaluation evaluate_pattern_gain_fixed(const Position& position) noexcept;
// Hot-path overload for callers that have already computed legal moves and their count.
[[nodiscard]] FixedEvaluation evaluate_pattern_gain_fixed(const Position& position,
                                                          Bitboard legal_moves,
                                                          int empty_count) noexcept;

// Round a scale-32 pattern/gain value to the public whole-half-point score, with ties away from
// zero.
[[nodiscard]] Score round_pattern_gain_evaluation(FixedEvaluation value) noexcept;

}  // namespace poe2::minimax

#endif  // POE2_MINIMAX_EVALUATION_HPP
