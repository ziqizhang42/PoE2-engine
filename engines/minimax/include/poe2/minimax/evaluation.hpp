#ifndef POE2_MINIMAX_EVALUATION_HPP
#define POE2_MINIMAX_EVALUATION_HPP

#include "poe2/board.hpp"

namespace poe2::minimax {

// Returns the score difference from the side-to-move perspective.
// Values are measured in half-points so Player Two's 5.5-point handicap remains exact.
[[nodiscard]] Score evaluate(const Position& position) noexcept;

}  // namespace poe2::minimax

#endif  // POE2_MINIMAX_EVALUATION_HPP
