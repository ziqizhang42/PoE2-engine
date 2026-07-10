#include "poe2/minimax/search.hpp"

#include <bit>

#include "poe2/move.hpp"

namespace poe2::minimax {

void Search::new_game() noexcept {}

engine::EngineResult Search::run(const Position& position, const engine::EngineLimits&,
                                 const engine::InfoSink&) {
  const Bitboard legal_moves = position.legal_moves();
  if (legal_moves == 0) {
    return {};
  }

  // Placeholder until the first minimax implementation lands.
  const int move_index = std::countr_zero(legal_moves);
  return engine::EngineResult{
      .best_move = Move{.square = square_from_index(move_index)},
  };
}

}  // namespace poe2::minimax
