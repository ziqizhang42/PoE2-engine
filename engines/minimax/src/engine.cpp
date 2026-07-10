#include "poe2/minimax/engine.hpp"

namespace poe2::minimax {

void MinimaxEngine::new_game() { search_.new_game(); }

engine::EngineResult MinimaxEngine::choose_move(const Position& position,
                                                const engine::EngineLimits& limits,
                                                const engine::InfoSink& info) {
  return search_.run(position, limits, info);
}

}  // namespace poe2::minimax
