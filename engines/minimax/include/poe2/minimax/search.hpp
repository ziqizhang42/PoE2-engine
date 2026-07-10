#ifndef POE2_MINIMAX_SEARCH_HPP
#define POE2_MINIMAX_SEARCH_HPP

#include "poe2/engine.hpp"

namespace poe2::minimax {

class Search final {
 public:
  void new_game() noexcept;
  [[nodiscard]] engine::EngineResult run(const Position& position,
                                         const engine::EngineLimits& limits,
                                         const engine::InfoSink& info);
};

}  // namespace poe2::minimax

#endif  // POE2_MINIMAX_SEARCH_HPP
