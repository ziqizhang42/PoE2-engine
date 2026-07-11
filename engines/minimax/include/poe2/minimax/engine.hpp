#ifndef POE2_MINIMAX_ENGINE_HPP
#define POE2_MINIMAX_ENGINE_HPP

#include "poe2/engine.hpp"
#include "poe2/minimax/search.hpp"

namespace poe2::minimax {

class MinimaxEngine final : public engine::Engine {
 public:
  explicit MinimaxEngine(SearchOptions options = {});

  void new_game() override;
  [[nodiscard]] engine::EngineResult choose_move(const Position& position,
                                                 const engine::EngineLimits& limits,
                                                 const engine::InfoSink& info) override;

 private:
  Search search_;
};

}  // namespace poe2::minimax

#endif  // POE2_MINIMAX_ENGINE_HPP
