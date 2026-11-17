#ifndef POE2_MINIMAX_ENGINE_HPP
#define POE2_MINIMAX_ENGINE_HPP

#include <string>
#include <string_view>
#include <vector>

#include "poe2/engine.hpp"
#include "poe2/minimax/net.hpp"
#include "poe2/minimax/search.hpp"

namespace poe2::minimax {

[[nodiscard]] std::string_view engine_version() noexcept;

// "version <v>", "designs <d>...", and "net <name> <id> <sha256>" for the given net.
[[nodiscard]] std::vector<std::string> engine_identity(const Net& net);

class MinimaxEngine final : public engine::Engine {
 public:
  explicit MinimaxEngine(const SearchOptions& options = {});

  void new_game() override;
  [[nodiscard]] std::vector<std::string> identity() const override;
  [[nodiscard]] engine::EngineResult choose_move(const Position& position,
                                                 const engine::EngineLimits& limits,
                                                 const engine::InfoSink& info) override;

 private:
  Search search_;
};

}  // namespace poe2::minimax

#endif  // POE2_MINIMAX_ENGINE_HPP
