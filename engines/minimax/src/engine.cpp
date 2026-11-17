#include "poe2/minimax/engine.hpp"

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "poe2/minimax/engine_version.hpp"

namespace poe2::minimax {

std::string_view engine_version() noexcept { return kEngineVersion; }

std::vector<std::string> engine_identity(const Net& net) {
  std::string designs = "designs";
  for (const std::string_view design : supported_net_designs()) {
    designs += ' ';
    designs += design;
  }
  return {
      "version " + std::string{kEngineVersion},
      std::move(designs),
      "net " + std::string{net.name()} + ' ' + std::string{net.id()} + ' ' +
          std::string{net.sha256()},
  };
}

MinimaxEngine::MinimaxEngine(const SearchOptions& options) : search_(options) {}

void MinimaxEngine::new_game() { search_.new_game(); }

std::vector<std::string> MinimaxEngine::identity() const { return engine_identity(search_.net()); }

engine::EngineResult MinimaxEngine::choose_move(const Position& position,
                                                const engine::EngineLimits& limits,
                                                const engine::InfoSink& info) {
  return search_.run(position, limits, info);
}

}  // namespace poe2::minimax
