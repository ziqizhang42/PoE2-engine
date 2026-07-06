#ifndef POE2_ENGINE_HPP
#define POE2_ENGINE_HPP

#include <chrono>
#include <cstdint>
#include <functional>
#include <optional>
#include <string_view>
#include <vector>

#include "poe2/move.hpp"

namespace poe2::engine {

struct EngineLimits {
  std::optional<int> depth;
  std::optional<std::chrono::milliseconds> move_time;
  std::optional<std::uint64_t> nodes;
};

struct EngineResult {
  std::optional<Move> best_move;
  std::optional<Score> score;
  int depth = 0;
  std::uint64_t nodes = 0;
  std::vector<Move> principal_variation;
};

using InfoSink = std::function<void(std::string_view)>;

class Engine {
 public:
  virtual ~Engine() = default;

  virtual void new_game() {}
  [[nodiscard]] virtual EngineResult choose_move(const Position& position,
                                                 const EngineLimits& limits,
                                                 const InfoSink& info) = 0;
};

}  // namespace poe2::engine

#endif  // POE2_ENGINE_HPP
