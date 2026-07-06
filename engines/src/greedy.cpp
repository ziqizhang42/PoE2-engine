#include <exception>
#include <iostream>
#include <optional>

#include "poe2/engine_stdio.hpp"
#include "poe2/move.hpp"

namespace {

[[nodiscard]] std::optional<poe2::Move> greedy_move(const poe2::Position& position) {
  const poe2::Bitboard legal_moves = position.legal_moves();
  const poe2::Player player = position.side_to_move();
  const poe2::Score score_before = position.score(player);

  poe2::Position trial = position;
  std::optional<poe2::Move> best_move;
  std::optional<poe2::Score> best_gain;

  for (int index = 0; index < poe2::kCellCount; ++index) {
    const poe2::Bitboard bit = poe2::Bitboard{1} << index;
    if ((legal_moves & bit) == 0) {
      continue;
    }

    const poe2::Square square = poe2::square_from_index(index);
    poe2::MoveUndo undo;
    if (!trial.make_move(square, undo)) {
      continue;
    }

    const poe2::Score gain = trial.score(player) - score_before;
    trial.unmake_move(undo);

    if (!best_gain.has_value() || gain > *best_gain) {
      best_move = poe2::Move{.square = square};
      best_gain = gain;
    }
  }

  return best_move;
}

class GreedyEngine final : public poe2::engine_stdio::Engine {
 public:
  [[nodiscard]] poe2::engine_stdio::EngineResult choose_move(
      const poe2::Position& position, const poe2::engine_stdio::EngineLimits&,
      const poe2::engine_stdio::InfoSink&) override {
    return poe2::engine_stdio::EngineResult{.best_move = greedy_move(position)};
  }
};

int run() {
  GreedyEngine engine;
  return poe2::engine_stdio::run_engine_stdio("greedy", engine);
}

}  // namespace

int main() {
  try {
    return run();
  } catch (const std::exception& error) {
    std::cerr << "fatal " << error.what() << '\n';
  } catch (...) {
    std::cerr << "fatal unknown\n";
  }

  return 1;
}
