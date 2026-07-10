#include "poe2/minimax/search.hpp"

#include <bit>
#include <cassert>
#include <chrono>
#include <optional>
#include <utility>
#include <vector>

#include "poe2/minimax/evaluation.hpp"
#include "poe2/move.hpp"

namespace poe2::minimax {

namespace {

using Clock = std::chrono::steady_clock;

class SearchState final {
 public:
  explicit SearchState(std::chrono::milliseconds move_time) : deadline_(Clock::now() + move_time) {}

  [[nodiscard]] bool enter_node() noexcept {
    if (Clock::now() >= deadline_) {
      return false;
    }

    ++nodes_;
    return true;
  }

  [[nodiscard]] std::uint64_t nodes() const noexcept { return nodes_; }

 private:
  Clock::time_point deadline_;
  std::uint64_t nodes_ = 0;
};

struct NodeResult {
  Score value = 0;
  std::vector<Move> principal_variation;
};

void set_principal_variation(NodeResult& result, Move move, const NodeResult& child) {
  result.principal_variation.clear();
  result.principal_variation.reserve(child.principal_variation.size() + 1);
  result.principal_variation.push_back(move);
  result.principal_variation.insert(result.principal_variation.end(),
                                    child.principal_variation.begin(),
                                    child.principal_variation.end());
}

[[nodiscard]] std::optional<NodeResult> negamax(Position& position, int depth, SearchState& state) {
  if (!state.enter_node()) {
    return std::nullopt;
  }

  Bitboard legal_moves = position.legal_moves();
  if (depth == 0 || legal_moves == 0) {
    return NodeResult{.value = evaluate(position)};
  }

  NodeResult best;
  bool found_move = false;

  while (legal_moves != 0) {
    const int move_index = std::countr_zero(legal_moves);
    legal_moves &= legal_moves - Bitboard{1};

    const Move move{.square = square_from_index(move_index)};
    MoveUndo undo;
    const bool made_move = position.make_move(move.square, undo);
    assert(made_move);
    if (!made_move) {
      continue;
    }

    std::optional<NodeResult> child = negamax(position, depth - 1, state);
    position.unmake_move(undo);

    if (!child.has_value()) {
      return std::nullopt;
    }

    const Score value = -child->value;
    if (!found_move || value > best.value) {
      found_move = true;
      best.value = value;
      set_principal_variation(best, move, *child);
    }
  }

  assert(found_move);
  return best;
}

void commit_iteration(engine::EngineResult& result, NodeResult iteration, int depth) {
  assert(!iteration.principal_variation.empty());
  result.best_move = iteration.principal_variation.front();
  result.score = iteration.value;
  result.depth = depth;
  result.principal_variation = std::move(iteration.principal_variation);
}

}  // namespace

void Search::new_game() noexcept {}

engine::EngineResult Search::run(const Position& position, const engine::EngineLimits& limits,
                                 const engine::InfoSink& info) {
  const Bitboard legal_moves = position.legal_moves();
  if (legal_moves == 0) {
    return {};
  }

  engine::EngineResult result{
      .best_move = Move{.square = square_from_index(std::countr_zero(legal_moves))},
  };

  if (!limits.move_time.has_value() || limits.move_time->count() <= 0) {
    if (info) {
      info("error minimax_requires_movetime");
    }
    return result;
  }

  SearchState state{*limits.move_time};
  Position search_position = position;
  const int maximum_depth = search_position.board().empty_count();

  for (int depth = 1; depth <= maximum_depth; ++depth) {
    std::optional<NodeResult> iteration = negamax(search_position, depth, state);
    if (!iteration.has_value()) {
      break;
    }

    commit_iteration(result, std::move(*iteration), depth);
  }

  result.nodes = state.nodes();
  return result;
}

}  // namespace poe2::minimax
