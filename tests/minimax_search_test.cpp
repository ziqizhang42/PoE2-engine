#include <algorithm>
#include <array>
#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstdint>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "poe2/minimax/evaluation.hpp"
#include "poe2/minimax/search.hpp"
#include "poe2/move.hpp"
#include "poe2/symmetry.hpp"

namespace {

[[nodiscard]] poe2::engine::EngineLimits timed_limits(std::chrono::milliseconds move_time,
                                                      std::optional<int> depth = std::nullopt) {
  return poe2::engine::EngineLimits{.depth = depth, .move_time = move_time};
}

[[nodiscard]] poe2::Score value_for(const poe2::Position& position, poe2::Player perspective) {
  const poe2::Score value = poe2::minimax::evaluate(position);
  return perspective == position.side_to_move() ? value : -value;
}

[[nodiscard]] poe2::Position position_with_empty_squares(poe2::Bitboard empty_squares) {
  poe2::Position position;
  for (int index = 0; index < poe2::kCellCount; ++index) {
    const poe2::Bitboard bit = poe2::Bitboard{1} << index;
    if ((empty_squares & bit) == 0) {
      REQUIRE(position.play(poe2::square_from_index(index)));
    }
  }

  return position;
}

struct SearchDiagnostics {
  std::uint64_t tt_probes = 0;
  std::uint64_t tt_hits = 0;
  std::uint64_t tt_cutoffs = 0;
  std::uint64_t alpha_beta_cutoffs = 0;
  std::uint64_t tt_stores = 0;
  std::uint64_t pvs_scouts = 0;
  std::uint64_t pvs_researches = 0;
  std::uint64_t symmetry_prunes = 0;
  std::uint64_t move_order_evaluations = 0;
  std::uint64_t static_evaluations = 0;
  std::uint64_t closure_evaluations = 0;
  std::uint64_t closure_gain_queries = 0;
  std::uint64_t gain_queries = 0;
  std::uint64_t history_updates = 0;
  std::uint64_t history_max = 0;
  std::uint64_t hash_entries = 0;
  std::uint64_t hash_capacity = 0;
  std::uint64_t hash_bytes = 0;
};

[[nodiscard]] SearchDiagnostics parse_diagnostics(std::string_view text) {
  SearchDiagnostics diagnostics;
  std::istringstream input{std::string{text}};
  std::string name;
  std::uint64_t value = 0;
  while (input >> name >> value) {
    if (name == "ttprobes") {
      diagnostics.tt_probes = value;
    } else if (name == "tthits") {
      diagnostics.tt_hits = value;
    } else if (name == "ttcutoffs") {
      diagnostics.tt_cutoffs = value;
    } else if (name == "abcutoffs") {
      diagnostics.alpha_beta_cutoffs = value;
    } else if (name == "ttstores") {
      diagnostics.tt_stores = value;
    } else if (name == "pvsscouts") {
      diagnostics.pvs_scouts = value;
    } else if (name == "pvsresearches") {
      diagnostics.pvs_researches = value;
    } else if (name == "symmetryprunes") {
      diagnostics.symmetry_prunes = value;
    } else if (name == "moveorderevals") {
      diagnostics.move_order_evaluations = value;
    } else if (name == "staticevals") {
      diagnostics.static_evaluations = value;
    } else if (name == "closureevals") {
      diagnostics.closure_evaluations = value;
    } else if (name == "closuregainqueries") {
      diagnostics.closure_gain_queries = value;
    } else if (name == "gainqueries") {
      diagnostics.gain_queries = value;
    } else if (name == "historyupdates") {
      diagnostics.history_updates = value;
    } else if (name == "historymax") {
      diagnostics.history_max = value;
    } else if (name == "hashentries") {
      diagnostics.hash_entries = value;
    } else if (name == "hashcapacity") {
      diagnostics.hash_capacity = value;
    } else if (name == "hashbytes") {
      diagnostics.hash_bytes = value;
    }
  }
  return diagnostics;
}

[[nodiscard]] poe2::engine::EngineResult run_with_diagnostics(poe2::minimax::Search& search,
                                                              const poe2::Position& position,
                                                              std::chrono::milliseconds move_time,
                                                              SearchDiagnostics& diagnostics) {
  std::string message;
  const poe2::engine::EngineResult result = search.run(
      position, timed_limits(move_time), [&message](std::string_view text) { message = text; });
  diagnostics = parse_diagnostics(message);
  return result;
}

[[nodiscard]] poe2::engine::EngineResult run_fixed_depth_with_diagnostics(
    poe2::minimax::Search& search, const poe2::Position& position, int depth,
    SearchDiagnostics& diagnostics) {
  std::string message;
  const poe2::engine::EngineResult result =
      search.run(position, timed_limits(std::chrono::seconds{5}, depth),
                 [&message](std::string_view text) { message = text; });
  diagnostics = parse_diagnostics(message);
  return result;
}

[[nodiscard]] std::vector<poe2::Square> history_with_empty_squares(poe2::Bitboard empty_squares) {
  std::vector<poe2::Square> history;
  for (int index = 0; index < poe2::kCellCount; ++index) {
    if ((empty_squares & (poe2::Bitboard{1} << index)) == 0) {
      history.push_back(poe2::square_from_index(index));
    }
  }
  return history;
}

[[nodiscard]] poe2::Position position_from_history(
    const std::vector<poe2::Square>& history, poe2::Symmetry symmetry = poe2::Symmetry::kIdentity) {
  poe2::Position position;
  for (const poe2::Square square : history) {
    REQUIRE(position.play(poe2::transform_square(symmetry, square)));
  }
  return position;
}

void require_legal_principal_variation(const poe2::Position& root,
                                       const poe2::engine::EngineResult& result) {
  REQUIRE(result.best_move.has_value());
  REQUIRE_FALSE(result.principal_variation.empty());
  REQUIRE(result.principal_variation.front() == result.best_move);
  REQUIRE(result.principal_variation.size() <= static_cast<std::size_t>(result.depth));

  poe2::Position position = root;
  for (const poe2::Move move : result.principal_variation) {
    REQUIRE(position.play(move.square));
  }
}

[[nodiscard]] poe2::Score ordinary_negamax_oracle(poe2::Position& position, int depth) {
  if (depth == 0 || position.legal_moves() == 0) {
    return poe2::minimax::evaluate(position);
  }

  poe2::Score best = std::numeric_limits<poe2::Score>::lowest();
  poe2::Bitboard moves = position.legal_moves();
  while (moves != 0) {
    const int move_index = std::countr_zero(moves);
    moves &= moves - poe2::Bitboard{1};
    poe2::MoveUndo undo;
    REQUIRE(position.make_move(poe2::square_from_index(move_index), undo));
    best = std::max(best, -ordinary_negamax_oracle(position, depth - 1));
    position.unmake_move(undo);
  }
  return best;
}

struct RootOracleResult {
  poe2::Score value = std::numeric_limits<poe2::Score>::lowest();
  poe2::Bitboard optimal_moves = 0;
};

struct RootGroupOracleLine {
  poe2::PositionKey canonical_key;
  poe2::Bitboard moves = 0;
  poe2::Score value = std::numeric_limits<poe2::Score>::lowest();
};

[[nodiscard]] std::vector<RootGroupOracleLine> root_group_oracle(const poe2::Position& root,
                                                                 int depth) {
  REQUIRE(depth > 0);
  std::vector<RootGroupOracleLine> result;
  poe2::Position position = root;
  poe2::Bitboard moves = root.legal_moves();
  while (moves != 0) {
    const int move_index = std::countr_zero(moves);
    moves &= moves - poe2::Bitboard{1};
    const poe2::Square square = poe2::square_from_index(move_index);
    poe2::MoveUndo undo;
    REQUIRE(position.make_move(square, undo));
    const poe2::PositionKey canonical_key = poe2::canonicalize_position_key(position.key()).key;
    const poe2::Score value = -ordinary_negamax_oracle(position, depth - 1);
    position.unmake_move(undo);

    const auto group = std::find_if(result.begin(), result.end(),
                                    [canonical_key](const RootGroupOracleLine& candidate) {
                                      return candidate.canonical_key == canonical_key;
                                    });
    if (group == result.end()) {
      result.push_back(RootGroupOracleLine{
          .canonical_key = canonical_key,
          .moves = poe2::square_bit(square),
          .value = value,
      });
    } else {
      REQUIRE(group->value == value);
      group->moves |= poe2::square_bit(square);
    }
  }
  return result;
}

[[nodiscard]] poe2::PositionKey canonical_successor_key(const poe2::Position& root,
                                                        poe2::Move move) {
  poe2::Position child = root;
  REQUIRE(child.play(move.square));
  return poe2::canonicalize_position_key(child.key()).key;
}

void require_legal_analysis_line(const poe2::Position& root,
                                 const poe2::minimax::AnalysisLine& line, int depth) {
  REQUIRE(line.rank > 0);
  REQUIRE_FALSE(line.equivalent_moves.empty());
  REQUIRE_FALSE(line.principal_variation.empty());
  REQUIRE(line.principal_variation.front() == line.move);
  REQUIRE(line.principal_variation.size() <= static_cast<std::size_t>(depth));

  const poe2::PositionKey group_key = canonical_successor_key(root, line.move);
  bool includes_representative = false;
  for (const poe2::Move equivalent : line.equivalent_moves) {
    REQUIRE((root.legal_moves() & poe2::square_bit(equivalent.square)) != 0);
    REQUIRE(canonical_successor_key(root, equivalent) == group_key);
    includes_representative = includes_representative || equivalent == line.move;
  }
  REQUIRE(includes_representative);

  poe2::Position position = root;
  for (const poe2::Move move : line.principal_variation) {
    REQUIRE(position.play(move.square));
  }
}

[[nodiscard]] RootOracleResult ordinary_root_oracle(const poe2::Position& root, int depth) {
  REQUIRE(depth > 0);
  RootOracleResult result;
  poe2::Position position = root;
  poe2::Bitboard moves = position.legal_moves();
  while (moves != 0) {
    const int move_index = std::countr_zero(moves);
    moves &= moves - poe2::Bitboard{1};
    const poe2::Square square = poe2::square_from_index(move_index);
    poe2::MoveUndo undo;
    REQUIRE(position.make_move(square, undo));
    const poe2::Score value = -ordinary_negamax_oracle(position, depth - 1);
    position.unmake_move(undo);

    if (value > result.value) {
      result.value = value;
      result.optimal_moves = poe2::square_bit(square);
    } else if (value == result.value) {
      result.optimal_moves |= poe2::square_bit(square);
    }
  }
  return result;
}

void require_result_move_is_optimal(const poe2::engine::EngineResult& result,
                                    poe2::Bitboard optimal_moves) {
  REQUIRE(result.best_move.has_value());
  const poe2::Move best_move = result.best_move.value_or(poe2::Move{.square = {-1, -1}});
  REQUIRE((optimal_moves & poe2::square_bit(best_move.square)) != 0);
}

}  // namespace

TEST_CASE("minimax requires a move timer and returns a safe fallback", "[minimax]") {
  poe2::Position position;
  std::string message;

  poe2::minimax::Search search;
  const poe2::engine::EngineResult result =
      search.run(position, {}, [&message](std::string_view text) { message = text; });

  REQUIRE(result.best_move == poe2::Move{.square = {0, 0}});
  REQUIRE_FALSE(result.score.has_value());
  REQUIRE(result.depth == 0);
  REQUIRE(result.nodes == 0);
  REQUIRE(result.principal_variation.empty());
  REQUIRE(message == "error minimax_requires_movetime");
}

TEST_CASE("iterative deepening keeps the last completed negamax result", "[minimax]") {
  poe2::Position position;
  const poe2::PositionKey original_key = position.key();
  const poe2::Player perspective = position.side_to_move();

  poe2::minimax::Search search;
  const auto start = std::chrono::steady_clock::now();
  const poe2::engine::EngineResult result =
      search.run(position, timed_limits(std::chrono::milliseconds{50}), {});
  const auto elapsed = std::chrono::steady_clock::now() - start;

  REQUIRE(result.best_move.has_value());
  const poe2::Move best_move = result.best_move.value_or(poe2::Move{.square = {-1, -1}});
  REQUIRE((position.legal_moves() & poe2::square_bit(best_move.square)) != 0);
  REQUIRE(result.score.has_value());
  REQUIRE(result.depth > 0);
  REQUIRE(result.depth < poe2::kCellCount);
  REQUIRE(result.nodes > 0);
  REQUIRE(result.principal_variation.size() <= static_cast<std::size_t>(result.depth));
  REQUIRE_FALSE(result.principal_variation.empty());
  REQUIRE(result.principal_variation.front() == result.best_move);
  REQUIRE(elapsed < std::chrono::seconds{1});
  REQUIRE(position.key() == original_key);

  poe2::Position leaf = position;
  for (const poe2::Move move : result.principal_variation) {
    REQUIRE(leaf.play(move.square));
  }
  if (result.principal_variation.size() == static_cast<std::size_t>(result.depth)) {
    const poe2::Score leaf_value = poe2::minimax::evaluate_two_ply_closure(leaf);
    const poe2::Score expected_value =
        perspective == leaf.side_to_move() ? leaf_value : -leaf_value;
    REQUIRE(result.score == expected_value);
  }
}

TEST_CASE("minimax honors a positive maximum depth without changing uncapped behavior",
          "[minimax][depth]") {
  const poe2::Position position;
  poe2::minimax::Search search{poe2::minimax::SearchOptions{
      .hash_bytes = 0,
      .use_symmetry = false,
      .use_two_ply_closure = false,
  }};
  SearchDiagnostics diagnostics;
  const poe2::engine::EngineResult result =
      run_fixed_depth_with_diagnostics(search, position, 2, diagnostics);

  REQUIRE(result.depth == 2);
  REQUIRE(result.score.has_value());
  REQUIRE(result.nodes > 0);
  REQUIRE(diagnostics.static_evaluations > 0);
  REQUIRE(diagnostics.closure_evaluations == 0);
  require_legal_principal_variation(position, result);
}

TEST_CASE("closure stops at its exact terminal horizon with one two or three empty squares",
          "[minimax][closure][depth][terminal]") {
  const std::array<poe2::Bitboard, 3> empty_square_sets{{
      poe2::square_bit({0, 0}),
      poe2::square_bit({0, 0}) | poe2::square_bit({3, 4}),
      poe2::square_bit({0, 0}) | poe2::square_bit({3, 4}) | poe2::square_bit({6, 6}),
  }};

  for (const poe2::Bitboard empty_squares : empty_square_sets) {
    const poe2::Position position = position_with_empty_squares(empty_squares);
    const int empty_count = position.board().empty_count();
    const RootOracleResult oracle = ordinary_root_oracle(position, empty_count);
    poe2::minimax::Search closure{poe2::minimax::SearchOptions{
        .hash_bytes = 0,
        .use_symmetry = false,
        .use_two_ply_closure = true,
    }};
    SearchDiagnostics diagnostics;
    const poe2::engine::EngineResult result =
        run_with_diagnostics(closure, position, std::chrono::milliseconds{100}, diagnostics);

    CAPTURE(empty_count);
    REQUIRE(result.depth == 1);
    REQUIRE(result.score == oracle.value);
    require_result_move_is_optimal(result, oracle.optimal_moves);
    require_legal_principal_variation(position, result);
    REQUIRE(diagnostics.static_evaluations > 0);
    REQUIRE(diagnostics.closure_evaluations == diagnostics.static_evaluations);
  }
}

TEST_CASE("closure search depth d equals ordinary search depth d plus two without TT or symmetry",
          "[minimax][closure][depth]") {
  struct EquivalenceCase {
    poe2::Bitboard empty_squares = 0;
    int closure_depth = 1;
  };
  const std::array<EquivalenceCase, 5> cases{{
      {
          .empty_squares = poe2::square_bit({0, 0}),
          .closure_depth = 1,
      },
      {
          .empty_squares = poe2::square_bit({0, 0}) | poe2::square_bit({6, 6}),
          .closure_depth = 1,
      },
      {
          .empty_squares = poe2::square_bit({0, 1}) | poe2::square_bit({1, 4}) |
                           poe2::square_bit({2, 6}) | poe2::square_bit({4, 0}) |
                           poe2::square_bit({5, 3}) | poe2::square_bit({6, 5}),
          .closure_depth = 1,
      },
      {
          .empty_squares = poe2::square_bit({0, 1}) | poe2::square_bit({1, 4}) |
                           poe2::square_bit({2, 6}) | poe2::square_bit({4, 0}) |
                           poe2::square_bit({5, 3}) | poe2::square_bit({6, 5}),
          .closure_depth = 2,
      },
      {
          .empty_squares = poe2::square_bit({0, 1}) | poe2::square_bit({1, 4}) |
                           poe2::square_bit({2, 6}) | poe2::square_bit({3, 2}) |
                           poe2::square_bit({4, 0}) | poe2::square_bit({5, 3}) |
                           poe2::square_bit({6, 5}),
          .closure_depth = 3,
      },
  }};

  for (const EquivalenceCase& test_case : cases) {
    const poe2::Position position = position_with_empty_squares(test_case.empty_squares);
    const int ordinary_depth =
        std::min(test_case.closure_depth + 2, position.board().empty_count());
    const RootOracleResult oracle = ordinary_root_oracle(position, ordinary_depth);
    poe2::minimax::Search control{poe2::minimax::SearchOptions{
        .hash_bytes = 0,
        .use_symmetry = false,
        .use_two_ply_closure = false,
    }};
    poe2::minimax::Search closure{poe2::minimax::SearchOptions{
        .hash_bytes = 0,
        .use_symmetry = false,
        .use_two_ply_closure = true,
    }};
    SearchDiagnostics control_diagnostics;
    SearchDiagnostics closure_diagnostics;
    const poe2::engine::EngineResult control_result = run_fixed_depth_with_diagnostics(
        control, position, test_case.closure_depth + 2, control_diagnostics);
    const poe2::engine::EngineResult closure_result = run_fixed_depth_with_diagnostics(
        closure, position, test_case.closure_depth, closure_diagnostics);

    CAPTURE(position.board().empty_count(), test_case.closure_depth, ordinary_depth);
    REQUIRE(control_result.depth == ordinary_depth);
    REQUIRE(closure_result.depth ==
            std::min(test_case.closure_depth, position.board().empty_count()));
    REQUIRE(control_result.score == oracle.value);
    REQUIRE(closure_result.score == oracle.value);
    require_result_move_is_optimal(control_result, oracle.optimal_moves);
    require_result_move_is_optimal(closure_result, oracle.optimal_moves);
    require_legal_principal_variation(position, control_result);
    require_legal_principal_variation(position, closure_result);
    REQUIRE(control_diagnostics.static_evaluations > 0);
    REQUIRE(control_diagnostics.closure_evaluations == 0);
    REQUIRE(control_diagnostics.closure_gain_queries == 0);
    REQUIRE(closure_diagnostics.static_evaluations > 0);
    REQUIRE(closure_diagnostics.closure_evaluations == closure_diagnostics.static_evaluations);
    REQUIRE(closure_diagnostics.gain_queries ==
            closure_diagnostics.move_order_evaluations + closure_diagnostics.closure_gain_queries);
  }
}

TEST_CASE("closure depth equivalence holds with production TT and D4 pruning",
          "[minimax][closure][depth][tt][symmetry]") {
  const poe2::Bitboard empty_squares =
      poe2::square_bit({0, 1}) | poe2::square_bit({1, 4}) | poe2::square_bit({2, 6}) |
      poe2::square_bit({3, 2}) | poe2::square_bit({4, 0}) | poe2::square_bit({4, 5}) |
      poe2::square_bit({5, 3}) | poe2::square_bit({6, 0}) | poe2::square_bit({6, 5});
  const std::vector<poe2::Square> history = history_with_empty_squares(empty_squares);

  for (const poe2::Symmetry symmetry : poe2::kAllSymmetries) {
    const poe2::Position position = position_from_history(history, symmetry);
    const RootOracleResult oracle = ordinary_root_oracle(position, 4);
    poe2::minimax::Search control{poe2::minimax::SearchOptions{
        .hash_bytes = poe2::minimax::kMebibyte,
        .use_symmetry = true,
        .use_two_ply_closure = false,
    }};
    poe2::minimax::Search closure{poe2::minimax::SearchOptions{
        .hash_bytes = poe2::minimax::kMebibyte,
        .use_symmetry = true,
        .use_two_ply_closure = true,
    }};
    SearchDiagnostics control_diagnostics;
    SearchDiagnostics closure_diagnostics;
    const poe2::engine::EngineResult control_result =
        run_fixed_depth_with_diagnostics(control, position, 4, control_diagnostics);
    const poe2::engine::EngineResult closure_result =
        run_fixed_depth_with_diagnostics(closure, position, 2, closure_diagnostics);

    REQUIRE(control_result.depth == 4);
    REQUIRE(closure_result.depth == 2);
    REQUIRE(control_result.score == oracle.value);
    REQUIRE(closure_result.score == oracle.value);
    require_result_move_is_optimal(control_result, oracle.optimal_moves);
    require_result_move_is_optimal(closure_result, oracle.optimal_moves);
    require_legal_principal_variation(position, control_result);
    require_legal_principal_variation(position, closure_result);
    REQUIRE(control_diagnostics.tt_probes > 0);
    REQUIRE(closure_diagnostics.tt_probes > 0);
  }
}

TEST_CASE("negamax searches a small game to completion with the exact handicap", "[minimax]") {
  const poe2::Bitboard empty_squares = poe2::square_bit({0, 0}) | poe2::square_bit({3, 4});
  const poe2::Position position = position_with_empty_squares(empty_squares);
  const poe2::Player perspective = position.side_to_move();

  std::optional<poe2::Move> expected_move;
  poe2::Score expected_value = std::numeric_limits<poe2::Score>::lowest();
  poe2::Bitboard legal_moves = position.legal_moves();
  while (legal_moves != 0) {
    const int move_index = std::countr_zero(legal_moves);
    legal_moves &= legal_moves - poe2::Bitboard{1};

    poe2::Position final_position = position;
    const poe2::Move candidate{.square = poe2::square_from_index(move_index)};
    REQUIRE(final_position.play(candidate.square));

    const poe2::Bitboard reply = final_position.legal_moves();
    REQUIRE(std::popcount(reply) == 1);
    REQUIRE(final_position.play(poe2::square_from_index(std::countr_zero(reply))));

    const poe2::Score candidate_value = value_for(final_position, perspective);
    if (!expected_move.has_value() || candidate_value > expected_value) {
      expected_move = candidate;
      expected_value = candidate_value;
    }
  }

  poe2::minimax::Search search{poe2::minimax::SearchOptions{
      .hash_bytes = poe2::minimax::kMebibyte,
      .use_symmetry = false,
      .use_two_ply_closure = false,
  }};
  SearchDiagnostics diagnostics;
  const poe2::engine::EngineResult result =
      run_with_diagnostics(search, position, std::chrono::milliseconds{100}, diagnostics);

  REQUIRE(result.best_move == expected_move);
  REQUIRE(result.score == expected_value);
  REQUIRE(result.depth == 2);
  REQUIRE(result.nodes == 8);
  REQUIRE(result.principal_variation.size() == 2);
  REQUIRE(diagnostics.tt_hits > 0);
  REQUIRE(diagnostics.tt_cutoffs == 0);

  poe2::Position final_position = position;
  for (const poe2::Move move : result.principal_variation) {
    REQUIRE(final_position.play(move.square));
  }
  REQUIRE(final_position.is_full());
  REQUIRE(result.score == value_for(final_position, perspective));
}

TEST_CASE("principal variation search remains exact on an asymmetric fixed-depth position",
          "[minimax][alphabeta][pvs][move-ordering][oracle]") {
  const poe2::Bitboard empty_squares = poe2::square_bit({0, 0}) | poe2::square_bit({0, 1}) |
                                       poe2::square_bit({0, 2}) | poe2::square_bit({0, 3});
  const poe2::Position position = position_with_empty_squares(empty_squares);
  for (std::size_t index = 1; index < poe2::kAllSymmetries.size(); ++index) {
    REQUIRE(poe2::transform_position_key(poe2::kAllSymmetries[index], position.key()) !=
            position.key());
  }

  const int depth = 4;
  const RootOracleResult oracle = ordinary_root_oracle(position, depth);
  REQUIRE(std::has_single_bit(oracle.optimal_moves));
  const poe2::Move expected_move{
      .square = poe2::square_from_index(std::countr_zero(oracle.optimal_moves)),
  };

  for (const bool use_table : {false, true}) {
    for (const bool use_symmetry : {false, true}) {
      CAPTURE(use_table, use_symmetry);
      poe2::minimax::Search search{poe2::minimax::SearchOptions{
          .hash_bytes = use_table ? poe2::minimax::kMebibyte : 0,
          .use_symmetry = use_symmetry,
          .use_two_ply_closure = false,
      }};
      SearchDiagnostics diagnostics;
      const poe2::engine::EngineResult result =
          run_fixed_depth_with_diagnostics(search, position, depth, diagnostics);

      REQUIRE(result.depth == depth);
      REQUIRE(result.score == oracle.value);
      REQUIRE(result.best_move == expected_move);
      REQUIRE(result.nodes == (use_table ? 74 : 118));
      require_legal_principal_variation(position, result);
      REQUIRE(diagnostics.alpha_beta_cutoffs > 0);
      REQUIRE(diagnostics.pvs_scouts > 0);
      REQUIRE(diagnostics.pvs_researches > 0);
      REQUIRE(diagnostics.move_order_evaluations > 0);
      REQUIRE((diagnostics.tt_probes > 0) == use_table);
      REQUIRE(diagnostics.symmetry_prunes == 0);
    }
  }
}

TEST_CASE("cutoff history breaks equal-gain ties and ages only when the root changes",
          "[minimax][alphabeta][history][move-ordering]") {
  const poe2::Position position = position_from_history({
      {1, 2},
      {6, 0},
      {3, 1},
      {4, 3},
      {2, 6},
      {4, 1},
      {3, 4},
      {1, 5},
  });
  const poe2::Square row_major_move{2, 1};
  const poe2::Square learned_move{2, 2};
  REQUIRE(position.score_gain_unchecked(position.side_to_move(), row_major_move) ==
          position.score_gain_unchecked(position.side_to_move(), learned_move));

  poe2::minimax::Search search{poe2::minimax::SearchOptions{
      .hash_bytes = 0,
      .use_symmetry = false,
      .use_two_ply_closure = false,
  }};
  SearchDiagnostics first_diagnostics;
  const poe2::engine::EngineResult first =
      run_fixed_depth_with_diagnostics(search, position, 4, first_diagnostics);
  SearchDiagnostics second_diagnostics;
  const poe2::engine::EngineResult second =
      run_fixed_depth_with_diagnostics(search, position, 4, second_diagnostics);

  REQUIRE(first.depth == 4);
  REQUIRE(first.score == -11);
  REQUIRE(first.best_move == poe2::Move{.square = row_major_move});
  REQUIRE(first.nodes == 7039);
  REQUIRE(first_diagnostics.history_updates == first_diagnostics.alpha_beta_cutoffs);
  REQUIRE(first_diagnostics.history_updates > 0);
  REQUIRE(first_diagnostics.history_max == 1031);
  require_legal_principal_variation(position, first);

  REQUIRE(second.depth == first.depth);
  REQUIRE(second.score == first.score);
  REQUIRE(second.best_move == poe2::Move{.square = learned_move});
  REQUIRE(second.nodes == 6802);
  REQUIRE(second.nodes < first.nodes);
  REQUIRE(second_diagnostics.history_updates == second_diagnostics.alpha_beta_cutoffs);
  REQUIRE(second_diagnostics.history_max == 2041);
  require_legal_principal_variation(position, second);

  SearchDiagnostics same_root_diagnostics;
  static_cast<void>(run_fixed_depth_with_diagnostics(search, position, 1, same_root_diagnostics));
  REQUIRE(same_root_diagnostics.history_updates == 0);
  REQUIRE(same_root_diagnostics.history_max == second_diagnostics.history_max);

  SearchDiagnostics terminal_diagnostics;
  static_cast<void>(run_with_diagnostics(search, position_with_empty_squares(0),
                                         std::chrono::milliseconds{100}, terminal_diagnostics));
  REQUIRE(terminal_diagnostics.history_updates == 0);
  REQUIRE(terminal_diagnostics.history_max == second_diagnostics.history_max);

  poe2::Position advanced = position;
  REQUIRE(advanced.play({0, 0}));
  SearchDiagnostics changed_root_diagnostics;
  static_cast<void>(
      run_fixed_depth_with_diagnostics(search, advanced, 1, changed_root_diagnostics));
  REQUIRE(changed_root_diagnostics.history_updates == 0);
  REQUIRE(changed_root_diagnostics.history_max == second_diagnostics.history_max / 2);

  SearchDiagnostics repeated_changed_root_diagnostics;
  static_cast<void>(
      run_fixed_depth_with_diagnostics(search, advanced, 1, repeated_changed_root_diagnostics));
  REQUIRE(repeated_changed_root_diagnostics.history_updates == 0);
  REQUIRE(repeated_changed_root_diagnostics.history_max == changed_root_diagnostics.history_max);

  search.new_game();
  SearchDiagnostics reset_diagnostics;
  static_cast<void>(run_fixed_depth_with_diagnostics(search, advanced, 1, reset_diagnostics));
  REQUIRE(reset_diagnostics.history_updates == 0);
  REQUIRE(reset_diagnostics.history_max == 0);
}

TEST_CASE("fail-soft bounds are cached and reused", "[minimax][alphabeta][tt]") {
  const poe2::Bitboard empty_squares = poe2::square_bit({0, 0}) | poe2::square_bit({0, 1}) |
                                       poe2::square_bit({0, 2}) | poe2::square_bit({0, 3});
  const poe2::Position position = position_with_empty_squares(empty_squares);
  poe2::minimax::Search search{poe2::minimax::SearchOptions{
      .hash_bytes = poe2::minimax::kMebibyte,
      .use_symmetry = false,
      .use_two_ply_closure = false,
  }};

  const poe2::engine::EngineResult first =
      search.run(position, timed_limits(std::chrono::milliseconds{100}), {});

  const auto require_cached_move_legal = [&search](const poe2::Position& cached_position, int depth,
                                                   poe2::TranspositionBound bound) {
    const std::optional<poe2::TranspositionEntry> entry =
        search.transposition_table().probe(cached_position);
    REQUIRE(entry.has_value());
    REQUIRE(entry->value.depth == depth);
    REQUIRE(entry->value.bound == bound);
    REQUIRE(entry->value.best_move.has_value());
    REQUIRE((cached_position.legal_moves() & poe2::square_bit(*entry->value.best_move)) != 0);
    return *entry;
  };

  const poe2::TranspositionEntry root_entry =
      require_cached_move_legal(position, 4, poe2::TranspositionBound::kExact);
  REQUIRE(root_entry.value.score == -41);

  poe2::Position b1_child = position;
  REQUIRE(b1_child.play({0, 1}));
  static_cast<void>(require_cached_move_legal(b1_child, 3, poe2::TranspositionBound::kLower));

  poe2::Position b1_a1_descendant = b1_child;
  REQUIRE(b1_a1_descendant.play({0, 0}));
  static_cast<void>(
      require_cached_move_legal(b1_a1_descendant, 2, poe2::TranspositionBound::kUpper));

  const poe2::engine::EngineResult second =
      search.run(position, timed_limits(std::chrono::milliseconds{100}), {});
  REQUIRE(second.depth == 4);
  REQUIRE(second.score == first.score);
  REQUIRE(second.best_move == first.best_move);
  REQUIRE(second.principal_variation == first.principal_variation);

  const std::optional<poe2::TranspositionEntry> repeated_root_entry =
      search.transposition_table().probe(position);
  if (!repeated_root_entry.has_value()) {
    FAIL("repeated search should preserve the completed root entry");
    return;
  }
  REQUIRE(repeated_root_entry->value.depth == 4);
  REQUIRE(repeated_root_entry->value.bound == poe2::TranspositionBound::kExact);
  REQUIRE(repeated_root_entry->value.score == first.score);
}

TEST_CASE("minimax search handles a full board", "[minimax]") {
  const poe2::Position position = position_with_empty_squares(0);

  poe2::minimax::Search search;
  const poe2::engine::EngineResult result =
      search.run(position, timed_limits(std::chrono::milliseconds{10}), {});

  REQUIRE_FALSE(result.best_move.has_value());
  REQUIRE_FALSE(result.score.has_value());
  REQUIRE(result.depth == 0);
  REQUIRE(result.nodes == 0);
  REQUIRE(result.principal_variation.empty());
}

TEST_CASE("symmetry pruning works without a transposition table", "[minimax][symmetry]") {
  const poe2::Bitboard empty_squares =
      poe2::square_bit({0, 0}) | poe2::square_bit({0, poe2::kBoardSize - 1});
  const poe2::Position position = position_with_empty_squares(empty_squares);
  REQUIRE(poe2::transform_position_key(poe2::Symmetry::kReflectVertical, position.key()) ==
          position.key());

  poe2::minimax::Search symmetric_search{poe2::minimax::SearchOptions{
      .hash_bytes = 0,
      .use_symmetry = true,
      .use_two_ply_closure = false,
  }};
  SearchDiagnostics symmetric_diagnostics;
  const poe2::engine::EngineResult symmetric_result = run_with_diagnostics(
      symmetric_search, position, std::chrono::milliseconds{100}, symmetric_diagnostics);

  poe2::minimax::Search raw_search{poe2::minimax::SearchOptions{
      .hash_bytes = 0,
      .use_symmetry = false,
      .use_two_ply_closure = false,
  }};
  SearchDiagnostics raw_diagnostics;
  const poe2::engine::EngineResult raw_result =
      run_with_diagnostics(raw_search, position, std::chrono::milliseconds{100}, raw_diagnostics);

  REQUIRE(symmetric_result.depth == 2);
  REQUIRE(symmetric_result.nodes == 5);
  REQUIRE(symmetric_diagnostics.symmetry_prunes == 2);
  REQUIRE(symmetric_diagnostics.tt_probes == 0);
  REQUIRE(symmetric_diagnostics.hash_capacity == 0);
  REQUIRE(symmetric_diagnostics.hash_bytes == 0);
  REQUIRE(raw_result.depth == 2);
  REQUIRE(raw_result.nodes == 8);
  REQUIRE(raw_diagnostics.symmetry_prunes == 0);
  REQUIRE(raw_result.score == symmetric_result.score);
}

TEST_CASE("all board orientations reuse canonical entries and remap cached moves",
          "[minimax][symmetry][tt]") {
  const poe2::Bitboard empty_squares =
      poe2::square_bit({0, 1}) | poe2::square_bit({2, 5}) | poe2::square_bit({5, 3});
  const std::vector<poe2::Square> history = history_with_empty_squares(empty_squares);
  const poe2::Position seed_position = position_from_history(history);
  poe2::minimax::Search search{poe2::minimax::SearchOptions{
      .hash_bytes = poe2::minimax::kMebibyte,
      .use_symmetry = true,
      .use_two_ply_closure = false,
  }};

  SearchDiagnostics seed_diagnostics;
  const poe2::engine::EngineResult seed_result =
      run_with_diagnostics(search, seed_position, std::chrono::milliseconds{100}, seed_diagnostics);
  REQUIRE(seed_result.depth == 3);
  REQUIRE(seed_diagnostics.tt_stores > 0);

  for (const poe2::Symmetry symmetry : poe2::kAllSymmetries) {
    const poe2::Position equivalent = position_from_history(history, symmetry);
    SearchDiagnostics diagnostics;
    const poe2::engine::EngineResult result =
        run_with_diagnostics(search, equivalent, std::chrono::milliseconds{100}, diagnostics);

    REQUIRE(result.depth == 3);
    REQUIRE(result.nodes == 3);
    REQUIRE(diagnostics.tt_probes == 3);
    REQUIRE(diagnostics.tt_hits == 3);
    REQUIRE(diagnostics.tt_cutoffs == 3);
    REQUIRE(diagnostics.tt_stores == 0);
    require_legal_principal_variation(equivalent, result);
  }
}

TEST_CASE("raw-key transposition entries are reused when symmetry is disabled", "[minimax][tt]") {
  const poe2::Bitboard empty_squares =
      poe2::square_bit({0, 1}) | poe2::square_bit({2, 5}) | poe2::square_bit({5, 3});
  const poe2::Position position = position_from_history(history_with_empty_squares(empty_squares));
  poe2::minimax::Search search{poe2::minimax::SearchOptions{
      .hash_bytes = poe2::minimax::kMebibyte,
      .use_symmetry = false,
      .use_two_ply_closure = false,
  }};

  SearchDiagnostics first_diagnostics;
  const poe2::engine::EngineResult first =
      run_with_diagnostics(search, position, std::chrono::milliseconds{100}, first_diagnostics);
  SearchDiagnostics second_diagnostics;
  const poe2::engine::EngineResult second =
      run_with_diagnostics(search, position, std::chrono::milliseconds{100}, second_diagnostics);

  REQUIRE(first.depth == 3);
  REQUIRE(second.depth == 3);
  REQUIRE(second.score == first.score);
  REQUIRE(second.nodes == 3);
  REQUIRE(second_diagnostics.tt_hits == 3);
  REQUIRE(second_diagnostics.tt_cutoffs == 3);
  REQUIRE(second_diagnostics.symmetry_prunes == 0);
  require_legal_principal_variation(position, second);
}

TEST_CASE("an interrupted root is not stored at its unfinished depth", "[minimax][tt]") {
  const poe2::Position position;
  poe2::minimax::Search search;

  const poe2::engine::EngineResult result =
      search.run(position, timed_limits(std::chrono::milliseconds{10}), {});

  REQUIRE(result.depth > 0);
  REQUIRE(result.depth < poe2::kCellCount);
  const poe2::PositionKey root_key = poe2::canonicalize_position_key(position.key()).key;
  const std::optional<poe2::TranspositionEntry> root_entry =
      search.transposition_table().probe(root_key);
  if (!root_entry.has_value()) {
    FAIL("completed root iteration should remain cached");
    return;
  }
  REQUIRE(root_entry->value.depth == result.depth);
  REQUIRE(root_entry->value.bound == poe2::TranspositionBound::kExact);
}

TEST_CASE("newgame clears cached entries while per-search counters reset", "[minimax][tt]") {
  const poe2::Bitboard empty_squares = poe2::square_bit({0, 1}) | poe2::square_bit({2, 5});
  const poe2::Position position = position_from_history(history_with_empty_squares(empty_squares));
  poe2::minimax::Search search{
      poe2::minimax::SearchOptions{.hash_bytes = poe2::minimax::kMebibyte, .use_symmetry = true}};

  SearchDiagnostics populated_diagnostics;
  static_cast<void>(run_with_diagnostics(search, position, std::chrono::milliseconds{100},
                                         populated_diagnostics));
  const std::size_t populated_entries = search.transposition_table().size();
  REQUIRE(populated_entries > 0);

  const poe2::Position full_position = position_with_empty_squares(0);
  SearchDiagnostics idle_diagnostics;
  static_cast<void>(
      run_with_diagnostics(search, full_position, std::chrono::milliseconds{10}, idle_diagnostics));
  REQUIRE(idle_diagnostics.tt_probes == 0);
  REQUIRE(idle_diagnostics.tt_hits == 0);
  REQUIRE(idle_diagnostics.tt_cutoffs == 0);
  REQUIRE(idle_diagnostics.alpha_beta_cutoffs == 0);
  REQUIRE(idle_diagnostics.tt_stores == 0);
  REQUIRE(idle_diagnostics.pvs_scouts == 0);
  REQUIRE(idle_diagnostics.pvs_researches == 0);
  REQUIRE(idle_diagnostics.symmetry_prunes == 0);
  REQUIRE(idle_diagnostics.move_order_evaluations == 0);
  REQUIRE(idle_diagnostics.history_updates == 0);
  REQUIRE(idle_diagnostics.hash_entries == populated_entries);

  search.new_game();
  REQUIRE(search.transposition_table().empty());
  REQUIRE(search.transposition_table().capacity() == idle_diagnostics.hash_capacity);

  SearchDiagnostics cleared_diagnostics;
  static_cast<void>(run_with_diagnostics(search, full_position, std::chrono::milliseconds{10},
                                         cleared_diagnostics));
  REQUIRE(cleared_diagnostics.hash_entries == 0);
}

TEST_CASE("collision-broken cached principal variations remain legal prefixes", "[minimax][tt]") {
  poe2::minimax::Search search{poe2::minimax::SearchOptions{
      .hash_bytes = 64,
      .use_symmetry = true,
      .use_two_ply_closure = false,
  }};
  REQUIRE(search.transposition_table().capacity() == 2);
  REQUIRE(search.transposition_table().storage_bytes() == 64);

  const poe2::Bitboard empty_squares = poe2::square_bit({0, 1}) | poe2::square_bit({2, 5}) |
                                       poe2::square_bit({4, 0}) | poe2::square_bit({5, 3});
  const poe2::Position position = position_from_history(history_with_empty_squares(empty_squares));
  const poe2::engine::EngineResult result =
      search.run(position, timed_limits(std::chrono::milliseconds{100}), {});

  REQUIRE(result.depth == 4);
  REQUIRE(result.principal_variation.size() < 4);
  require_legal_principal_variation(position, result);
}

TEST_CASE("single-PV analysis exactly preserves the native search result",
          "[minimax][analysis][multipv][parity]") {
  const poe2::Position position;
  const poe2::minimax::SearchOptions options{
      .hash_bytes = poe2::minimax::kMebibyte,
      .use_symmetry = true,
      .use_two_ply_closure = false,
  };
  poe2::minimax::Search native_search{options};
  poe2::minimax::Search analysis_search{options};
  poe2::minimax::Search streamed_search{options};
  const poe2::engine::EngineLimits limits = timed_limits(std::chrono::seconds{5}, 3);

  const poe2::engine::EngineResult native = native_search.run(position, limits, {});
  const poe2::minimax::AnalysisResult analysis = analysis_search.analyze(position, limits, 1);
  std::vector<poe2::minimax::AnalysisResult> updates;
  const poe2::minimax::AnalysisResult streamed = streamed_search.analyze(
      position, limits, 1,
      [&updates](const poe2::minimax::AnalysisResult& update) { updates.push_back(update); });

  REQUIRE(analysis.completed_depth == native.depth);
  REQUIRE(analysis.nodes == native.nodes);
  REQUIRE(analysis.lines.size() == 1);
  REQUIRE(analysis.lines.front().rank == 1);
  REQUIRE(analysis.lines.front().move == native.best_move);
  REQUIRE(analysis.lines.front().score == native.score);
  REQUIRE(analysis.lines.front().principal_variation == native.principal_variation);
  require_legal_analysis_line(position, analysis.lines.front(), analysis.completed_depth);
  REQUIRE(streamed == analysis);
  REQUIRE(updates.size() == 3);
  for (std::size_t index = 0; index < updates.size(); ++index) {
    REQUIRE(updates[index].completed_depth == static_cast<int>(index) + 1);
    if (index > 0) {
      REQUIRE(updates[index - 1].nodes < updates[index].nodes);
    }
  }
}

TEST_CASE("fixed-depth Multi-PV matches a root oracle and survives a warm root entry",
          "[minimax][analysis][multipv][oracle][tt]") {
  const poe2::Bitboard empty_squares = poe2::square_bit({0, 0}) | poe2::square_bit({1, 2}) |
                                       poe2::square_bit({2, 5}) | poe2::square_bit({4, 1}) |
                                       poe2::square_bit({5, 4}) | poe2::square_bit({6, 6});
  const poe2::Position position = position_with_empty_squares(empty_squares);
  const int depth = 3;
  const std::vector<RootGroupOracleLine> oracle = root_group_oracle(position, depth);
  poe2::minimax::Search search{poe2::minimax::SearchOptions{
      .hash_bytes = poe2::minimax::kMebibyte,
      .use_symmetry = true,
      .use_two_ply_closure = false,
  }};

  const poe2::engine::EngineLimits limits = timed_limits(std::chrono::seconds{5}, depth);
  const poe2::engine::EngineResult warmup = search.run(position, limits, {});
  REQUIRE(warmup.depth == depth);
  const poe2::minimax::AnalysisResult result = search.analyze(position, limits, 5);

  REQUIRE(result.completed_depth == depth);
  REQUIRE(result.lines.size() == std::min<std::size_t>(5, oracle.size()));
  std::vector<poe2::PositionKey> returned_keys;
  for (std::size_t index = 0; index < result.lines.size(); ++index) {
    const poe2::minimax::AnalysisLine& line = result.lines[index];
    REQUIRE(line.rank == static_cast<int>(index) + 1);
    require_legal_analysis_line(position, line, depth);
    if (index > 0) {
      REQUIRE(result.lines[index - 1].score >= line.score);
    }

    const poe2::PositionKey key = canonical_successor_key(position, line.move);
    REQUIRE(std::find(returned_keys.begin(), returned_keys.end(), key) == returned_keys.end());
    returned_keys.push_back(key);
    const auto expected = std::find_if(
        oracle.begin(), oracle.end(),
        [key](const RootGroupOracleLine& candidate) { return candidate.canonical_key == key; });
    REQUIRE(expected != oracle.end());
    REQUIRE(line.score == expected->value);

    poe2::Bitboard returned_moves = 0;
    for (const poe2::Move equivalent : line.equivalent_moves) {
      returned_moves |= poe2::square_bit(equivalent.square);
    }
    REQUIRE(returned_moves == expected->moves);
  }

  const poe2::minimax::AnalysisResult repeated = search.analyze(position, limits, 5);
  REQUIRE(repeated.completed_depth == result.completed_depth);
  REQUIRE(repeated.lines == result.lines);
  REQUIRE(repeated.nodes <= result.nodes);
}

TEST_CASE("symmetric placements share ranks while equal unrelated groups remain separate",
          "[minimax][analysis][multipv][symmetry]") {
  const poe2::Position position;
  poe2::minimax::Search search{poe2::minimax::SearchOptions{
      .hash_bytes = 0,
      .use_symmetry = false,
      .use_two_ply_closure = false,
  }};
  const poe2::minimax::AnalysisResult result =
      search.analyze(position, timed_limits(std::chrono::seconds{5}, 1), 5);

  REQUIRE(result.completed_depth == 1);
  REQUIRE(result.lines.size() == 5);
  std::size_t physical_move_count = 0;
  std::vector<poe2::PositionKey> keys;
  for (const poe2::minimax::AnalysisLine& line : result.lines) {
    REQUIRE(line.score == result.lines.front().score);
    require_legal_analysis_line(position, line, 1);
    physical_move_count += line.equivalent_moves.size();
    const poe2::PositionKey key = canonical_successor_key(position, line.move);
    REQUIRE(std::find(keys.begin(), keys.end(), key) == keys.end());
    keys.push_back(key);
  }
  REQUIRE(physical_move_count > result.lines.size());
}

TEST_CASE("Multi-PV returns fewer lines when fewer successor groups exist",
          "[minimax][analysis][multipv][terminal]") {
  const poe2::Position position = position_with_empty_squares(poe2::square_bit({3, 4}));
  poe2::minimax::Search search;
  const poe2::minimax::AnalysisResult result =
      search.analyze(position, timed_limits(std::chrono::seconds{5}), 5);

  REQUIRE(result.completed_depth == 1);
  REQUIRE(result.lines.size() == 1);
  REQUIRE(result.lines.front().equivalent_moves == std::vector<poe2::Move>{{.square = {3, 4}}});
  require_legal_analysis_line(position, result.lines.front(), 1);
}

TEST_CASE(
    "Multi-PV progress contains only complete common depths and keeps incomplete work private",
    "[minimax][analysis][multipv][progress][timeout]") {
  const poe2::Position position;
  poe2::minimax::Search search{poe2::minimax::SearchOptions{
      .hash_bytes = poe2::minimax::kMebibyte,
      .use_symmetry = true,
      .use_two_ply_closure = false,
  }};
  std::vector<poe2::minimax::AnalysisResult> updates;
  const poe2::minimax::AnalysisResult result =
      search.analyze(position, timed_limits(std::chrono::milliseconds{100}, 6), 5,
                     [&updates](const poe2::minimax::AnalysisResult& update) {
                       updates.push_back(update);
                       if (update.completed_depth == 1) {
                         std::this_thread::sleep_for(std::chrono::milliseconds{150});
                       }
                     });

  REQUIRE(updates.size() == 1);
  REQUIRE(updates.front().completed_depth == 1);
  REQUIRE(updates.front().lines.size() == 5);
  REQUIRE(result.completed_depth == updates.front().completed_depth);
  REQUIRE(result.lines == updates.front().lines);
  REQUIRE(result.nodes > updates.front().nodes);
}
