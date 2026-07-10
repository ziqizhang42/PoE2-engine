#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

#include "poe2/minimax/evaluation.hpp"
#include "poe2/minimax/search.hpp"
#include "poe2/move.hpp"

namespace {

[[nodiscard]] poe2::engine::EngineLimits timed_limits(std::chrono::milliseconds move_time) {
  return poe2::engine::EngineLimits{.move_time = move_time};
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
  REQUIRE(result.principal_variation.size() == static_cast<std::size_t>(result.depth));
  REQUIRE(result.principal_variation.front() == result.best_move);
  REQUIRE(elapsed < std::chrono::seconds{1});
  REQUIRE(position.key() == original_key);

  poe2::Position leaf = position;
  for (const poe2::Move move : result.principal_variation) {
    REQUIRE(leaf.play(move.square));
  }
  REQUIRE(result.score == value_for(leaf, perspective));
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

  poe2::minimax::Search search;
  const poe2::engine::EngineResult result =
      search.run(position, timed_limits(std::chrono::milliseconds{100}), {});

  REQUIRE(result.best_move == expected_move);
  REQUIRE(result.score == expected_value);
  REQUIRE(result.depth == 2);
  REQUIRE(result.nodes == 8);
  REQUIRE(result.principal_variation.size() == 2);

  poe2::Position final_position = position;
  for (const poe2::Move move : result.principal_variation) {
    REQUIRE(final_position.play(move.square));
  }
  REQUIRE(final_position.is_full());
  REQUIRE(result.score == value_for(final_position, perspective));
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
