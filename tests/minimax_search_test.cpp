#include <catch2/catch_test_macros.hpp>

#include "poe2/minimax/search.hpp"
#include "poe2/move.hpp"

TEST_CASE("placeholder minimax search returns a legal move", "[minimax]") {
  poe2::Position position;
  REQUIRE(position.play(poe2::Square{0, 0}));

  poe2::minimax::Search search;
  const poe2::engine::EngineResult result = search.run(position, {}, {});

  REQUIRE(result.best_move == poe2::Move{.square = {0, 1}});
}

TEST_CASE("placeholder minimax search handles a full board", "[minimax]") {
  poe2::Position position;
  for (int index = 0; index < poe2::kCellCount; ++index) {
    REQUIRE(position.play(poe2::square_from_index(index)));
  }

  poe2::minimax::Search search;
  const poe2::engine::EngineResult result = search.run(position, {}, {});

  REQUIRE_FALSE(result.best_move.has_value());
}
