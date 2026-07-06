#include "poe2/engine_stdio.hpp"

#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <optional>
#include <sstream>
#include <string>

namespace {

poe2::engine_stdio::EngineLimits require_limits(
    std::optional<poe2::engine_stdio::EngineLimits> limits) {
  if (!limits.has_value()) {
    FAIL("go limits should parse");
    return {};
  }

  return *limits;
}

class RecordingEngine final : public poe2::engine_stdio::Engine {
 public:
  void new_game() override { ++new_games; }

  [[nodiscard]] poe2::engine_stdio::EngineResult choose_move(
      const poe2::Position&, const poe2::engine_stdio::EngineLimits& limits,
      const poe2::engine_stdio::InfoSink& info) override {
    called = true;
    observed = limits;
    info("depth 2 nodes 40");
    return poe2::engine_stdio::EngineResult{
        .best_move = poe2::Move{.square = {0, 0}},
        .score = 7,
        .depth = 3,
        .nodes = 99,
        .principal_variation = {poe2::Move{.square = {0, 0}}, poe2::Move{.square = {1, 1}}},
    };
  }

  poe2::engine_stdio::EngineLimits observed;
  int new_games = 0;
  bool called = false;
};

}  // namespace

TEST_CASE("go command parser accepts engine limits", "[engine_stdio]") {
  std::ostringstream output;

  const poe2::engine_stdio::EngineLimits empty =
      require_limits(poe2::engine_stdio::parse_go_limits("go", output));
  REQUIRE_FALSE(empty.depth.has_value());
  REQUIRE_FALSE(empty.move_time.has_value());
  REQUIRE_FALSE(empty.nodes.has_value());
  REQUIRE(output.str().empty());

  const poe2::engine_stdio::EngineLimits limits = require_limits(
      poe2::engine_stdio::parse_go_limits("go depth 5 movetime 250 nodes 1000", output));
  REQUIRE(limits.depth == 5);
  REQUIRE(limits.move_time == std::chrono::milliseconds{250});
  REQUIRE(limits.nodes == 1000);
  REQUIRE(poe2::engine_stdio::format_go_command(limits) == "go depth 5 movetime 250 nodes 1000");
  REQUIRE(output.str().empty());
}

TEST_CASE("go command parser rejects malformed limits", "[engine_stdio]") {
  std::ostringstream output;

  REQUIRE_FALSE(poe2::engine_stdio::parse_go_limits("go depth nope", output).has_value());
  REQUIRE(output.str() == "info error malformed_go_value depth nope\n");

  output.str({});
  output.clear();
  REQUIRE_FALSE(poe2::engine_stdio::parse_go_limits("go movetime", output).has_value());
  REQUIRE(output.str() == "info error missing_go_value movetime\n");

  output.str({});
  output.clear();
  REQUIRE_FALSE(poe2::engine_stdio::parse_go_limits("go visits 10", output).has_value());
  REQUIRE(output.str() == "info error unknown_go_option visits\n");

  output.str({});
  output.clear();
  REQUIRE_FALSE(poe2::engine_stdio::parse_go_limits("go visits", output).has_value());
  REQUIRE(output.str() == "info error unknown_go_option visits\n");
}

TEST_CASE("engine stdio adapter passes limits and formats engine result info", "[engine_stdio]") {
  std::istringstream input{"newgame\ngo depth 3 movetime 25 nodes 99\nquit\n"};
  std::ostringstream output;
  RecordingEngine engine;

  REQUIRE(poe2::engine_stdio::run_engine_stdio("test", engine, input, output) == 0);

  REQUIRE(engine.new_games == 1);
  REQUIRE(engine.called);
  REQUIRE(engine.observed.depth == 3);
  REQUIRE(engine.observed.move_time == std::chrono::milliseconds{25});
  REQUIRE(engine.observed.nodes == 99);
  REQUIRE(output.str() ==
          "info depth 2 nodes 40\n"
          "info depth 3 score 7 nodes 99 pv a1 b2\n"
          "bestmove a1\n");
}
