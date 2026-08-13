#include <catch2/catch_test_macros.hpp>
#include <cctype>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "poe2/wasm/adapter.hpp"

namespace {

[[nodiscard]] std::vector<std::string> row_major_history(int plies) {
  std::vector<std::string> moves;
  moves.reserve(static_cast<std::size_t>(plies));
  for (int index = 0; index < plies; ++index) {
    moves.push_back(poe2::format_move(poe2::Move{.square = poe2::square_from_index(index)}));
  }
  return moves;
}

[[nodiscard]] poe2::wasm::AnalysisSuccess require_success(
    const poe2::wasm::AnalysisResponse& response) {
  const auto* success = std::get_if<poe2::wasm::AnalysisSuccess>(&response);
  REQUIRE(success != nullptr);
  if (success == nullptr) {
    return {};
  }
  return *success;
}

[[nodiscard]] poe2::wasm::AnalysisError require_error(const poe2::wasm::AnalysisResponse& response,
                                                      poe2::wasm::AnalysisErrorCode code) {
  const auto* error = std::get_if<poe2::wasm::AnalysisError>(&response);
  REQUIRE(error != nullptr);
  if (error == nullptr) {
    return {};
  }
  REQUIRE(error->code == code);
  return *error;
}

}  // namespace

TEST_CASE("WASM adapter exposes stable versions and a 16 MiB hash", "[wasm][adapter]") {
  const poe2::wasm::Analyzer analyzer;

  REQUIRE(poe2::wasm::engine_version() == "0.1.0");
  REQUIRE(poe2::wasm::kApiVersion == 1);
  REQUIRE(analyzer.hash_storage_bytes() == 16 * poe2::minimax::kMebibyte);
}

TEST_CASE("WASM adapter normalizes search results to Player 1 and reuses its hash",
          "[wasm][adapter][parity]") {
  poe2::wasm::Analyzer analyzer;
  const poe2::wasm::AnalysisRequest request{
      .moves = row_major_history(45),
      .search_time_ms = 5000,
      .max_depth = 2,
  };

  const poe2::wasm::AnalysisSuccess first = require_success(analyzer.analyze(request));
  REQUIRE(first.best_move == "g7");
  REQUIRE(first.evaluation_half_points == 41);
  REQUIRE(first.completed_depth == 2);
  REQUIRE(first.nodes == 16);
  REQUIRE(first.principal_variation == std::vector<std::string>{"g7", "f7"});

  const poe2::wasm::AnalysisSuccess second = require_success(analyzer.analyze(request));
  REQUIRE(second.best_move == first.best_move);
  REQUIRE(second.evaluation_half_points == first.evaluation_half_points);
  REQUIRE(second.completed_depth == first.completed_depth);
  REQUIRE(second.nodes == 2);
  REQUIRE(second.principal_variation == first.principal_variation);
}

TEST_CASE("WASM adapter preserves Player 1 search scores when Player 1 is to move",
          "[wasm][adapter][parity]") {
  poe2::wasm::Analyzer analyzer;
  const poe2::wasm::AnalysisSuccess result =
      require_success(analyzer.analyze(poe2::wasm::AnalysisRequest{
          .moves = row_major_history(46),
          .search_time_ms = 5000,
          .max_depth = 1,
      }));

  REQUIRE(result.best_move == "g7");
  REQUIRE(result.evaluation_half_points == 69);
  REQUIRE(result.completed_depth == 1);
  REQUIRE(result.nodes == 4);
  REQUIRE(result.principal_variation == std::vector<std::string>{"g7"});
}

TEST_CASE("WASM adapter returns structured history errors", "[wasm][adapter][errors]") {
  poe2::wasm::Analyzer analyzer;

  std::vector<std::string> overlong_history = row_major_history(poe2::kCellCount);
  overlong_history.emplace_back("a1");
  const poe2::wasm::AnalysisError overlong =
      require_error(analyzer.analyze(poe2::wasm::AnalysisRequest{
                        .moves = std::move(overlong_history),
                        .search_time_ms = 10,
                    }),
                    poe2::wasm::AnalysisErrorCode::kInvalidHistory);
  REQUIRE(overlong.field == poe2::wasm::AnalysisErrorField::kMoves);

  const poe2::wasm::AnalysisError malformed =
      require_error(analyzer.analyze(poe2::wasm::AnalysisRequest{
                        .moves = {"h1"},
                        .search_time_ms = 10,
                    }),
                    poe2::wasm::AnalysisErrorCode::kMalformedHistoryMove);
  REQUIRE(malformed.field == poe2::wasm::AnalysisErrorField::kMoves);
  REQUIRE(malformed.move_index == 0);
  REQUIRE(malformed.move == "h1");
  REQUIRE_FALSE(malformed.reason.has_value());

  const poe2::wasm::AnalysisError illegal =
      require_error(analyzer.analyze(poe2::wasm::AnalysisRequest{
                        .moves = {"a1", "a1"},
                        .search_time_ms = 10,
                    }),
                    poe2::wasm::AnalysisErrorCode::kIllegalHistoryMove);
  REQUIRE(illegal.field == poe2::wasm::AnalysisErrorField::kMoves);
  REQUIRE(illegal.move_index == 1);
  REQUIRE(illegal.move == "a1");
  REQUIRE(illegal.reason == poe2::MoveError::kOccupied);

  const poe2::wasm::AnalysisError terminal =
      require_error(analyzer.analyze(poe2::wasm::AnalysisRequest{
                        .moves = row_major_history(poe2::kCellCount),
                        .search_time_ms = 10,
                    }),
                    poe2::wasm::AnalysisErrorCode::kTerminalPosition);
  REQUIRE(terminal.field == poe2::wasm::AnalysisErrorField::kMoves);
}

TEST_CASE("WASM adapter validates search limits", "[wasm][adapter][errors]") {
  poe2::wasm::Analyzer analyzer;

  const poe2::wasm::AnalysisError time_error =
      require_error(analyzer.analyze(poe2::wasm::AnalysisRequest{}),
                    poe2::wasm::AnalysisErrorCode::kInvalidSearchTime);
  REQUIRE(time_error.field == poe2::wasm::AnalysisErrorField::kSearchTimeMs);

  for (const int depth : {0, poe2::kCellCount + 1}) {
    const poe2::wasm::AnalysisError depth_error =
        require_error(analyzer.analyze(poe2::wasm::AnalysisRequest{
                          .search_time_ms = 10,
                          .max_depth = depth,
                      }),
                      poe2::wasm::AnalysisErrorCode::kInvalidMaxDepth);
    REQUIRE(depth_error.field == poe2::wasm::AnalysisErrorField::kMaxDepth);
  }
}

TEST_CASE("WASM adapter accepts uppercase input and emits canonical notation", "[wasm][adapter]") {
  poe2::wasm::Analyzer analyzer;
  const poe2::wasm::AnalysisSuccess result =
      require_success(analyzer.analyze(poe2::wasm::AnalysisRequest{
          .moves = {"A1"},
          .search_time_ms = 5000,
          .max_depth = 1,
      }));

  REQUIRE(std::islower(static_cast<unsigned char>(result.best_move.front())) != 0);
  for (const std::string& move : result.principal_variation) {
    REQUIRE(std::islower(static_cast<unsigned char>(move.front())) != 0);
  }
}
