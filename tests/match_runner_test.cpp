#include "poe2/match_runner.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace {

enum class TestOutcome : std::uint8_t {
  kEngineOneWin,
  kEngineTwoWin,
  kNoWinner,
};

[[nodiscard]] poe2::match_runner::SeriesGameResult make_game(int game_number,
                                                             poe2::Player engine_one_player,
                                                             TestOutcome outcome) {
  std::optional<poe2::Player> winner;
  if (outcome == TestOutcome::kEngineOneWin) {
    winner = engine_one_player;
  } else if (outcome == TestOutcome::kEngineTwoWin) {
    winner = poe2::opponent(engine_one_player);
  }

  return poe2::match_runner::SeriesGameResult{
      .game_number = game_number,
      .engine_one_player = engine_one_player,
      .opening_line_number = (game_number + 1) / 2,
      .opening_moves = "a1 b1",
      .match =
          poe2::match_runner::MatchResult{
              .reason = poe2::match_runner::MatchEndReason::kNormal,
              .winner = winner,
          },
  };
}

void add_pair(poe2::match_runner::SeriesResult& result, TestOutcome first, TestOutcome second) {
  const int first_game = static_cast<int>(result.games.size()) + 1;
  result.games.push_back(make_game(first_game, poe2::Player::kOne, first));
  result.games.push_back(make_game(first_game + 1, poe2::Player::kTwo, second));
  result.games_played = static_cast<int>(result.games.size());
}

}  // namespace

TEST_CASE("opening book parser accepts comments and normalizes moves", "[match_runner]") {
  constexpr std::string_view kBook =
      "# generated\n"
      "\n"
      "A1 b1 c1 d1 # trailing comment\n"
      "g7 f7 e7 d7\n";

  const poe2::match_runner::OpeningBook book =
      poe2::match_runner::parse_opening_book_text("inline", kBook);

  REQUIRE(book.path == "inline");
  REQUIRE(book.lines.size() == 2);
  REQUIRE(book.lines[0].line_number == 3);
  REQUIRE(book.lines[0].text == "a1 b1 c1 d1");
  REQUIRE(book.lines[1].line_number == 4);
  REQUIRE(book.lines[1].text == "g7 f7 e7 d7");
}

TEST_CASE("opening book parser rejects illegal prefixes", "[match_runner]") {
  REQUIRE_THROWS_AS(poe2::match_runner::parse_opening_book_text("inline", "a1 a1\n"),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(poe2::match_runner::parse_opening_book_text("inline", "z9\n"),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(poe2::match_runner::parse_opening_book_text("inline", "# empty\n"),
                    std::invalid_argument);
}

TEST_CASE("series statistics use complete side-swapped pairs as samples", "[match_runner]") {
  poe2::match_runner::SeriesResult result;
  add_pair(result, TestOutcome::kEngineOneWin, TestOutcome::kEngineOneWin);
  add_pair(result, TestOutcome::kEngineOneWin, TestOutcome::kEngineTwoWin);
  add_pair(result, TestOutcome::kEngineTwoWin, TestOutcome::kEngineTwoWin);
  add_pair(result, TestOutcome::kEngineOneWin, TestOutcome::kNoWinner);
  add_pair(result, TestOutcome::kNoWinner, TestOutcome::kEngineTwoWin);
  result.games.push_back(make_game(11, poe2::Player::kOne, TestOutcome::kEngineOneWin));
  result.games_played = static_cast<int>(result.games.size());

  poe2::match_runner::SeriesOptions options;
  options.alternate_sides = true;
  poe2::match_runner::analyze_series_result(result, options);

  REQUIRE(result.statistical_unit == poe2::match_runner::StatisticalUnit::kOpeningPair);
  REQUIRE(result.statistical_samples == 5);
  REQUIRE(result.statistical_games == 10);
  REQUIRE(result.statistical_score_counts[0] == 1);
  REQUIRE(result.statistical_score_counts[1] == 1);
  REQUIRE(result.statistical_score_counts[2] == 1);
  REQUIRE(result.statistical_score_counts[3] == 1);
  REQUIRE(result.statistical_score_counts[4] == 1);
  REQUIRE(result.engine_one_result_rate == Catch::Approx(0.5));
  REQUIRE(result.confidence_low < 0.5);
  REQUIRE(result.confidence_high > 0.5);
  REQUIRE(result.sequential_decision == poe2::match_runner::SequentialDecision::kContinue);
}

TEST_CASE("paired sequential test accepts decisive alternative and null results",
          "[match_runner]") {
  poe2::match_runner::SeriesResult stronger;
  for (int pair = 0; pair < 17; ++pair) {
    add_pair(stronger, TestOutcome::kEngineOneWin, TestOutcome::kEngineOneWin);
  }
  for (int pair = 0; pair < 14; ++pair) {
    add_pair(stronger, TestOutcome::kEngineOneWin, TestOutcome::kEngineTwoWin);
  }

  poe2::match_runner::SeriesOptions options;
  options.alternate_sides = true;
  poe2::match_runner::analyze_series_result(stronger, options);

  REQUIRE(stronger.statistical_samples == 31);
  REQUIRE(stronger.engine_one_result_rate == Catch::Approx(48.0 / 62.0));
  REQUIRE(stronger.statistical_score_counts[2] == 14);
  REQUIRE(stronger.statistical_score_counts[4] == 17);
  REQUIRE(stronger.confidence_low > 0.5);
  REQUIRE(stronger.sequential_alt_log_evidence > stronger.sequential_upper_bound);
  REQUIRE(stronger.sequential_decision ==
          poe2::match_runner::SequentialDecision::kAcceptAlternative);

  poe2::match_runner::SeriesResult weaker;
  for (int pair = 0; pair < 17; ++pair) {
    add_pair(weaker, TestOutcome::kEngineTwoWin, TestOutcome::kEngineTwoWin);
  }
  for (int pair = 0; pair < 14; ++pair) {
    add_pair(weaker, TestOutcome::kEngineOneWin, TestOutcome::kEngineTwoWin);
  }
  poe2::match_runner::analyze_series_result(weaker, options);

  REQUIRE(weaker.sequential_null_log_evidence > -weaker.sequential_lower_bound);
  REQUIRE(weaker.sequential_decision == poe2::match_runner::SequentialDecision::kAcceptNull);
}

TEST_CASE("fixed-side series retain game-level statistical samples", "[match_runner]") {
  poe2::match_runner::SeriesResult result;
  result.games.push_back(make_game(1, poe2::Player::kOne, TestOutcome::kEngineOneWin));
  result.games.push_back(make_game(2, poe2::Player::kOne, TestOutcome::kNoWinner));
  result.games.push_back(make_game(3, poe2::Player::kOne, TestOutcome::kEngineTwoWin));
  result.games_played = static_cast<int>(result.games.size());

  poe2::match_runner::SeriesOptions options;
  options.alternate_sides = false;
  poe2::match_runner::analyze_series_result(result, options);

  REQUIRE(result.statistical_unit == poe2::match_runner::StatisticalUnit::kGame);
  REQUIRE(result.statistical_samples == 3);
  REQUIRE(result.statistical_games == 3);
  REQUIRE(result.statistical_score_counts[0] == 1);
  REQUIRE(result.statistical_score_counts[2] == 1);
  REQUIRE(result.statistical_score_counts[4] == 1);
  REQUIRE(result.engine_one_result_rate == Catch::Approx(0.5));
}
