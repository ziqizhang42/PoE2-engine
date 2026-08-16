#include "poe2/match_runner.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>

#include "poe2/symmetry.hpp"

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
      .opening_slot = (game_number + 1) / 2,
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

[[nodiscard]] std::string canonical_key_text(const poe2::Position& position) {
  const poe2::CanonicalPositionKey canonical = poe2::canonicalize_position_key(position.key());
  return std::to_string(canonical.key.low) + ':' + std::to_string(canonical.key.high);
}

[[nodiscard]] std::uint64_t file_fnv1a64(const std::filesystem::path& path) {
  std::ifstream input{path, std::ios::binary};
  REQUIRE(input);
  std::uint64_t hash = 14695981039346656037ULL;
  char ch = 0;
  while (input.get(ch)) {
    hash ^= static_cast<unsigned char>(ch);
    hash *= 1099511628211ULL;
  }
  return hash;
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

TEST_CASE("opening book parser treats startpos as a normal opening", "[match_runner]") {
  const poe2::match_runner::OpeningBook book =
      poe2::match_runner::parse_opening_book_text("inline", "startpos\na1 b1\n");

  REQUIRE(book.lines.size() == 2);
  REQUIRE(book.lines[0].line_number == 1);
  REQUIRE(book.lines[0].moves.empty());
  REQUIRE(book.lines[0].text == "startpos");
}

TEST_CASE("opening book parser rejects illegal prefixes", "[match_runner]") {
  REQUIRE_THROWS_AS(poe2::match_runner::parse_opening_book_text("inline", "a1 a1\n"),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(poe2::match_runner::parse_opening_book_text("inline", "z9\n"),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(poe2::match_runner::parse_opening_book_text("inline", "# empty\n"),
                    std::invalid_argument);
}

TEST_CASE("committed opening corpus satisfies its generation invariants", "[match_runner]") {
  const std::filesystem::path opening_root =
      std::filesystem::path{POE2_SOURCE_DIR} / "eval" / "openings";
  const std::filesystem::path development_path = opening_root / "development.txt";
  const std::filesystem::path holdout_path = opening_root / "holdout.txt";
  const poe2::match_runner::OpeningBook development =
      poe2::match_runner::load_opening_book(development_path.string());
  const poe2::match_runner::OpeningBook holdout =
      poe2::match_runner::load_opening_book(holdout_path.string());

  REQUIRE(development.lines.size() == 10001);
  REQUIRE(holdout.lines.size() == 10001);
  REQUIRE(development.lines.front().text == "startpos");
  REQUIRE(holdout.lines.front().text == "startpos");
  REQUIRE(file_fnv1a64(development_path) == 0x553d3b06c145a713ULL);
  REQUIRE(file_fnv1a64(holdout_path) == 0xb91e0cd88d5a6275ULL);

  std::unordered_set<std::string> canonical_positions;
  std::map<int, int> development_depths;
  std::map<int, int> holdout_depths;
  const auto validate_partition = [&](const poe2::match_runner::OpeningBook& book,
                                      std::map<int, int>& depths) {
    for (const poe2::match_runner::OpeningLine& line : book.lines) {
      if (line.moves.empty()) {
        continue;
      }
      poe2::Position position;
      for (const std::string& move_text : line.moves) {
        const std::optional<poe2::Move> move = poe2::parse_move(move_text);
        REQUIRE(move.has_value());
        REQUIRE(poe2::apply_move(position, *move).accepted);
      }
      const poe2::ScoreByPlayer scores = position.scores();
      REQUIRE(std::abs(scores.player_one - scores.player_two) <= 4);
      REQUIRE(canonical_positions.insert(canonical_key_text(position)).second);
      ++depths[static_cast<int>(line.moves.size())];
    }
  };
  validate_partition(development, development_depths);
  validate_partition(holdout, holdout_depths);

  REQUIRE(canonical_positions.size() == 20000);
  REQUIRE(development_depths[2] + holdout_depths[2] == 315);
  REQUIRE(development_depths[4] + holdout_depths[4] == 3281);
  REQUIRE(development_depths[6] + holdout_depths[6] == 3281);
  REQUIRE(development_depths[8] + holdout_depths[8] == 3281);
  REQUIRE(development_depths[10] + holdout_depths[10] == 3281);
  REQUIRE(development_depths[12] + holdout_depths[12] == 3281);
  REQUIRE(development_depths[14] + holdout_depths[14] == 3280);
  for (const int depth : {2, 4, 6, 8, 10, 12, 14}) {
    REQUIRE(std::abs(development_depths[depth] - holdout_depths[depth]) <= 1);
  }
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
  for (int pair = 0; pair < 60; ++pair) {
    add_pair(stronger, TestOutcome::kEngineOneWin, TestOutcome::kEngineOneWin);
  }
  for (int pair = 0; pair < 20; ++pair) {
    add_pair(stronger, TestOutcome::kEngineOneWin, TestOutcome::kEngineTwoWin);
  }

  poe2::match_runner::SeriesOptions options;
  options.alternate_sides = true;
  poe2::match_runner::analyze_series_result(stronger, options);

  REQUIRE(stronger.statistical_samples == 80);
  REQUIRE(stronger.engine_one_result_rate == Catch::Approx(70.0 / 80.0));
  REQUIRE(stronger.statistical_score_counts[2] == 20);
  REQUIRE(stronger.statistical_score_counts[4] == 60);
  REQUIRE(stronger.confidence_low > 0.5);
  REQUIRE(stronger.sequential_llr > stronger.sequential_upper_bound);
  REQUIRE(stronger.sequential_decision ==
          poe2::match_runner::SequentialDecision::kAcceptAlternative);

  poe2::match_runner::SeriesResult weaker;
  for (int pair = 0; pair < 60; ++pair) {
    add_pair(weaker, TestOutcome::kEngineTwoWin, TestOutcome::kEngineTwoWin);
  }
  for (int pair = 0; pair < 20; ++pair) {
    add_pair(weaker, TestOutcome::kEngineOneWin, TestOutcome::kEngineTwoWin);
  }
  poe2::match_runner::analyze_series_result(weaker, options);

  REQUIRE(weaker.sequential_llr < weaker.sequential_lower_bound);
  REQUIRE(weaker.sequential_decision == poe2::match_runner::SequentialDecision::kAcceptNull);
}

TEST_CASE("paired normalized-Elo GSPRT matches a reference vector", "[match_runner]") {
  poe2::match_runner::SeriesResult result;
  for (int pair = 0; pair < 3; ++pair) {
    add_pair(result, TestOutcome::kEngineTwoWin, TestOutcome::kEngineTwoWin);
  }
  for (int pair = 0; pair < 20; ++pair) {
    add_pair(result, TestOutcome::kEngineOneWin, TestOutcome::kEngineTwoWin);
  }
  for (int pair = 0; pair < 7; ++pair) {
    add_pair(result, TestOutcome::kEngineOneWin, TestOutcome::kEngineOneWin);
  }

  poe2::match_runner::SeriesOptions options;
  poe2::match_runner::analyze_series_result(result, options);

  REQUIRE(result.sequential_llr == Catch::Approx(0.4673233262).epsilon(1.0e-8));
  REQUIRE(result.normalized_elo.has_value());
  REQUIRE(result.normalized_elo.value_or(std::numeric_limits<double>::quiet_NaN()) ==
          Catch::Approx(58.30870036).epsilon(1.0e-8));
  REQUIRE(result.sequential_lower_bound == Catch::Approx(-2.944438979));
  REQUIRE(result.sequential_upper_bound == Catch::Approx(2.944438979));
}

TEST_CASE("abnormal games are excluded from statistical pairs", "[match_runner]") {
  poe2::match_runner::SeriesResult result;
  add_pair(result, TestOutcome::kEngineOneWin, TestOutcome::kEngineOneWin);
  add_pair(result, TestOutcome::kEngineOneWin, TestOutcome::kEngineTwoWin);
  result.games[1].match.reason = poe2::match_runner::MatchEndReason::kIllegalMove;

  poe2::match_runner::SeriesOptions options;
  poe2::match_runner::analyze_series_result(result, options);

  REQUIRE(result.statistical_samples == 1);
  REQUIRE(result.statistical_games == 2);
  REQUIRE(result.statistical_score_counts[2] == 1);
  REQUIRE(result.statistical_score_counts[4] == 0);
}

TEST_CASE("paired GSPRT simulation respects its configured error risks", "[match_runner]") {
  constexpr int kTrials = 100;
  constexpr int kMaximumPairs = 4000;
  constexpr double kAlternativeLossProbability = 0.08185648911941386;
  constexpr double kSplitProbability = 0.8;
  std::mt19937_64 generator{0x504f453247535052ULL};  // NOLINT(bugprone-random-generator-seed)
  const auto unit_random = [&]() {
    return static_cast<double>(generator()) /
           static_cast<double>(std::numeric_limits<std::uint64_t>::max());
  };
  const auto simulate = [&](double loss_probability) {
    std::array<int, 5> counts{};
    for (int sample = 0; sample < kMaximumPairs; ++sample) {
      const double draw = unit_random();
      if (draw < loss_probability) {
        ++counts[0];
      } else if (draw < loss_probability + kSplitProbability) {
        ++counts[2];
      } else {
        ++counts[4];
      }
      const poe2::match_runner::GsprtAnalysis analysis =
          poe2::match_runner::analyze_normalized_elo_gsprt(
              counts, poe2::match_runner::StatisticalUnit::kOpeningPair, 0.0, 20.0, 0.05, 0.05);
      if (analysis.decision != poe2::match_runner::SequentialDecision::kContinue) {
        return analysis.decision;
      }
    }
    return poe2::match_runner::SequentialDecision::kContinue;
  };

  int null_false_positives = 0;
  int null_undecided = 0;
  int alternative_false_negatives = 0;
  int alternative_undecided = 0;
  for (int trial = 0; trial < kTrials; ++trial) {
    const poe2::match_runner::SequentialDecision null_decision = simulate(0.1);
    null_false_positives +=
        null_decision == poe2::match_runner::SequentialDecision::kAcceptAlternative ? 1 : 0;
    null_undecided += null_decision == poe2::match_runner::SequentialDecision::kContinue ? 1 : 0;

    const poe2::match_runner::SequentialDecision alternative_decision =
        simulate(kAlternativeLossProbability);
    alternative_false_negatives +=
        alternative_decision == poe2::match_runner::SequentialDecision::kAcceptNull ? 1 : 0;
    alternative_undecided +=
        alternative_decision == poe2::match_runner::SequentialDecision::kContinue ? 1 : 0;
  }

  REQUIRE(null_false_positives <= 12);
  REQUIRE(alternative_false_negatives <= 12);
  REQUIRE(null_undecided <= 10);
  REQUIRE(alternative_undecided <= 10);
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
