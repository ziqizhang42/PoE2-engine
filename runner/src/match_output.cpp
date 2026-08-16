#include <iomanip>
#include <ostream>
#include <string_view>

#include "poe2/match_runner.hpp"

namespace poe2::match_runner {

namespace {

[[nodiscard]] double percentage(int count, int total) noexcept {
  if (total <= 0) {
    return 0.0;
  }

  return static_cast<double>(count) * 100.0 / static_cast<double>(total);
}

[[nodiscard]] double average(long long total, int count) noexcept {
  if (count <= 0) {
    return 0.0;
  }

  return static_cast<double>(total) / static_cast<double>(count);
}

[[nodiscard]] double percent_from_rate(double rate) noexcept { return rate * 100.0; }

}  // namespace

std::string_view player_name(Player player) noexcept {
  return player == Player::kOne ? "p1" : "p2";
}

std::string_view reason_name(MatchEndReason reason) noexcept {
  switch (reason) {
    case MatchEndReason::kNormal:
      return "normal";
    case MatchEndReason::kTimeout:
      return "timeout";
    case MatchEndReason::kDisconnected:
      return "disconnected";
    case MatchEndReason::kMalformedMove:
      return "malformed_move";
    case MatchEndReason::kIllegalMove:
      return "illegal_move";
    case MatchEndReason::kProtocolError:
      return "protocol_error";
  }

  return "unknown";
}

std::string_view sequential_decision_name(SequentialDecision decision) noexcept {
  switch (decision) {
    case SequentialDecision::kContinue:
      return "continue";
    case SequentialDecision::kAcceptNull:
      return "accept_null";
    case SequentialDecision::kAcceptAlternative:
      return "accept_alt";
    case SequentialDecision::kInvalid:
      return "invalid";
  }

  return "unknown";
}

std::string_view statistical_unit_name(StatisticalUnit unit) noexcept {
  switch (unit) {
    case StatisticalUnit::kGame:
      return "game";
    case StatisticalUnit::kOpeningPair:
      return "opening_pair";
  }

  return "unknown";
}

std::string_view confidence_method_name(StatisticalUnit unit) noexcept {
  return unit == StatisticalUnit::kOpeningPair ? "paired_betting_confidence_sequence"
                                               : "game_betting_confidence_sequence";
}

std::string_view sequential_test_method_name(StatisticalUnit unit) noexcept {
  return unit == StatisticalUnit::kOpeningPair ? "paired_normalized_elo_gsprt"
                                               : "game_normalized_elo_gsprt";
}

std::string_view engine_name_from_winner(const SeriesGameResult& game) noexcept {
  if (!game.match.winner.has_value()) {
    return "none";
  }

  return *game.match.winner == game.engine_one_player ? "engine_one" : "engine_two";
}

void print_state(const Position& position, std::ostream& output) {
  const ScoreByPlayer scores = position.scores();
  output << "state"
         << " ply=" << position.ply() << " side=" << player_name(position.side_to_move())
         << " p1=" << scores.player_one << " p2=" << scores.player_two
         << " empty=" << position.board().empty_count() << '\n';
}

void print_final(const GameResult& result, std::ostream& output) {
  output << "final"
         << " p1=" << result.scores.player_one << " p2=" << result.scores.player_two
         << " winner=" << player_name(result.winner) << '\n';
}

void print_match_result(const MatchResult& result, std::ostream& output) {
  output << "result"
         << " reason=" << reason_name(result.reason) << " winner="
         << (result.winner.has_value() ? player_name(*result.winner) : std::string_view{"none"})
         << " p1=" << result.scores.player_one << " p2=" << result.scores.player_two
         << " plies=" << result.moves.size() << '\n';
  if (!result.detail.empty()) {
    output << "detail " << result.detail << '\n';
  }
}

void print_series_result(const SeriesResult& result, std::ostream& output) {
  output << "series_result"
         << " games_requested=" << result.games_requested << " games_played=" << result.games_played
         << " valid=" << (result.valid ? 1 : 0) << " engine_one_wins=" << result.engine_one_wins
         << " engine_two_wins=" << result.engine_two_wins << " no_winner=" << result.no_winner
         << " engine_one_win_pct=" << std::fixed << std::setprecision(1)
         << percentage(result.engine_one_wins, result.games_played)
         << " engine_two_win_pct=" << percentage(result.engine_two_wins, result.games_played)
         << " avg_plies=" << average(result.plies_total, result.games_played)
         << " engine_one_avg_score=" << average(result.engine_one_score_total, result.games_played)
         << " engine_two_avg_score=" << average(result.engine_two_score_total, result.games_played)
         << " unique_openings=" << result.unique_openings_used << '\n';
  output << "series_reasons"
         << " normal=" << result.normal_games << " timeout=" << result.timeout_games
         << " disconnected=" << result.disconnected_games
         << " malformed_move=" << result.malformed_move_games
         << " illegal_move=" << result.illegal_move_games
         << " protocol_error=" << result.protocol_error_games << '\n';
  output << "series_confidence"
         << " confidence_pct=" << std::fixed << std::setprecision(1)
         << percent_from_rate(result.confidence_level)
         << " method=" << confidence_method_name(result.statistical_unit)
         << " unit=" << statistical_unit_name(result.statistical_unit)
         << " samples=" << result.statistical_samples << " games=" << result.statistical_games
         << " unpaired_games=" << result.games_played - result.statistical_games
         << " engine_one_score_pct=" << percent_from_rate(result.engine_one_result_rate)
         << " low_pct=" << percent_from_rate(result.confidence_low)
         << " high_pct=" << percent_from_rate(result.confidence_high) << '\n';
  if (result.statistical_unit == StatisticalUnit::kOpeningPair) {
    output << "series_pairs"
           << " score_rate_0=" << result.statistical_score_counts[0]
           << " score_rate_0_25=" << result.statistical_score_counts[1]
           << " score_rate_0_5=" << result.statistical_score_counts[2]
           << " score_rate_0_75=" << result.statistical_score_counts[3]
           << " score_rate_1=" << result.statistical_score_counts[4] << '\n';
  }
  output << "series_betting"
         << " method=" << confidence_method_name(result.statistical_unit)
         << " above_even_log_e=" << std::setprecision(3) << result.betting_log_evidence_above_even
         << " below_even_log_e=" << result.betting_log_evidence_below_even << '\n';
  output << "series_sequential"
         << " method=" << sequential_test_method_name(result.statistical_unit)
         << " bound_unit=normalized_elo"
         << " null_nelo=" << std::fixed << std::setprecision(2) << result.sequential_null_nelo
         << " alt_nelo=" << result.sequential_alt_nelo
         << " alpha_pct=" << percent_from_rate(result.sequential_alpha)
         << " beta_pct=" << percent_from_rate(result.sequential_beta) << " estimate_nelo=";
  if (result.normalized_elo.has_value()) {
    output << *result.normalized_elo;
  } else {
    output << "null";
  }
  output << " llr=" << std::setprecision(3) << result.sequential_llr
         << " lower=" << result.sequential_lower_bound << " upper=" << result.sequential_upper_bound
         << " decision=" << sequential_decision_name(result.sequential_decision) << '\n';
  if (!result.invalid_reason.empty()) {
    output << "invalid_reason " << result.invalid_reason << '\n';
  }
  if (!result.detail.empty()) {
    output << "detail " << result.detail << '\n';
  }
}

}  // namespace poe2::match_runner
