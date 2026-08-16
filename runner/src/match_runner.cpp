#include "poe2/match_runner.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <future>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <ostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "engine_process.hpp"
#include "poe2/engine_stdio.hpp"

namespace poe2::match_runner {

namespace {

using Clock = std::chrono::steady_clock;
using detail::EngineProcess;
using detail::LineReadResult;
using detail::LineReadStatus;

[[nodiscard]] Player other_player(Player player) noexcept {
  return player == Player::kOne ? Player::kTwo : Player::kOne;
}

[[nodiscard]] std::optional<std::string> apply_opening_moves(
    Position& position, std::vector<std::string>& played_moves,
    const std::vector<std::string>& opening_moves) {
  for (const std::string& move_text : opening_moves) {
    const std::optional<Move> parsed = parse_move(move_text);
    if (!parsed.has_value()) {
      return "malformed opening move: " + move_text;
    }

    const std::string formatted = format_move(*parsed);
    const MoveResult result = apply_move(position, *parsed);
    if (!result.accepted) {
      const std::string_view error =
          result.error.has_value() ? move_error_name(*result.error) : "unknown";
      return "illegal opening move " + formatted + ": " + std::string{error};
    }
    if (result.game_result.has_value()) {
      return "opening line reaches a terminal position";
    }

    played_moves.push_back(formatted);
  }

  return std::nullopt;
}

struct BestMoveResult {
  MatchEndReason reason = MatchEndReason::kProtocolError;
  std::optional<std::string> move_text;
  std::string detail;
};

[[nodiscard]] bool write_engine_command(EngineProcess& engine, std::string_view command,
                                        std::string& detail) {
  if (engine.write_line(command)) {
    return true;
  }

  detail = "failed to write command: ";
  detail += command;
  return false;
}

[[nodiscard]] bool wait_for_ready(EngineProcess& engine, std::chrono::milliseconds timeout,
                                  std::string& detail) {
  if (!write_engine_command(engine, engine_stdio::kCommandPoe2, detail)) {
    return false;
  }
  if (!write_engine_command(engine, engine_stdio::kCommandIsReady, detail)) {
    return false;
  }

  const Clock::time_point deadline = Clock::now() + timeout;
  while (Clock::now() < deadline) {
    const auto remaining =
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now());
    const LineReadResult line = engine.read_line(remaining);
    if (line.status == LineReadStatus::kTimeout) {
      detail = "engine did not reply readyok";
      return false;
    }
    if (line.status == LineReadStatus::kClosed) {
      detail = "engine closed before readyok";
      return false;
    }
    if (line.line == engine_stdio::kResponseReadyOk) {
      return true;
    }
  }

  detail = "engine did not reply readyok";
  return false;
}

[[nodiscard]] BestMoveResult read_bestmove(EngineProcess& engine,
                                           std::chrono::milliseconds timeout) {
  const Clock::time_point deadline = Clock::now() + timeout;
  while (Clock::now() < deadline) {
    const auto remaining =
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now());
    const LineReadResult line = engine.read_line(remaining);
    if (line.status == LineReadStatus::kTimeout) {
      return BestMoveResult{
          .reason = MatchEndReason::kTimeout,
          .detail = "engine did not reply bestmove",
      };
    }
    if (line.status == LineReadStatus::kClosed) {
      return BestMoveResult{
          .reason = MatchEndReason::kDisconnected,
          .detail = "engine closed before bestmove",
      };
    }

    if (std::optional<std::string> move = engine_stdio::parse_bestmove_text(line.line);
        move.has_value()) {
      return BestMoveResult{
          .reason = MatchEndReason::kNormal,
          .move_text = std::move(move),
      };
    }
  }

  return BestMoveResult{
      .reason = MatchEndReason::kTimeout,
      .detail = "engine did not reply bestmove",
  };
}

[[nodiscard]] MatchResult failed_match(const Position& position,
                                       const std::vector<std::string>& moves, MatchEndReason reason,
                                       Player failed_player, std::string detail) {
  return MatchResult{
      .reason = reason,
      .winner = other_player(failed_player),
      .scores = position.scores(),
      .moves = moves,
      .detail = std::move(detail),
  };
}

[[nodiscard]] MatchResult play_ready_match(EngineProcess& player_one, EngineProcess& player_two,
                                           const MatchOptions& options, std::ostream& output) {
  Position position;
  std::vector<std::string> moves;
  std::string detail;
  if (!write_engine_command(player_one, engine_stdio::kCommandNewGame, detail)) {
    return failed_match(position, moves, MatchEndReason::kDisconnected, Player::kOne,
                        std::move(detail));
  }
  if (!write_engine_command(player_two, engine_stdio::kCommandNewGame, detail)) {
    return failed_match(position, moves, MatchEndReason::kDisconnected, Player::kTwo,
                        std::move(detail));
  }

  if (std::optional<std::string> opening_error =
          apply_opening_moves(position, moves, options.opening_moves);
      opening_error.has_value()) {
    return MatchResult{
        .reason = MatchEndReason::kProtocolError,
        .scores = position.scores(),
        .moves = std::move(moves),
        .detail = std::move(*opening_error),
    };
  }

  if (options.verbose) {
    print_state(position, output);
  }

  while (true) {
    const Player side_to_move = position.side_to_move();
    EngineProcess& engine = side_to_move == Player::kOne ? player_one : player_two;
    const std::string go_command = engine_stdio::format_go_command(options.go_limits);

    if (!write_engine_command(engine, engine_stdio::format_position_command(moves), detail)) {
      return failed_match(position, moves, MatchEndReason::kDisconnected, side_to_move,
                          std::move(detail));
    }
    if (!write_engine_command(engine, go_command, detail)) {
      return failed_match(position, moves, MatchEndReason::kDisconnected, side_to_move,
                          std::move(detail));
    }

    const int ply = position.ply();
    const BestMoveResult best_move = read_bestmove(engine, options.move_timeout);
    if (!best_move.move_text.has_value()) {
      return failed_match(position, moves, best_move.reason, side_to_move, best_move.detail);
    }

    const std::optional<Move> parsed_move = parse_move(*best_move.move_text);
    if (!parsed_move.has_value()) {
      return failed_match(position, moves, MatchEndReason::kMalformedMove, side_to_move,
                          "bestmove was not a valid coordinate: " + *best_move.move_text);
    }

    const std::string formatted_move = format_move(*parsed_move);
    const MoveResult move_result = apply_move(position, *parsed_move);
    if (!move_result.accepted) {
      const std::string_view error =
          move_result.error.has_value() ? move_error_name(*move_result.error) : "unknown";
      return failed_match(position, moves, MatchEndReason::kIllegalMove, side_to_move,
                          "illegal bestmove " + formatted_move + " " + std::string{error});
    }

    moves.push_back(formatted_move);
    if (options.verbose) {
      output << "move"
             << " ply=" << ply << " side=" << player_name(side_to_move)
             << " move=" << formatted_move << '\n';
      print_state(position, output);
    }

    if (move_result.game_result.has_value()) {
      if (options.verbose) {
        print_final(*move_result.game_result, output);
      }
      return MatchResult{
          .reason = MatchEndReason::kNormal,
          .winner = move_result.game_result->winner,
          .scores = move_result.game_result->scores,
          .moves = std::move(moves),
      };
    }
  }
}

[[nodiscard]] MatchResult play_match(EngineProcess& player_one, EngineProcess& player_two,
                                     const MatchOptions& options, std::ostream& output) {
  std::string detail;
  if (!wait_for_ready(player_one, options.move_timeout, detail)) {
    return MatchResult{
        .reason = MatchEndReason::kProtocolError,
        .winner = Player::kTwo,
        .detail = std::move(detail),
    };
  }
  if (!wait_for_ready(player_two, options.move_timeout, detail)) {
    return MatchResult{
        .reason = MatchEndReason::kProtocolError,
        .winner = Player::kOne,
        .detail = std::move(detail),
    };
  }

  return play_ready_match(player_one, player_two, options, output);
}

[[nodiscard]] bool can_continue_series_after(MatchEndReason reason) noexcept {
  return reason == MatchEndReason::kNormal || reason == MatchEndReason::kMalformedMove ||
         reason == MatchEndReason::kIllegalMove;
}

[[nodiscard]] Player engine_one_player_for_game(int zero_based_game_index,
                                                bool alternate_sides) noexcept {
  if (alternate_sides && (zero_based_game_index % 2) != 0) {
    return Player::kTwo;
  }

  return Player::kOne;
}

void count_reason(SeriesResult& result, MatchEndReason reason) noexcept {
  switch (reason) {
    case MatchEndReason::kNormal:
      ++result.normal_games;
      return;
    case MatchEndReason::kTimeout:
      ++result.timeout_games;
      return;
    case MatchEndReason::kDisconnected:
      ++result.disconnected_games;
      return;
    case MatchEndReason::kMalformedMove:
      ++result.malformed_move_games;
      return;
    case MatchEndReason::kIllegalMove:
      ++result.illegal_move_games;
      return;
    case MatchEndReason::kProtocolError:
      ++result.protocol_error_games;
      return;
  }
}

[[nodiscard]] Score score_for_player(ScoreByPlayer scores, Player player) noexcept {
  return player == Player::kOne ? scores.player_one : scores.player_two;
}

void update_series_result(SeriesResult& result, SeriesGameResult game) {
  ++result.games_played;
  count_reason(result, game.match.reason);

  const Player engine_two_player = other_player(game.engine_one_player);
  result.engine_one_score_total += score_for_player(game.match.scores, game.engine_one_player);
  result.engine_two_score_total += score_for_player(game.match.scores, engine_two_player);
  result.plies_total += static_cast<long long>(game.match.moves.size());

  if (!game.match.winner.has_value()) {
    ++result.no_winner;
  } else if (*game.match.winner == game.engine_one_player) {
    ++result.engine_one_wins;
  } else {
    ++result.engine_two_wins;
  }

  result.games.push_back(std::move(game));
}

constexpr int kStatisticalScoreBinCount = 5;
constexpr int kBettingMixtureSize = 64;
constexpr int kConfidenceBisectionSteps = 60;
constexpr double kMinimumProbability = 1.0e-12;
static_assert(std::tuple_size_v<decltype(SeriesResult::statistical_score_counts)> ==
              kStatisticalScoreBinCount);

enum class EvidenceDirection : std::uint8_t {
  kAbove,
  kBelow,
};

[[nodiscard]] int engine_one_game_half_points(const SeriesGameResult& game) noexcept {
  if (!game.match.winner.has_value()) {
    return 1;
  }
  return *game.match.winner == game.engine_one_player ? 2 : 0;
}

[[nodiscard]] constexpr double score_rate_for_bin(int bin) noexcept {
  return static_cast<double>(bin) / static_cast<double>(kStatisticalScoreBinCount - 1);
}

void collect_statistical_samples(SeriesResult& result, const SeriesOptions& options) noexcept {
  result.statistical_score_counts.fill(0);
  result.statistical_samples = 0;
  result.statistical_games = 0;
  result.statistical_unit =
      options.alternate_sides ? StatisticalUnit::kOpeningPair : StatisticalUnit::kGame;

  if (options.alternate_sides) {
    const std::size_t complete_pairs = result.games.size() / 2;
    for (std::size_t pair = 0; pair < complete_pairs; ++pair) {
      const std::size_t first = pair * 2;
      if (result.games[first].match.reason != MatchEndReason::kNormal ||
          result.games[first + 1].match.reason != MatchEndReason::kNormal) {
        continue;
      }
      const int pair_half_points = engine_one_game_half_points(result.games[first]) +
                                   engine_one_game_half_points(result.games[first + 1]);
      ++result.statistical_score_counts[static_cast<std::size_t>(pair_half_points)];
      ++result.statistical_samples;
    }
    result.statistical_games = result.statistical_samples * 2;
  } else {
    for (const SeriesGameResult& game : result.games) {
      if (game.match.reason != MatchEndReason::kNormal) {
        continue;
      }
      const int score_bin = engine_one_game_half_points(game) * 2;
      ++result.statistical_score_counts[static_cast<std::size_t>(score_bin)];
      ++result.statistical_samples;
    }
    result.statistical_games = result.statistical_samples;
  }

  double normalized_score_total = 0.0;
  for (int bin = 0; bin < kStatisticalScoreBinCount; ++bin) {
    normalized_score_total +=
        static_cast<double>(result.statistical_score_counts[static_cast<std::size_t>(bin)]) *
        score_rate_for_bin(bin);
  }
  result.engine_one_result_rate =
      result.statistical_samples > 0
          ? normalized_score_total / static_cast<double>(result.statistical_samples)
          : 0.0;
  result.engine_one_result_score =
      result.engine_one_result_rate * static_cast<double>(result.statistical_games);
}

[[nodiscard]] double log_betting_evidence(
    const std::array<int, kStatisticalScoreBinCount>& score_counts, double mean_bound,
    EvidenceDirection direction) noexcept {
  // For each fixed betting fraction, 1 + lambda * (score - mean_bound) is non-negative
  // and has conditional expectation at most one when the score mean is at most the bound.
  // Products of those factors are e-processes, and the uniform mixture remains an e-process.
  // Reversing the centered score tests a lower mean bound. This permits optional stopping without
  // modeling the distribution of the five possible opening-pair scores.
  const double bounded_mean =
      std::clamp(mean_bound, kMinimumProbability, 1.0 - kMinimumProbability);
  std::array<double, kBettingMixtureSize> component_logs{};
  double maximum_log = -std::numeric_limits<double>::infinity();

  for (int component = 0; component < kBettingMixtureSize; ++component) {
    const double betting_fraction =
        (static_cast<double>(component) + 0.5) / static_cast<double>(kBettingMixtureSize);
    const double lambda = direction == EvidenceDirection::kAbove
                              ? betting_fraction / bounded_mean
                              : betting_fraction / (1.0 - bounded_mean);
    double component_log = 0.0;
    for (int bin = 0; bin < kStatisticalScoreBinCount; ++bin) {
      const int count = score_counts[static_cast<std::size_t>(bin)];
      if (count == 0) {
        continue;
      }
      const double score_rate = score_rate_for_bin(bin);
      const double centered_score = direction == EvidenceDirection::kAbove
                                        ? score_rate - bounded_mean
                                        : bounded_mean - score_rate;
      const double factor = 1.0 + (lambda * centered_score);
      component_log += static_cast<double>(count) * std::log(factor);
    }
    component_logs[static_cast<std::size_t>(component)] = component_log;
    maximum_log = std::max(maximum_log, component_log);
  }

  double scaled_sum = 0.0;
  for (const double component_log : component_logs) {
    scaled_sum += std::exp(component_log - maximum_log);
  }
  return maximum_log + std::log(scaled_sum / static_cast<double>(kBettingMixtureSize));
}

void update_confidence_sequence(SeriesResult& result) noexcept {
  if (result.statistical_samples <= 0 || result.confidence_level <= 0.0 ||
      result.confidence_level >= 1.0) {
    result.confidence_low = 0.0;
    result.confidence_high = 1.0;
    return;
  }

  // Invert the two one-sided e-processes. Splitting the error probability between the tails
  // produces an anytime-valid confidence sequence at every completed statistical unit.
  const double tail_probability = (1.0 - result.confidence_level) / 2.0;
  const double threshold = std::log(1.0 / tail_probability);

  double rejected_low = 0.0;
  double retained_low = result.engine_one_result_rate;
  if (log_betting_evidence(result.statistical_score_counts, rejected_low,
                           EvidenceDirection::kAbove) >= threshold) {
    for (int step = 0; step < kConfidenceBisectionSteps; ++step) {
      const double candidate = (rejected_low + retained_low) / 2.0;
      if (log_betting_evidence(result.statistical_score_counts, candidate,
                               EvidenceDirection::kAbove) >= threshold) {
        rejected_low = candidate;
      } else {
        retained_low = candidate;
      }
    }
  }
  result.confidence_low = retained_low;

  double retained_high = result.engine_one_result_rate;
  double rejected_high = 1.0;
  if (log_betting_evidence(result.statistical_score_counts, rejected_high,
                           EvidenceDirection::kBelow) >= threshold) {
    for (int step = 0; step < kConfidenceBisectionSteps; ++step) {
      const double candidate = (retained_high + rejected_high) / 2.0;
      if (log_betting_evidence(result.statistical_score_counts, candidate,
                               EvidenceDirection::kBelow) >= threshold) {
        rejected_high = candidate;
      } else {
        retained_high = candidate;
      }
    }
  }
  result.confidence_high = retained_high;
}

struct Distribution {
  std::vector<double> values;
  std::vector<double> probabilities;
  double count = 0.0;
};

struct DistributionStats {
  double mean = 0.0;
  double variance = 0.0;
};

inline constexpr double kResultRegularization = 1.0e-3;
// 800 / ln(10), matching the normalized-Elo scale used by Fishtest.
inline constexpr double kNormalizedEloPerT = 347.43558552260146;

[[nodiscard]] DistributionStats distribution_stats(const Distribution& distribution) noexcept {
  DistributionStats stats;
  for (std::size_t index = 0; index < distribution.values.size(); ++index) {
    stats.mean += distribution.values[index] * distribution.probabilities[index];
  }
  for (std::size_t index = 0; index < distribution.values.size(); ++index) {
    const double centered = distribution.values[index] - stats.mean;
    stats.variance += distribution.probabilities[index] * centered * centered;
  }
  return stats;
}

[[nodiscard]] Distribution result_distribution(const SeriesResult& result) {
  Distribution distribution;
  if (result.statistical_unit == StatisticalUnit::kOpeningPair) {
    distribution.values = {0.0, 0.25, 0.5, 0.75, 1.0};
    distribution.probabilities.reserve(kStatisticalScoreBinCount);
    for (const int count : result.statistical_score_counts) {
      const double regularized = count == 0 ? kResultRegularization : static_cast<double>(count);
      distribution.probabilities.push_back(regularized);
      distribution.count += regularized;
    }
  } else {
    distribution.values = {0.0, 0.5, 1.0};
    for (const int bin : {0, 2, 4}) {
      const int count = result.statistical_score_counts[static_cast<std::size_t>(bin)];
      const double regularized = count == 0 ? kResultRegularization : static_cast<double>(count);
      distribution.probabilities.push_back(regularized);
      distribution.count += regularized;
    }
  }

  for (double& probability : distribution.probabilities) {
    probability /= distribution.count;
  }
  return distribution;
}

[[nodiscard]] std::optional<double> secular_root(
    const std::vector<double>& centered_values, const std::vector<double>& probabilities) noexcept {
  const auto [minimum, maximum] =
      std::minmax_element(centered_values.begin(), centered_values.end());
  if (minimum == centered_values.end() || *minimum >= 0.0 || *maximum <= 0.0) {
    return std::nullopt;
  }

  double lower = std::nextafter(-1.0 / *maximum, std::numeric_limits<double>::infinity());
  double upper = std::nextafter(-1.0 / *minimum, -std::numeric_limits<double>::infinity());
  const auto value_at = [&](double value) noexcept {
    double sum = 0.0;
    for (std::size_t index = 0; index < centered_values.size(); ++index) {
      sum += probabilities[index] * centered_values[index] / (1.0 + value * centered_values[index]);
    }
    return sum;
  };

  if (!(value_at(lower) > 0.0) || !(value_at(upper) < 0.0)) {
    return std::nullopt;
  }
  for (int step = 0; step < 100; ++step) {
    const double middle = std::midpoint(lower, upper);
    if (value_at(middle) > 0.0) {
      lower = middle;
    } else {
      upper = middle;
    }
  }
  return std::midpoint(lower, upper);
}

[[nodiscard]] std::optional<Distribution> maximum_likelihood_for_t(const Distribution& empirical,
                                                                   double reference,
                                                                   double target_t) noexcept {
  // Iterative constrained multinomial MLE for a fixed t-value, as used by Fishtest's
  // normalized-Elo GSPRT. The secular equation enforces the constraint at each iteration.
  Distribution estimate{
      .values = empirical.values,
      .probabilities = std::vector<double>(empirical.values.size(),
                                           1.0 / static_cast<double>(empirical.values.size())),
      .count = empirical.count,
  };

  for (int iteration = 0; iteration < 100; ++iteration) {
    const DistributionStats stats = distribution_stats(estimate);
    if (!(stats.variance > 0.0) || !std::isfinite(stats.variance)) {
      return std::nullopt;
    }
    const double sigma = std::sqrt(stats.variance);
    std::vector<double> centered_values;
    centered_values.reserve(empirical.values.size());
    for (const double value : empirical.values) {
      const double standardized = (stats.mean - value) / sigma;
      centered_values.push_back(value - reference -
                                target_t * sigma * (1.0 + standardized * standardized) / 2.0);
    }

    const std::optional<double> root = secular_root(centered_values, empirical.probabilities);
    if (!root.has_value()) {
      return std::nullopt;
    }

    std::vector<double> next_probabilities;
    next_probabilities.reserve(empirical.probabilities.size());
    double probability_sum = 0.0;
    double maximum_change = 0.0;
    for (std::size_t index = 0; index < empirical.probabilities.size(); ++index) {
      const double probability =
          empirical.probabilities[index] / (1.0 + *root * centered_values[index]);
      if (!(probability > 0.0) || !std::isfinite(probability)) {
        return std::nullopt;
      }
      next_probabilities.push_back(probability);
      probability_sum += probability;
    }
    for (std::size_t index = 0; index < next_probabilities.size(); ++index) {
      next_probabilities[index] /= probability_sum;
      maximum_change = std::max(
          maximum_change, std::abs(next_probabilities[index] - estimate.probabilities[index]));
    }
    estimate.probabilities = std::move(next_probabilities);
    if (maximum_change < 1.0e-12) {
      return estimate;
    }
  }

  return estimate;
}

[[nodiscard]] std::optional<double> normalized_elo_llr(const SeriesResult& result) noexcept {
  const Distribution empirical = result_distribution(result);
  const double pair_scale =
      result.statistical_unit == StatisticalUnit::kOpeningPair ? std::sqrt(2.0) : 1.0;
  const double null_t = result.sequential_null_nelo * pair_scale / kNormalizedEloPerT;
  const double alternative_t = result.sequential_alt_nelo * pair_scale / kNormalizedEloPerT;
  const std::optional<Distribution> null_estimate =
      maximum_likelihood_for_t(empirical, 0.5, null_t);
  const std::optional<Distribution> alternative_estimate =
      maximum_likelihood_for_t(empirical, 0.5, alternative_t);
  if (!null_estimate.has_value() || !alternative_estimate.has_value()) {
    return std::nullopt;
  }

  double llr_per_sample = 0.0;
  for (std::size_t index = 0; index < empirical.probabilities.size(); ++index) {
    llr_per_sample +=
        empirical.probabilities[index] * (std::log(alternative_estimate->probabilities[index]) -
                                          std::log(null_estimate->probabilities[index]));
  }
  const double llr = empirical.count * llr_per_sample;
  return std::isfinite(llr) ? std::optional<double>{llr} : std::nullopt;
}

void update_normalized_elo(SeriesResult& result) noexcept {
  result.normalized_elo.reset();
  if (result.statistical_samples <= 0) {
    return;
  }

  const Distribution empirical = result_distribution(result);
  const DistributionStats stats = distribution_stats(empirical);
  const double sigma_per_game = result.statistical_unit == StatisticalUnit::kOpeningPair
                                    ? std::sqrt(2.0 * stats.variance)
                                    : std::sqrt(stats.variance);
  if (!(sigma_per_game > 0.0) || !std::isfinite(sigma_per_game)) {
    return;
  }
  const double normalized_elo = ((stats.mean - 0.5) / sigma_per_game) * kNormalizedEloPerT;
  if (std::isfinite(normalized_elo)) {
    result.normalized_elo = normalized_elo;
  }
}

void update_sequential_test(SeriesResult& result) noexcept {
  if (!std::isfinite(result.sequential_null_nelo) || !std::isfinite(result.sequential_alt_nelo) ||
      result.sequential_alt_nelo <= result.sequential_null_nelo || result.sequential_alpha <= 0.0 ||
      result.sequential_alpha >= 1.0 || result.sequential_beta <= 0.0 ||
      result.sequential_beta >= 1.0) {
    result.sequential_decision = SequentialDecision::kInvalid;
    return;
  }
  result.sequential_lower_bound =
      std::log(result.sequential_beta / (1.0 - result.sequential_alpha));
  result.sequential_upper_bound =
      std::log((1.0 - result.sequential_beta) / result.sequential_alpha);
  if (!result.valid) {
    result.sequential_decision = SequentialDecision::kInvalid;
    return;
  }
  if (result.statistical_samples <= 0) {
    result.sequential_decision = SequentialDecision::kContinue;
    return;
  }

  const std::optional<double> llr = normalized_elo_llr(result);
  if (!llr.has_value()) {
    result.sequential_decision = SequentialDecision::kInvalid;
    return;
  }
  result.sequential_llr = *llr;
  if (result.sequential_llr >= result.sequential_upper_bound) {
    result.sequential_decision = SequentialDecision::kAcceptAlternative;
  } else if (result.sequential_llr <= result.sequential_lower_bound) {
    result.sequential_decision = SequentialDecision::kAcceptNull;
  } else {
    result.sequential_decision = SequentialDecision::kContinue;
  }
}

[[nodiscard]] bool is_decisive_sequential(SequentialDecision decision) noexcept {
  return decision == SequentialDecision::kAcceptAlternative ||
         decision == SequentialDecision::kAcceptNull;
}

[[nodiscard]] bool is_side_pair_complete(int zero_based_game_index, bool alternate_sides) noexcept {
  return !alternate_sides || (zero_based_game_index % 2) != 0;
}

[[nodiscard]] std::uint64_t bounded_random(std::mt19937_64& generator,
                                           std::uint64_t bound) noexcept {
  const std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  const std::uint64_t limit = maximum - (maximum % bound);
  std::uint64_t value = 0;
  do {
    value = generator();
  } while (value >= limit);
  return value % bound;
}

void deterministic_shuffle(std::vector<OpeningLine>& lines, std::uint64_t seed) noexcept {
  std::mt19937_64 generator{seed};
  for (std::size_t remaining = lines.size(); remaining > 1; --remaining) {
    const std::size_t selected = static_cast<std::size_t>(bounded_random(generator, remaining));
    std::swap(lines[remaining - 1], lines[selected]);
  }
}

void print_series_game(const SeriesGameResult& game, std::ostream& output) {
  output << "game"
         << " index=" << game.game_number
         << " engine_one_as=" << player_name(game.engine_one_player)
         << " opening_slot=" << game.opening_slot << " opening_line=" << game.opening_line_number
         << " reason=" << reason_name(game.match.reason)
         << " winner=" << engine_name_from_winner(game) << " p1=" << game.match.scores.player_one
         << " p2=" << game.match.scores.player_two << " plies=" << game.match.moves.size() << '\n';
  if (!game.match.detail.empty()) {
    output << "detail game=" << game.game_number << ' ' << game.match.detail << '\n';
  }
}

struct SeriesWorker {
  SeriesWorker(std::string engine_one_command, std::string engine_two_command)
      : engine_one{std::move(engine_one_command)}, engine_two{std::move(engine_two_command)} {}

  EngineProcess engine_one;
  EngineProcess engine_two;
};

struct CompletedSeriesGame {
  SeriesGameResult game;
  std::string output;
};

struct SeriesWorkUnitResult {
  std::vector<CompletedSeriesGame> games;
};

struct ScheduledSeriesWork {
  int work_unit_index = -1;
  std::future<SeriesWorkUnitResult> future;
};

[[nodiscard]] CompletedSeriesGame play_series_game(SeriesWorker& worker,
                                                   const SeriesOptions& options,
                                                   const OpeningBook& opening_book,
                                                   int game_index) {
  const Player engine_one_player = engine_one_player_for_game(game_index, options.alternate_sides);
  EngineProcess& player_one =
      engine_one_player == Player::kOne ? worker.engine_one : worker.engine_two;
  EngineProcess& player_two =
      engine_one_player == Player::kOne ? worker.engine_two : worker.engine_one;

  MatchOptions match_options{
      .move_timeout = options.move_timeout,
      .go_limits = options.go_limits,
      .verbose = options.verbose_games,
  };
  const bool has_openings = !opening_book.lines.empty();
  const int opening_index =
      has_openings ? (options.alternate_sides ? (game_index / 2) : game_index) : -1;
  const OpeningLine* opening = has_openings ? &opening_book.lines[opening_index] : nullptr;
  if (opening != nullptr) {
    match_options.opening_moves = opening->moves;
  }

  std::ostringstream game_output;
  MatchResult match = play_ready_match(player_one, player_two, match_options, game_output);
  return CompletedSeriesGame{
      .game =
          SeriesGameResult{
              .game_number = game_index + 1,
              .engine_one_player = engine_one_player,
              .opening_slot = opening != nullptr ? opening_index + 1 : 0,
              .opening_line_number = opening != nullptr ? opening->line_number : 0,
              .opening_moves = opening != nullptr ? opening->text : std::string{},
              .match = std::move(match),
          },
      .output = std::move(game_output).str(),
  };
}

[[nodiscard]] SeriesWorkUnitResult play_series_work_unit(SeriesWorker& worker,
                                                         const SeriesOptions& options,
                                                         const OpeningBook& opening_book,
                                                         int first_game_index, int games_in_unit) {
  SeriesWorkUnitResult work;
  work.games.reserve(static_cast<std::size_t>(games_in_unit));
  for (int offset = 0; offset < games_in_unit; ++offset) {
    CompletedSeriesGame completed =
        play_series_game(worker, options, opening_book, first_game_index + offset);
    const MatchEndReason reason = completed.game.match.reason;
    work.games.push_back(std::move(completed));
    if ((options.require_normal_games && reason != MatchEndReason::kNormal) ||
        !can_continue_series_after(reason)) {
      break;
    }
  }
  return work;
}

[[nodiscard]] bool commit_series_game(SeriesResult& result, CompletedSeriesGame completed,
                                      const SeriesOptions& options, std::ostream& output) {
  output << completed.output;
  if (options.print_game_results) {
    print_series_game(completed.game, output);
  }

  const MatchEndReason reason = completed.game.match.reason;
  const int game_number = completed.game.game_number;
  const int opening_slot = completed.game.opening_slot;
  update_series_result(result, std::move(completed.game));
  result.unique_openings_used = std::max(result.unique_openings_used, opening_slot);

  if (options.require_normal_games && reason != MatchEndReason::kNormal) {
    result.valid = false;
    result.invalid_reason =
        "game " + std::to_string(game_number) + " ended with " + std::string{reason_name(reason)};
    if (!result.games.back().match.detail.empty()) {
      result.invalid_reason += ": ";
      result.invalid_reason += result.games.back().match.detail;
    }
    result.detail = "evaluation invalid: " + result.invalid_reason;
    return true;
  }

  if (!can_continue_series_after(reason)) {
    result.detail = "series stopped after unrecoverable game result: ";
    result.detail += reason_name(reason);
    return true;
  }

  return false;
}

}  // namespace

GsprtAnalysis analyze_normalized_elo_gsprt(const std::array<int, 5>& score_counts,
                                           StatisticalUnit unit, double null_nelo,
                                           double alternative_nelo, double alpha,
                                           double beta) noexcept {
  SeriesResult result;
  result.statistical_unit = unit;
  result.statistical_score_counts = score_counts;
  if (unit == StatisticalUnit::kOpeningPair) {
    for (const int count : score_counts) {
      result.statistical_samples += count;
    }
    result.statistical_games = result.statistical_samples * 2;
  } else {
    for (const int bin : {0, 2, 4}) {
      result.statistical_samples += score_counts[static_cast<std::size_t>(bin)];
    }
    result.statistical_games = result.statistical_samples;
  }
  result.sequential_null_nelo = null_nelo;
  result.sequential_alt_nelo = alternative_nelo;
  result.sequential_alpha = alpha;
  result.sequential_beta = beta;
  update_normalized_elo(result);
  update_sequential_test(result);
  return GsprtAnalysis{
      .normalized_elo = result.normalized_elo,
      .llr = result.sequential_llr,
      .lower_bound = result.sequential_lower_bound,
      .upper_bound = result.sequential_upper_bound,
      .decision = result.sequential_decision,
  };
}

void analyze_series_result(SeriesResult& result, const SeriesOptions& options) noexcept {
  result.confidence_level = kDefaultConfidenceLevel;
  result.sequential_null_nelo = options.sequential_null_nelo;
  result.sequential_alt_nelo = options.sequential_alt_nelo;
  result.sequential_alpha = options.sequential_alpha;
  result.sequential_beta = options.sequential_beta;
  result.betting_log_evidence_above_even = 0.0;
  result.betting_log_evidence_below_even = 0.0;
  result.sequential_llr = 0.0;
  result.sequential_lower_bound = 0.0;
  result.sequential_upper_bound = 0.0;
  result.sequential_decision = SequentialDecision::kContinue;

  collect_statistical_samples(result, options);
  update_confidence_sequence(result);
  result.betting_log_evidence_above_even =
      log_betting_evidence(result.statistical_score_counts, 0.5, EvidenceDirection::kAbove);
  result.betting_log_evidence_below_even =
      log_betting_evidence(result.statistical_score_counts, 0.5, EvidenceDirection::kBelow);
  update_normalized_elo(result);
  update_sequential_test(result);
}

MatchResult run_process_match(const MatchOptions& options, std::ostream& output) {
  std::signal(SIGPIPE, SIG_IGN);

  EngineProcess player_one{options.player_one_command};
  EngineProcess player_two{options.player_two_command};
  player_one.start();
  player_two.start();

  return play_match(player_one, player_two, options, output);
}

SeriesResult run_process_series(const SeriesOptions& options, std::ostream& output) {
  std::signal(SIGPIPE, SIG_IGN);

  SeriesResult result{
      .games_requested = options.games,
      .workers_requested = options.workers,
  };
  if (options.games <= 0) {
    throw std::invalid_argument{"series games must be positive"};
  }
  if (options.workers <= 0) {
    throw std::invalid_argument{"series workers must be positive"};
  }
  OpeningBook opening_book = options.opening_book;
  if (!opening_book.lines.empty()) {
    const int required_openings =
        options.alternate_sides ? options.games / 2 + options.games % 2 : options.games;
    if (required_openings > static_cast<int>(opening_book.lines.size())) {
      throw std::invalid_argument{"opening book has " + std::to_string(opening_book.lines.size()) +
                                  " lines but the series needs " +
                                  std::to_string(required_openings) + " unique openings"};
    }
  }
  if (options.shuffle_openings) {
    std::uint64_t seed = 0;
    if (options.opening_seed.has_value()) {
      seed = *options.opening_seed;
    } else {
      std::random_device random;
      seed = (static_cast<std::uint64_t>(random()) << 32U) ^ static_cast<std::uint64_t>(random());
    }
    deterministic_shuffle(opening_book.lines, seed);
  }

  const int games_per_work_unit = options.alternate_sides ? 2 : 1;
  const int work_unit_count =
      options.games / games_per_work_unit + (options.games % games_per_work_unit != 0 ? 1 : 0);
  const int worker_count = std::min(options.workers, work_unit_count);

  // Start every child before launching worker threads so EngineProcess::start never forks a
  // multithreaded parent.
  std::vector<std::unique_ptr<SeriesWorker>> workers;
  workers.reserve(static_cast<std::size_t>(worker_count));
  for (int worker_index = 0; worker_index < worker_count; ++worker_index) {
    auto worker =
        std::make_unique<SeriesWorker>(options.engine_one_command, options.engine_two_command);
    worker->engine_one.start();
    worker->engine_two.start();

    std::string detail;
    if (!wait_for_ready(worker->engine_one, options.move_timeout, detail)) {
      result.detail =
          "worker " + std::to_string(worker_index + 1) + " engine_one startup failed: " + detail;
      result.valid = false;
      result.invalid_reason = result.detail;
      ++result.protocol_error_games;
      analyze_series_result(result, options);
      return result;
    }
    if (!wait_for_ready(worker->engine_two, options.move_timeout, detail)) {
      result.detail =
          "worker " + std::to_string(worker_index + 1) + " engine_two startup failed: " + detail;
      result.valid = false;
      result.invalid_reason = result.detail;
      ++result.protocol_error_games;
      analyze_series_result(result, options);
      return result;
    }
    workers.push_back(std::move(worker));
  }
  result.workers_used = worker_count;

  // Keep at most one ordered work unit in flight per worker. A unit is a whole side-swapped pair
  // when sides alternate, so no worker can split a statistical sample across a cutoff.
  std::vector<ScheduledSeriesWork> scheduled(static_cast<std::size_t>(worker_count));
  const auto launch_work = [&](int worker_index, int work_unit_index) {
    ScheduledSeriesWork& slot = scheduled[static_cast<std::size_t>(worker_index)];
    const int first_game_index = work_unit_index * games_per_work_unit;
    const int games_in_unit = std::min(games_per_work_unit, options.games - first_game_index);
    SeriesWorker* const worker = workers[static_cast<std::size_t>(worker_index)].get();
    slot.work_unit_index = work_unit_index;
    slot.future = std::async(std::launch::async, [worker, &options, &opening_book, first_game_index,
                                                  games_in_unit]() {
      return play_series_work_unit(*worker, options, opening_book, first_game_index, games_in_unit);
    });
  };

  int next_work_unit = 0;
  for (int worker_index = 0; worker_index < worker_count; ++worker_index) {
    launch_work(worker_index, next_work_unit++);
  }

  bool stopped = false;
  for (int next_to_commit = 0; next_to_commit < work_unit_count && !stopped; ++next_to_commit) {
    // Waiting for the canonical next unit makes completion timing irrelevant to result order.
    const auto slot_position = std::find_if(
        scheduled.begin(), scheduled.end(),
        [next_to_commit](const auto& slot) { return slot.work_unit_index == next_to_commit; });
    if (slot_position == scheduled.end()) {
      throw std::logic_error{"scheduled series work unit was lost"};
    }

    SeriesWorkUnitResult work = slot_position->future.get();
    slot_position->work_unit_index = -1;
    for (CompletedSeriesGame& completed : work.games) {
      if (commit_series_game(result, std::move(completed), options, output)) {
        stopped = true;
        break;
      }
    }

    if (!stopped && options.sequential_stop && !result.games.empty() &&
        is_side_pair_complete(result.games.back().game_number - 1, options.alternate_sides)) {
      analyze_series_result(result, options);
      if (is_decisive_sequential(result.sequential_decision)) {
        result.detail = "series stopped after sequential decision: ";
        result.detail += sequential_decision_name(result.sequential_decision);
        stopped = true;
      }
    }

    if (!stopped && next_work_unit < work_unit_count) {
      const int worker_index = static_cast<int>(std::distance(scheduled.begin(), slot_position));
      launch_work(worker_index, next_work_unit++);
    }
  }

  if (stopped) {
    for (ScheduledSeriesWork& slot : scheduled) {
      if (slot.work_unit_index < 0) {
        continue;
      }
      try {
        const SeriesWorkUnitResult discarded = slot.future.get();
        result.games_discarded += static_cast<int>(discarded.games.size());
      } catch (const std::exception& error) {
        output << "discarded_work_error work_unit=" << slot.work_unit_index + 1
               << " detail=" << error.what() << '\n';
      } catch (...) {
        output << "discarded_work_error work_unit=" << slot.work_unit_index + 1
               << " detail=unknown\n";
      }
      slot.work_unit_index = -1;
    }
  }

  analyze_series_result(result, options);
  return result;
}

}  // namespace poe2::match_runner
