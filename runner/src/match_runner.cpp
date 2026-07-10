#include "poe2/match_runner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <optional>
#include <ostream>
#include <random>
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

[[nodiscard]] double engine_one_result_score(const SeriesResult& result) noexcept {
  return static_cast<double>(result.engine_one_wins) + (0.5 * result.no_winner);
}

[[nodiscard]] SprtDecision sprt_decision(double log_likelihood_ratio, double lower_bound,
                                         double upper_bound) noexcept {
  if (log_likelihood_ratio >= upper_bound) {
    return SprtDecision::kAcceptAlternative;
  }
  if (log_likelihood_ratio <= lower_bound) {
    return SprtDecision::kAcceptNull;
  }

  return SprtDecision::kContinue;
}

void update_confidence_interval(SeriesResult& result) noexcept {
  const int samples = result.statistical_samples;
  if (samples <= 0) {
    result.confidence_low = 0.0;
    result.confidence_high = 0.0;
    return;
  }

  constexpr double kWilsonZ95 = 1.959963984540054;
  const double sample_count = static_cast<double>(samples);
  const double z_squared = kWilsonZ95 * kWilsonZ95;
  const double denominator = 1.0 + (z_squared / sample_count);
  const double center =
      (result.engine_one_result_rate + (z_squared / (2.0 * sample_count))) / denominator;
  const double margin =
      (kWilsonZ95 *
       std::sqrt((result.engine_one_result_rate * (1.0 - result.engine_one_result_rate) +
                  (z_squared / (4.0 * sample_count))) /
                 sample_count)) /
      denominator;

  result.confidence_low = std::clamp(center - margin, 0.0, 1.0);
  result.confidence_high = std::clamp(center + margin, 0.0, 1.0);
}

void update_sprt(SeriesResult& result) noexcept {
  const int samples = result.statistical_samples;
  if (samples <= 0 || result.sprt_null_rate <= 0.0 || result.sprt_null_rate >= 1.0 ||
      result.sprt_alt_rate <= 0.0 || result.sprt_alt_rate >= 1.0 ||
      result.sprt_alt_rate <= result.sprt_null_rate || result.sprt_alpha <= 0.0 ||
      result.sprt_alpha >= 1.0 || result.sprt_beta <= 0.0 || result.sprt_beta >= 1.0) {
    result.sprt_decision = SprtDecision::kInvalid;
    return;
  }

  const double losses = static_cast<double>(samples) - result.engine_one_result_score;
  result.sprt_log_likelihood_ratio =
      (result.engine_one_result_score * std::log(result.sprt_alt_rate / result.sprt_null_rate)) +
      (losses * std::log((1.0 - result.sprt_alt_rate) / (1.0 - result.sprt_null_rate)));
  result.sprt_lower_bound = std::log(result.sprt_beta / (1.0 - result.sprt_alpha));
  result.sprt_upper_bound = std::log((1.0 - result.sprt_beta) / result.sprt_alpha);
  result.sprt_decision = sprt_decision(result.sprt_log_likelihood_ratio, result.sprt_lower_bound,
                                       result.sprt_upper_bound);
}

void update_series_statistics(SeriesResult& result, const SeriesOptions& options) noexcept {
  result.statistical_samples = result.games_played;
  result.engine_one_result_score = engine_one_result_score(result);
  result.engine_one_result_rate =
      result.statistical_samples > 0
          ? result.engine_one_result_score / static_cast<double>(result.statistical_samples)
          : 0.0;
  result.confidence_level = kDefaultConfidenceLevel;
  result.sprt_null_rate = options.sprt_null_rate;
  result.sprt_alt_rate = options.sprt_alt_rate;
  result.sprt_alpha = options.sprt_alpha;
  result.sprt_beta = options.sprt_beta;

  update_confidence_interval(result);
  update_sprt(result);
}

[[nodiscard]] bool is_decisive_sprt(SprtDecision decision) noexcept {
  return decision == SprtDecision::kAcceptAlternative || decision == SprtDecision::kAcceptNull;
}

[[nodiscard]] bool is_side_pair_complete(int zero_based_game_index, bool alternate_sides) noexcept {
  return !alternate_sides || (zero_based_game_index % 2) != 0;
}

void print_series_game(const SeriesGameResult& game, std::ostream& output) {
  output << "game"
         << " index=" << game.game_number
         << " engine_one_as=" << player_name(game.engine_one_player)
         << " opening_line=" << game.opening_line_number
         << " reason=" << reason_name(game.match.reason)
         << " winner=" << engine_name_from_winner(game) << " p1=" << game.match.scores.player_one
         << " p2=" << game.match.scores.player_two << " plies=" << game.match.moves.size() << '\n';
  if (!game.match.detail.empty()) {
    output << "detail game=" << game.game_number << ' ' << game.match.detail << '\n';
  }
}

}  // namespace

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
  };
  OpeningBook opening_book = options.opening_book;
  if (options.shuffle_openings) {
    std::mt19937_64 generator{std::random_device{}()};
    std::shuffle(opening_book.lines.begin(), opening_book.lines.end(), generator);
  }

  EngineProcess engine_one{options.engine_one_command};
  EngineProcess engine_two{options.engine_two_command};
  engine_one.start();
  engine_two.start();

  std::string detail;
  if (!wait_for_ready(engine_one, options.move_timeout, detail)) {
    result.detail = "engine_one startup failed: " + detail;
    ++result.protocol_error_games;
    update_series_statistics(result, options);
    return result;
  }
  if (!wait_for_ready(engine_two, options.move_timeout, detail)) {
    result.detail = "engine_two startup failed: " + detail;
    ++result.protocol_error_games;
    update_series_statistics(result, options);
    return result;
  }

  for (int game_index = 0; game_index < options.games; ++game_index) {
    const Player engine_one_player =
        engine_one_player_for_game(game_index, options.alternate_sides);
    EngineProcess& player_one = engine_one_player == Player::kOne ? engine_one : engine_two;
    EngineProcess& player_two = engine_one_player == Player::kOne ? engine_two : engine_one;

    MatchOptions match_options{
        .move_timeout = options.move_timeout,
        .go_limits = options.go_limits,
        .verbose = options.verbose_games,
    };
    const bool has_openings = !opening_book.lines.empty();
    const int opening_index = has_openings
                                  ? (options.alternate_sides ? (game_index / 2) : game_index) %
                                        static_cast<int>(opening_book.lines.size())
                                  : -1;
    const OpeningLine* opening = has_openings ? &opening_book.lines[opening_index] : nullptr;
    if (opening != nullptr) {
      match_options.opening_moves = opening->moves;
    }

    MatchResult match = play_ready_match(player_one, player_two, match_options, output);
    const MatchEndReason reason = match.reason;

    SeriesGameResult game{
        .game_number = game_index + 1,
        .engine_one_player = engine_one_player,
        .opening_line_number = opening != nullptr ? opening->line_number : 0,
        .opening_moves = opening != nullptr ? opening->text : std::string{},
        .match = std::move(match),
    };
    if (options.print_game_results) {
      print_series_game(game, output);
    }
    update_series_result(result, std::move(game));

    if (!can_continue_series_after(reason)) {
      result.detail = "series stopped after unrecoverable game result: ";
      result.detail += reason_name(reason);
      break;
    }

    if (options.sprt_stop && is_side_pair_complete(game_index, options.alternate_sides)) {
      update_series_statistics(result, options);
      if (is_decisive_sprt(result.sprt_decision)) {
        result.detail = "series stopped after sprt decision: ";
        result.detail += sprt_decision_name(result.sprt_decision);
        break;
      }
    }
  }

  update_series_statistics(result, options);
  return result;
}

}  // namespace poe2::match_runner
