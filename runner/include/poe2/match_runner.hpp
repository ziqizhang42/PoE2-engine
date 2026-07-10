#ifndef POE2_MATCH_RUNNER_HPP
#define POE2_MATCH_RUNNER_HPP

#include <chrono>
#include <cstdint>
#include <iosfwd>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "poe2/engine.hpp"
#include "poe2/move.hpp"

namespace poe2::match_runner {

inline constexpr std::chrono::milliseconds kDefaultMoveTimeout{5000};
inline constexpr double kDefaultConfidenceLevel = 0.95;
inline constexpr double kDefaultSprtNullRate = 0.50;
inline constexpr double kDefaultSprtAltRate = 0.55;
inline constexpr double kDefaultSprtAlpha = 0.05;
inline constexpr double kDefaultSprtBeta = 0.05;

struct MatchOptions {
  std::string player_one_command;
  std::string player_two_command;
  std::chrono::milliseconds move_timeout = kDefaultMoveTimeout;
  engine::EngineLimits go_limits;
  std::vector<std::string> opening_moves;
  bool verbose = true;
};

struct OpeningLine {
  int line_number = 0;
  std::vector<std::string> moves;
  std::string text;
};

struct OpeningBook {
  std::string path;
  std::vector<OpeningLine> lines;
};

enum class MatchEndReason : std::uint8_t {
  kNormal,
  kTimeout,
  kDisconnected,
  kMalformedMove,
  kIllegalMove,
  kProtocolError,
};

enum class SprtDecision : std::uint8_t {
  kContinue,
  kAcceptNull,
  kAcceptAlternative,
  kInvalid,
};

struct MatchResult {
  MatchEndReason reason = MatchEndReason::kProtocolError;
  std::optional<Player> winner;
  ScoreByPlayer scores;
  std::vector<std::string> moves;
  std::string detail;
};

struct SeriesOptions {
  std::string engine_one_command;
  std::string engine_two_command;
  int games = 1;
  std::chrono::milliseconds move_timeout = kDefaultMoveTimeout;
  engine::EngineLimits go_limits;
  OpeningBook opening_book;
  bool shuffle_openings = false;
  bool alternate_sides = true;
  bool print_game_results = true;
  bool verbose_games = false;
  bool sprt_stop = false;
  double sprt_null_rate = kDefaultSprtNullRate;
  double sprt_alt_rate = kDefaultSprtAltRate;
  double sprt_alpha = kDefaultSprtAlpha;
  double sprt_beta = kDefaultSprtBeta;
};

struct SeriesGameResult {
  int game_number = 0;
  Player engine_one_player = Player::kOne;
  int opening_line_number = 0;
  std::string opening_moves;
  MatchResult match;
};

struct SeriesResult {
  int games_requested = 0;
  int games_played = 0;
  int engine_one_wins = 0;
  int engine_two_wins = 0;
  int no_winner = 0;
  int normal_games = 0;
  int timeout_games = 0;
  int disconnected_games = 0;
  int malformed_move_games = 0;
  int illegal_move_games = 0;
  int protocol_error_games = 0;
  long long engine_one_score_total = 0;
  long long engine_two_score_total = 0;
  long long plies_total = 0;
  int statistical_samples = 0;
  double engine_one_result_score = 0.0;
  double engine_one_result_rate = 0.0;
  double confidence_level = kDefaultConfidenceLevel;
  double confidence_low = 0.0;
  double confidence_high = 0.0;
  double sprt_null_rate = kDefaultSprtNullRate;
  double sprt_alt_rate = kDefaultSprtAltRate;
  double sprt_alpha = kDefaultSprtAlpha;
  double sprt_beta = kDefaultSprtBeta;
  double sprt_log_likelihood_ratio = 0.0;
  double sprt_lower_bound = 0.0;
  double sprt_upper_bound = 0.0;
  SprtDecision sprt_decision = SprtDecision::kContinue;
  std::vector<SeriesGameResult> games;
  std::string detail;
};

[[nodiscard]] std::string_view player_name(Player player) noexcept;
[[nodiscard]] std::string_view reason_name(MatchEndReason reason) noexcept;
[[nodiscard]] std::string_view sprt_decision_name(SprtDecision decision) noexcept;
[[nodiscard]] std::string_view engine_name_from_winner(const SeriesGameResult& game) noexcept;
[[nodiscard]] std::string format_opening_moves(const std::vector<std::string>& moves);
[[nodiscard]] OpeningBook parse_opening_book_text(std::string_view path, std::string_view text);
[[nodiscard]] OpeningBook load_opening_book(std::string_view path);

void print_state(const Position& position, std::ostream& output);
void print_final(const GameResult& result, std::ostream& output);
void print_match_result(const MatchResult& result, std::ostream& output);
void print_series_result(const SeriesResult& result, std::ostream& output);

[[nodiscard]] MatchResult run_process_match(const MatchOptions& options, std::ostream& output);
[[nodiscard]] SeriesResult run_process_series(const SeriesOptions& options, std::ostream& output);

}  // namespace poe2::match_runner

#endif  // POE2_MATCH_RUNNER_HPP
