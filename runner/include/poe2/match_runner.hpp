#ifndef POE2_MATCH_RUNNER_HPP
#define POE2_MATCH_RUNNER_HPP

#include <array>
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
inline constexpr double kDefaultSequentialNullNelo = 0.0;
inline constexpr double kDefaultSequentialAltNelo = 20.0;
inline constexpr double kDefaultSequentialAlpha = 0.05;
inline constexpr double kDefaultSequentialBeta = 0.05;
inline constexpr int kAnalysisVersion = 2;

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

enum class SequentialDecision : std::uint8_t {
  kContinue,
  kAcceptNull,
  kAcceptAlternative,
  kInvalid,
};

enum class StatisticalUnit : std::uint8_t {
  kGame,
  kOpeningPair,
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
  std::optional<std::uint64_t> opening_seed;
  bool alternate_sides = true;
  bool print_game_results = true;
  bool verbose_games = false;
  bool require_normal_games = false;
  bool sequential_stop = false;
  double sequential_null_nelo = kDefaultSequentialNullNelo;
  double sequential_alt_nelo = kDefaultSequentialAltNelo;
  double sequential_alpha = kDefaultSequentialAlpha;
  double sequential_beta = kDefaultSequentialBeta;
};

struct SeriesGameResult {
  int game_number = 0;
  Player engine_one_player = Player::kOne;
  int opening_slot = 0;
  int opening_line_number = 0;
  std::string opening_moves;
  MatchResult match;
};

struct SeriesResult {
  int games_requested = 0;
  int games_played = 0;
  bool valid = true;
  std::string invalid_reason;
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
  int unique_openings_used = 0;
  StatisticalUnit statistical_unit = StatisticalUnit::kGame;
  int statistical_samples = 0;
  int statistical_games = 0;
  std::array<int, 5> statistical_score_counts{};
  double engine_one_result_score = 0.0;
  double engine_one_result_rate = 0.0;
  double confidence_level = kDefaultConfidenceLevel;
  double confidence_low = 0.0;
  double confidence_high = 0.0;
  std::optional<double> normalized_elo;
  double sequential_null_nelo = kDefaultSequentialNullNelo;
  double sequential_alt_nelo = kDefaultSequentialAltNelo;
  double sequential_alpha = kDefaultSequentialAlpha;
  double sequential_beta = kDefaultSequentialBeta;
  double betting_log_evidence_above_even = 0.0;
  double betting_log_evidence_below_even = 0.0;
  double sequential_llr = 0.0;
  double sequential_lower_bound = 0.0;
  double sequential_upper_bound = 0.0;
  SequentialDecision sequential_decision = SequentialDecision::kContinue;
  std::vector<SeriesGameResult> games;
  std::string detail;
};

struct GsprtAnalysis {
  std::optional<double> normalized_elo;
  double llr = 0.0;
  double lower_bound = 0.0;
  double upper_bound = 0.0;
  SequentialDecision decision = SequentialDecision::kContinue;
};

[[nodiscard]] std::string_view player_name(Player player) noexcept;
[[nodiscard]] std::string_view reason_name(MatchEndReason reason) noexcept;
[[nodiscard]] std::string_view sequential_decision_name(SequentialDecision decision) noexcept;
[[nodiscard]] std::string_view statistical_unit_name(StatisticalUnit unit) noexcept;
[[nodiscard]] std::string_view confidence_method_name(StatisticalUnit unit) noexcept;
[[nodiscard]] std::string_view sequential_test_method_name(StatisticalUnit unit) noexcept;
[[nodiscard]] std::string_view engine_name_from_winner(const SeriesGameResult& game) noexcept;
[[nodiscard]] std::string format_opening_moves(const std::vector<std::string>& moves);
[[nodiscard]] OpeningBook parse_opening_book_text(std::string_view path, std::string_view text);
[[nodiscard]] OpeningBook load_opening_book(std::string_view path);

void print_state(const Position& position, std::ostream& output);
void print_final(const GameResult& result, std::ostream& output);
void print_match_result(const MatchResult& result, std::ostream& output);
void print_series_result(const SeriesResult& result, std::ostream& output);
[[nodiscard]] GsprtAnalysis analyze_normalized_elo_gsprt(const std::array<int, 5>& score_counts,
                                                         StatisticalUnit unit, double null_nelo,
                                                         double alternative_nelo, double alpha,
                                                         double beta) noexcept;
void analyze_series_result(SeriesResult& result, const SeriesOptions& options) noexcept;

[[nodiscard]] MatchResult run_process_match(const MatchOptions& options, std::ostream& output);
[[nodiscard]] SeriesResult run_process_series(const SeriesOptions& options, std::ostream& output);

}  // namespace poe2::match_runner

#endif  // POE2_MATCH_RUNNER_HPP
