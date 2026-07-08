#include "poe2/match_runner.hpp"

#include <fcntl.h>
#include <poll.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "poe2/engine_stdio.hpp"

namespace poe2::match_runner {

namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] Player other_player(Player player) noexcept {
  return player == Player::kOne ? Player::kTwo : Player::kOne;
}

[[nodiscard]] std::string trim_copy(std::string_view text) {
  const std::size_t first = text.find_first_not_of(" \t\r\n");
  if (first == std::string_view::npos) {
    return {};
  }

  const std::size_t last = text.find_last_not_of(" \t\r\n");
  return std::string{text.substr(first, last - first + 1)};
}

[[nodiscard]] std::string line_location(std::string_view path, int line_number) {
  std::ostringstream output;
  if (!path.empty()) {
    output << path << ':';
  }
  output << line_number;
  return output.str();
}

[[nodiscard]] std::string strip_comment(std::string_view line) {
  const std::size_t comment = line.find('#');
  if (comment != std::string_view::npos) {
    line = line.substr(0, comment);
  }

  return trim_copy(line);
}

[[nodiscard]] std::optional<OpeningLine> parse_opening_line(std::string_view path, int line_number,
                                                            std::string_view line) {
  const std::string opening_text = strip_comment(line);
  if (opening_text.empty()) {
    return std::nullopt;
  }

  Position position;
  std::vector<std::string> moves;
  std::istringstream input{opening_text};
  std::string token;
  while (input >> token) {
    const std::optional<Move> parsed = parse_move(token);
    if (!parsed.has_value()) {
      throw std::invalid_argument{line_location(path, line_number) +
                                  " malformed opening move: " + token};
    }

    const std::string formatted = format_move(*parsed);
    const MoveResult result = apply_move(position, *parsed);
    if (!result.accepted) {
      const std::string_view error =
          result.error.has_value() ? move_error_name(*result.error) : "unknown";
      throw std::invalid_argument{line_location(path, line_number) + " illegal opening move " +
                                  formatted + ": " + std::string{error}};
    }
    if (result.game_result.has_value()) {
      throw std::invalid_argument{line_location(path, line_number) +
                                  " opening line reaches a terminal position"};
    }

    moves.push_back(formatted);
  }

  const std::string normalized_text = format_opening_moves(moves);
  return OpeningLine{
      .line_number = line_number,
      .moves = std::move(moves),
      .text = normalized_text,
  };
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

enum class LineReadStatus : std::uint8_t {
  kLine,
  kTimeout,
  kClosed,
};

struct LineReadResult {
  LineReadStatus status = LineReadStatus::kClosed;
  std::string line;
};

class EngineProcess final {
 public:
  explicit EngineProcess(std::string command) : command_(std::move(command)) {}

  EngineProcess(const EngineProcess&) = delete;
  EngineProcess& operator=(const EngineProcess&) = delete;

  EngineProcess(EngineProcess&&) = delete;
  EngineProcess& operator=(EngineProcess&&) = delete;

  ~EngineProcess() { stop(); }

  void start() {
    int stdin_pipe[2] = {-1, -1};
    int stdout_pipe[2] = {-1, -1};

    if (::pipe(stdin_pipe) != 0) {
      throw std::runtime_error(std::string{"pipe failed: "} + std::strerror(errno));
    }
    if (::pipe(stdout_pipe) != 0) {
      close_pipe(stdin_pipe);
      throw std::runtime_error(std::string{"pipe failed: "} + std::strerror(errno));
    }

    const pid_t child = ::fork();
    if (child < 0) {
      close_pipe(stdin_pipe);
      close_pipe(stdout_pipe);
      throw std::runtime_error(std::string{"fork failed: "} + std::strerror(errno));
    }

    if (child == 0) {
      std::signal(SIGPIPE, SIG_DFL);

      ::dup2(stdin_pipe[0], STDIN_FILENO);
      ::dup2(stdout_pipe[1], STDOUT_FILENO);

      close_pipe(stdin_pipe);
      close_pipe(stdout_pipe);

      ::execl("/bin/sh", "sh", "-c", command_.c_str(), static_cast<char*>(nullptr));
      ::_exit(127);
    }

    pid_ = child;
    stdin_fd_ = stdin_pipe[1];
    stdout_fd_ = stdout_pipe[0];

    ::close(stdin_pipe[0]);
    ::close(stdout_pipe[1]);

    const int flags = ::fcntl(stdout_fd_, F_GETFL, 0);
    if (flags >= 0) {
      ::fcntl(stdout_fd_, F_SETFL, flags | O_NONBLOCK);
    }
  }

  bool write_line(std::string_view line) noexcept {
    if (stdin_fd_ < 0) {
      return false;
    }

    return write_all(line.data(), line.size()) && write_all("\n", 1);
  }

  [[nodiscard]] LineReadResult read_line(std::chrono::milliseconds timeout) {
    if (stdout_fd_ < 0) {
      return LineReadResult{.status = LineReadStatus::kClosed};
    }

    const Clock::time_point deadline = Clock::now() + timeout;
    while (true) {
      if (std::optional<std::string> line = extract_buffered_line(); line.has_value()) {
        return LineReadResult{.status = LineReadStatus::kLine, .line = *line};
      }

      const Clock::time_point now = Clock::now();
      if (now >= deadline) {
        return LineReadResult{.status = LineReadStatus::kTimeout};
      }

      const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
      pollfd descriptor{
          .fd = stdout_fd_,
          .events = POLLIN | POLLHUP,
          .revents = 0,
      };

      const int poll_result = ::poll(&descriptor, 1, static_cast<int>(remaining.count()));
      if (poll_result < 0) {
        if (errno == EINTR) {
          continue;
        }
        throw std::runtime_error(std::string{"poll failed: "} + std::strerror(errno));
      }
      if (poll_result == 0) {
        return LineReadResult{.status = LineReadStatus::kTimeout};
      }

      read_available_output();
      if (std::optional<std::string> line = extract_buffered_line(); line.has_value()) {
        return LineReadResult{.status = LineReadStatus::kLine, .line = *line};
      }

      if ((descriptor.revents & POLLHUP) != 0) {
        if (!read_buffer_.empty()) {
          std::string line = std::move(read_buffer_);
          read_buffer_.clear();
          trim_carriage_return(line);
          return LineReadResult{.status = LineReadStatus::kLine, .line = std::move(line)};
        }
        return LineReadResult{.status = LineReadStatus::kClosed};
      }
    }
  }

  void stop() noexcept {
    if (stdin_fd_ >= 0) {
      write_line(engine_stdio::kCommandQuit);
      ::close(stdin_fd_);
      stdin_fd_ = -1;
    }

    if (stdout_fd_ >= 0) {
      ::close(stdout_fd_);
      stdout_fd_ = -1;
    }

    if (pid_ <= 0) {
      return;
    }

    int status = 0;
    if (::waitpid(pid_, &status, WNOHANG) == 0) {
      ::kill(pid_, SIGTERM);
      for (int attempt = 0; attempt < 20; ++attempt) {
        if (::waitpid(pid_, &status, WNOHANG) == pid_) {
          pid_ = -1;
          return;
        }
        ::usleep(10'000);
      }

      ::kill(pid_, SIGKILL);
      ::waitpid(pid_, &status, 0);
    }

    pid_ = -1;
  }

 private:
  static void close_pipe(const int pipe_fds[2]) noexcept {
    if (pipe_fds[0] >= 0) {
      ::close(pipe_fds[0]);
    }
    if (pipe_fds[1] >= 0) {
      ::close(pipe_fds[1]);
    }
  }

  static void trim_carriage_return(std::string& line) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
  }

  [[nodiscard]] std::optional<std::string> extract_buffered_line() {
    const std::size_t newline = read_buffer_.find('\n');
    if (newline == std::string::npos) {
      return std::nullopt;
    }

    std::string line = read_buffer_.substr(0, newline);
    read_buffer_.erase(0, newline + 1);
    trim_carriage_return(line);
    return line;
  }

  void read_available_output() {
    while (true) {
      char buffer[4096];
      const ssize_t bytes_read = ::read(stdout_fd_, buffer, sizeof(buffer));
      if (bytes_read > 0) {
        read_buffer_.append(buffer, static_cast<std::size_t>(bytes_read));
        continue;
      }
      if (bytes_read == 0) {
        return;
      }
      if (errno == EINTR) {
        continue;
      }
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        return;
      }

      throw std::runtime_error(std::string{"read failed: "} + std::strerror(errno));
    }
  }

  bool write_all(const char* data, std::size_t size) noexcept {
    const char* cursor = data;
    std::size_t remaining = size;
    while (remaining > 0) {
      const ssize_t written = ::write(stdin_fd_, cursor, remaining);
      if (written < 0) {
        if (errno == EINTR) {
          continue;
        }
        return false;
      }
      if (written == 0) {
        return false;
      }

      cursor += written;
      remaining -= static_cast<std::size_t>(written);
    }

    return true;
  }

  std::string command_;
  pid_t pid_ = -1;
  int stdin_fd_ = -1;
  int stdout_fd_ = -1;
  std::string read_buffer_;
};

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

std::string_view sprt_decision_name(SprtDecision decision) noexcept {
  switch (decision) {
    case SprtDecision::kContinue:
      return "continue";
    case SprtDecision::kAcceptNull:
      return "accept_null";
    case SprtDecision::kAcceptAlternative:
      return "accept_alt";
    case SprtDecision::kInvalid:
      return "invalid";
  }

  return "unknown";
}

std::string_view engine_name_from_winner(const SeriesGameResult& game) noexcept {
  if (!game.match.winner.has_value()) {
    return "none";
  }

  return *game.match.winner == game.engine_one_player ? "engine_one" : "engine_two";
}

std::string format_opening_moves(const std::vector<std::string>& moves) {
  std::string text;
  for (std::size_t index = 0; index < moves.size(); ++index) {
    if (index != 0) {
      text += ' ';
    }
    text += moves[index];
  }
  return text;
}

OpeningBook parse_opening_book_text(std::string_view path, std::string_view text) {
  OpeningBook book{
      .path = std::string{path},
  };

  std::istringstream input{std::string{text}};
  std::string line;
  int line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (std::optional<OpeningLine> opening = parse_opening_line(path, line_number, line);
        opening.has_value()) {
      book.lines.push_back(std::move(*opening));
    }
  }

  if (book.lines.empty()) {
    throw std::invalid_argument{"opening book has no opening lines: " + std::string{path}};
  }

  return book;
}

OpeningBook load_opening_book(std::string_view path) {
  std::ifstream input{std::string{path}};
  if (!input) {
    throw std::invalid_argument{"failed to open opening book: " + std::string{path}};
  }

  std::ostringstream text;
  text << input.rdbuf();
  return parse_opening_book_text(path, text.str());
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
         << " engine_one_wins=" << result.engine_one_wins
         << " engine_two_wins=" << result.engine_two_wins << " no_winner=" << result.no_winner
         << " engine_one_win_pct=" << std::fixed << std::setprecision(1)
         << percentage(result.engine_one_wins, result.games_played)
         << " engine_two_win_pct=" << percentage(result.engine_two_wins, result.games_played)
         << " avg_plies=" << average(result.plies_total, result.games_played)
         << " engine_one_avg_score=" << average(result.engine_one_score_total, result.games_played)
         << " engine_two_avg_score=" << average(result.engine_two_score_total, result.games_played)
         << '\n';
  output << "series_reasons"
         << " normal=" << result.normal_games << " timeout=" << result.timeout_games
         << " disconnected=" << result.disconnected_games
         << " malformed_move=" << result.malformed_move_games
         << " illegal_move=" << result.illegal_move_games
         << " protocol_error=" << result.protocol_error_games << '\n';
  output << "series_confidence"
         << " confidence_pct=" << std::fixed << std::setprecision(1)
         << percent_from_rate(result.confidence_level) << " samples=" << result.statistical_samples
         << " engine_one_score_pct=" << percent_from_rate(result.engine_one_result_rate)
         << " low_pct=" << percent_from_rate(result.confidence_low)
         << " high_pct=" << percent_from_rate(result.confidence_high) << '\n';
  output << "series_sprt"
         << " null_pct=" << std::fixed << std::setprecision(1)
         << percent_from_rate(result.sprt_null_rate)
         << " alt_pct=" << percent_from_rate(result.sprt_alt_rate)
         << " alpha_pct=" << percent_from_rate(result.sprt_alpha)
         << " beta_pct=" << percent_from_rate(result.sprt_beta) << " llr=" << std::setprecision(3)
         << result.sprt_log_likelihood_ratio << " lower=" << result.sprt_lower_bound
         << " upper=" << result.sprt_upper_bound
         << " decision=" << sprt_decision_name(result.sprt_decision) << '\n';
  if (!result.detail.empty()) {
    output << "detail " << result.detail << '\n';
  }
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
  };

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
    const bool has_openings = !options.opening_book.lines.empty();
    const int opening_index = has_openings
                                  ? (options.alternate_sides ? (game_index / 2) : game_index) %
                                        static_cast<int>(options.opening_book.lines.size())
                                  : -1;
    const OpeningLine* opening =
        has_openings ? &options.opening_book.lines[opening_index] : nullptr;
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

    if (options.sprt_stop) {
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
