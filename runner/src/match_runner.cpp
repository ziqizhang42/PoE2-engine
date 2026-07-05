#include "poe2/match_runner.hpp"

#include <fcntl.h>
#include <poll.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <optional>
#include <ostream>
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

  Position position;
  std::vector<std::string> moves;
  if (!write_engine_command(player_one, engine_stdio::kCommandNewGame, detail)) {
    return failed_match(position, moves, MatchEndReason::kDisconnected, Player::kOne,
                        std::move(detail));
  }
  if (!write_engine_command(player_two, engine_stdio::kCommandNewGame, detail)) {
    return failed_match(position, moves, MatchEndReason::kDisconnected, Player::kTwo,
                        std::move(detail));
  }

  print_state(position, output);

  while (true) {
    const Player side_to_move = position.side_to_move();
    EngineProcess& engine = side_to_move == Player::kOne ? player_one : player_two;

    if (!write_engine_command(engine, engine_stdio::format_position_command(moves), detail)) {
      return failed_match(position, moves, MatchEndReason::kDisconnected, side_to_move,
                          std::move(detail));
    }
    if (!write_engine_command(engine, engine_stdio::kCommandGo, detail)) {
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
    output << "move"
           << " ply=" << ply << " side=" << player_name(side_to_move) << " move=" << formatted_move
           << '\n';
    print_state(position, output);

    if (move_result.game_result.has_value()) {
      print_final(*move_result.game_result, output);
      return MatchResult{
          .reason = MatchEndReason::kNormal,
          .winner = move_result.game_result->winner,
          .scores = move_result.game_result->scores,
          .moves = std::move(moves),
      };
    }
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

MatchResult run_process_match(const MatchOptions& options, std::ostream& output) {
  std::signal(SIGPIPE, SIG_IGN);

  EngineProcess player_one{options.player_one_command};
  EngineProcess player_two{options.player_two_command};
  player_one.start();
  player_two.start();

  return play_match(player_one, player_two, options, output);
}

}  // namespace poe2::match_runner
