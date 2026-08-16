#include "engine_process.hpp"

#include <fcntl.h>
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <csignal>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "poe2/engine_stdio.hpp"

namespace poe2::match_runner::detail {

namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] bool mark_close_on_exec(const int pipe_fds[2]) noexcept {
  for (int index = 0; index < 2; ++index) {
    const int fd = pipe_fds[index];
    const int flags = ::fcntl(fd, F_GETFD, 0);
    if (flags < 0 || ::fcntl(fd, F_SETFD, flags | FD_CLOEXEC) != 0) {
      return false;
    }
  }
  return true;
}

}  // namespace

EngineProcess::EngineProcess(std::string command) : command_(std::move(command)) {}

EngineProcess::~EngineProcess() { stop(); }

void EngineProcess::start() {
  int stdin_pipe[2] = {-1, -1};
  int stdout_pipe[2] = {-1, -1};

  if (::pipe(stdin_pipe) != 0) {
    throw std::runtime_error(std::string{"pipe failed: "} + std::strerror(errno));
  }
  if (::pipe(stdout_pipe) != 0) {
    close_pipe(stdin_pipe);
    throw std::runtime_error(std::string{"pipe failed: "} + std::strerror(errno));
  }
  if (!mark_close_on_exec(stdin_pipe) || !mark_close_on_exec(stdout_pipe)) {
    const int error = errno;
    close_pipe(stdin_pipe);
    close_pipe(stdout_pipe);
    throw std::runtime_error(std::string{"fcntl FD_CLOEXEC failed: "} + std::strerror(error));
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

bool EngineProcess::write_line(std::string_view line) noexcept {
  if (stdin_fd_ < 0) {
    return false;
  }

  return write_all(line.data(), line.size()) && write_all("\n", 1);
}

LineReadResult EngineProcess::read_line(std::chrono::milliseconds timeout) {
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

void EngineProcess::stop() noexcept {
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

void EngineProcess::close_pipe(const int pipe_fds[2]) noexcept {
  if (pipe_fds[0] >= 0) {
    ::close(pipe_fds[0]);
  }
  if (pipe_fds[1] >= 0) {
    ::close(pipe_fds[1]);
  }
}

void EngineProcess::trim_carriage_return(std::string& line) {
  if (!line.empty() && line.back() == '\r') {
    line.pop_back();
  }
}

std::optional<std::string> EngineProcess::extract_buffered_line() {
  const std::size_t newline = read_buffer_.find('\n');
  if (newline == std::string::npos) {
    return std::nullopt;
  }

  std::string line = read_buffer_.substr(0, newline);
  read_buffer_.erase(0, newline + 1);
  trim_carriage_return(line);
  return line;
}

void EngineProcess::read_available_output() {
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

bool EngineProcess::write_all(const char* data, std::size_t size) noexcept {
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

}  // namespace poe2::match_runner::detail
