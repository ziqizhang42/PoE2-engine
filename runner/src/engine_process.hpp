#ifndef POE2_RUNNER_ENGINE_PROCESS_HPP
#define POE2_RUNNER_ENGINE_PROCESS_HPP

#include <sys/types.h>

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace poe2::match_runner::detail {

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
  explicit EngineProcess(std::string command);

  EngineProcess(const EngineProcess&) = delete;
  EngineProcess& operator=(const EngineProcess&) = delete;

  EngineProcess(EngineProcess&&) = delete;
  EngineProcess& operator=(EngineProcess&&) = delete;

  ~EngineProcess();

  void start();
  bool write_line(std::string_view line) noexcept;
  [[nodiscard]] LineReadResult read_line(std::chrono::milliseconds timeout);
  void stop() noexcept;

 private:
  static void close_pipe(const int pipe_fds[2]) noexcept;
  static void trim_carriage_return(std::string& line);

  [[nodiscard]] std::optional<std::string> extract_buffered_line();
  void read_available_output();
  bool write_all(const char* data, std::size_t size) noexcept;

  std::string command_;
  pid_t pid_ = -1;
  int stdin_fd_ = -1;
  int stdout_fd_ = -1;
  std::string read_buffer_;
};

}  // namespace poe2::match_runner::detail

#endif  // POE2_RUNNER_ENGINE_PROCESS_HPP
