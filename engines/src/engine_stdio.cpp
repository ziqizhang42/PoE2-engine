#include "poe2/engine_stdio.hpp"

#include <charconv>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

namespace poe2::engine_stdio {

namespace {

[[nodiscard]] std::optional<std::uint64_t> parse_positive_u64(std::string_view text) noexcept {
  std::uint64_t value = 0;
  const char* begin = text.data();
  const char* end = begin + text.size();
  const auto result = std::from_chars(begin, end, value);
  if (result.ec != std::errc{} || result.ptr != end || value == 0) {
    return std::nullopt;
  }

  return value;
}

void write_info(std::ostream& output, std::string_view text) {
  output << "info";
  if (!text.empty()) {
    output << ' ' << text;
  }
  output << '\n';
}

[[nodiscard]] bool append_move_text(std::string& text, Move move) {
  const std::string formatted = format_move(move);
  if (formatted.empty()) {
    return false;
  }

  if (!text.empty()) {
    text.push_back(' ');
  }
  text += formatted;
  return true;
}

}  // namespace

bool is_command(std::string_view line, std::string_view command) noexcept {
  return line == command ||
         (line.starts_with(command) && line.size() > command.size() && line[command.size()] == ' ');
}

bool is_position_command(std::string_view line) noexcept {
  return is_command(line, kCommandPosition);
}

std::string format_position_command(std::span<const std::string> moves) {
  std::string command{kCommandPosition};
  command += " startpos";

  if (moves.empty()) {
    return command;
  }

  command += " moves";
  for (const std::string& move : moves) {
    command.push_back(' ');
    command += move;
  }

  return command;
}

std::string format_go_command(const EngineLimits& limits) {
  std::string command{kCommandGo};

  if (limits.depth.has_value()) {
    command += " depth ";
    command += std::to_string(*limits.depth);
  }
  if (limits.move_time.has_value()) {
    command += " movetime ";
    command += std::to_string(limits.move_time->count());
  }
  if (limits.nodes.has_value()) {
    command += " nodes ";
    command += std::to_string(*limits.nodes);
  }

  return command;
}

std::string format_bestmove(Move move) {
  std::string command{kResponseBestMove};
  command.push_back(' ');
  command += format_move(move);
  return command;
}

std::string format_no_move() {
  std::string command{kResponseBestMove};
  command.push_back(' ');
  command += kNoMove;
  return command;
}

std::optional<std::string> format_info_result(const EngineResult& result) {
  if (!result.score.has_value() && result.depth == 0 && result.nodes == 0 &&
      result.principal_variation.empty()) {
    return std::nullopt;
  }

  std::string info{"info"};
  if (result.depth != 0) {
    info += " depth ";
    info += std::to_string(result.depth);
  }
  if (result.score.has_value()) {
    info += " score ";
    info += std::to_string(*result.score);
  }
  if (result.nodes != 0) {
    info += " nodes ";
    info += std::to_string(result.nodes);
  }
  if (!result.principal_variation.empty()) {
    std::string pv;
    for (const Move move : result.principal_variation) {
      if (!append_move_text(pv, move)) {
        return std::nullopt;
      }
    }
    info += " pv ";
    info += pv;
  }

  return info;
}

std::optional<std::string> parse_bestmove_text(std::string_view line) {
  if (!is_command(line, kResponseBestMove)) {
    return std::nullopt;
  }

  std::istringstream input{std::string{line}};
  std::string keyword;
  std::string move;
  input >> keyword >> move;
  if (keyword != kResponseBestMove || move.empty()) {
    return std::nullopt;
  }

  return move;
}

std::optional<EngineLimits> parse_go_limits(std::string_view command, std::ostream& output) {
  std::istringstream input{std::string{command}};

  std::string token;
  input >> token;
  if (token != kCommandGo) {
    return std::nullopt;
  }

  EngineLimits limits;
  while (input >> token) {
    if (token != "depth" && token != "movetime" && token != "nodes") {
      output << "info error unknown_go_option " << token << '\n';
      return std::nullopt;
    }

    std::string value_text;
    if (!(input >> value_text)) {
      output << "info error missing_go_value " << token << '\n';
      return std::nullopt;
    }

    const std::optional<std::uint64_t> value = parse_positive_u64(value_text);
    if (!value.has_value()) {
      output << "info error malformed_go_value " << token << ' ' << value_text << '\n';
      return std::nullopt;
    }

    if (token == "depth") {
      if (*value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        output << "info error go_value_too_large depth " << value_text << '\n';
        return std::nullopt;
      }
      limits.depth = static_cast<int>(*value);
      continue;
    }
    if (token == "movetime") {
      using Rep = std::chrono::milliseconds::rep;
      if (*value > static_cast<std::uint64_t>(std::numeric_limits<Rep>::max())) {
        output << "info error go_value_too_large movetime " << value_text << '\n';
        return std::nullopt;
      }
      limits.move_time = std::chrono::milliseconds{static_cast<Rep>(*value)};
      continue;
    }
    if (token == "nodes") {
      limits.nodes = value;
      continue;
    }
  }

  return limits;
}

bool set_position_from_command(std::string_view command, Position& position, std::ostream& output) {
  std::istringstream input{std::string{command}};

  std::string token;
  input >> token;
  if (token != kCommandPosition) {
    return false;
  }

  input >> token;
  if (token != "startpos") {
    output << "info error unsupported_position\n";
    return false;
  }

  Position next_position;
  if (!(input >> token)) {
    position = next_position;
    return true;
  }

  if (token != "moves") {
    output << "info error expected_moves\n";
    return false;
  }

  while (input >> token) {
    const std::optional<Move> move = parse_move(token);
    if (!move.has_value()) {
      output << "info error malformed_history_move " << token << '\n';
      return false;
    }

    const MoveResult result = apply_move(next_position, *move);
    if (!result.accepted) {
      const std::string_view error =
          result.error.has_value() ? move_error_name(*result.error) : "unknown";
      output << "info error illegal_history_move " << token << ' ' << error << '\n';
      return false;
    }
  }

  position = next_position;
  return true;
}

int run_engine_stdio(std::string_view name, Engine& engine) {
  return run_engine_stdio(name, engine, std::cin, std::cout);
}

int run_engine_stdio(std::string_view name, Engine& engine, std::istream& input,
                     std::ostream& output) {
  output.setf(std::ios::unitbuf);

  Position position;
  std::string line;

  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }
    if (line == kCommandQuit || line == kCommandExit) {
      return 0;
    }
    if (line == kCommandPoe2) {
      output << "id name " << name << '\n';
      output << kResponsePoe2Ok << '\n';
      continue;
    }
    if (line == kCommandIsReady) {
      output << kResponseReadyOk << '\n';
      continue;
    }
    if (line == kCommandNewGame) {
      position = Position{};
      engine.new_game();
      continue;
    }
    if (is_position_command(line)) {
      static_cast<void>(set_position_from_command(line, position, output));
      continue;
    }
    if (is_command(line, kCommandGo)) {
      const std::optional<EngineLimits> limits = parse_go_limits(line, output);
      if (!limits.has_value()) {
        output << format_no_move() << '\n';
        continue;
      }

      const InfoSink info = [&output](std::string_view text) { write_info(output, text); };
      const EngineResult result = engine.choose_move(position, *limits, info);
      if (const std::optional<std::string> result_info = format_info_result(result);
          result_info.has_value()) {
        output << *result_info << '\n';
      }
      output << (result.best_move.has_value() ? format_bestmove(*result.best_move)
                                              : format_no_move())
             << '\n';
      continue;
    }

    output << "info unknown_command " << line << '\n';
  }

  return 0;
}

}  // namespace poe2::engine_stdio
