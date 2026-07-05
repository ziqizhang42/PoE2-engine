#include "poe2/engine_stdio.hpp"

#include <iostream>
#include <sstream>
#include <string>

namespace poe2::engine_stdio {

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

int run_engine_stdio(std::string_view name, const MoveChooser& choose_move) {
  return run_engine_stdio(name, choose_move, std::cin, std::cout);
}

int run_engine_stdio(std::string_view name, const MoveChooser& choose_move, std::istream& input,
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
      continue;
    }
    if (is_position_command(line)) {
      static_cast<void>(set_position_from_command(line, position, output));
      continue;
    }
    if (is_command(line, kCommandGo)) {
      const std::optional<Move> move = choose_move(position);
      output << (move.has_value() ? format_bestmove(*move) : format_no_move()) << '\n';
      continue;
    }

    output << "info unknown_command " << line << '\n';
  }

  return 0;
}

}  // namespace poe2::engine_stdio
