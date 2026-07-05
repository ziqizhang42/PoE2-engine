#include <exception>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "poe2/match_runner.hpp"
#include "poe2/move.hpp"

namespace {

void print_help() { std::cout << "commands: a1-g7, state, quit, help\n"; }

int run_manual() {
  poe2::Position position;
  std::string line;

  print_help();
  poe2::match_runner::print_state(position, std::cout);

  while (std::getline(std::cin, line)) {
    if (line.empty()) {
      continue;
    }
    if (line == "quit" || line == "exit") {
      return 0;
    }
    if (line == "help") {
      print_help();
      continue;
    }
    if (line == "state") {
      poe2::match_runner::print_state(position, std::cout);
      continue;
    }

    const std::optional<poe2::Move> move = poe2::parse_move(line);
    if (!move.has_value()) {
      std::cout << "rejected " << line << " malformed\n";
      continue;
    }

    const std::string formatted_move = poe2::format_move(*move);
    const poe2::MoveResult result = poe2::apply_move(position, *move);
    if (!result.accepted) {
      const std::string_view error =
          result.error.has_value() ? poe2::move_error_name(*result.error) : "unknown";
      std::cout << "rejected " << formatted_move << ' ' << error << '\n';
      if (result.game_result.has_value()) {
        poe2::match_runner::print_final(*result.game_result, std::cout);
      }
      continue;
    }

    std::cout << "accepted " << formatted_move << '\n';
    poe2::match_runner::print_state(position, std::cout);
    if (result.game_result.has_value()) {
      poe2::match_runner::print_final(*result.game_result, std::cout);
      return 0;
    }
  }

  return 0;
}

[[nodiscard]] std::optional<int> parse_positive_int(std::string_view text) {
  std::istringstream input{std::string{text}};
  int value = 0;
  input >> value;
  if (!input || value <= 0) {
    return std::nullopt;
  }

  std::string trailing;
  if (input >> trailing) {
    return std::nullopt;
  }

  return value;
}

void print_usage(std::ostream& output) {
  output << "usage:\n"
         << "  poe2_runner\n"
         << "  poe2_runner manual\n"
         << "  poe2_runner match --p1 <command> --p2 <command> [--timeout-ms <ms>]\n";
}

[[nodiscard]] poe2::match_runner::MatchOptions parse_match_options(int argc, char** argv) {
  poe2::match_runner::MatchOptions options;

  for (int index = 0; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument == "--p1") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--p1 requires a command"};
      }
      options.player_one_command = argv[++index];
      continue;
    }
    if (argument == "--p2") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--p2 requires a command"};
      }
      options.player_two_command = argv[++index];
      continue;
    }
    if (argument == "--timeout-ms") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--timeout-ms requires a positive integer"};
      }
      const std::optional<int> timeout_ms = parse_positive_int(argv[++index]);
      if (!timeout_ms.has_value()) {
        throw std::invalid_argument{"--timeout-ms requires a positive integer"};
      }
      options.move_timeout = std::chrono::milliseconds{*timeout_ms};
      continue;
    }

    throw std::invalid_argument{"unknown match argument: " + std::string{argument}};
  }

  if (options.player_one_command.empty()) {
    throw std::invalid_argument{"missing --p1 command"};
  }
  if (options.player_two_command.empty()) {
    throw std::invalid_argument{"missing --p2 command"};
  }

  return options;
}

int run_match(int argc, char** argv) {
  const poe2::match_runner::MatchOptions options = parse_match_options(argc, argv);

  std::cout << "match"
            << " timeout_ms=" << options.move_timeout.count() << '\n';
  std::cout << "engine p1 " << options.player_one_command << '\n';
  std::cout << "engine p2 " << options.player_two_command << '\n';

  const poe2::match_runner::MatchResult result =
      poe2::match_runner::run_process_match(options, std::cout);
  poe2::match_runner::print_match_result(result, std::cout);
  return 0;
}

int run(int argc, char** argv) {
  if (argc <= 1) {
    return run_manual();
  }

  const std::string_view command = argv[1];
  if (command == "manual") {
    return run_manual();
  }
  if (command == "match") {
    return run_match(argc - 2, argv + 2);
  }
  if (command == "--help" || command == "help") {
    print_usage(std::cout);
    return 0;
  }

  throw std::invalid_argument{"unknown command: " + std::string{command}};
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "fatal " << error.what() << '\n';
  } catch (...) {
    std::cerr << "fatal unknown\n";
  }

  return 1;
}
