#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "eval_cli.hpp"
#include "openings_cli.hpp"
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

[[nodiscard]] std::optional<double> parse_probability(std::string_view text) {
  std::istringstream input{std::string{text}};
  double value = 0.0;
  input >> value;
  if (!input || !std::isfinite(value) || value <= 0.0 || value >= 1.0) {
    return std::nullopt;
  }

  std::string trailing;
  if (input >> trailing) {
    return std::nullopt;
  }

  return value;
}

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

bool parse_go_limit_option(std::string_view argument, int& index, int argc, char** argv,
                           poe2::engine::EngineLimits& limits) {
  if (argument == "--go-depth") {
    if (index + 1 >= argc) {
      throw std::invalid_argument{"--go-depth requires a positive integer"};
    }
    const std::optional<int> depth = parse_positive_int(argv[++index]);
    if (!depth.has_value()) {
      throw std::invalid_argument{"--go-depth requires a positive integer"};
    }
    limits.depth = depth;
    return true;
  }
  if (argument == "--go-movetime-ms") {
    if (index + 1 >= argc) {
      throw std::invalid_argument{"--go-movetime-ms requires a positive integer"};
    }
    const std::optional<int> move_time_ms = parse_positive_int(argv[++index]);
    if (!move_time_ms.has_value()) {
      throw std::invalid_argument{"--go-movetime-ms requires a positive integer"};
    }
    limits.move_time = std::chrono::milliseconds{*move_time_ms};
    return true;
  }
  if (argument == "--go-nodes") {
    if (index + 1 >= argc) {
      throw std::invalid_argument{"--go-nodes requires a positive integer"};
    }
    const std::optional<std::uint64_t> nodes = parse_positive_u64(argv[++index]);
    if (!nodes.has_value()) {
      throw std::invalid_argument{"--go-nodes requires a positive integer"};
    }
    limits.nodes = nodes;
    return true;
  }

  return false;
}

void print_go_limits(const poe2::engine::EngineLimits& limits, std::ostream& output) {
  if (limits.depth.has_value()) {
    output << " go_depth=" << *limits.depth;
  }
  if (limits.move_time.has_value()) {
    output << " go_movetime_ms=" << limits.move_time->count();
  }
  if (limits.nodes.has_value()) {
    output << " go_nodes=" << *limits.nodes;
  }
}

void print_usage(std::ostream& output) {
  output
      << "usage:\n"
      << "  poe2_runner\n"
      << "  poe2_runner manual\n"
      << "  poe2_runner match --p1 <command> --p2 <command> [--timeout-ms <ms>]\n"
      << "                    [--go-depth <n>] [--go-movetime-ms <ms>] [--go-nodes <n>]\n"
      << "                    [--opening-book <path>] [--quiet]\n"
      << "  poe2_runner series --engine-one <command> --engine-two <command> --games <n>\n"
      << "                     [--timeout-ms <ms>] [--fixed-sides] [--summary-only]\n"
      << "                     [--opening-book <path>]\n"
      << "                     [--verbose-games] [--sprt-stop] [--sprt-null <p>] [--sprt-alt <p>]\n"
      << "                     [--sprt-alpha <p>] [--sprt-beta <p>]\n"
      << "                     [--go-depth <n>] [--go-movetime-ms <ms>] [--go-nodes <n>]\n"
      << "  poe2_runner eval --new-build <dir> --base <id|dir|binary>\n"
      << "                   --new-engine <name> --base-engine <name>\n"
      << "                   [--new-engine-args <args>] [--base-engine-args <args>]\n"
      << "                   [--games <n>] [--timeout-ms <ms>] [--sprt-stop]\n"
      << "                   [--opening-book <path>]\n"
      << "                   [--require-accept-alt] [--ledger <path>] [--no-ledger]\n"
      << "                   [--go-depth <n>] [--go-movetime-ms <ms>] [--go-nodes <n>]\n"
      << "  poe2_runner openings generate-random --out <path> --count <n> --plies <n> --seed <n>\n"
      << "                                     [--max-score-gap <n>]\n"
      << "  poe2_runner openings generate-systematic --out <path> --plies 2\n";
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
    if (argument == "--quiet") {
      options.verbose = false;
      continue;
    }
    if (argument == "--opening-book") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--opening-book requires a path"};
      }
      const poe2::match_runner::OpeningBook book =
          poe2::match_runner::load_opening_book(argv[++index]);
      options.opening_moves = book.lines.front().moves;
      continue;
    }
    if (parse_go_limit_option(argument, index, argc, argv, options.go_limits)) {
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

[[nodiscard]] poe2::match_runner::SeriesOptions parse_series_options(int argc, char** argv) {
  poe2::match_runner::SeriesOptions options;

  for (int index = 0; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument == "--engine-one" || argument == "--p1") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--engine-one requires a command"};
      }
      options.engine_one_command = argv[++index];
      continue;
    }
    if (argument == "--engine-two" || argument == "--p2") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--engine-two requires a command"};
      }
      options.engine_two_command = argv[++index];
      continue;
    }
    if (argument == "--games") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--games requires a positive integer"};
      }
      const std::optional<int> games = parse_positive_int(argv[++index]);
      if (!games.has_value()) {
        throw std::invalid_argument{"--games requires a positive integer"};
      }
      options.games = *games;
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
    if (argument == "--fixed-sides") {
      options.alternate_sides = false;
      continue;
    }
    if (argument == "--summary-only") {
      options.print_game_results = false;
      continue;
    }
    if (argument == "--opening-book") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--opening-book requires a path"};
      }
      options.opening_book = poe2::match_runner::load_opening_book(argv[++index]);
      continue;
    }
    if (argument == "--verbose-games") {
      options.verbose_games = true;
      continue;
    }
    if (argument == "--sprt-stop") {
      options.sprt_stop = true;
      continue;
    }
    if (argument == "--sprt-null") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--sprt-null requires a probability between 0 and 1"};
      }
      const std::optional<double> rate = parse_probability(argv[++index]);
      if (!rate.has_value()) {
        throw std::invalid_argument{"--sprt-null requires a probability between 0 and 1"};
      }
      options.sprt_null_rate = *rate;
      continue;
    }
    if (argument == "--sprt-alt") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--sprt-alt requires a probability between 0 and 1"};
      }
      const std::optional<double> rate = parse_probability(argv[++index]);
      if (!rate.has_value()) {
        throw std::invalid_argument{"--sprt-alt requires a probability between 0 and 1"};
      }
      options.sprt_alt_rate = *rate;
      continue;
    }
    if (argument == "--sprt-alpha") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--sprt-alpha requires a probability between 0 and 1"};
      }
      const std::optional<double> alpha = parse_probability(argv[++index]);
      if (!alpha.has_value()) {
        throw std::invalid_argument{"--sprt-alpha requires a probability between 0 and 1"};
      }
      options.sprt_alpha = *alpha;
      continue;
    }
    if (argument == "--sprt-beta") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--sprt-beta requires a probability between 0 and 1"};
      }
      const std::optional<double> beta = parse_probability(argv[++index]);
      if (!beta.has_value()) {
        throw std::invalid_argument{"--sprt-beta requires a probability between 0 and 1"};
      }
      options.sprt_beta = *beta;
      continue;
    }
    if (parse_go_limit_option(argument, index, argc, argv, options.go_limits)) {
      continue;
    }

    throw std::invalid_argument{"unknown series argument: " + std::string{argument}};
  }

  if (options.engine_one_command.empty()) {
    throw std::invalid_argument{"missing --engine-one command"};
  }
  if (options.engine_two_command.empty()) {
    throw std::invalid_argument{"missing --engine-two command"};
  }
  if (options.sprt_alt_rate <= options.sprt_null_rate) {
    throw std::invalid_argument{"--sprt-alt must be greater than --sprt-null"};
  }

  return options;
}

int run_match(int argc, char** argv) {
  const poe2::match_runner::MatchOptions options = parse_match_options(argc, argv);

  std::cout << "match"
            << " timeout_ms=" << options.move_timeout.count();
  print_go_limits(options.go_limits, std::cout);
  std::cout << '\n';
  std::cout << "engine p1 " << options.player_one_command << '\n';
  std::cout << "engine p2 " << options.player_two_command << '\n';

  const poe2::match_runner::MatchResult result =
      poe2::match_runner::run_process_match(options, std::cout);
  poe2::match_runner::print_match_result(result, std::cout);
  return 0;
}

int run_series(int argc, char** argv) {
  const poe2::match_runner::SeriesOptions options = parse_series_options(argc, argv);

  std::cout << "series"
            << " games=" << options.games << " timeout_ms=" << options.move_timeout.count()
            << " alternate_sides=" << (options.alternate_sides ? 1 : 0) << " opening_book="
            << (options.opening_book.path.empty() ? "none" : options.opening_book.path)
            << " opening_count=" << options.opening_book.lines.size()
            << " sprt_stop=" << (options.sprt_stop ? 1 : 0)
            << " sprt_null=" << options.sprt_null_rate << " sprt_alt=" << options.sprt_alt_rate
            << " sprt_alpha=" << options.sprt_alpha << " sprt_beta=" << options.sprt_beta;
  print_go_limits(options.go_limits, std::cout);
  std::cout << '\n';
  std::cout << "engine engine_one " << options.engine_one_command << '\n';
  std::cout << "engine engine_two " << options.engine_two_command << '\n';

  const poe2::match_runner::SeriesResult result =
      poe2::match_runner::run_process_series(options, std::cout);
  poe2::match_runner::print_series_result(result, std::cout);
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
  if (command == "series") {
    return run_series(argc - 2, argv + 2);
  }
  if (command == "eval") {
    return poe2::runner_cli::run_eval(argc - 2, argv + 2);
  }
  if (command == "openings") {
    return poe2::runner_cli::run_openings(argc - 2, argv + 2);
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
