#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "poe2/match_runner.hpp"
#include "poe2/move.hpp"
#include "poe2/symmetry.hpp"

namespace {

namespace fs = std::filesystem;

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
      << "  poe2_runner eval --new-build <dir> --base <id|dir|binary> [--engine <name>]\n"
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

struct EvalOptions {
  poe2::match_runner::SeriesOptions series;
  std::string new_build;
  std::string base;
  std::string engine = "poe2_greedy";
  std::string preset = "release";
  std::string run_root = "build/eval/runs";
  std::string ledger = "eval/results.csv";
  std::string kind = "gate";
  bool write_ledger = true;
  bool require_accept_alt = false;
};

struct OpeningGenerateOptions {
  fs::path output_path;
  int count = 0;
  int plies = 0;
  int max_score_gap = 4;
  std::uint64_t seed = 0;
};

struct OpeningSystematicOptions {
  fs::path output_path;
  int plies = 0;
};

[[nodiscard]] std::string shell_quote(std::string_view text) {
  std::string quoted = "'";
  for (const char ch : text) {
    if (ch == '\'') {
      quoted += "'\\''";
    } else {
      quoted += ch;
    }
  }
  quoted += "'";
  return quoted;
}

[[nodiscard]] std::string json_escape(std::string_view text) {
  std::string escaped;
  for (const char ch : text) {
    switch (ch) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        escaped += ch;
        break;
    }
  }
  return escaped;
}

void write_csv_field(std::ostream& output, std::string_view text) {
  const bool needs_quotes = text.find_first_of(",\"\n\r") != std::string_view::npos;
  if (!needs_quotes) {
    output << text;
    return;
  }

  output << '"';
  for (const char ch : text) {
    if (ch == '"') {
      output << "\"\"";
    } else {
      output << ch;
    }
  }
  output << '"';
}

[[nodiscard]] std::string utc_timestamp() {
  const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  const std::tm utc = *std::gmtime(&now);
  std::ostringstream output;
  output << std::put_time(&utc, "%Y%m%dT%H%M%SZ");
  return output.str();
}

[[nodiscard]] bool is_file(const fs::path& path) {
  std::error_code error;
  return fs::is_regular_file(path, error);
}

[[nodiscard]] bool is_dir(const fs::path& path) {
  std::error_code error;
  return fs::is_directory(path, error);
}

[[nodiscard]] std::string build_id_from_build_dir(fs::path dir, std::string_view preset) {
  dir = dir.lexically_normal();
  if (dir.filename() == preset && dir.has_parent_path()) {
    return dir.parent_path().filename().string();
  }
  return dir.filename().string();
}

[[nodiscard]] fs::path resolve_engine_binary(std::string_view input, std::string_view engine,
                                             std::string_view preset) {
  fs::path input_path{std::string{input}};
  if (is_file(input_path)) {
    return input_path;
  }

  if (is_dir(input_path)) {
    fs::path direct = input_path / "engines" / engine;
    if (is_file(direct)) {
      return direct;
    }

    fs::path nested = input_path / preset / "engines" / engine;
    if (is_file(nested)) {
      return nested;
    }
  }

  fs::path build_id_path =
      fs::path{"build"} / "by-commit" / input_path / preset / "engines" / engine;
  if (is_file(build_id_path)) {
    return build_id_path;
  }

  throw std::invalid_argument{"could not resolve engine '" + std::string{engine} + "' from '" +
                              std::string{input} + "'"};
}

[[nodiscard]] std::string id_from_eval_input(std::string_view input, std::string_view preset) {
  const fs::path input_path{std::string{input}};
  if (is_dir(input_path)) {
    return build_id_from_build_dir(input_path, preset);
  }
  if (is_file(input_path)) {
    const fs::path engine_dir = input_path.parent_path();
    const fs::path preset_dir = engine_dir.parent_path();
    return build_id_from_build_dir(preset_dir, preset);
  }
  return std::string{input};
}

void create_parent_directories(const fs::path& path) {
  const fs::path parent = path.parent_path();
  if (!parent.empty()) {
    fs::create_directories(parent);
  }
}

[[nodiscard]] std::string score_text(double value) {
  std::ostringstream output;
  output << std::fixed << std::setprecision(3) << value;
  return output.str();
}

[[nodiscard]] std::string file_digest(const fs::path& path) {
  if (path.empty()) {
    return {};
  }

  std::ifstream input{path, std::ios::binary};
  if (!input) {
    throw std::runtime_error{"failed to read " + path.string()};
  }

  std::uint64_t hash = 14695981039346656037ULL;
  char buffer[4096];
  while (input) {
    input.read(buffer, sizeof(buffer));
    const std::streamsize bytes = input.gcount();
    for (std::streamsize index = 0; index < bytes; ++index) {
      hash ^= static_cast<unsigned char>(buffer[index]);
      hash *= 1099511628211ULL;
    }
  }

  std::ostringstream output;
  output << "fnv1a64:" << std::hex << std::setw(16) << std::setfill('0') << hash;
  return output.str();
}

[[nodiscard]] std::string opening_book_digest(const poe2::match_runner::OpeningBook& opening_book) {
  if (opening_book.path.empty()) {
    return {};
  }

  return file_digest(opening_book.path);
}

[[nodiscard]] std::string position_key_text(poe2::PositionKey key) {
  std::ostringstream output;
  output << key.low << ':' << key.high;
  return output.str();
}

[[nodiscard]] std::vector<poe2::Move> legal_moves(const poe2::Position& position) {
  std::vector<poe2::Move> moves;
  poe2::Bitboard bits = position.legal_moves();
  while (bits != 0) {
    const int index = std::countr_zero(bits);
    moves.push_back(poe2::Move{.square = poe2::square_from_index(index)});
    bits &= bits - poe2::Bitboard{1};
  }

  return moves;
}

[[nodiscard]] std::optional<std::vector<std::string>> generate_opening_line(
    std::mt19937_64& rng, int plies, int max_score_gap,
    std::unordered_set<std::string>& seen_positions) {
  poe2::Position position;
  std::vector<std::string> moves;
  moves.reserve(static_cast<std::size_t>(plies));

  for (int ply = 0; ply < plies; ++ply) {
    const std::vector<poe2::Move> candidates = legal_moves(position);
    if (candidates.empty()) {
      return std::nullopt;
    }

    std::uniform_int_distribution<std::size_t> distribution{0, candidates.size() - 1};
    const poe2::Move move = candidates[distribution(rng)];
    if (!position.play(move.square)) {
      return std::nullopt;
    }
    moves.push_back(poe2::format_move(move));
  }

  const poe2::ScoreByPlayer scores = position.scores();
  if (std::abs(scores.player_one - scores.player_two) > max_score_gap) {
    return std::nullopt;
  }

  const poe2::CanonicalPositionKey canonical = poe2::canonicalize_position_key(position.key());
  if (!seen_positions.insert(position_key_text(canonical.key)).second) {
    return std::nullopt;
  }

  return moves;
}

void write_generated_opening_book(const OpeningGenerateOptions& options) {
  if (options.output_path.empty()) {
    throw std::invalid_argument{"missing --out"};
  }
  if (options.count <= 0) {
    throw std::invalid_argument{"--count requires a positive integer"};
  }
  if (options.plies <= 0 || options.plies >= poe2::kCellCount) {
    throw std::invalid_argument{"--plies requires a positive integer below the board size"};
  }
  if (options.max_score_gap < 0) {
    throw std::invalid_argument{"--max-score-gap must be non-negative"};
  }

  create_parent_directories(options.output_path);
  std::ofstream output{options.output_path};
  if (!output) {
    throw std::runtime_error{"failed to write " + options.output_path.string()};
  }

  output << "# poe2 opening suite\n"
         << "# generator=random-legal-balanced\n"
         << "# seed=" << options.seed << "\n"
         << "# count=" << options.count << "\n"
         << "# plies=" << options.plies << "\n"
         << "# max_score_gap=" << options.max_score_gap << "\n";

  std::mt19937_64 rng{options.seed};
  std::unordered_set<std::string> seen_positions;
  int generated = 0;
  const int max_attempts = std::max(options.count * 1000, 1000);
  for (int attempt = 0; attempt < max_attempts && generated < options.count; ++attempt) {
    std::optional<std::vector<std::string>> moves =
        generate_opening_line(rng, options.plies, options.max_score_gap, seen_positions);
    if (!moves.has_value()) {
      continue;
    }

    output << poe2::match_runner::format_opening_moves(*moves) << '\n';
    ++generated;
  }

  if (generated != options.count) {
    throw std::runtime_error{"only generated " + std::to_string(generated) + " of " +
                             std::to_string(options.count) + " requested openings"};
  }
}

void write_systematic_opening_book(const OpeningSystematicOptions& options) {
  if (options.output_path.empty()) {
    throw std::invalid_argument{"missing --out"};
  }
  if (options.plies != 2) {
    throw std::invalid_argument{"generate-systematic currently supports --plies 2"};
  }

  create_parent_directories(options.output_path);
  std::ofstream output{options.output_path};
  if (!output) {
    throw std::runtime_error{"failed to write " + options.output_path.string()};
  }

  output << "# poe2 opening suite: systematic-2ply-v1\n"
         << "# generator=systematic-symmetry\n"
         << "# plies=2\n"
         << "# raw_ordered_sequences=2352\n"
         << "# expected_canonical_positions=315\n";

  std::unordered_set<std::string> seen_positions;
  int generated = 0;
  for (int first_index = 0; first_index < poe2::kCellCount; ++first_index) {
    for (int second_index = 0; second_index < poe2::kCellCount; ++second_index) {
      if (first_index == second_index) {
        continue;
      }

      poe2::Position position;
      const poe2::Move first{.square = poe2::square_from_index(first_index)};
      const poe2::Move second{.square = poe2::square_from_index(second_index)};
      if (!position.play(first.square) || !position.play(second.square)) {
        throw std::runtime_error{"failed to generate legal 2-ply opening"};
      }

      const poe2::CanonicalPositionKey canonical = poe2::canonicalize_position_key(position.key());
      if (!seen_positions.insert(position_key_text(canonical.key)).second) {
        continue;
      }

      output << poe2::format_move(first) << ' ' << poe2::format_move(second) << '\n';
      ++generated;
    }
  }

  if (generated != 315) {
    throw std::runtime_error{"generated " + std::to_string(generated) +
                             " systematic 2-ply openings; expected 315"};
  }
}

void write_command_file(const fs::path& path, const EvalOptions& options,
                        const fs::path& new_engine, const fs::path& base_engine) {
  std::ofstream output{path};
  if (!output) {
    throw std::runtime_error{"failed to write " + path.string()};
  }

  output << "poe2_runner eval"
         << " --new-build " << shell_quote(options.new_build) << " --base "
         << shell_quote(options.base) << " --engine " << shell_quote(options.engine) << " --games "
         << options.series.games << " --timeout-ms " << options.series.move_timeout.count();
  if (!options.series.opening_book.path.empty()) {
    output << " --opening-book " << shell_quote(options.series.opening_book.path);
  }
  if (options.series.go_limits.depth.has_value()) {
    output << " --go-depth " << *options.series.go_limits.depth;
  }
  if (options.series.go_limits.move_time.has_value()) {
    output << " --go-movetime-ms " << options.series.go_limits.move_time->count();
  }
  if (options.series.go_limits.nodes.has_value()) {
    output << " --go-nodes " << *options.series.go_limits.nodes;
  }
  if (options.series.sprt_stop) {
    output << " --sprt-stop";
  }
  if (options.require_accept_alt) {
    output << " --require-accept-alt";
  }
  output << '\n';
  output << "new_engine=" << new_engine.string() << '\n';
  output << "base_engine=" << base_engine.string() << '\n';
}

void write_summary_json(const fs::path& path, const poe2::match_runner::SeriesResult& result) {
  std::ofstream output{path};
  if (!output) {
    throw std::runtime_error{"failed to write " + path.string()};
  }

  output << "{\n"
         << "  \"games_requested\": " << result.games_requested << ",\n"
         << "  \"games_played\": " << result.games_played << ",\n"
         << "  \"engine_one_wins\": " << result.engine_one_wins << ",\n"
         << "  \"engine_two_wins\": " << result.engine_two_wins << ",\n"
         << "  \"no_winner\": " << result.no_winner << ",\n"
         << "  \"engine_one_score_rate\": " << score_text(result.engine_one_result_rate) << ",\n"
         << "  \"confidence_low\": " << score_text(result.confidence_low) << ",\n"
         << "  \"confidence_high\": " << score_text(result.confidence_high) << ",\n"
         << "  \"sprt_decision\": \""
         << poe2::match_runner::sprt_decision_name(result.sprt_decision) << "\",\n"
         << "  \"sprt_log_likelihood_ratio\": " << score_text(result.sprt_log_likelihood_ratio)
         << "\n"
         << "}\n";
}

void write_games_csv(const fs::path& path, const poe2::match_runner::SeriesResult& result) {
  std::ofstream output{path};
  if (!output) {
    throw std::runtime_error{"failed to write " + path.string()};
  }

  output
      << "game,engine_one_player,opening_line,opening_moves,winner,reason,plies,player_one_score,"
         "player_two_score\n";
  for (const poe2::match_runner::SeriesGameResult& game : result.games) {
    output << game.game_number << ',' << poe2::match_runner::player_name(game.engine_one_player)
           << ',' << game.opening_line_number << ',';
    write_csv_field(output, game.opening_moves);
    output << ',' << poe2::match_runner::engine_name_from_winner(game) << ','
           << poe2::match_runner::reason_name(game.match.reason) << ',' << game.match.moves.size()
           << ',' << game.match.scores.player_one << ',' << game.match.scores.player_two << '\n';
  }
}

void write_manifest_json(const fs::path& path, const EvalOptions& options,
                         const poe2::match_runner::SeriesResult& result, std::string_view timestamp,
                         std::string_view run_id, std::string_view new_id, std::string_view base_id,
                         const fs::path& new_engine, const fs::path& base_engine) {
  std::ofstream output{path};
  if (!output) {
    throw std::runtime_error{"failed to write " + path.string()};
  }

  output << "{\n"
         << "  \"run_id\": \"" << json_escape(run_id) << "\",\n"
         << "  \"created_at_utc\": \"" << json_escape(timestamp) << "\",\n"
         << "  \"kind\": \"" << json_escape(options.kind) << "\",\n"
         << "  \"new_id\": \"" << json_escape(new_id) << "\",\n"
         << "  \"base_id\": \"" << json_escape(base_id) << "\",\n"
         << "  \"engine\": \"" << json_escape(options.engine) << "\",\n"
         << "  \"new_engine\": \"" << json_escape(new_engine.string()) << "\",\n"
         << "  \"base_engine\": \"" << json_escape(base_engine.string()) << "\",\n"
         << "  \"opening_book\": ";
  if (options.series.opening_book.path.empty()) {
    output << "null";
  } else {
    output << "\"" << json_escape(options.series.opening_book.path) << "\"";
  }
  output << ",\n";

  const std::string book_digest = opening_book_digest(options.series.opening_book);
  output << "  \"opening_book_digest\": ";
  if (book_digest.empty()) {
    output << "null";
  } else {
    output << "\"" << json_escape(book_digest) << "\"";
  }
  output << ",\n"
         << "  \"opening_count\": " << options.series.opening_book.lines.size() << ",\n"
         << "  \"games\": " << options.series.games << ",\n"
         << "  \"games_played\": " << result.games_played << ",\n"
         << "  \"timeout_ms\": " << options.series.move_timeout.count() << ",\n"
         << "  \"go_depth\": ";
  if (options.series.go_limits.depth.has_value()) {
    output << *options.series.go_limits.depth;
  } else {
    output << "null";
  }
  output << ",\n"
         << "  \"go_movetime_ms\": ";
  if (options.series.go_limits.move_time.has_value()) {
    output << options.series.go_limits.move_time->count();
  } else {
    output << "null";
  }
  output << ",\n"
         << "  \"go_nodes\": ";
  if (options.series.go_limits.nodes.has_value()) {
    output << *options.series.go_limits.nodes;
  } else {
    output << "null";
  }
  output << ",\n"
         << "  \"sprt_stop\": " << (options.series.sprt_stop ? "true" : "false") << ",\n"
         << "  \"sprt_null\": " << options.series.sprt_null_rate << ",\n"
         << "  \"sprt_alt\": " << options.series.sprt_alt_rate << ",\n"
         << "  \"sprt_alpha\": " << options.series.sprt_alpha << ",\n"
         << "  \"sprt_beta\": " << options.series.sprt_beta << ",\n"
         << "  \"sprt_decision\": \""
         << poe2::match_runner::sprt_decision_name(result.sprt_decision) << "\"\n"
         << "}\n";
}

void append_ledger_row(const fs::path& path, const EvalOptions& options,
                       const poe2::match_runner::SeriesResult& result, std::string_view timestamp,
                       std::string_view run_id, std::string_view new_id, std::string_view base_id,
                       const fs::path& run_dir) {
  constexpr std::string_view kHeader =
      "run_id,created_at_utc,kind,new_id,base_id,engine,games_requested,games_played,"
      "engine_one_wins,engine_two_wins,no_winner,engine_one_score_pct,confidence_low_pct,"
      "confidence_high_pct,sprt_decision,sprt_null,sprt_alt,opening_book,opening_book_digest,"
      "opening_count,go_depth,go_movetime_ms,go_nodes,timeout_ms,run_dir\n";

  create_parent_directories(path);
  const bool needs_header = !is_file(path) || fs::file_size(path) == 0;
  std::ofstream output{path, std::ios::app};
  if (!output) {
    throw std::runtime_error{"failed to append " + path.string()};
  }
  if (needs_header) {
    output << kHeader;
  }

  write_csv_field(output, run_id);
  output << ',';
  write_csv_field(output, timestamp);
  output << ',';
  write_csv_field(output, options.kind);
  output << ',';
  write_csv_field(output, new_id);
  output << ',';
  write_csv_field(output, base_id);
  output << ',';
  write_csv_field(output, options.engine);
  output << ',' << result.games_requested << ',' << result.games_played << ','
         << result.engine_one_wins << ',' << result.engine_two_wins << ',' << result.no_winner
         << ',' << score_text(result.engine_one_result_rate * 100.0) << ','
         << score_text(result.confidence_low * 100.0) << ','
         << score_text(result.confidence_high * 100.0) << ',';
  write_csv_field(output, poe2::match_runner::sprt_decision_name(result.sprt_decision));
  output << ',' << options.series.sprt_null_rate << ',' << options.series.sprt_alt_rate << ',';
  write_csv_field(output, options.series.opening_book.path);
  output << ',';
  write_csv_field(output, opening_book_digest(options.series.opening_book));
  output << ',' << options.series.opening_book.lines.size() << ',';
  if (options.series.go_limits.depth.has_value()) {
    output << *options.series.go_limits.depth;
  }
  output << ',';
  if (options.series.go_limits.move_time.has_value()) {
    output << options.series.go_limits.move_time->count();
  }
  output << ',';
  if (options.series.go_limits.nodes.has_value()) {
    output << *options.series.go_limits.nodes;
  }
  output << ',' << options.series.move_timeout.count() << ',';
  write_csv_field(output, run_dir.string());
  output << '\n';
}

[[nodiscard]] EvalOptions parse_eval_options(int argc, char** argv) {
  EvalOptions options;
  options.series.games = 1000;
  options.series.move_timeout = std::chrono::milliseconds{1000};
  options.series.go_limits.move_time = std::chrono::milliseconds{900};
  options.series.print_game_results = false;

  for (int index = 0; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument == "--new-build") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--new-build requires a directory"};
      }
      options.new_build = argv[++index];
      continue;
    }
    if (argument == "--base") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--base requires a build id, directory, or binary"};
      }
      options.base = argv[++index];
      continue;
    }
    if (argument == "--engine") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--engine requires a binary name"};
      }
      options.engine = argv[++index];
      continue;
    }
    if (argument == "--preset") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--preset requires a name"};
      }
      options.preset = argv[++index];
      continue;
    }
    if (argument == "--run-root") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--run-root requires a directory"};
      }
      options.run_root = argv[++index];
      continue;
    }
    if (argument == "--ledger") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--ledger requires a path"};
      }
      options.ledger = argv[++index];
      options.write_ledger = true;
      continue;
    }
    if (argument == "--no-ledger") {
      options.write_ledger = false;
      continue;
    }
    if (argument == "--kind") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--kind requires a name"};
      }
      options.kind = argv[++index];
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
      options.series.games = *games;
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
      options.series.move_timeout = std::chrono::milliseconds{*timeout_ms};
      continue;
    }
    if (argument == "--fixed-sides") {
      options.series.alternate_sides = false;
      continue;
    }
    if (argument == "--verbose-games") {
      options.series.print_game_results = true;
      options.series.verbose_games = true;
      continue;
    }
    if (argument == "--opening-book") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--opening-book requires a path"};
      }
      options.series.opening_book = poe2::match_runner::load_opening_book(argv[++index]);
      continue;
    }
    if (argument == "--sprt-stop") {
      options.series.sprt_stop = true;
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
      options.series.sprt_null_rate = *rate;
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
      options.series.sprt_alt_rate = *rate;
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
      options.series.sprt_alpha = *alpha;
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
      options.series.sprt_beta = *beta;
      continue;
    }
    if (argument == "--require-accept-alt") {
      options.require_accept_alt = true;
      continue;
    }
    if (parse_go_limit_option(argument, index, argc, argv, options.series.go_limits)) {
      continue;
    }

    throw std::invalid_argument{"unknown eval argument: " + std::string{argument}};
  }

  if (options.new_build.empty()) {
    throw std::invalid_argument{"missing --new-build"};
  }
  if (options.base.empty()) {
    throw std::invalid_argument{"missing --base"};
  }
  if (options.series.sprt_alt_rate <= options.series.sprt_null_rate) {
    throw std::invalid_argument{"--sprt-alt must be greater than --sprt-null"};
  }

  return options;
}

[[nodiscard]] OpeningGenerateOptions parse_opening_generate_options(int argc, char** argv) {
  OpeningGenerateOptions options;

  for (int index = 0; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument == "--out") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--out requires a path"};
      }
      options.output_path = argv[++index];
      continue;
    }
    if (argument == "--count") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--count requires a positive integer"};
      }
      const std::optional<int> count = parse_positive_int(argv[++index]);
      if (!count.has_value()) {
        throw std::invalid_argument{"--count requires a positive integer"};
      }
      options.count = *count;
      continue;
    }
    if (argument == "--plies") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--plies requires a positive integer"};
      }
      const std::optional<int> plies = parse_positive_int(argv[++index]);
      if (!plies.has_value()) {
        throw std::invalid_argument{"--plies requires a positive integer"};
      }
      options.plies = *plies;
      continue;
    }
    if (argument == "--seed") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--seed requires a positive integer"};
      }
      const std::optional<std::uint64_t> seed = parse_positive_u64(argv[++index]);
      if (!seed.has_value()) {
        throw std::invalid_argument{"--seed requires a positive integer"};
      }
      options.seed = *seed;
      continue;
    }
    if (argument == "--max-score-gap") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--max-score-gap requires a non-negative integer"};
      }
      std::istringstream input{argv[++index]};
      int max_score_gap = -1;
      input >> max_score_gap;
      std::string trailing;
      if (!input || (input >> trailing) || max_score_gap < 0) {
        throw std::invalid_argument{"--max-score-gap requires a non-negative integer"};
      }
      options.max_score_gap = max_score_gap;
      continue;
    }

    throw std::invalid_argument{"unknown openings generate argument: " + std::string{argument}};
  }

  return options;
}

[[nodiscard]] OpeningSystematicOptions parse_opening_systematic_options(int argc, char** argv) {
  OpeningSystematicOptions options;

  for (int index = 0; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument == "--out") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--out requires a path"};
      }
      options.output_path = argv[++index];
      continue;
    }
    if (argument == "--plies") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--plies requires a positive integer"};
      }
      const std::optional<int> plies = parse_positive_int(argv[++index]);
      if (!plies.has_value()) {
        throw std::invalid_argument{"--plies requires a positive integer"};
      }
      options.plies = *plies;
      continue;
    }

    throw std::invalid_argument{"unknown openings generate-systematic argument: " +
                                std::string{argument}};
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

int run_eval(int argc, char** argv) {
  EvalOptions options = parse_eval_options(argc, argv);
  const fs::path new_build = fs::path{options.new_build}.lexically_normal();
  if (!is_dir(new_build)) {
    throw std::invalid_argument{"new build directory does not exist: " + new_build.string()};
  }

  const fs::path new_engine = new_build / "engines" / options.engine;
  if (!is_file(new_engine)) {
    throw std::invalid_argument{"new engine binary does not exist: " + new_engine.string()};
  }

  const fs::path base_engine = resolve_engine_binary(options.base, options.engine, options.preset);
  const std::string new_id = build_id_from_build_dir(new_build, options.preset);
  const std::string base_id = id_from_eval_input(options.base, options.preset);
  const std::string timestamp = utc_timestamp();
  const std::string run_id = timestamp + "__" + new_id + "__vs__" + base_id;
  const fs::path run_dir = fs::path{options.run_root} / run_id;
  fs::create_directories(run_dir);

  options.series.engine_one_command = shell_quote(new_engine.string());
  options.series.engine_two_command = shell_quote(base_engine.string());

  write_command_file(run_dir / "command.txt", options, new_engine, base_engine);

  std::ofstream log{run_dir / "runner.log"};
  if (!log) {
    throw std::runtime_error{"failed to write " + (run_dir / "runner.log").string()};
  }

  log << "eval"
      << " run_id=" << run_id << " kind=" << options.kind << " new_id=" << new_id
      << " base_id=" << base_id << '\n';
  log << "series"
      << " games=" << options.series.games << " timeout_ms=" << options.series.move_timeout.count()
      << " alternate_sides=" << (options.series.alternate_sides ? 1 : 0) << " opening_book="
      << (options.series.opening_book.path.empty() ? "none" : options.series.opening_book.path)
      << " opening_count=" << options.series.opening_book.lines.size()
      << " sprt_stop=" << (options.series.sprt_stop ? 1 : 0)
      << " sprt_null=" << options.series.sprt_null_rate
      << " sprt_alt=" << options.series.sprt_alt_rate << " sprt_alpha=" << options.series.sprt_alpha
      << " sprt_beta=" << options.series.sprt_beta;
  print_go_limits(options.series.go_limits, log);
  log << '\n';
  log << "engine engine_one " << new_engine.string() << '\n';
  log << "engine engine_two " << base_engine.string() << '\n';

  const poe2::match_runner::SeriesResult result =
      poe2::match_runner::run_process_series(options.series, log);
  poe2::match_runner::print_series_result(result, log);
  log.close();

  write_summary_json(run_dir / "summary.json", result);
  write_games_csv(run_dir / "games.csv", result);
  write_manifest_json(run_dir / "manifest.json", options, result, timestamp, run_id, new_id,
                      base_id, new_engine, base_engine);
  if (options.write_ledger) {
    append_ledger_row(options.ledger, options, result, timestamp, run_id, new_id, base_id, run_dir);
  }

  std::cout << "eval_run"
            << " run_dir=" << run_dir.string()
            << " ledger=" << (options.write_ledger ? options.ledger : "disabled") << '\n';
  poe2::match_runner::print_series_result(result, std::cout);

  if (options.require_accept_alt &&
      result.sprt_decision != poe2::match_runner::SprtDecision::kAcceptAlternative) {
    std::cerr << "eval_gate sprt_decision="
              << poe2::match_runner::sprt_decision_name(result.sprt_decision)
              << " required=accept_alt\n";
    return 2;
  }

  return 0;
}

int run_openings(int argc, char** argv) {
  if (argc <= 0) {
    throw std::invalid_argument{"missing openings command"};
  }

  const std::string_view command = argv[0];
  if (command == "generate-random" || command == "generate") {
    const OpeningGenerateOptions options = parse_opening_generate_options(argc - 1, argv + 1);
    write_generated_opening_book(options);
    const poe2::match_runner::OpeningBook book =
        poe2::match_runner::load_opening_book(options.output_path.string());
    std::cout << "openings_generated"
              << " path=" << options.output_path.string() << " count=" << book.lines.size()
              << " plies=" << options.plies << " seed=" << options.seed
              << " max_score_gap=" << options.max_score_gap << '\n';
    return 0;
  }
  if (command == "generate-systematic") {
    const OpeningSystematicOptions options = parse_opening_systematic_options(argc - 1, argv + 1);
    write_systematic_opening_book(options);
    const poe2::match_runner::OpeningBook book =
        poe2::match_runner::load_opening_book(options.output_path.string());
    std::cout << "openings_generated"
              << " path=" << options.output_path.string() << " count=" << book.lines.size()
              << " plies=" << options.plies << " mode=systematic-symmetry\n";
    return 0;
  }

  throw std::invalid_argument{"unknown openings command: " + std::string{command}};
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
    return run_eval(argc - 2, argv + 2);
  }
  if (command == "openings") {
    return run_openings(argc - 2, argv + 2);
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
