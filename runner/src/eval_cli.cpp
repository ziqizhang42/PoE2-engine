#include "eval_cli.hpp"

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "poe2/match_runner.hpp"

namespace poe2::runner_cli {

namespace {

namespace fs = std::filesystem;

struct EvalOptions {
  match_runner::SeriesOptions series;
  std::string new_build;
  std::string base;
  std::string new_engine;
  std::string base_engine;
  std::string new_engine_args;
  std::string base_engine_args;
  std::string preset = "release";
  std::string run_root = "build/eval/runs";
  std::string ledger = "eval/results.csv";
  std::string kind = "gate";
  std::string opening_seed_source;
  std::string opening_book_digest;
  std::uint64_t resolved_opening_seed = 0;
  bool write_ledger = true;
  bool require_accept_alt = false;
};

struct ReservedRunDirectory {
  std::string run_id;
  fs::path path;
};

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

[[nodiscard]] std::optional<double> parse_finite_double(std::string_view text) {
  std::istringstream input{std::string{text}};
  double value = 0.0;
  input >> value;
  if (!input || !std::isfinite(value)) {
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

[[nodiscard]] std::optional<std::uint64_t> parse_u64(std::string_view text) noexcept {
  std::uint64_t value = 0;
  const char* begin = text.data();
  const char* end = begin + text.size();
  const auto result = std::from_chars(begin, end, value);
  if (result.ec != std::errc{} || result.ptr != end) {
    return std::nullopt;
  }
  return value;
}

bool parse_go_limit_option(std::string_view argument, int& index, int argc, char** argv,
                           engine::EngineLimits& limits) {
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

void print_go_limits(const engine::EngineLimits& limits, std::ostream& output) {
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

[[nodiscard]] std::string engine_command(const fs::path& engine_path, std::string_view args) {
  std::string command = shell_quote(engine_path.string());
  if (!args.empty()) {
    command += ' ';
    command += args;
  }

  return command;
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

class LedgerFileLock {
 public:
  explicit LedgerFileLock(const fs::path& ledger_path) {
    fs::path lock_path = ledger_path;
    lock_path += ".lock";
    descriptor_ = ::open(lock_path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0666);
    if (descriptor_ < 0) {
      throw std::runtime_error{"failed to open ledger lock " + lock_path.string() + ": " +
                               std::strerror(errno)};
    }
    while (::flock(descriptor_, LOCK_EX) != 0) {
      if (errno == EINTR) {
        continue;
      }
      const int error = errno;
      ::close(descriptor_);
      descriptor_ = -1;
      throw std::runtime_error{"failed to lock ledger " + ledger_path.string() + ": " +
                               std::strerror(error)};
    }
  }

  LedgerFileLock(const LedgerFileLock&) = delete;
  LedgerFileLock& operator=(const LedgerFileLock&) = delete;

  ~LedgerFileLock() {
    if (descriptor_ >= 0) {
      ::flock(descriptor_, LOCK_UN);
      ::close(descriptor_);
    }
  }

 private:
  int descriptor_ = -1;
};

[[nodiscard]] std::string score_text(double value) {
  std::ostringstream output;
  output << std::fixed << std::setprecision(3) << value;
  return output.str();
}

[[nodiscard]] std::string round_trip_text(double value) {
  std::array<char, 64> buffer{};
  const auto [end, error] = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
  if (error != std::errc{}) {
    throw std::runtime_error{"failed to serialize a floating-point evaluation setting"};
  }
  return std::string{buffer.data(), end};
}

[[nodiscard]] std::string fnv1a64_hex(std::string_view text) {
  std::uint64_t hash = 14695981039346656037ULL;
  for (const char ch : text) {
    hash ^= static_cast<unsigned char>(ch);
    hash *= 1099511628211ULL;
  }

  std::ostringstream output;
  output << std::hex << std::setw(16) << std::setfill('0') << hash;
  return output.str();
}

[[nodiscard]] std::uint64_t fnv1a64(std::string_view text) noexcept {
  std::uint64_t hash = 14695981039346656037ULL;
  for (const char ch : text) {
    hash ^= static_cast<unsigned char>(ch);
    hash *= 1099511628211ULL;
  }
  return hash;
}

[[nodiscard]] std::uint64_t derive_opening_seed(std::string_view new_id, std::string_view base_id,
                                                std::string_view book_digest,
                                                std::string_view kind) {
  std::string material;
  const auto append_part = [&](std::string_view part) {
    material += std::to_string(part.size());
    material += ':';
    material += part;
    material += '|';
  };
  append_part(new_id);
  append_part(base_id);
  append_part(book_digest);
  append_part(kind);
  return fnv1a64(material);
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

[[nodiscard]] std::string opening_book_digest(const match_runner::OpeningBook& opening_book) {
  if (opening_book.path.empty()) {
    return {};
  }

  return file_digest(opening_book.path);
}

[[nodiscard]] std::string engine_args_run_id_suffix(const EvalOptions& options) {
  std::string suffix;
  if (!options.new_engine_args.empty()) {
    suffix += "__newargs_";
    suffix += fnv1a64_hex(options.new_engine_args);
  }
  if (!options.base_engine_args.empty()) {
    suffix += "__baseargs_";
    suffix += fnv1a64_hex(options.base_engine_args);
  }

  return suffix;
}

[[nodiscard]] std::string run_id_component(std::string_view text) {
  std::string component;
  component.reserve(text.size());
  for (const char ch : text) {
    const bool safe = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                      (ch >= '0' && ch <= '9') || ch == '-' || ch == '_' || ch == '.';
    component += safe ? ch : '_';
  }
  return component.empty() ? "unknown" : component;
}

[[nodiscard]] ReservedRunDirectory reserve_run_directory(const fs::path& run_root,
                                                         const std::string& base_run_id) {
  for (int attempt = 0; attempt < 1000; ++attempt) {
    std::string run_id = base_run_id;
    if (attempt != 0) {
      run_id += "__r";
      run_id += std::to_string(attempt + 1);
    }

    fs::path run_dir = run_root / run_id;
    std::error_code error;
    if (fs::create_directories(run_dir, error)) {
      return ReservedRunDirectory{.run_id = std::move(run_id), .path = std::move(run_dir)};
    }
    if (error) {
      throw std::runtime_error{"failed to create " + run_dir.string() + ": " + error.message()};
    }
  }

  throw std::runtime_error{"failed to reserve a unique eval run directory under " +
                           run_root.string()};
}

void write_command_file(const fs::path& path, const EvalOptions& options,
                        const fs::path& new_engine, const fs::path& base_engine) {
  std::ofstream output{path};
  if (!output) {
    throw std::runtime_error{"failed to write " + path.string()};
  }

  output << "poe2_runner eval"
         << " --new-build " << shell_quote(options.new_build) << " --base "
         << shell_quote(options.base) << " --new-engine " << shell_quote(options.new_engine)
         << " --base-engine " << shell_quote(options.base_engine);
  if (!options.new_engine_args.empty()) {
    output << " --new-engine-args " << shell_quote(options.new_engine_args);
  }
  if (!options.base_engine_args.empty()) {
    output << " --base-engine-args " << shell_quote(options.base_engine_args);
  }
  output << " --preset " << shell_quote(options.preset) << " --kind " << shell_quote(options.kind)
         << " --run-root " << shell_quote(options.run_root);
  if (options.write_ledger) {
    output << " --ledger " << shell_quote(options.ledger);
  } else {
    output << " --no-ledger";
  }
  output << " --games " << options.series.games << " --workers " << options.series.workers
         << " --timeout-ms " << options.series.move_timeout.count();
  if (!options.series.opening_book.path.empty()) {
    output << " --opening-book " << shell_quote(options.series.opening_book.path);
  }
  if (options.series.shuffle_openings) {
    output << " --shuffle-openings";
  }
  if (options.series.opening_seed.has_value()) {
    output << " --opening-seed " << *options.series.opening_seed;
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
  if (options.series.sequential_stop) {
    output << " --sequential-stop";
  }
  output << " --sequential-null " << round_trip_text(options.series.sequential_null_nelo)
         << " --sequential-alt " << round_trip_text(options.series.sequential_alt_nelo)
         << " --sequential-alpha " << round_trip_text(options.series.sequential_alpha)
         << " --sequential-beta " << round_trip_text(options.series.sequential_beta);
  if (options.require_accept_alt) {
    output << " --require-accept-alt";
  }
  output << '\n';
  output << "new_engine_path=" << new_engine.string() << '\n';
  if (!options.new_engine_args.empty()) {
    output << "new_engine_args=" << options.new_engine_args << '\n';
  }
  output << "base_engine_path=" << base_engine.string() << '\n';
  if (!options.base_engine_args.empty()) {
    output << "base_engine_args=" << options.base_engine_args << '\n';
  }
}

void write_summary_json(const fs::path& path, const match_runner::SeriesResult& result) {
  std::ofstream output{path};
  if (!output) {
    throw std::runtime_error{"failed to write " + path.string()};
  }

  output << "{\n"
         << "  \"analysis_version\": " << match_runner::kAnalysisVersion << ",\n"
         << "  \"valid\": " << (result.valid ? "true" : "false") << ",\n"
         << "  \"invalid_reason\": ";
  if (result.invalid_reason.empty()) {
    output << "null";
  } else {
    output << "\"" << json_escape(result.invalid_reason) << "\"";
  }
  output << ",\n"
         << "  \"games_requested\": " << result.games_requested << ",\n"
         << "  \"games_played\": " << result.games_played << ",\n"
         << "  \"workers_requested\": " << result.workers_requested << ",\n"
         << "  \"workers_used\": " << result.workers_used << ",\n"
         << "  \"games_discarded\": " << result.games_discarded << ",\n"
         << "  \"unique_openings\": " << result.unique_openings_used << ",\n"
         << "  \"engine_one_wins\": " << result.engine_one_wins << ",\n"
         << "  \"engine_two_wins\": " << result.engine_two_wins << ",\n"
         << "  \"no_winner\": " << result.no_winner << ",\n"
         << "  \"reasons\": {\n"
         << "    \"normal\": " << result.normal_games << ",\n"
         << "    \"timeout\": " << result.timeout_games << ",\n"
         << "    \"disconnected\": " << result.disconnected_games << ",\n"
         << "    \"malformed_move\": " << result.malformed_move_games << ",\n"
         << "    \"illegal_move\": " << result.illegal_move_games << ",\n"
         << "    \"protocol_error\": " << result.protocol_error_games << "\n"
         << "  },\n"
         << "  \"statistical_unit\": \""
         << match_runner::statistical_unit_name(result.statistical_unit) << "\",\n"
         << "  \"statistical_samples\": " << result.statistical_samples << ",\n"
         << "  \"statistical_games\": " << result.statistical_games << ",\n"
         << "  \"unpaired_games\": " << result.games_played - result.statistical_games << ",\n"
         << "  \"statistical_score_rate_counts\": {\n"
         << "    \"0\": " << result.statistical_score_counts[0] << ",\n"
         << "    \"0.25\": " << result.statistical_score_counts[1] << ",\n"
         << "    \"0.5\": " << result.statistical_score_counts[2] << ",\n"
         << "    \"0.75\": " << result.statistical_score_counts[3] << ",\n"
         << "    \"1\": " << result.statistical_score_counts[4] << "\n"
         << "  },\n"
         << "  \"engine_one_score_rate\": " << score_text(result.engine_one_result_rate) << ",\n"
         << "  \"normalized_elo\": ";
  if (result.normalized_elo.has_value()) {
    output << score_text(*result.normalized_elo);
  } else {
    output << "null";
  }
  output << ",\n"
         << "  \"confidence_method\": \""
         << match_runner::confidence_method_name(result.statistical_unit) << "\",\n"
         << "  \"confidence_low\": " << score_text(result.confidence_low) << ",\n"
         << "  \"confidence_high\": " << score_text(result.confidence_high) << ",\n"
         << "  \"betting_log_evidence_above_even\": "
         << score_text(result.betting_log_evidence_above_even) << ",\n"
         << "  \"betting_log_evidence_below_even\": "
         << score_text(result.betting_log_evidence_below_even) << ",\n"
         << "  \"sequential_test_method\": \""
         << match_runner::sequential_test_method_name(result.statistical_unit) << "\",\n"
         << "  \"sequential_bound_unit\": \"normalized_elo\",\n"
         << "  \"sequential_null\": " << round_trip_text(result.sequential_null_nelo) << ",\n"
         << "  \"sequential_alt\": " << round_trip_text(result.sequential_alt_nelo) << ",\n"
         << "  \"sequential_alpha\": " << round_trip_text(result.sequential_alpha) << ",\n"
         << "  \"sequential_beta\": " << round_trip_text(result.sequential_beta) << ",\n"
         << "  \"sequential_llr\": " << score_text(result.sequential_llr) << ",\n"
         << "  \"sequential_lower_bound\": " << score_text(result.sequential_lower_bound) << ",\n"
         << "  \"sequential_upper_bound\": " << score_text(result.sequential_upper_bound) << ",\n"
         << "  \"sequential_decision\": \""
         << match_runner::sequential_decision_name(result.sequential_decision) << "\"\n"
         << "}\n";
}

void write_games_csv(const fs::path& path, const match_runner::SeriesResult& result) {
  std::ofstream output{path};
  if (!output) {
    throw std::runtime_error{"failed to write " + path.string()};
  }

  output << "game,engine_one_player,opening_slot,opening_line,opening_moves,winner,reason,plies,"
            "player_one_score,"
            "player_two_score\n";
  for (const match_runner::SeriesGameResult& game : result.games) {
    output << game.game_number << ',' << match_runner::player_name(game.engine_one_player) << ','
           << game.opening_slot << ',' << game.opening_line_number << ',';
    write_csv_field(output, game.opening_moves);
    output << ',' << match_runner::engine_name_from_winner(game) << ','
           << match_runner::reason_name(game.match.reason) << ',' << game.match.moves.size() << ','
           << game.match.scores.player_one << ',' << game.match.scores.player_two << '\n';
  }
}

void write_manifest_json(const fs::path& path, const EvalOptions& options,
                         const match_runner::SeriesResult& result, std::string_view timestamp,
                         std::string_view run_id, std::string_view new_id, std::string_view base_id,
                         const fs::path& new_engine_path, const fs::path& base_engine_path) {
  std::ofstream output{path};
  if (!output) {
    throw std::runtime_error{"failed to write " + path.string()};
  }

  output << "{\n"
         << "  \"run_id\": \"" << json_escape(run_id) << "\",\n"
         << "  \"created_at_utc\": \"" << json_escape(timestamp) << "\",\n"
         << "  \"kind\": \"" << json_escape(options.kind) << "\",\n"
         << "  \"new_id\": \"" << json_escape(new_id) << "\",\n"
         << "  \"new_engine\": \"" << json_escape(options.new_engine) << "\",\n"
         << "  \"new_engine_args\": \"" << json_escape(options.new_engine_args) << "\",\n"
         << "  \"base_id\": \"" << json_escape(base_id) << "\",\n"
         << "  \"base_engine\": \"" << json_escape(options.base_engine) << "\",\n"
         << "  \"base_engine_args\": \"" << json_escape(options.base_engine_args) << "\",\n"
         << "  \"new_engine_path\": \"" << json_escape(new_engine_path.string()) << "\",\n"
         << "  \"base_engine_path\": \"" << json_escape(base_engine_path.string()) << "\",\n"
         << "  \"opening_book\": ";
  if (options.series.opening_book.path.empty()) {
    output << "null";
  } else {
    output << "\"" << json_escape(options.series.opening_book.path) << "\"";
  }
  output << ",\n";

  const std::string& book_digest = options.opening_book_digest;
  output << "  \"opening_book_digest\": ";
  if (book_digest.empty()) {
    output << "null";
  } else {
    output << "\"" << json_escape(book_digest) << "\"";
  }
  output << ",\n"
         << "  \"analysis_version\": " << match_runner::kAnalysisVersion << ",\n"
         << "  \"statistical_unit\": \""
         << match_runner::statistical_unit_name(result.statistical_unit) << "\",\n"
         << "  \"confidence_method\": \""
         << match_runner::confidence_method_name(result.statistical_unit) << "\",\n"
         << "  \"sequential_test_method\": \""
         << match_runner::sequential_test_method_name(result.statistical_unit) << "\",\n"
         << "  \"opening_count\": " << options.series.opening_book.lines.size() << ",\n"
         << "  \"unique_openings\": " << result.unique_openings_used << ",\n"
         << "  \"shuffle_openings\": " << (options.series.shuffle_openings ? "true" : "false")
         << ",\n"
         << "  \"opening_sampling\": \"deterministic_shuffle_without_replacement\",\n"
         << "  \"opening_seed\": " << options.resolved_opening_seed << ",\n"
         << "  \"opening_seed_source\": \"" << json_escape(options.opening_seed_source) << "\",\n"
         << "  \"games\": " << options.series.games << ",\n"
         << "  \"games_played\": " << result.games_played << ",\n"
         << "  \"workers_requested\": " << result.workers_requested << ",\n"
         << "  \"workers_used\": " << result.workers_used << ",\n"
         << "  \"games_discarded\": " << result.games_discarded << ",\n"
         << "  \"valid\": " << (result.valid ? "true" : "false") << ",\n"
         << "  \"invalid_reason\": ";
  if (result.invalid_reason.empty()) {
    output << "null";
  } else {
    output << "\"" << json_escape(result.invalid_reason) << "\"";
  }
  output << ",\n"
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
         << "  \"sequential_stop\": " << (options.series.sequential_stop ? "true" : "false")
         << ",\n"
         << "  \"sequential_bound_unit\": \"normalized_elo\",\n"
         << "  \"sequential_null\": " << round_trip_text(options.series.sequential_null_nelo)
         << ",\n"
         << "  \"sequential_alt\": " << round_trip_text(options.series.sequential_alt_nelo) << ",\n"
         << "  \"sequential_alpha\": " << round_trip_text(options.series.sequential_alpha) << ",\n"
         << "  \"sequential_beta\": " << round_trip_text(options.series.sequential_beta) << ",\n"
         << "  \"sequential_llr\": " << score_text(result.sequential_llr) << ",\n"
         << "  \"sequential_lower_bound\": " << score_text(result.sequential_lower_bound) << ",\n"
         << "  \"sequential_upper_bound\": " << score_text(result.sequential_upper_bound) << ",\n"
         << "  \"normalized_elo\": ";
  if (result.normalized_elo.has_value()) {
    output << score_text(*result.normalized_elo);
  } else {
    output << "null";
  }
  output << ",\n"
         << "  \"betting_log_evidence_above_even\": "
         << score_text(result.betting_log_evidence_above_even) << ",\n"
         << "  \"betting_log_evidence_below_even\": "
         << score_text(result.betting_log_evidence_below_even) << ",\n"
         << "  \"sequential_decision\": \""
         << match_runner::sequential_decision_name(result.sequential_decision) << "\"\n"
         << "}\n";
}

constexpr std::string_view kLegacyLedgerHeader =
    "run_id,created_at_utc,kind,new_id,new_engine,new_engine_args,base_id,base_engine,"
    "base_engine_args,games_requested,games_played,"
    "engine_one_wins,engine_two_wins,no_winner,engine_one_score_pct,confidence_low_pct,"
    "confidence_high_pct,sequential_decision,sequential_null,sequential_alt,opening_book,opening_"
    "book_digest,"
    "opening_count,go_depth,go_movetime_ms,go_nodes,timeout_ms,run_dir,analysis_version,"
    "statistical_unit,statistical_samples,statistical_games,pair_score_0,pair_score_0_5,"
    "pair_score_1,pair_score_1_5,pair_score_2,confidence_method,sequential_test_method,"
    "analysis_note";

constexpr std::string_view kPreviousLedgerHeader =
    "run_id,created_at_utc,kind,new_id,new_engine,new_engine_args,base_id,base_engine,"
    "base_engine_args,games_requested,games_played,"
    "engine_one_wins,engine_two_wins,no_winner,engine_one_score_pct,confidence_low_pct,"
    "confidence_high_pct,sequential_decision,sequential_null,sequential_alt,opening_book,opening_"
    "book_digest,"
    "opening_count,go_depth,go_movetime_ms,go_nodes,timeout_ms,run_dir,analysis_version,"
    "statistical_unit,statistical_samples,statistical_games,pair_score_0,pair_score_0_5,"
    "pair_score_1,pair_score_1_5,pair_score_2,confidence_method,sequential_test_method,"
    "analysis_note,valid,invalid_reason,opening_seed,opening_seed_source,opening_sampling,"
    "unique_openings,sequential_model,sequential_bound_unit,sequential_llr,sequential_lower_bound,"
    "sequential_upper_bound,normalized_elo,betting_log_evidence_above_even,"
    "betting_log_evidence_below_even";

constexpr std::string_view kLedgerHeader =
    "run_id,created_at_utc,kind,new_id,new_engine,new_engine_args,base_id,base_engine,"
    "base_engine_args,games_requested,games_played,"
    "engine_one_wins,engine_two_wins,no_winner,engine_one_score_pct,confidence_low_pct,"
    "confidence_high_pct,sequential_decision,sequential_null,sequential_alt,opening_book,opening_"
    "book_digest,"
    "opening_count,go_depth,go_movetime_ms,go_nodes,timeout_ms,run_dir,analysis_version,"
    "statistical_unit,statistical_samples,statistical_games,pair_score_0,pair_score_0_5,"
    "pair_score_1,pair_score_1_5,pair_score_2,confidence_method,sequential_test_method,"
    "analysis_note,valid,invalid_reason,opening_seed,opening_seed_source,opening_sampling,"
    "unique_openings,sequential_model,sequential_bound_unit,sequential_llr,sequential_lower_bound,"
    "sequential_upper_bound,normalized_elo,betting_log_evidence_above_even,"
    "betting_log_evidence_below_even,workers_requested,workers_used,games_discarded";

[[nodiscard]] std::vector<std::string> parse_csv_fields(std::string_view row) {
  std::vector<std::string> fields;
  std::string field;
  bool quoted = false;
  for (std::size_t index = 0; index < row.size(); ++index) {
    const char ch = row[index];
    if (ch == '"') {
      if (quoted && index + 1 < row.size() && row[index + 1] == '"') {
        field += '"';
        ++index;
      } else {
        quoted = !quoted;
      }
    } else if (ch == ',' && !quoted) {
      fields.push_back(std::move(field));
      field.clear();
    } else {
      field += ch;
    }
  }
  if (quoted) {
    throw std::runtime_error{"unterminated quoted field in evaluation ledger"};
  }
  fields.push_back(std::move(field));
  return fields;
}

void migrate_ledger_if_needed(const fs::path& path) {
  if (!is_file(path) || fs::file_size(path) == 0) {
    return;
  }

  std::ifstream input{path};
  std::string header;
  std::getline(input, header);
  if (header == kLedgerHeader) {
    return;
  }
  const bool is_legacy = header == kLegacyLedgerHeader;
  if (!is_legacy && header != kPreviousLedgerHeader) {
    throw std::runtime_error{"unsupported ledger schema: " + path.string()};
  }

  const std::vector<std::string> header_fields = parse_csv_fields(header);
  std::size_t method_index = 0;
  if (is_legacy) {
    const auto method_position =
        std::find(header_fields.begin(), header_fields.end(), "sequential_test_method");
    if (method_position == header_fields.end()) {
      throw std::runtime_error{"legacy ledger is missing sequential_test_method: " + path.string()};
    }
    method_index = static_cast<std::size_t>(std::distance(header_fields.begin(), method_position));
  }

  fs::path temporary = path;
  temporary += ".migration.tmp";
  std::ofstream output{temporary, std::ios::trunc};
  if (!output) {
    throw std::runtime_error{"failed to write " + temporary.string()};
  }
  output << kLedgerHeader << '\n';

  std::string row;
  while (std::getline(input, row)) {
    if (row.empty()) {
      continue;
    }
    const std::vector<std::string> fields = parse_csv_fields(row);
    if (fields.size() != header_fields.size()) {
      throw std::runtime_error{"legacy ledger row has the wrong number of fields: " +
                               path.string()};
    }
    output << row;
    if (is_legacy) {
      output << ',';
      const std::array<std::string, 14> additions{
          "true", "", "", "", "", "", fields[method_index], "score_rate", "", "", "", "", "", ""};
      for (std::size_t index = 0; index < additions.size(); ++index) {
        if (index != 0) {
          output << ',';
        }
        write_csv_field(output, additions[index]);
      }
    }
    output << ",1,1,0\n";
  }
  output.close();
  if (!output) {
    throw std::runtime_error{"failed to finish " + temporary.string()};
  }
  fs::rename(temporary, path);
}

void append_ledger_row(const fs::path& path, const EvalOptions& options,
                       const match_runner::SeriesResult& result, std::string_view timestamp,
                       std::string_view run_id, std::string_view new_id, std::string_view base_id,
                       const fs::path& run_dir, bool coordinate_writers = false) {
  create_parent_directories(path);
  std::optional<LedgerFileLock> ledger_lock;
  if (coordinate_writers) {
    ledger_lock.emplace(path);
  }
  migrate_ledger_if_needed(path);
  const bool needs_header = !is_file(path) || fs::file_size(path) == 0;
  if (!needs_header) {
    std::ifstream existing{path};
    std::string existing_header;
    std::getline(existing, existing_header);
    if (existing_header != kLedgerHeader) {
      throw std::runtime_error{"unsupported ledger schema: " + path.string()};
    }
  }
  std::ofstream output{path, std::ios::app};
  if (!output) {
    throw std::runtime_error{"failed to append " + path.string()};
  }
  if (needs_header) {
    output << kLedgerHeader << '\n';
  }

  write_csv_field(output, run_id);
  output << ',';
  write_csv_field(output, timestamp);
  output << ',';
  write_csv_field(output, options.kind);
  output << ',';
  write_csv_field(output, new_id);
  output << ',';
  write_csv_field(output, options.new_engine);
  output << ',';
  write_csv_field(output, options.new_engine_args);
  output << ',';
  write_csv_field(output, base_id);
  output << ',';
  write_csv_field(output, options.base_engine);
  output << ',';
  write_csv_field(output, options.base_engine_args);
  output << ',' << result.games_requested << ',' << result.games_played << ','
         << result.engine_one_wins << ',' << result.engine_two_wins << ',' << result.no_winner
         << ',' << score_text(result.engine_one_result_rate * 100.0) << ','
         << score_text(result.confidence_low * 100.0) << ','
         << score_text(result.confidence_high * 100.0) << ',';
  write_csv_field(output, match_runner::sequential_decision_name(result.sequential_decision));
  output << ',' << round_trip_text(options.series.sequential_null_nelo) << ','
         << round_trip_text(options.series.sequential_alt_nelo) << ',';
  write_csv_field(output, options.series.opening_book.path);
  output << ',';
  write_csv_field(output, options.opening_book_digest);
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
  output << ',' << match_runner::kAnalysisVersion << ',';
  write_csv_field(output, match_runner::statistical_unit_name(result.statistical_unit));
  output << ',' << result.statistical_samples << ',' << result.statistical_games << ',';
  if (result.statistical_unit == match_runner::StatisticalUnit::kOpeningPair) {
    output << result.statistical_score_counts[0] << ',' << result.statistical_score_counts[1] << ','
           << result.statistical_score_counts[2] << ',' << result.statistical_score_counts[3] << ','
           << result.statistical_score_counts[4];
  } else {
    output << ",,,,";
  }
  output << ',';
  write_csv_field(output, match_runner::confidence_method_name(result.statistical_unit));
  output << ',';
  write_csv_field(output, match_runner::sequential_test_method_name(result.statistical_unit));
  output << ',';
  write_csv_field(output, "paired normalized-Elo GSPRT");
  output << ',' << (result.valid ? "true" : "false") << ',';
  write_csv_field(output, result.invalid_reason);
  output << ',' << options.resolved_opening_seed << ',';
  write_csv_field(output, options.opening_seed_source);
  output << ',';
  write_csv_field(output, "deterministic_shuffle_without_replacement");
  output << ',' << result.unique_openings_used << ',';
  write_csv_field(output, match_runner::sequential_test_method_name(result.statistical_unit));
  output << ',';
  write_csv_field(output, "normalized_elo");
  output << ',' << score_text(result.sequential_llr) << ','
         << score_text(result.sequential_lower_bound) << ','
         << score_text(result.sequential_upper_bound) << ',';
  if (result.normalized_elo.has_value()) {
    output << score_text(*result.normalized_elo);
  }
  output << ',' << score_text(result.betting_log_evidence_above_even) << ','
         << score_text(result.betting_log_evidence_below_even) << ',' << result.workers_requested
         << ',' << result.workers_used << ',' << result.games_discarded;
  output << '\n';
}

[[nodiscard]] EvalOptions parse_eval_options(int argc, char** argv) {
  EvalOptions options;
  options.series.games = 2000;
  options.series.move_timeout = std::chrono::milliseconds{1000};
  options.series.go_limits.move_time = std::chrono::milliseconds{100};
  options.series.print_game_results = false;
  options.series.require_normal_games = true;

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
    if (argument == "--new-engine") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--new-engine requires a binary name"};
      }
      options.new_engine = argv[++index];
      continue;
    }
    if (argument == "--new-engine-args") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--new-engine-args requires an argument string"};
      }
      options.new_engine_args = argv[++index];
      continue;
    }
    if (argument == "--base-engine") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--base-engine requires a binary name"};
      }
      options.base_engine = argv[++index];
      continue;
    }
    if (argument == "--base-engine-args") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--base-engine-args requires an argument string"};
      }
      options.base_engine_args = argv[++index];
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
    if (argument == "--workers") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--workers requires a positive integer"};
      }
      const std::optional<int> workers = parse_positive_int(argv[++index]);
      if (!workers.has_value()) {
        throw std::invalid_argument{"--workers requires a positive integer"};
      }
      options.series.workers = *workers;
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
      options.series.opening_book = match_runner::load_opening_book(argv[++index]);
      continue;
    }
    if (argument == "--shuffle-openings") {
      options.series.shuffle_openings = true;
      continue;
    }
    if (argument == "--opening-seed") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--opening-seed requires an unsigned integer"};
      }
      const std::optional<std::uint64_t> seed = parse_u64(argv[++index]);
      if (!seed.has_value()) {
        throw std::invalid_argument{"--opening-seed requires an unsigned integer"};
      }
      options.series.opening_seed = seed;
      options.opening_seed_source = "explicit";
      continue;
    }
    if (argument == "--sequential-stop") {
      options.series.sequential_stop = true;
      continue;
    }
    if (argument == "--sequential-null") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--sequential-null requires a finite normalized-Elo value"};
      }
      const std::optional<double> nelo = parse_finite_double(argv[++index]);
      if (!nelo.has_value()) {
        throw std::invalid_argument{"--sequential-null requires a finite normalized-Elo value"};
      }
      options.series.sequential_null_nelo = *nelo;
      continue;
    }
    if (argument == "--sequential-alt") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--sequential-alt requires a finite normalized-Elo value"};
      }
      const std::optional<double> nelo = parse_finite_double(argv[++index]);
      if (!nelo.has_value()) {
        throw std::invalid_argument{"--sequential-alt requires a finite normalized-Elo value"};
      }
      options.series.sequential_alt_nelo = *nelo;
      continue;
    }
    if (argument == "--sequential-alpha") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--sequential-alpha requires a probability between 0 and 1"};
      }
      const std::optional<double> alpha = parse_probability(argv[++index]);
      if (!alpha.has_value()) {
        throw std::invalid_argument{"--sequential-alpha requires a probability between 0 and 1"};
      }
      options.series.sequential_alpha = *alpha;
      continue;
    }
    if (argument == "--sequential-beta") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--sequential-beta requires a probability between 0 and 1"};
      }
      const std::optional<double> beta = parse_probability(argv[++index]);
      if (!beta.has_value()) {
        throw std::invalid_argument{"--sequential-beta requires a probability between 0 and 1"};
      }
      options.series.sequential_beta = *beta;
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
  if (options.new_engine.empty()) {
    throw std::invalid_argument{"missing --new-engine"};
  }
  if (options.base_engine.empty()) {
    throw std::invalid_argument{"missing --base-engine"};
  }
  if (options.series.sequential_alt_nelo <= options.series.sequential_null_nelo) {
    throw std::invalid_argument{"--sequential-alt must be greater than --sequential-null"};
  }
  if (!options.series.alternate_sides) {
    throw std::invalid_argument{"eval requires alternating sides"};
  }
  if ((options.series.games % 2) != 0) {
    throw std::invalid_argument{"eval requires an even --games value for complete opening pairs"};
  }
  if (options.series.opening_book.lines.empty()) {
    throw std::invalid_argument{"eval requires --opening-book"};
  }
  if (!options.series.shuffle_openings) {
    throw std::invalid_argument{"eval requires --shuffle-openings"};
  }
  if (options.series.games / 2 > static_cast<int>(options.series.opening_book.lines.size())) {
    throw std::invalid_argument{"opening book does not contain enough unique openings for --games"};
  }

  return options;
}

}  // namespace

int run_eval(int argc, char** argv) {
  EvalOptions options = parse_eval_options(argc, argv);
  const fs::path new_build = fs::path{options.new_build}.lexically_normal();
  if (!is_dir(new_build)) {
    throw std::invalid_argument{"new build directory does not exist: " + new_build.string()};
  }

  const fs::path new_engine_path = new_build / "engines" / options.new_engine;
  if (!is_file(new_engine_path)) {
    throw std::invalid_argument{"new engine binary does not exist: " + new_engine_path.string()};
  }

  const fs::path base_engine_path =
      resolve_engine_binary(options.base, options.base_engine, options.preset);
  const std::string new_id = build_id_from_build_dir(new_build, options.preset);
  const std::string base_id = id_from_eval_input(options.base, options.preset);
  options.opening_book_digest = opening_book_digest(options.series.opening_book);
  if (!options.series.opening_seed.has_value()) {
    options.series.opening_seed =
        derive_opening_seed(new_id, base_id, options.opening_book_digest, options.kind);
    options.opening_seed_source = "derived";
  }
  options.resolved_opening_seed = options.series.opening_seed.value_or(0);
  const std::string timestamp = utc_timestamp();
  const std::string base_run_id =
      timestamp + "__" + run_id_component(new_id) + "__" + run_id_component(options.new_engine) +
      "__vs__" + run_id_component(base_id) + "__" + run_id_component(options.base_engine) +
      engine_args_run_id_suffix(options);
  const ReservedRunDirectory run = reserve_run_directory(options.run_root, base_run_id);

  options.series.engine_one_command = engine_command(new_engine_path, options.new_engine_args);
  options.series.engine_two_command = engine_command(base_engine_path, options.base_engine_args);

  write_command_file(run.path / "command.txt", options, new_engine_path, base_engine_path);

  std::ofstream log{run.path / "runner.log"};
  if (!log) {
    throw std::runtime_error{"failed to write " + (run.path / "runner.log").string()};
  }

  log << "eval"
      << " run_id=" << run.run_id << " kind=" << options.kind << " new_id=" << new_id
      << " new_engine=" << options.new_engine;
  if (!options.new_engine_args.empty()) {
    log << " new_engine_args=" << shell_quote(options.new_engine_args);
  }
  log << " base_id=" << base_id << " base_engine=" << options.base_engine;
  if (!options.base_engine_args.empty()) {
    log << " base_engine_args=" << shell_quote(options.base_engine_args);
  }
  log << '\n';
  log << "series"
      << " games=" << options.series.games << " workers=" << options.series.workers
      << " timeout_ms=" << options.series.move_timeout.count()
      << " alternate_sides=" << (options.series.alternate_sides ? 1 : 0) << " opening_book="
      << (options.series.opening_book.path.empty() ? "none" : options.series.opening_book.path)
      << " opening_count=" << options.series.opening_book.lines.size()
      << " shuffle_openings=" << (options.series.shuffle_openings ? 1 : 0)
      << " opening_seed=" << options.resolved_opening_seed
      << " opening_seed_source=" << options.opening_seed_source
      << " sequential_stop=" << (options.series.sequential_stop ? 1 : 0)
      << " sequential_null_nelo=" << options.series.sequential_null_nelo
      << " sequential_alt_nelo=" << options.series.sequential_alt_nelo
      << " sequential_alpha=" << options.series.sequential_alpha
      << " sequential_beta=" << options.series.sequential_beta;
  print_go_limits(options.series.go_limits, log);
  log << '\n';
  log << "engine engine_one " << options.series.engine_one_command << '\n';
  log << "engine engine_two " << options.series.engine_two_command << '\n';

  match_runner::SeriesResult result;
  try {
    result = match_runner::run_process_series(options.series, log);
  } catch (const std::exception& error) {
    result.games_requested = options.series.games;
    result.workers_requested = options.series.workers;
    result.valid = false;
    result.invalid_reason = "runner exception: " + std::string{error.what()};
    result.detail = "evaluation invalid: " + result.invalid_reason;
    ++result.protocol_error_games;
    match_runner::analyze_series_result(result, options.series);
  } catch (...) {
    result.games_requested = options.series.games;
    result.workers_requested = options.series.workers;
    result.valid = false;
    result.invalid_reason = "runner exception: unknown";
    result.detail = "evaluation invalid: " + result.invalid_reason;
    ++result.protocol_error_games;
    match_runner::analyze_series_result(result, options.series);
  }
  if (result.valid && result.sequential_decision == match_runner::SequentialDecision::kInvalid) {
    result.valid = false;
    result.invalid_reason = "statistical analysis failed";
    result.detail = "evaluation invalid: " + result.invalid_reason;
    match_runner::analyze_series_result(result, options.series);
  }
  match_runner::print_series_result(result, log);
  log.close();

  write_summary_json(run.path / "summary.json", result);
  write_games_csv(run.path / "games.csv", result);
  write_manifest_json(run.path / "manifest.json", options, result, timestamp, run.run_id, new_id,
                      base_id, new_engine_path, base_engine_path);
  append_ledger_row(run.path / "ledger-row.csv", options, result, timestamp, run.run_id, new_id,
                    base_id, run.path);
  if (options.write_ledger) {
    append_ledger_row(options.ledger, options, result, timestamp, run.run_id, new_id, base_id,
                      run.path, true);
  }

  std::cout << "eval_run"
            << " run_dir=" << run.path.string()
            << " ledger=" << (options.write_ledger ? options.ledger : "disabled") << '\n';
  match_runner::print_series_result(result, std::cout);

  if (!result.valid) {
    std::cerr << "eval_invalid reason=" << result.invalid_reason << '\n';
    return 3;
  }

  if (options.require_accept_alt &&
      result.sequential_decision != match_runner::SequentialDecision::kAcceptAlternative) {
    std::cerr << "eval_gate sequential_decision="
              << match_runner::sequential_decision_name(result.sequential_decision)
              << " required=accept_alt\n";
    return 2;
  }

  return 0;
}

}  // namespace poe2::runner_cli
