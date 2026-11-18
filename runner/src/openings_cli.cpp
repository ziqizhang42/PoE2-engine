#include "openings_cli.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "poe2/match_runner.hpp"
#include "poe2/move.hpp"
#include "poe2/opening_balance.hpp"
#include "poe2/opening_generation.hpp"
#include "poe2/symmetry.hpp"

namespace poe2::runner_cli {

namespace {

namespace fs = std::filesystem;

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

struct OpeningCorpusOptions {
  fs::path development_output_path;
  fs::path holdout_output_path;
  int count = 0;
  std::vector<int> plies;
  int max_score_gap = 4;
  std::uint64_t seed = 0;
};

struct FitBalanceOptions {
  std::vector<fs::path> runs;
  fs::path engine_path;
  fs::path net_path;
  fs::path output_path;
  match_runner::BalanceSearch search;
  int workers = 1;
  int min_openings = 30;
  double threshold = 2.0;
};

struct BalancedCorpusOptions {
  fs::path development_output_path;
  fs::path holdout_output_path;
  fs::path correction_path;
  fs::path engine_path;
  fs::path net_path;
  std::optional<fs::path> previous_development_path;
  std::optional<fs::path> previous_holdout_path;
  std::vector<int> plies;
  int per_length = 0;
  double threshold = 2.0;
  int max_score_gap = 4;
  std::uint64_t seed = 0;
  int workers = 1;
};

constexpr std::chrono::milliseconds kHandshakeTimeout{30000};
constexpr std::size_t kCandidateBatch = 256;

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

void create_parent_directories(const fs::path& path) {
  const fs::path parent = path.parent_path();
  if (!parent.empty()) {
    fs::create_directories(parent);
  }
}

void write_generated_opening_book(const OpeningGenerateOptions& options) {
  if (options.output_path.empty()) {
    throw std::invalid_argument{"missing --out"};
  }
  if (options.count <= 0) {
    throw std::invalid_argument{"--count requires a positive integer"};
  }
  if (options.plies <= 0 || options.plies >= kCellCount) {
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
    std::optional<match_runner::GeneratedOpening> opening = match_runner::generate_random_opening(
        rng, options.plies, options.max_score_gap, seen_positions);
    if (!opening.has_value()) {
      continue;
    }

    output << match_runner::format_opening_moves(opening->moves) << '\n';
    ++generated;
  }

  if (generated != options.count) {
    throw std::runtime_error{"only generated " + std::to_string(generated) + " of " +
                             std::to_string(options.count) + " requested openings"};
  }
}

[[nodiscard]] std::string plies_text(const std::vector<int>& plies) {
  std::ostringstream output;
  for (std::size_t index = 0; index < plies.size(); ++index) {
    if (index != 0) {
      output << ',';
    }
    output << plies[index];
  }
  return output.str();
}

void write_corpus_partition(const fs::path& path, std::string_view partition,
                            const OpeningCorpusOptions& options,
                            const std::vector<match_runner::GeneratedOpening>& openings) {
  std::ofstream output{path, std::ios::trunc};
  if (!output) {
    throw std::runtime_error{"failed to write " + path.string()};
  }

  std::map<int, int> depth_counts;
  for (const match_runner::GeneratedOpening& opening : openings) {
    ++depth_counts[opening.plies];
  }
  output << "# poe2 opening suite\n"
         << "# generator=rule-generated-corpus\n"
         << "# partition=" << partition << "\n"
         << "# seed=" << options.seed << "\n"
         << "# non_empty_count=" << openings.size() << "\n"
         << "# total_count=" << openings.size() + 1 << "\n"
         << "# plies=" << plies_text(options.plies) << "\n"
         << "# max_score_gap=" << options.max_score_gap << "\n";
  for (const auto& [depth, count] : depth_counts) {
    output << "# depth_count_" << depth << '=' << count << '\n';
  }
  output << "startpos\n";
  for (const match_runner::GeneratedOpening& opening : openings) {
    output << match_runner::format_opening_moves(opening.moves) << '\n';
  }
}

void write_opening_corpus(const OpeningCorpusOptions& options) {
  if (options.development_output_path.empty()) {
    throw std::invalid_argument{"missing --development-out"};
  }
  if (options.holdout_output_path.empty()) {
    throw std::invalid_argument{"missing --holdout-out"};
  }
  if (options.development_output_path.lexically_normal() ==
      options.holdout_output_path.lexically_normal()) {
    throw std::invalid_argument{"development and holdout outputs must be different paths"};
  }
  if (options.count < 315 || (options.count % 2) != 0) {
    throw std::invalid_argument{"--count must be an even integer of at least 315"};
  }
  if (options.plies.size() < 2 || options.plies.front() != 2) {
    throw std::invalid_argument{"--plies must include 2 and at least one deeper stratum"};
  }
  if (options.max_score_gap < 0) {
    throw std::invalid_argument{"--max-score-gap must be non-negative"};
  }

  std::map<int, int> target_counts;
  target_counts[2] = 315;
  const int remaining = options.count - 315;
  const int deeper_strata = static_cast<int>(options.plies.size()) - 1;
  const int per_stratum = remaining / deeper_strata;
  int extra = remaining % deeper_strata;
  for (const int depth : options.plies) {
    if (depth == 2) {
      continue;
    }
    target_counts[depth] = per_stratum + (extra > 0 ? 1 : 0);
    extra = std::max(extra - 1, 0);
  }

  std::map<int, std::vector<match_runner::GeneratedOpening>> strata;
  strata[2] = match_runner::generate_systematic_two_ply_openings();
  std::unordered_set<std::string> seen_positions;
  for (const match_runner::GeneratedOpening& opening : strata[2]) {
    seen_positions.insert(match_runner::canonical_key_text(opening.canonical_key));
  }

  std::mt19937_64 rng{options.seed};
  for (const int depth : options.plies) {
    if (depth == 2) {
      continue;
    }
    const int target = target_counts[depth];
    const int max_attempts = std::max(target * 2000, 100000);
    for (int attempt = 0; attempt < max_attempts && static_cast<int>(strata[depth].size()) < target;
         ++attempt) {
      std::optional<match_runner::GeneratedOpening> opening =
          match_runner::generate_random_opening(rng, depth, options.max_score_gap, seen_positions);
      if (opening.has_value()) {
        strata[depth].push_back(std::move(*opening));
      }
    }
    if (static_cast<int>(strata[depth].size()) != target) {
      throw std::runtime_error{"only generated " + std::to_string(strata[depth].size()) + " of " +
                               std::to_string(target) + " requested openings at depth " +
                               std::to_string(depth)};
    }
  }

  int development_extras = options.count / 2;
  for (const auto& [depth, openings] : strata) {
    static_cast<void>(depth);
    development_extras -= static_cast<int>(openings.size()) / 2;
  }

  std::vector<match_runner::GeneratedOpening> development;
  std::vector<match_runner::GeneratedOpening> holdout;
  development.reserve(static_cast<std::size_t>(options.count / 2));
  holdout.reserve(static_cast<std::size_t>(options.count / 2));
  for (auto& [depth, openings] : strata) {
    for (match_runner::GeneratedOpening& opening : openings) {
      opening.partition_hash =
          match_runner::hash_opening(options.seed, depth, opening.canonical_key);
    }
    std::sort(
        openings.begin(), openings.end(),
        [](const match_runner::GeneratedOpening& lhs, const match_runner::GeneratedOpening& rhs) {
          if (lhs.partition_hash != rhs.partition_hash) {
            return lhs.partition_hash < rhs.partition_hash;
          }
          if (lhs.canonical_key.low != rhs.canonical_key.low) {
            return lhs.canonical_key.low < rhs.canonical_key.low;
          }
          return lhs.canonical_key.high < rhs.canonical_key.high;
        });

    int development_count = static_cast<int>(openings.size()) / 2;
    if ((openings.size() % 2) != 0 && development_extras > 0) {
      ++development_count;
      --development_extras;
    }
    for (int index = 0; index < static_cast<int>(openings.size()); ++index) {
      if (index < development_count) {
        development.push_back(std::move(openings[static_cast<std::size_t>(index)]));
      } else {
        holdout.push_back(std::move(openings[static_cast<std::size_t>(index)]));
      }
    }
  }
  if (development_extras != 0 ||
      development.size() != static_cast<std::size_t>(options.count / 2) ||
      holdout.size() != static_cast<std::size_t>(options.count / 2)) {
    throw std::runtime_error{"failed to balance development and holdout opening partitions"};
  }

  create_parent_directories(options.development_output_path);
  create_parent_directories(options.holdout_output_path);
  fs::path development_temporary = options.development_output_path;
  development_temporary += ".tmp-" + std::to_string(options.seed);
  fs::path holdout_temporary = options.holdout_output_path;
  holdout_temporary += ".tmp-" + std::to_string(options.seed);
  try {
    write_corpus_partition(development_temporary, "development", options, development);
    write_corpus_partition(holdout_temporary, "holdout", options, holdout);
    const match_runner::OpeningBook development_book =
        match_runner::load_opening_book(development_temporary.string());
    const match_runner::OpeningBook holdout_book =
        match_runner::load_opening_book(holdout_temporary.string());
    const std::size_t expected_size = static_cast<std::size_t>(options.count) / 2U + 1U;
    if (development_book.lines.size() != expected_size ||
        holdout_book.lines.size() != expected_size) {
      throw std::runtime_error{"generated corpus failed opening-book validation"};
    }
    fs::rename(development_temporary, options.development_output_path);
    fs::rename(holdout_temporary, options.holdout_output_path);
  } catch (...) {
    std::error_code ignored;
    fs::remove(development_temporary, ignored);
    fs::remove(holdout_temporary, ignored);
    throw;
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
  for (int first_index = 0; first_index < kCellCount; ++first_index) {
    for (int second_index = 0; second_index < kCellCount; ++second_index) {
      if (first_index == second_index) {
        continue;
      }

      Position position;
      const Move first{.square = square_from_index(first_index)};
      const Move second{.square = square_from_index(second_index)};
      if (!position.play(first.square) || !position.play(second.square)) {
        throw std::runtime_error{"failed to generate legal 2-ply opening"};
      }

      const CanonicalPositionKey canonical = canonicalize_position_key(position.key());
      if (!seen_positions.insert(match_runner::canonical_key_text(canonical.key)).second) {
        continue;
      }

      output << format_move(first) << ' ' << format_move(second) << '\n';
      ++generated;
    }
  }

  if (generated != 315) {
    throw std::runtime_error{"generated " + std::to_string(generated) +
                             " systematic 2-ply openings; expected 315"};
  }
}

void write_text_atomically(const fs::path& path, std::string_view text) {
  create_parent_directories(path);
  fs::path temporary = path;
  temporary += ".tmp";
  try {
    {
      std::ofstream output{temporary, std::ios::binary | std::ios::trunc};
      output << text;
      if (!output.flush()) {
        throw std::runtime_error{"failed to write " + temporary.string()};
      }
    }
    fs::rename(temporary, path);
  } catch (...) {
    std::error_code ignored;
    fs::remove(temporary, ignored);
    throw;
  }
}

[[nodiscard]] fs::path run_games_file(const fs::path& run) {
  if (!fs::is_regular_file(run / "summary.json")) {
    throw std::invalid_argument{"run has no summary.json: " + run.string()};
  }
  return run / "games.csv";
}

void fit_balance(const FitBalanceOptions& options) {
  if (options.runs.empty()) {
    throw std::invalid_argument{"fit-balance needs at least one --run"};
  }
  if (options.engine_path.empty()) {
    throw std::invalid_argument{"missing --engine"};
  }
  if (options.output_path.empty()) {
    throw std::invalid_argument{"missing --out"};
  }

  std::vector<fs::path> games_files;
  games_files.reserve(options.runs.size());
  for (const fs::path& run : options.runs) {
    games_files.push_back(run_games_file(run));
  }
  const std::vector<match_runner::OpeningRecord> records =
      match_runner::collect_opening_records(games_files);
  if (records.empty()) {
    throw std::runtime_error{"the runs contain no normal games from an opening"};
  }

  const std::string command = match_runner::balance_engine_command(
      options.engine_path.string(), options.search, options.net_path.string());
  match_runner::CorrectionRecord record{
      .engine = match_runner::query_engine_identity(command, kHandshakeTimeout),
      .search = options.search,
  };
  std::vector<std::vector<std::string>> openings;
  openings.reserve(records.size());
  for (const match_runner::OpeningRecord& opening : records) {
    openings.push_back(opening.moves);
  }
  const std::vector<int> scores =
      match_runner::search_opening_scores(command, options.search.nodes, openings, options.workers);

  std::vector<match_runner::LeadSample> samples;
  samples.reserve(records.size());
  long long games = 0;
  for (std::size_t index = 0; index < records.size(); ++index) {
    const match_runner::OpeningRecord& opening = records[index];
    samples.push_back(match_runner::LeadSample{
        .plies = opening.plies(),
        .points = match_runner::first_player_points(scores[index], opening.plies()),
        .lead = opening.mean_lead(),
    });
    games += opening.games;
  }
  record.correction = match_runner::fit_lead_correction(samples, options.min_openings);
  if (record.correction.by_plies.empty()) {
    throw std::runtime_error{"no opening length has --min-openings openings"};
  }

  std::vector<std::string> notes{"fitted on " + std::to_string(games) + " games of " +
                                 std::to_string(records.size()) + " openings from " +
                                 std::to_string(games_files.size()) + " runs:"};
  for (const fs::path& games_file : games_files) {
    notes.push_back("  " + games_file.parent_path().filename().string() + ' ' +
                    match_runner::file_digest(games_file.string()));
  }
  for (std::string& line : match_runner::describe_held_out(
           match_runner::check_held_out(records, scores, options.threshold, options.min_openings),
           options.threshold)) {
    notes.push_back(std::move(line));
  }

  const std::string text = match_runner::format_correction_record(record, notes);
  write_text_atomically(options.output_path, text);
  std::cout << text << "correction_written path=" << options.output_path.string()
            << " openings=" << records.size() << " games=" << games << '\n';
}

using PreviousPartitions = std::unordered_map<std::string, match_runner::BookPartition>;

[[nodiscard]] PreviousPartitions previous_partitions(const BalancedCorpusOptions& options) {
  PreviousPartitions previous;
  const auto add = [&](const std::optional<fs::path>& path, match_runner::BookPartition partition) {
    if (!path.has_value()) {
      return;
    }
    const match_runner::OpeningBook book = match_runner::load_opening_book(path->string());
    for (const match_runner::OpeningLine& line : book.lines) {
      if (line.moves.empty()) {
        continue;
      }
      const auto [found, inserted] = previous.try_emplace(
          match_runner::canonical_key_text(match_runner::canonical_opening_key(line.moves)),
          partition);
      if (!inserted && found->second != partition) {
        throw std::runtime_error{"a position is in both previous books: " + line.text};
      }
    }
  };
  add(options.previous_development_path, match_runner::BookPartition::kDevelopment);
  add(options.previous_holdout_path, match_runner::BookPartition::kHoldout);
  return previous;
}

struct BalanceContext {
  const BalancedCorpusOptions& options;
  const match_runner::CorrectionRecord& correction;
  const PreviousPartitions& previous;
  std::string command;
};

void offer_candidates(const BalanceContext& context, match_runner::QuotaFiller& filler,
                      const std::vector<match_runner::GeneratedOpening>& candidates) {
  std::vector<std::vector<std::string>> openings;
  openings.reserve(candidates.size());
  for (const match_runner::GeneratedOpening& candidate : candidates) {
    openings.push_back(candidate.moves);
  }
  const std::vector<int> scores = match_runner::search_opening_scores(
      context.command, context.correction.search.nodes, openings, context.options.workers);

  for (std::size_t index = 0; index < candidates.size() && !filler.complete(); ++index) {
    const match_runner::GeneratedOpening& candidate = candidates[index];
    const std::optional<double> predicted = context.correction.correction.predict(
        candidate.plies, match_runner::first_player_points(scores[index], candidate.plies));
    if (!predicted.has_value()) {
      throw std::runtime_error{"the correction has no constants for " +
                               std::to_string(candidate.plies) + " plies"};
    }
    const bool inherited =
        context.previous.contains(match_runner::canonical_key_text(candidate.canonical_key));
    filler.offer(
        match_runner::SelectedOpening{
            .opening = candidate,
            .score = scores[index],
            .predicted_lead = *predicted,
            .inherited = inherited,
        },
        match_runner::assign_partition(context.options.seed, candidate, context.previous));
  }
}

// 2-ply positions are few, so every passing one is kept; deeper lengths draw random lines from
// their own stream until both partitions hold the quota.
[[nodiscard]] match_runner::QuotaFiller select_length(const BalanceContext& context, int plies) {
  const BalancedCorpusOptions& options = context.options;
  if (plies == 2) {
    match_runner::QuotaFiller filler{std::nullopt, options.threshold};
    offer_candidates(context, filler, match_runner::generate_systematic_two_ply_openings());
    return filler;
  }

  match_runner::QuotaFiller filler{options.per_length, options.threshold};
  std::mt19937_64 rng{match_runner::hash_opening(options.seed, plies, PositionKey{})};
  std::unordered_set<std::string> seen_positions;
  const long long max_attempts =
      std::max(static_cast<long long>(options.per_length) * 2000LL, static_cast<long long>(100000));
  long long attempts = 0;
  while (!filler.complete()) {
    std::vector<match_runner::GeneratedOpening> batch;
    while (batch.size() < kCandidateBatch && attempts < max_attempts) {
      ++attempts;
      if (std::optional<match_runner::GeneratedOpening> opening =
              match_runner::generate_random_opening(rng, plies, options.max_score_gap,
                                                    seen_positions);
          opening.has_value()) {
        batch.push_back(std::move(*opening));
      }
    }
    if (batch.empty()) {
      throw std::runtime_error{"ran out of candidate openings at " + std::to_string(plies) +
                               " plies"};
    }
    offer_candidates(context, filler, batch);
  }
  return filler;
}

[[nodiscard]] match_runner::BookSource book_source(const fs::path& path) {
  return match_runner::BookSource{
      .path = path.string(),
      .digest = match_runner::file_digest(path.string()),
  };
}

void write_balanced_corpus(const BalancedCorpusOptions& options) {
  if (options.development_output_path.empty()) {
    throw std::invalid_argument{"missing --development-out"};
  }
  if (options.holdout_output_path.empty()) {
    throw std::invalid_argument{"missing --holdout-out"};
  }
  if (options.development_output_path.lexically_normal() ==
      options.holdout_output_path.lexically_normal()) {
    throw std::invalid_argument{"development and holdout outputs must be different paths"};
  }
  if (options.correction_path.empty()) {
    throw std::invalid_argument{"missing --correction"};
  }
  if (options.engine_path.empty()) {
    throw std::invalid_argument{"missing --engine"};
  }
  if (options.plies.empty()) {
    throw std::invalid_argument{"missing --plies"};
  }
  if (options.per_length <= 0) {
    throw std::invalid_argument{"--per-length requires a positive integer"};
  }
  if (options.seed == 0) {
    throw std::invalid_argument{"--seed requires a positive integer"};
  }

  std::ifstream correction_input{options.correction_path};
  if (!correction_input) {
    throw std::invalid_argument{"failed to read " + options.correction_path.string()};
  }
  std::ostringstream correction_text;
  correction_text << correction_input.rdbuf();
  match_runner::BalancedBookSettings settings{
      .seed = options.seed,
      .plies = options.plies,
      .max_score_gap = options.max_score_gap,
      .threshold = options.threshold,
      .per_length = options.per_length,
      .correction = match_runner::parse_correction_record(correction_text.str()),
      .correction_file = book_source(options.correction_path),
  };
  for (const int plies : options.plies) {
    if (!settings.correction.correction.by_plies.contains(plies)) {
      throw std::invalid_argument{"the correction has no constants for " + std::to_string(plies) +
                                  " plies"};
    }
  }
  if (options.previous_development_path.has_value()) {
    settings.previous_development = book_source(*options.previous_development_path);
  }
  if (options.previous_holdout_path.has_value()) {
    settings.previous_holdout = book_source(*options.previous_holdout_path);
  }

  const std::string command = match_runner::balance_engine_command(
      options.engine_path.string(), settings.correction.search, options.net_path.string());
  const match_runner::EngineIdentity engine =
      match_runner::query_engine_identity(command, kHandshakeTimeout);
  if (engine.net_sha256 != settings.correction.engine.net_sha256) {
    throw std::invalid_argument{"the engine plays net " + engine.net_sha256 +
                                " but the correction was fitted for net " +
                                settings.correction.engine.net_sha256};
  }
  settings.correction.engine = engine;

  const PreviousPartitions previous = previous_partitions(options);
  const BalanceContext context{
      .options = options,
      .correction = settings.correction,
      .previous = previous,
      .command = command,
  };
  std::map<int, match_runner::QuotaFiller> lengths;
  for (const int plies : options.plies) {
    const match_runner::QuotaFiller& filler =
        lengths.try_emplace(plies, select_length(context, plies)).first->second;
    std::cout << "balanced_length plies=" << plies;
    for (const match_runner::BookPartition partition :
         {match_runner::BookPartition::kDevelopment, match_runner::BookPartition::kHoldout}) {
      const match_runner::PartitionSelection& selection = filler.selection(partition);
      std::cout << ' ' << match_runner::partition_name(partition)
                << "_candidates=" << selection.candidates << ' '
                << match_runner::partition_name(partition) << "_kept=" << selection.kept.size();
    }
    std::cout << '\n' << std::flush;
  }

  const std::string development_text = match_runner::format_balanced_book(
      settings, match_runner::BookPartition::kDevelopment, lengths);
  const std::string holdout_text =
      match_runner::format_balanced_book(settings, match_runner::BookPartition::kHoldout, lengths);
  static_cast<void>(match_runner::parse_opening_book_text("development", development_text));
  static_cast<void>(match_runner::parse_opening_book_text("holdout", holdout_text));
  write_text_atomically(options.development_output_path, development_text);
  write_text_atomically(options.holdout_output_path, holdout_text);
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

[[nodiscard]] std::vector<int> parse_plies_list(std::string_view text) {
  std::vector<int> plies;
  std::size_t start = 0;
  while (start <= text.size()) {
    const std::size_t comma = text.find(',', start);
    const std::size_t end = comma == std::string_view::npos ? text.size() : comma;
    const std::optional<int> value = parse_positive_int(text.substr(start, end - start));
    if (!value.has_value() || *value >= kCellCount) {
      throw std::invalid_argument{
          "--plies requires comma-separated positive integers below the board size"};
    }
    plies.push_back(*value);
    if (comma == std::string_view::npos) {
      break;
    }
    start = comma + 1;
  }
  std::sort(plies.begin(), plies.end());
  if (std::adjacent_find(plies.begin(), plies.end()) != plies.end()) {
    throw std::invalid_argument{"--plies values must be unique"};
  }
  return plies;
}

[[nodiscard]] OpeningCorpusOptions parse_opening_corpus_options(int argc, char** argv) {
  OpeningCorpusOptions options;
  for (int index = 0; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument == "--development-out") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--development-out requires a path"};
      }
      options.development_output_path = argv[++index];
      continue;
    }
    if (argument == "--holdout-out") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--holdout-out requires a path"};
      }
      options.holdout_output_path = argv[++index];
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
        throw std::invalid_argument{"--plies requires a comma-separated list"};
      }
      options.plies = parse_plies_list(argv[++index]);
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
    throw std::invalid_argument{"unknown openings generate-corpus argument: " +
                                std::string{argument}};
  }
  return options;
}

[[nodiscard]] std::string_view next_value(int argc, char** argv, int& index,
                                          std::string_view what) {
  if (index + 1 >= argc) {
    throw std::invalid_argument{std::string{argv[index]} + " requires " + std::string{what}};
  }
  return argv[++index];
}

[[nodiscard]] int next_positive_int(int argc, char** argv, int& index) {
  const std::string_view argument = argv[index];
  const std::optional<int> value =
      parse_positive_int(next_value(argc, argv, index, "a positive integer"));
  if (!value.has_value()) {
    throw std::invalid_argument{std::string{argument} + " requires a positive integer"};
  }
  return *value;
}

[[nodiscard]] double next_positive_double(int argc, char** argv, int& index) {
  const std::string_view argument = argv[index];
  const std::string_view text = next_value(argc, argv, index, "a positive number");
  double value = 0.0;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || !(value > 0.0)) {
    throw std::invalid_argument{std::string{argument} + " requires a positive number"};
  }
  return value;
}

[[nodiscard]] FitBalanceOptions parse_fit_balance_options(int argc, char** argv) {
  FitBalanceOptions options;
  for (int index = 0; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument == "--run") {
      options.runs.emplace_back(next_value(argc, argv, index, "a run directory"));
    } else if (argument == "--engine") {
      options.engine_path = next_value(argc, argv, index, "a path");
    } else if (argument == "--net") {
      options.net_path = next_value(argc, argv, index, "a path");
    } else if (argument == "--out") {
      options.output_path = next_value(argc, argv, index, "a path");
    } else if (argument == "--nodes") {
      options.search.nodes = static_cast<std::uint64_t>(next_positive_int(argc, argv, index));
    } else if (argument == "--hash-mb") {
      options.search.hash_mb = next_positive_int(argc, argv, index);
    } else if (argument == "--workers") {
      options.workers = next_positive_int(argc, argv, index);
    } else if (argument == "--min-openings") {
      options.min_openings = next_positive_int(argc, argv, index);
    } else if (argument == "--threshold") {
      options.threshold = next_positive_double(argc, argv, index);
    } else {
      throw std::invalid_argument{"unknown openings fit-balance argument: " +
                                  std::string{argument}};
    }
  }
  return options;
}

[[nodiscard]] BalancedCorpusOptions parse_balanced_corpus_options(int argc, char** argv) {
  BalancedCorpusOptions options;
  for (int index = 0; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument == "--development-out") {
      options.development_output_path = next_value(argc, argv, index, "a path");
    } else if (argument == "--holdout-out") {
      options.holdout_output_path = next_value(argc, argv, index, "a path");
    } else if (argument == "--correction") {
      options.correction_path = next_value(argc, argv, index, "a path");
    } else if (argument == "--engine") {
      options.engine_path = next_value(argc, argv, index, "a path");
    } else if (argument == "--net") {
      options.net_path = next_value(argc, argv, index, "a path");
    } else if (argument == "--previous-development") {
      options.previous_development_path = fs::path{next_value(argc, argv, index, "a path")};
    } else if (argument == "--previous-holdout") {
      options.previous_holdout_path = fs::path{next_value(argc, argv, index, "a path")};
    } else if (argument == "--plies") {
      options.plies = parse_plies_list(next_value(argc, argv, index, "a comma-separated list"));
    } else if (argument == "--per-length") {
      options.per_length = next_positive_int(argc, argv, index);
    } else if (argument == "--threshold") {
      options.threshold = next_positive_double(argc, argv, index);
    } else if (argument == "--max-score-gap") {
      options.max_score_gap = next_positive_int(argc, argv, index);
    } else if (argument == "--seed") {
      const std::optional<std::uint64_t> seed =
          parse_positive_u64(next_value(argc, argv, index, "a positive integer"));
      if (!seed.has_value()) {
        throw std::invalid_argument{"--seed requires a positive integer"};
      }
      options.seed = *seed;
    } else if (argument == "--workers") {
      options.workers = next_positive_int(argc, argv, index);
    } else {
      throw std::invalid_argument{"unknown openings generate-balanced argument: " +
                                  std::string{argument}};
    }
  }
  return options;
}

}  // namespace

int run_openings(int argc, char** argv) {
  if (argc <= 0) {
    throw std::invalid_argument{"missing openings command"};
  }

  const std::string_view command = argv[0];
  if (command == "generate-random" || command == "generate") {
    const OpeningGenerateOptions options = parse_opening_generate_options(argc - 1, argv + 1);
    write_generated_opening_book(options);
    const match_runner::OpeningBook book =
        match_runner::load_opening_book(options.output_path.string());
    std::cout << "openings_generated"
              << " path=" << options.output_path.string() << " count=" << book.lines.size()
              << " plies=" << options.plies << " seed=" << options.seed
              << " max_score_gap=" << options.max_score_gap << '\n';
    return 0;
  }
  if (command == "generate-systematic") {
    const OpeningSystematicOptions options = parse_opening_systematic_options(argc - 1, argv + 1);
    write_systematic_opening_book(options);
    const match_runner::OpeningBook book =
        match_runner::load_opening_book(options.output_path.string());
    std::cout << "openings_generated"
              << " path=" << options.output_path.string() << " count=" << book.lines.size()
              << " plies=" << options.plies << " mode=systematic-symmetry\n";
    return 0;
  }
  if (command == "generate-corpus") {
    const OpeningCorpusOptions options = parse_opening_corpus_options(argc - 1, argv + 1);
    write_opening_corpus(options);
    const match_runner::OpeningBook development =
        match_runner::load_opening_book(options.development_output_path.string());
    const match_runner::OpeningBook holdout =
        match_runner::load_opening_book(options.holdout_output_path.string());
    std::cout << "openings_generated"
              << " development_path=" << options.development_output_path.string()
              << " development_count=" << development.lines.size()
              << " holdout_path=" << options.holdout_output_path.string()
              << " holdout_count=" << holdout.lines.size() << " non_empty_count=" << options.count
              << " plies=" << plies_text(options.plies) << " seed=" << options.seed
              << " max_score_gap=" << options.max_score_gap << '\n';
    return 0;
  }

  if (command == "fit-balance") {
    fit_balance(parse_fit_balance_options(argc - 1, argv + 1));
    return 0;
  }
  if (command == "generate-balanced") {
    const BalancedCorpusOptions options = parse_balanced_corpus_options(argc - 1, argv + 1);
    write_balanced_corpus(options);
    const match_runner::OpeningBook development =
        match_runner::load_opening_book(options.development_output_path.string());
    const match_runner::OpeningBook holdout =
        match_runner::load_opening_book(options.holdout_output_path.string());
    std::cout << "openings_generated"
              << " development_path=" << options.development_output_path.string()
              << " development_count=" << development.lines.size()
              << " holdout_path=" << options.holdout_output_path.string()
              << " holdout_count=" << holdout.lines.size() << " seed=" << options.seed
              << " per_length=" << options.per_length << '\n';
    return 0;
  }

  throw std::invalid_argument{"unknown openings command: " + std::string{command}};
}

}  // namespace poe2::runner_cli
