#include "openings_cli.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
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
#include <unordered_set>
#include <utility>
#include <vector>

#include "poe2/match_runner.hpp"
#include "poe2/move.hpp"
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

struct GeneratedOpening {
  int plies = 0;
  std::vector<std::string> moves;
  PositionKey canonical_key;
  std::uint64_t partition_hash = 0;
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

[[nodiscard]] std::string position_key_text(PositionKey key) {
  std::ostringstream output;
  output << key.low << ':' << key.high;
  return output.str();
}

[[nodiscard]] std::vector<Move> legal_moves(const Position& position) {
  std::vector<Move> moves;
  Bitboard bits = position.legal_moves();
  while (bits != 0) {
    const int index = std::countr_zero(bits);
    moves.push_back(Move{.square = square_from_index(index)});
    bits &= bits - Bitboard{1};
  }

  return moves;
}

[[nodiscard]] std::optional<GeneratedOpening> generate_opening_line(
    std::mt19937_64& rng, int plies, int max_score_gap,
    std::unordered_set<std::string>& seen_positions) {
  Position position;
  std::vector<std::string> moves;
  moves.reserve(static_cast<std::size_t>(plies));

  for (int ply = 0; ply < plies; ++ply) {
    const std::vector<Move> candidates = legal_moves(position);
    if (candidates.empty()) {
      return std::nullopt;
    }

    std::uniform_int_distribution<std::size_t> distribution{0, candidates.size() - 1};
    const Move move = candidates[distribution(rng)];
    if (!position.play(move.square)) {
      return std::nullopt;
    }
    moves.push_back(format_move(move));
  }

  const ScoreByPlayer scores = position.scores();
  if (std::abs(scores.player_one - scores.player_two) > max_score_gap) {
    return std::nullopt;
  }

  const CanonicalPositionKey canonical = canonicalize_position_key(position.key());
  if (!seen_positions.insert(position_key_text(canonical.key)).second) {
    return std::nullopt;
  }

  return GeneratedOpening{
      .plies = plies,
      .moves = std::move(moves),
      .canonical_key = canonical.key,
  };
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
    std::optional<GeneratedOpening> opening =
        generate_opening_line(rng, options.plies, options.max_score_gap, seen_positions);
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

[[nodiscard]] std::uint64_t hash_opening(std::uint64_t seed, int plies, PositionKey key) noexcept {
  std::uint64_t hash = 14695981039346656037ULL;
  const auto mix = [&](std::uint64_t value) {
    for (int byte = 0; byte < 8; ++byte) {
      hash ^= static_cast<unsigned char>(value & 0xffU);
      hash *= 1099511628211ULL;
      value >>= 8U;
    }
  };
  mix(seed);
  mix(static_cast<std::uint64_t>(plies));
  mix(key.low);
  mix(key.high);
  return hash;
}

[[nodiscard]] std::vector<GeneratedOpening> generate_systematic_two_ply_openings() {
  std::vector<GeneratedOpening> openings;
  std::unordered_set<std::string> seen_positions;
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
      if (!seen_positions.insert(position_key_text(canonical.key)).second) {
        continue;
      }
      openings.push_back(GeneratedOpening{
          .plies = 2,
          .moves = {format_move(first), format_move(second)},
          .canonical_key = canonical.key,
      });
    }
  }
  if (openings.size() != 315) {
    throw std::runtime_error{"generated " + std::to_string(openings.size()) +
                             " systematic 2-ply openings; expected 315"};
  }
  return openings;
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
                            const std::vector<GeneratedOpening>& openings) {
  std::ofstream output{path, std::ios::trunc};
  if (!output) {
    throw std::runtime_error{"failed to write " + path.string()};
  }

  std::map<int, int> depth_counts;
  for (const GeneratedOpening& opening : openings) {
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
  for (const GeneratedOpening& opening : openings) {
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

  std::map<int, std::vector<GeneratedOpening>> strata;
  strata[2] = generate_systematic_two_ply_openings();
  std::unordered_set<std::string> seen_positions;
  for (const GeneratedOpening& opening : strata[2]) {
    seen_positions.insert(position_key_text(opening.canonical_key));
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
      std::optional<GeneratedOpening> opening =
          generate_opening_line(rng, depth, options.max_score_gap, seen_positions);
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

  std::vector<GeneratedOpening> development;
  std::vector<GeneratedOpening> holdout;
  development.reserve(static_cast<std::size_t>(options.count / 2));
  holdout.reserve(static_cast<std::size_t>(options.count / 2));
  for (auto& [depth, openings] : strata) {
    for (GeneratedOpening& opening : openings) {
      opening.partition_hash = hash_opening(options.seed, depth, opening.canonical_key);
    }
    std::sort(openings.begin(), openings.end(),
              [](const GeneratedOpening& lhs, const GeneratedOpening& rhs) {
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
      if (!seen_positions.insert(position_key_text(canonical.key)).second) {
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

  throw std::invalid_argument{"unknown openings command: " + std::string{command}};
}

}  // namespace poe2::runner_cli
