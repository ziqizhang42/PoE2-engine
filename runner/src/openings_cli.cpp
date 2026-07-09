#include "openings_cli.hpp"

#include <bit>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
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

[[nodiscard]] std::optional<std::vector<std::string>> generate_opening_line(
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

  return moves;
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
    std::optional<std::vector<std::string>> moves =
        generate_opening_line(rng, options.plies, options.max_score_gap, seen_positions);
    if (!moves.has_value()) {
      continue;
    }

    output << match_runner::format_opening_moves(*moves) << '\n';
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

  throw std::invalid_argument{"unknown openings command: " + std::string{command}};
}

}  // namespace poe2::runner_cli
