#include <bit>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "poe2/minimax/evaluation.hpp"

namespace {

enum class Evaluator : std::uint8_t {
  kTwoPlyClosure,
  kPatternGain,
};

struct Options {
  Evaluator evaluator = Evaluator::kPatternGain;
  std::uint64_t iterations = 0;
};

[[nodiscard]] std::uint64_t parse_u64(std::string_view text, int base, std::string_view field) {
  std::uint64_t value = 0;
  const char* const begin = text.data();
  const char* const end = begin + text.size();
  const auto result = std::from_chars(begin, end, value, base);
  if (text.empty() || result.ec != std::errc{} || result.ptr != end) {
    throw std::invalid_argument{std::string{"malformed "} + std::string{field} + ": " +
                                std::string{text}};
  }
  return value;
}

[[nodiscard]] Options parse_options(int argc, char* argv[]) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument{argv[index]};
    if (argument == "--help") {
      std::cout << "usage: poe2_minimax_infer"
                   " [--evaluator two-ply-closure|pattern-gain] [--iterations <passes>]\n"
                   "Reads hexadecimal player-one/player-two bitboards from stdin.\n";
      std::exit(0);
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument{"missing value for " + std::string{argument}};
    }
    const std::string_view value{argv[++index]};
    if (argument == "--evaluator") {
      if (value == "two-ply-closure") {
        options.evaluator = Evaluator::kTwoPlyClosure;
      } else if (value == "pattern-gain") {
        options.evaluator = Evaluator::kPatternGain;
      } else {
        throw std::invalid_argument{"--evaluator must be two-ply-closure or pattern-gain"};
      }
    } else if (argument == "--iterations") {
      options.iterations = parse_u64(value, 10, "--iterations");
      if (options.iterations == 0) {
        throw std::invalid_argument{"--iterations must be positive"};
      }
    } else {
      throw std::invalid_argument{"unknown argument: " + std::string{argument}};
    }
  }
  return options;
}

[[nodiscard]] poe2::Position reconstruct_position(poe2::Bitboard player_one,
                                                  poe2::Bitboard player_two) {
  if (((player_one | player_two) & ~poe2::kBoardMask) != 0 || (player_one & player_two) != 0) {
    throw std::invalid_argument{"position contains invalid or overlapping bitboards"};
  }
  const int ply = std::popcount(player_one | player_two);
  if (std::popcount(player_one) != (ply + 1) / 2 || std::popcount(player_two) != ply / 2) {
    throw std::invalid_argument{"position bitboards have inconsistent alternating counts"};
  }

  poe2::Position position;
  poe2::Bitboard remaining_one = player_one;
  poe2::Bitboard remaining_two = player_two;
  for (int move_number = 0; move_number < ply; ++move_number) {
    poe2::Bitboard& remaining = move_number % 2 == 0 ? remaining_one : remaining_two;
    if (remaining == 0) {
      throw std::logic_error{"validated position could not be reconstructed"};
    }
    const int move_index = std::countr_zero(remaining);
    remaining &= remaining - poe2::Bitboard{1};
    if (!position.play(poe2::square_from_index(move_index))) {
      throw std::logic_error{"validated position contains an illegal move"};
    }
  }
  return position;
}

[[nodiscard]] std::vector<poe2::Position> read_positions(std::istream& input) {
  std::vector<poe2::Position> positions;
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.empty()) {
      continue;
    }
    std::istringstream fields{line};
    std::string player_one_text;
    std::string player_two_text;
    std::string trailing;
    if (!(fields >> player_one_text >> player_two_text) || (fields >> trailing)) {
      throw std::invalid_argument{"input line " + std::to_string(line_number) +
                                  " must contain exactly two hexadecimal bitboards"};
    }
    const poe2::Bitboard player_one = parse_u64(player_one_text, 16, "player-one bitboard");
    const poe2::Bitboard player_two = parse_u64(player_two_text, 16, "player-two bitboard");
    positions.push_back(reconstruct_position(player_one, player_two));
  }
  if (positions.empty()) {
    throw std::invalid_argument{"stdin contained no positions"};
  }
  return positions;
}

[[nodiscard]] poe2::minimax::FixedEvaluation evaluate(const poe2::Position& position,
                                                      Evaluator evaluator) noexcept {
  if (evaluator == Evaluator::kPatternGain) {
    return poe2::minimax::evaluate_pattern_gain_fixed(position);
  }
  return static_cast<poe2::minimax::FixedEvaluation>(
      poe2::minimax::evaluate_two_ply_closure(position) * poe2::minimax::kPatternGainScale);
}

void run_benchmark(const std::vector<poe2::Position>& positions, const Options& options) {
  std::int64_t checksum = 0;
  for (const poe2::Position& position : positions) {
    checksum += evaluate(position, options.evaluator);
  }

  const auto begin = std::chrono::steady_clock::now();
  for (std::uint64_t iteration = 0; iteration < options.iterations; ++iteration) {
    for (const poe2::Position& position : positions) {
      checksum += evaluate(position, options.evaluator);
    }
  }
  const auto end = std::chrono::steady_clock::now();
  const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
  if (positions.size() > std::numeric_limits<std::uint64_t>::max() / options.iterations) {
    throw std::overflow_error{"benchmark evaluation count overflow"};
  }
  const std::uint64_t evaluations =
      static_cast<std::uint64_t>(positions.size()) * options.iterations;
  const double nanoseconds_per_evaluation =
      evaluations == 0 ? 0.0 : static_cast<double>(elapsed) / static_cast<double>(evaluations);
  std::cout << "benchmark evaluator="
            << (options.evaluator == Evaluator::kPatternGain ? "pattern-gain" : "two-ply-closure")
            << " positions=" << positions.size() << " iterations=" << options.iterations
            << " evaluations=" << evaluations << " elapsed_ns=" << elapsed
            << " ns_per_evaluation=" << std::fixed << std::setprecision(3)
            << nanoseconds_per_evaluation << " checksum=" << checksum << '\n';
}

int run(int argc, char* argv[]) {
  const Options options = parse_options(argc, argv);
  const std::vector<poe2::Position> positions = read_positions(std::cin);
  if (options.iterations > 0) {
    run_benchmark(positions, options);
    return 0;
  }
  for (const poe2::Position& position : positions) {
    std::cout << evaluate(position, options.evaluator) << '\n';
  }
  return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "error " << error.what() << '\n';
  } catch (...) {
    std::cerr << "error unknown\n";
  }
  return 1;
}
