#include <bit>
#include <charconv>
#include <cstdint>
#include <exception>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>

#include "poe2/engine_stdio.hpp"
#include "poe2/move.hpp"

namespace {

[[nodiscard]] std::optional<std::uint64_t> parse_seed(std::string_view text) noexcept {
  std::uint64_t seed = 0;
  const char* begin = text.data();
  const char* end = begin + text.size();
  const auto result = std::from_chars(begin, end, seed);
  if (result.ec != std::errc{} || result.ptr != end) {
    return std::nullopt;
  }

  return seed;
}

[[nodiscard]] std::uint64_t random_seed() {
  std::random_device device;
  return (std::uint64_t{device()} << 32U) ^ std::uint64_t{device()};
}

[[nodiscard]] std::optional<poe2::Move> random_legal_move(const poe2::Position& position,
                                                          std::mt19937_64& rng) {
  const poe2::Bitboard legal_moves = position.legal_moves();
  const int legal_count = std::popcount(legal_moves);
  if (legal_count == 0) {
    return std::nullopt;
  }

  std::uniform_int_distribution<int> distribution{0, legal_count - 1};
  int selected = distribution(rng);
  for (int index = 0; index < poe2::kCellCount; ++index) {
    const poe2::Bitboard bit = poe2::Bitboard{1} << index;
    if ((legal_moves & bit) == 0) {
      continue;
    }
    if (selected == 0) {
      return poe2::Move{.square = poe2::square_from_index(index)};
    }
    --selected;
  }

  return std::nullopt;
}

class RandomLegalEngine final : public poe2::engine_stdio::Engine {
 public:
  explicit RandomLegalEngine(std::uint64_t seed) : rng_{seed} {}

  [[nodiscard]] poe2::engine_stdio::EngineResult choose_move(
      const poe2::Position& position, const poe2::engine_stdio::EngineLimits&,
      const poe2::engine_stdio::InfoSink&) override {
    return poe2::engine_stdio::EngineResult{
        .best_move = random_legal_move(position, rng_),
    };
  }

 private:
  std::mt19937_64 rng_;
};

int run(int argc, char** argv) {
  std::optional<std::uint64_t> seed;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument == "--seed") {
      if (index + 1 >= argc) {
        throw std::invalid_argument{"--seed requires an unsigned integer"};
      }
      seed = parse_seed(argv[++index]);
      if (!seed.has_value()) {
        throw std::invalid_argument{"--seed requires an unsigned integer"};
      }
      continue;
    }

    throw std::invalid_argument{"unknown argument: " + std::string{argument}};
  }

  RandomLegalEngine engine{seed.value_or(random_seed())};
  return poe2::engine_stdio::run_engine_stdio("random_legal", engine);
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
