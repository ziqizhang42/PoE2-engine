#include <exception>
#include <iostream>
#include <optional>

#include "poe2/engine_stdio.hpp"
#include "poe2/move.hpp"

namespace {

[[nodiscard]] std::optional<poe2::Move> first_legal_move(const poe2::Position& position) {
  const poe2::Bitboard legal_moves = position.legal_moves();
  for (int index = 0; index < poe2::kCellCount; ++index) {
    const poe2::Bitboard bit = poe2::Bitboard{1} << index;
    if ((legal_moves & bit) != 0) {
      return poe2::Move{.square = poe2::square_from_index(index)};
    }
  }

  return std::nullopt;
}

int run() { return poe2::engine_stdio::run_engine_stdio("first_legal", first_legal_move); }

}  // namespace

int main() {
  try {
    return run();
  } catch (const std::exception& error) {
    std::cerr << "fatal " << error.what() << '\n';
  } catch (...) {
    std::cerr << "fatal unknown\n";
  }

  return 1;
}
