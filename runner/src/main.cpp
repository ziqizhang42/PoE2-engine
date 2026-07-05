#include <exception>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>

#include "poe2/move.hpp"

namespace {

[[nodiscard]] std::string_view player_name(poe2::Player player) noexcept {
  return player == poe2::Player::kOne ? "p1" : "p2";
}

void print_state(const poe2::Position& position) {
  const poe2::ScoreByPlayer scores = position.scores();
  std::cout << "state"
            << " ply=" << position.ply() << " side=" << player_name(position.side_to_move())
            << " p1=" << scores.player_one << " p2=" << scores.player_two
            << " empty=" << position.board().empty_count() << '\n';
}

void print_final(const poe2::GameResult& result) {
  std::cout << "final"
            << " p1=" << result.scores.player_one << " p2=" << result.scores.player_two
            << " winner=" << player_name(result.winner) << '\n';
}

void print_help() { std::cout << "commands: a1-g7, state, quit, help\n"; }

int run() {
  poe2::Position position;
  std::string line;

  print_help();
  print_state(position);

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
      print_state(position);
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
        print_final(*result.game_result);
      }
      continue;
    }

    std::cout << "accepted " << formatted_move << '\n';
    print_state(position);
    if (result.game_result.has_value()) {
      print_final(*result.game_result);
      return 0;
    }
  }

  return 0;
}

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
