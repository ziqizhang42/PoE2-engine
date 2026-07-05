#ifndef POE2_MATCH_RUNNER_HPP
#define POE2_MATCH_RUNNER_HPP

#include <chrono>
#include <cstdint>
#include <iosfwd>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "poe2/move.hpp"

namespace poe2::match_runner {

inline constexpr std::chrono::milliseconds kDefaultMoveTimeout{5000};

struct MatchOptions {
  std::string player_one_command;
  std::string player_two_command;
  std::chrono::milliseconds move_timeout = kDefaultMoveTimeout;
};

enum class MatchEndReason : std::uint8_t {
  kNormal,
  kTimeout,
  kDisconnected,
  kMalformedMove,
  kIllegalMove,
  kProtocolError,
};

struct MatchResult {
  MatchEndReason reason = MatchEndReason::kProtocolError;
  std::optional<Player> winner;
  ScoreByPlayer scores;
  std::vector<std::string> moves;
  std::string detail;
};

[[nodiscard]] std::string_view player_name(Player player) noexcept;
[[nodiscard]] std::string_view reason_name(MatchEndReason reason) noexcept;

void print_state(const Position& position, std::ostream& output);
void print_final(const GameResult& result, std::ostream& output);
void print_match_result(const MatchResult& result, std::ostream& output);

[[nodiscard]] MatchResult run_process_match(const MatchOptions& options, std::ostream& output);

}  // namespace poe2::match_runner

#endif  // POE2_MATCH_RUNNER_HPP
