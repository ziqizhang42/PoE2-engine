#ifndef POE2_OPENING_GENERATION_HPP
#define POE2_OPENING_GENERATION_HPP

#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "poe2/board.hpp"

namespace poe2::match_runner {

struct GeneratedOpening {
  int plies = 0;
  std::vector<std::string> moves;
  PositionKey canonical_key;
  std::uint64_t partition_hash = 0;
};

// Positions equal under any of the 8 board symmetries share one canonical key, so this text
// identifies an opening position up to symmetry.
[[nodiscard]] std::string canonical_key_text(PositionKey canonical_key);

// The canonical key of the position the moves reach. Throws std::invalid_argument for a malformed
// or illegal line.
[[nodiscard]] PositionKey canonical_opening_key(const std::vector<std::string>& moves);

// Plays uniformly random legal moves. Returns nothing when the raw score gap exceeds
// max_score_gap or the position (up to symmetry) is already in seen_positions; otherwise records
// it there.
[[nodiscard]] std::optional<GeneratedOpening> generate_random_opening(
    std::mt19937_64& rng, int plies, int max_score_gap,
    std::unordered_set<std::string>& seen_positions);

// Every 2-ply position up to symmetry (315), each under its first move order in square order.
[[nodiscard]] std::vector<GeneratedOpening> generate_systematic_two_ply_openings();

[[nodiscard]] std::uint64_t hash_opening(std::uint64_t seed, int plies, PositionKey key) noexcept;

}  // namespace poe2::match_runner

#endif  // POE2_OPENING_GENERATION_HPP
