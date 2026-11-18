#include "poe2/opening_generation.hpp"

#include <bit>
#include <cstdlib>
#include <stdexcept>
#include <utility>

#include "poe2/move.hpp"
#include "poe2/symmetry.hpp"

namespace poe2::match_runner {

namespace {

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

}  // namespace

std::string canonical_key_text(PositionKey canonical_key) {
  return std::to_string(canonical_key.low) + ':' + std::to_string(canonical_key.high);
}

PositionKey canonical_opening_key(const std::vector<std::string>& moves) {
  Position position;
  for (const std::string& text : moves) {
    const std::optional<Move> move = parse_move(text);
    if (!move.has_value() || !apply_move(position, *move).accepted) {
      throw std::invalid_argument{"illegal opening move: " + text};
    }
  }
  return canonicalize_position_key(position.key()).key;
}

std::optional<GeneratedOpening> generate_random_opening(
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
  if (!seen_positions.insert(canonical_key_text(canonical.key)).second) {
    return std::nullopt;
  }

  return GeneratedOpening{
      .plies = plies,
      .moves = std::move(moves),
      .canonical_key = canonical.key,
  };
}

std::vector<GeneratedOpening> generate_systematic_two_ply_openings() {
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
      if (!seen_positions.insert(canonical_key_text(canonical.key)).second) {
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

std::uint64_t hash_opening(std::uint64_t seed, int plies, PositionKey key) noexcept {
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

}  // namespace poe2::match_runner
