#include "poe2/minimax/evaluation.hpp"

namespace poe2::minimax {

Score evaluate(const Position& position) noexcept {
  const Score player_one_half_points = position.score(Player::kOne) * 2;
  const Score player_two_half_points =
      position.score(Player::kTwo) * 2 + kPlayerTwoHandicapHalfPoints;
  const Score player_one_advantage = player_one_half_points - player_two_half_points;

  return position.side_to_move() == Player::kOne ? player_one_advantage : -player_one_advantage;
}

}  // namespace poe2::minimax
