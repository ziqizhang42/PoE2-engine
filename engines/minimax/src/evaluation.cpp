#include "poe2/minimax/evaluation.hpp"

#include <algorithm>
#include <bit>
#include <cassert>
#include <limits>

namespace poe2::minimax {

namespace {

constexpr Score kConservativeMaximumPlayerScore =
    4 * kCellCount * (Score{1} << (kBoardSize - 1)) + kCellCount;
static_assert(6 * kConservativeMaximumPlayerScore + kPlayerTwoHandicapHalfPoints <=
              std::numeric_limits<Score>::max());

}  // namespace

Score evaluate(const Position& position) noexcept {
  const Score player_one_half_points = position.score(Player::kOne) * 2;
  const Score player_two_half_points =
      position.score(Player::kTwo) * 2 + kPlayerTwoHandicapHalfPoints;
  const Score player_one_advantage = player_one_half_points - player_two_half_points;

  return position.side_to_move() == Player::kOne ? player_one_advantage : -player_one_advantage;
}

Score evaluate_two_ply_closure(const Position& position) noexcept {
  const Score base = evaluate(position);
  Bitboard legal_moves = position.legal_moves();
  const int empty_count = std::popcount(legal_moves);
  if (empty_count == 0) {
    return base;
  }

  const Player player = position.side_to_move();
  if (empty_count == 1) {
    const Square square = square_from_index(std::countr_zero(legal_moves));
    return base + 2 * position.score_gain_unchecked(player, square);
  }

  const Player reply_player = opponent(player);
  Score best_reply_gain = std::numeric_limits<Score>::lowest();
  Score second_reply_gain = std::numeric_limits<Score>::lowest();
  Square best_reply_square;

  Bitboard reply_moves = legal_moves;
  while (reply_moves != 0) {
    const int move_index = std::countr_zero(reply_moves);
    reply_moves &= reply_moves - Bitboard{1};
    const Square square = square_from_index(move_index);
    const Score gain = position.score_gain_unchecked(reply_player, square);

    if (gain > best_reply_gain) {
      second_reply_gain = best_reply_gain;
      best_reply_gain = gain;
      best_reply_square = square;
    } else if (gain > second_reply_gain) {
      second_reply_gain = gain;
    }
  }

  Score best_pair_value = std::numeric_limits<Score>::lowest();
  while (legal_moves != 0) {
    const int move_index = std::countr_zero(legal_moves);
    legal_moves &= legal_moves - Bitboard{1};
    const Square square = square_from_index(move_index);
    const Score gain = position.score_gain_unchecked(player, square);

    const Score reply_gain = square == best_reply_square ? second_reply_gain : best_reply_gain;
    best_pair_value = std::max(best_pair_value, gain - reply_gain);
  }

  assert(best_pair_value != std::numeric_limits<Score>::lowest());
  return base + 2 * best_pair_value;
}

}  // namespace poe2::minimax
