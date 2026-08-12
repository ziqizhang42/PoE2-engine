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
  const Bitboard legal_moves = position.legal_moves();
  const int empty_count = std::popcount(legal_moves);
  return evaluate_two_ply_closure(position, legal_moves, empty_count);
}

Score evaluate_two_ply_closure(const Position& position, Bitboard legal_moves,
                               int empty_count) noexcept {
  assert(legal_moves == position.legal_moves());
  assert(empty_count == std::popcount(legal_moves));
  const Score base = evaluate(position);
  if (empty_count == 0) {
    return base;
  }

  const Player player = position.side_to_move();
  if (empty_count == 1) {
    const int move_index = std::countr_zero(legal_moves);
    return base + 2 * position.score_gain_unchecked(player, move_index);
  }

  const Player reply_player = opponent(player);
  Score best_reply_gain = std::numeric_limits<Score>::lowest();
  Score second_reply_gain = std::numeric_limits<Score>::lowest();
  int best_reply_index = 0;
  Score player_gain_at_best_reply = 0;
  Score best_player_gain = std::numeric_limits<Score>::lowest();
  Score second_player_gain = std::numeric_limits<Score>::lowest();
  int best_player_index = 0;

  Bitboard reply_moves = legal_moves;
  while (reply_moves != 0) {
    const int move_index = std::countr_zero(reply_moves);
    reply_moves &= reply_moves - Bitboard{1};
    const ScoreByPlayer gains = position.score_gains_unchecked(move_index);
    const Score player_gain = player == Player::kOne ? gains.player_one : gains.player_two;
    const Score gain = reply_player == Player::kOne ? gains.player_one : gains.player_two;

    if (gain > best_reply_gain) {
      second_reply_gain = best_reply_gain;
      best_reply_gain = gain;
      best_reply_index = move_index;
      player_gain_at_best_reply = player_gain;
    } else if (gain > second_reply_gain) {
      second_reply_gain = gain;
    }

    if (player_gain > best_player_gain) {
      second_player_gain = best_player_gain;
      best_player_gain = player_gain;
      best_player_index = move_index;
    } else if (player_gain > second_player_gain) {
      second_player_gain = player_gain;
    }
  }

  const Score best_other_player_gain =
      best_player_index == best_reply_index ? second_player_gain : best_player_gain;
  const Score best_pair_value = std::max(player_gain_at_best_reply - second_reply_gain,
                                         best_other_player_gain - best_reply_gain);
  return base + 2 * best_pair_value;
}

}  // namespace poe2::minimax
