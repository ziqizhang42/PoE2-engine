#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "pattern_gain_layout.hpp"
#include "poe2/minimax/evaluation.hpp"
#include "poe2/minimax/net.hpp"

namespace poe2::minimax {

namespace {

using pattern_gain::kCellLineContributions;
using pattern_gain::kPatternOffsets;
using pattern_gain::kScoringLineCount;
using pattern_gain::kScoringLines;

constexpr std::size_t kPackedCountLaneBits = 8;
constexpr std::uint32_t kPackedCountLaneMask = (1U << kPackedCountLaneBits) - 1U;
constexpr std::array<Score, 4> kGainThresholds{{2, 4, 8, 16}};

static_assert(kCellCount <= kPackedCountLaneMask);

template <std::uint16_t Digit>
void accumulate_line_codes(std::array<std::uint16_t, kScoringLineCount>& codes,
                           Bitboard pieces) noexcept {
  static_assert(Digit == 1 || Digit == 2);
  while (pieces != 0) {
    const int cell = std::countr_zero(pieces);
    pieces &= pieces - Bitboard{1};
    const pattern_gain::CellLineContributions& contributions = kCellLineContributions[cell];
    for (const pattern_gain::LineContribution contribution : contributions.values) {
      codes[contribution.line] =
          static_cast<std::uint16_t>(codes[contribution.line] + Digit * contribution.place);
    }
  }
}

template <std::size_t Size>
void insert_descending(std::array<Score, Size>& values, Score value) noexcept {
  if (value <= values.back()) {
    return;
  }
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (value > values[index]) {
      for (std::size_t shifted = values.size() - 1; shifted > index; --shifted) {
        values[shifted] = values[shifted - 1];
      }
      values[index] = value;
      return;
    }
  }
}

struct GainEvaluation {
  Score closure_value = 0;
  std::array<std::int64_t, kPatternGainFeatureCount> features{};
};

[[nodiscard]] constexpr std::uint32_t threshold_count_increment(Score gain) noexcept {
  std::uint32_t increment = 0;
  for (std::size_t index = 0; index < kGainThresholds.size(); ++index) {
    increment |= static_cast<std::uint32_t>(gain >= kGainThresholds[index])
                 << (index * kPackedCountLaneBits);
  }
  return increment;
}

[[nodiscard]] GainEvaluation evaluate_gains(const Position& position, Bitboard legal_moves,
                                            int empty_count) noexcept {
  assert(empty_count > 2);
  std::array<Score, 4> own_top{};
  std::array<Score, 4> opponent_top{};
  std::uint32_t own_counts = 0;
  std::uint32_t opponent_counts = 0;

  const Player player = position.side_to_move();
  int best_own_index = -1;
  Bitboard best_own_moves = 0;
  int opponent_best_index = -1;
  Score own_gain_at_opponent_best = 0;
  Bitboard opponent_best_moves = 0;
  Bitboard moves = legal_moves;
  while (moves != 0) {
    const int move_index = std::countr_zero(moves);
    moves &= moves - Bitboard{1};
    const ScoreByPlayer gains = position.score_gains_unchecked(move_index);
    const Score own = player == Player::kOne ? gains.player_one : gains.player_two;
    const Score reply = player == Player::kOne ? gains.player_two : gains.player_one;
    const Bitboard move = Bitboard{1} << move_index;

    if (own > own_top[0]) {
      best_own_index = move_index;
      best_own_moves = move;
    } else if (own == own_top[0]) {
      best_own_moves |= move;
    }
    if (reply > opponent_top[0]) {
      opponent_best_index = move_index;
      own_gain_at_opponent_best = own;
      opponent_best_moves = move;
    } else if (reply == opponent_top[0]) {
      opponent_best_moves |= move;
    }

    insert_descending(own_top, own);
    insert_descending(opponent_top, reply);
    own_counts += threshold_count_increment(own);
    opponent_counts += threshold_count_increment(reply);
  }

  const bool contested_best = (best_own_moves & opponent_best_moves) != 0;
  const bool unique_opponent_best = std::has_single_bit(opponent_best_moves);
  const Score best_other_own_gain = best_own_index == opponent_best_index ? own_top[1] : own_top[0];
  const Score opponent_best_candidate = own_gain_at_opponent_best - opponent_top[1];
  const Score other_candidate = best_other_own_gain - opponent_top[0];
  const bool select_opponent_best = opponent_best_candidate > other_candidate;
  const Score best_pair_value = select_opponent_best ? opponent_best_candidate : other_candidate;
  // On equal closure values, selecting the larger reply minimizes denial and preserves D4
  // invariance. The ordinary candidate always has the largest reply gain.
  const Score selected_reply = select_opponent_best ? opponent_top[1] : opponent_top[0];

  GainEvaluation result{
      .closure_value = evaluate(position) + 2 * best_pair_value,
  };
  std::size_t feature = 0;
  for (const Score value : own_top) {
    result.features[feature++] = value;
  }
  for (const Score value : opponent_top) {
    result.features[feature++] = value;
  }
  for (std::size_t threshold = 0; threshold < kGainThresholds.size(); ++threshold) {
    result.features[feature++] =
        (own_counts >> (threshold * kPackedCountLaneBits)) & kPackedCountLaneMask;
  }
  for (std::size_t threshold = 0; threshold < kGainThresholds.size(); ++threshold) {
    result.features[feature++] =
        (opponent_counts >> (threshold * kPackedCountLaneBits)) & kPackedCountLaneMask;
  }
  result.features[feature++] = contested_best ? 1 : 0;
  result.features[feature++] = unique_opponent_best ? 1 : 0;
  result.features[feature++] = opponent_top[0] - selected_reply;
  assert(feature == result.features.size());
  return result;
}

[[nodiscard]] constexpr std::int64_t rounded_divide(std::int64_t numerator,
                                                    std::int64_t denominator) noexcept {
  assert(denominator > 0);
  const bool negative = numerator < 0;
  const std::int64_t magnitude = negative ? -numerator : numerator;
  const std::int64_t rounded = (magnitude + denominator / 2) / denominator;
  return negative ? -rounded : rounded;
}

// The two phase knots bracketing a ply, the only ones interpolation reads. Summing weights for
// these alone, not for every knot, is what lets search with weights loaded at run time keep the
// speed of weights fixed at compile time, above all in the browser build.
struct KnotPair {
  std::size_t lower = 0;
  int left = 0;
  int right = 0;
};

template <std::size_t KnotCount>
[[nodiscard]] KnotPair bracketing_knots(const std::array<std::uint8_t, KnotCount>& knots,
                                        int ply) noexcept {
  static_assert(KnotCount >= 2);
  assert(ply >= knots.front() && ply <= knots.back());
  std::size_t upper = 1;
  while (ply > knots[upper]) {
    ++upper;
  }
  return KnotPair{.lower = upper - 1, .left = knots[upper - 1], .right = knots[upper]};
}

[[nodiscard]] std::int64_t interpolate(std::int64_t lower_value, std::int64_t upper_value,
                                       KnotPair pair, int ply) noexcept {
  const std::int64_t numerator = static_cast<std::int64_t>(pair.right - ply) * lower_value +
                                 static_cast<std::int64_t>(ply - pair.left) * upper_value;
  return rounded_divide(numerator, pair.right - pair.left);
}

[[nodiscard]] std::int64_t line_residual(const Position& position,
                                         const PatternGainTables& tables) noexcept {
  const Player own = position.side_to_move();
  const Bitboard player_one = position.board().bits(Player::kOne);
  const Bitboard player_two = position.board().bits(Player::kTwo);
  const Bitboard own_pieces = own == Player::kOne ? player_one : player_two;
  const Bitboard opponent_pieces = own == Player::kOne ? player_two : player_one;
  std::array<std::uint16_t, kScoringLineCount> codes{};
  accumulate_line_codes<1>(codes, own_pieces);
  accumulate_line_codes<2>(codes, opponent_pieces);
  const KnotPair pair = bracketing_knots(kPatternGainLineKnots, position.ply());
  const auto& lower_weights = tables.raw_line_weights[pair.lower];
  const auto& upper_weights = tables.raw_line_weights[pair.lower + 1];
  // 36 int16 weights cannot overflow an int32 sum.
  std::int32_t lower_sum = 0;
  std::int32_t upper_sum = 0;
  for (std::size_t line_index = 0; line_index < kScoringLines.size(); ++line_index) {
    const std::size_t pattern =
        kPatternOffsets[kScoringLines[line_index].length] + codes[line_index];
    lower_sum += lower_weights[pattern];
    upper_sum += upper_weights[pattern];
  }
  return interpolate(lower_sum, upper_sum, pair, position.ply());
}

[[nodiscard]] std::int64_t gain_residual(
    const std::array<std::int64_t, kPatternGainFeatureCount>& features, int ply,
    const PatternGainTables& tables) noexcept {
  const KnotPair pair = bracketing_knots(kPatternGainKnots, ply);
  const std::size_t lower = pair.lower * kPatternGainFeatureCount;
  const std::size_t upper = lower + kPatternGainFeatureCount;
  std::int64_t lower_sum = 0;
  std::int64_t upper_sum = 0;
  for (std::size_t feature = 0; feature < features.size(); ++feature) {
    lower_sum += features[feature] * tables.gain_weights[lower + feature];
    upper_sum += features[feature] * tables.gain_weights[upper + feature];
  }
  return interpolate(lower_sum, upper_sum, pair, ply);
}

}  // namespace

FixedEvaluation evaluate_pattern_gain_fixed(const Position& position) noexcept {
  return evaluate_pattern_gain_fixed(position, default_net()->pattern_gain());
}

FixedEvaluation evaluate_pattern_gain_fixed(const Position& position,
                                            const PatternGainTables& tables) noexcept {
  const Bitboard legal_moves = position.legal_moves();
  return evaluate_pattern_gain_fixed(position, legal_moves, std::popcount(legal_moves), tables);
}

FixedEvaluation evaluate_pattern_gain_fixed(const Position& position, Bitboard legal_moves,
                                            int empty_count,
                                            const PatternGainTables& tables) noexcept {
  assert(legal_moves == position.legal_moves());
  assert(empty_count == std::popcount(legal_moves));
  if (empty_count <= 2) {
    return static_cast<FixedEvaluation>(
        evaluate_two_ply_closure(position, legal_moves, empty_count) * kPatternGainScale);
  }

  const GainEvaluation gains = evaluate_gains(position, legal_moves, empty_count);
  const int ply = position.ply();
  const std::int64_t value = static_cast<std::int64_t>(gains.closure_value) * kPatternGainScale +
                             tables.intercept[static_cast<std::size_t>(ply)] +
                             line_residual(position, tables) +
                             gain_residual(gains.features, ply, tables);
  assert(value >= std::numeric_limits<FixedEvaluation>::min());
  assert(value <= std::numeric_limits<FixedEvaluation>::max());
  return static_cast<FixedEvaluation>(value);
}

Score round_pattern_gain_evaluation(FixedEvaluation value) noexcept {
  return static_cast<Score>(rounded_divide(value, kPatternGainScale));
}

}  // namespace poe2::minimax
