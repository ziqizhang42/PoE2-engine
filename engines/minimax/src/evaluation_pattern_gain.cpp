#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "evaluation_inputs.hpp"
#include "pattern_gain_layout.hpp"
#include "poe2/minimax/evaluation.hpp"
#include "poe2/minimax/net.hpp"

namespace poe2::minimax {

namespace {

using pattern_gain::bracketing_knots;
using pattern_gain::KnotPair;

static_assert(kGainThresholds[0] == 2);

struct RankedGains {
  std::array<Score, 4> top{};
  Bitboard best_moves = 0;
};

// Every legal move gains at least one point, so only the moves that gain more need ranking.
[[nodiscard]] RankedGains rank_gains(const MoveGains& gains, Player player, Bitboard legal_moves,
                                     int empty_count) noexcept {
  // The largest gains all lie in the highest threshold that enough moves reach.
  Bitboard candidates = gains.reaching(player, 0);
  for (std::size_t threshold = kGainThresholds.size() - 1; threshold > 0; --threshold) {
    const Bitboard reaching = gains.reaching(player, threshold);
    if (static_cast<std::size_t>(std::popcount(reaching)) >=
        std::tuple_size_v<decltype(RankedGains::top)>) {
      candidates = reaching;
      break;
    }
  }

  RankedGains ranked;
  if (candidates == 0) {
    ranked.best_moves = legal_moves;
  }
  const int candidate_count = std::popcount(candidates);
  while (candidates != 0) {
    const int move_index = std::countr_zero(candidates);
    const Bitboard move = Bitboard{1} << move_index;
    candidates ^= move;
    const Score gain = gains.gain(player, move_index);
    ranked.best_moves = (gain > ranked.top[0] ? Bitboard{0} : ranked.best_moves) |
                        (gain >= ranked.top[0] ? move : 0);
    Score sinking = gain;
    for (Score& rank : ranked.top) {
      const Score larger = std::max(rank, sinking);
      sinking = std::min(rank, sinking);
      rank = larger;
    }
  }
  const int ranked_count = std::min(empty_count, static_cast<int>(ranked.top.size()));
  for (int rank = candidate_count; rank < ranked_count; ++rank) {
    ranked.top[static_cast<std::size_t>(rank)] = 1;
  }
  return ranked;
}

struct GainEvaluation {
  Score closure_value = 0;
  std::array<std::int64_t, kPatternGainFeatureCount> features{};
};

[[nodiscard]] GainEvaluation evaluate_gains(const Position& position, Bitboard legal_moves,
                                            int empty_count, const MoveGains& gains) noexcept {
  assert(empty_count > 2);
  const Player player = position.side_to_move();
  const Player reply_player = opponent(player);
  const RankedGains own = rank_gains(gains, player, legal_moves, empty_count);
  const RankedGains reply = rank_gains(gains, reply_player, legal_moves, empty_count);
  const std::array<Score, 4>& own_top = own.top;
  const std::array<Score, 4>& opponent_top = reply.top;

  // The lowest-indexed best move stands for the others, as a scan in move order would pick it.
  const int best_own_index = std::countr_zero(own.best_moves);
  const int opponent_best_index = std::countr_zero(reply.best_moves);
  const Score own_gain_at_opponent_best = gains.gain(player, opponent_best_index);
  const bool contested_best = (own.best_moves & reply.best_moves) != 0;
  const bool unique_opponent_best = std::has_single_bit(reply.best_moves);
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
    result.features[feature++] = std::popcount(gains.reaching(player, threshold));
  }
  for (std::size_t threshold = 0; threshold < kGainThresholds.size(); ++threshold) {
    result.features[feature++] = std::popcount(gains.reaching(reply_player, threshold));
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

// 36 line numerators, each at most 28 * 2^15 in magnitude, cannot overflow an int32 sum.
static_assert(pattern_gain::kScoringLineCount * 28 * 32'768 <=
              std::numeric_limits<std::int32_t>::max());

[[nodiscard]] std::int64_t line_residual(const Position& position, const LinePatterns& patterns,
                                         const PatternGainTables& tables) noexcept {
  const int ply = position.ply();
  const auto& weights = tables.line_weights[static_cast<std::size_t>(ply)];
  std::int32_t numerator = 0;
  for (const std::uint16_t pattern : patterns.seen_by(position.side_to_move())) {
    numerator += weights[pattern];
  }
  const KnotPair knots = bracketing_knots(kPatternGainLineKnots, ply);
  return rounded_divide(numerator, knots.right - knots.left);
}

[[nodiscard]] std::int64_t gain_residual(
    const std::array<std::int64_t, kPatternGainFeatureCount>& features, int ply,
    const PatternGainTables& tables) noexcept {
  const auto& weights = tables.gain_weights[static_cast<std::size_t>(ply)];
  std::int64_t numerator = 0;
  for (std::size_t feature = 0; feature < features.size(); ++feature) {
    numerator += features[feature] * weights[feature];
  }
  const KnotPair knots = bracketing_knots(kPatternGainKnots, ply);
  return rounded_divide(numerator, knots.right - knots.left);
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

  EvaluationInputs inputs;
  inputs.reset(position);
  return evaluate_pattern_gain_fixed(position, legal_moves, empty_count, inputs, tables);
}

FixedEvaluation evaluate_pattern_gain_fixed(const Position& position, Bitboard legal_moves,
                                            int empty_count, const EvaluationInputs& inputs,
                                            const PatternGainTables& tables) noexcept {
  assert(legal_moves == position.legal_moves());
  assert(empty_count == std::popcount(legal_moves));
  const GainEvaluation gains = evaluate_gains(position, legal_moves, empty_count, inputs.gains);
  const int ply = position.ply();
  const std::int64_t value = static_cast<std::int64_t>(gains.closure_value) * kPatternGainScale +
                             tables.intercept[static_cast<std::size_t>(ply)] +
                             line_residual(position, inputs.patterns, tables) +
                             gain_residual(gains.features, ply, tables);
  assert(value >= std::numeric_limits<FixedEvaluation>::min());
  assert(value <= std::numeric_limits<FixedEvaluation>::max());
  return static_cast<FixedEvaluation>(value);
}

Score round_pattern_gain_evaluation(FixedEvaluation value) noexcept {
  return static_cast<Score>(rounded_divide(value, kPatternGainScale));
}

}  // namespace poe2::minimax
