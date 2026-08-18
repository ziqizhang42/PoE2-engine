#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "frozen_pattern_gain_model.hpp"
#include "poe2/minimax/evaluation.hpp"

namespace poe2::minimax {

namespace {

constexpr std::size_t kScoringLineCount = 36;
constexpr std::size_t kRawPatternCount = 3276;
constexpr std::uint16_t kUnassignedOrbit = std::numeric_limits<std::uint16_t>::max();
constexpr std::size_t kMaximumLinesPerCell = 4;
constexpr std::size_t kPackedCountLaneBits = 8;
constexpr std::uint32_t kPackedCountLaneMask = (1U << kPackedCountLaneBits) - 1U;
constexpr std::array<Score, 4> kGainThresholds{{2, 4, 8, 16}};

static_assert(kCellCount <= kPackedCountLaneMask);

struct ScoringLine {
  std::array<std::uint8_t, kBoardSize> cells{};
  std::uint8_t length = 0;
};

struct LineContribution {
  std::uint16_t place = 0;
  std::uint8_t line = 0;
};

struct CellLineContributions {
  // Corners use one zero-initialized entry, whose addition to line zero is a no-op.
  std::array<LineContribution, kMaximumLinesPerCell> values{};
};

[[nodiscard]] consteval std::array<ScoringLine, kScoringLineCount> make_scoring_lines() {
  std::array<ScoringLine, kScoringLineCount> lines{};
  std::size_t next = 0;
  const auto add = [&lines, &next](int row, int column, int row_step, int column_step) {
    ScoringLine line;
    while (row >= 0 && row < kBoardSize && column >= 0 && column < kBoardSize) {
      line.cells[line.length++] = static_cast<std::uint8_t>(square_index(Square{row, column}));
      row += row_step;
      column += column_step;
    }
    lines[next++] = line;
  };

  for (int row = 0; row < kBoardSize; ++row) {
    add(row, 0, 0, 1);
  }
  for (int column = 0; column < kBoardSize; ++column) {
    add(0, column, 1, 0);
  }
  for (int column = 0; column < kBoardSize - 1; ++column) {
    add(0, column, 1, 1);
  }
  for (int row = 1; row < kBoardSize - 1; ++row) {
    add(row, 0, 1, 1);
  }
  for (int column = 1; column < kBoardSize; ++column) {
    add(0, column, 1, -1);
  }
  for (int row = 1; row < kBoardSize - 1; ++row) {
    add(row, kBoardSize - 1, 1, -1);
  }
  return lines;
}

[[nodiscard]] consteval std::array<std::uint16_t, kBoardSize + 1> make_pattern_offsets() {
  std::array<std::uint16_t, kBoardSize + 1> offsets{};
  std::uint16_t offset = 0;
  std::uint16_t pattern_count = 1;
  for (int length = 0; length <= kBoardSize; ++length) {
    if (length >= 2) {
      offsets[static_cast<std::size_t>(length)] = offset;
      offset = static_cast<std::uint16_t>(offset + pattern_count);
    }
    pattern_count = static_cast<std::uint16_t>(pattern_count * 3);
  }
  return offsets;
}

[[nodiscard]] constexpr int reverse_ternary(int code, int length) noexcept {
  int reversed = 0;
  for (int index = 0; index < length; ++index) {
    reversed = reversed * 3 + code % 3;
    code /= 3;
  }
  return reversed;
}

[[nodiscard]] consteval std::array<std::uint16_t, kRawPatternCount> make_raw_to_reversal_orbit() {
  std::array<std::uint16_t, kRawPatternCount> result{};
  std::uint16_t next_orbit = 0;
  std::size_t raw_offset = 0;
  int pattern_count = 9;
  for (int length = 2; length <= kBoardSize; ++length) {
    std::array<std::uint16_t, 2187> representative_orbits{};
    representative_orbits.fill(kUnassignedOrbit);
    for (int code = 0; code < pattern_count; ++code) {
      const int representative = std::min(code, reverse_ternary(code, length));
      std::uint16_t& orbit = representative_orbits[static_cast<std::size_t>(representative)];
      if (orbit == kUnassignedOrbit) {
        orbit = next_orbit++;
      }
      result[raw_offset + static_cast<std::size_t>(code)] = orbit;
    }
    raw_offset += static_cast<std::size_t>(pattern_count);
    pattern_count *= 3;
  }
  return result;
}

[[nodiscard]] consteval std::array<CellLineContributions, kCellCount> make_cell_line_contributions(
    const std::array<ScoringLine, kScoringLineCount>& lines) {
  std::array<CellLineContributions, kCellCount> result{};
  std::array<std::uint8_t, kCellCount> counts{};
  for (std::size_t line_index = 0; line_index < lines.size(); ++line_index) {
    const ScoringLine& line = lines[line_index];
    std::uint16_t place = 1;
    for (std::uint8_t offset = 0; offset < line.length; ++offset) {
      CellLineContributions& contributions = result[line.cells[offset]];
      contributions.values[counts[line.cells[offset]]++] = LineContribution{
          .place = place,
          .line = static_cast<std::uint8_t>(line_index),
      };
      place = static_cast<std::uint16_t>(place * 3);
    }
  }
  return result;
}

using RawLineWeights =
    std::array<std::array<std::int16_t, frozen_pattern_gain_model::kLineKnots.size()>,
               kRawPatternCount>;

[[nodiscard]] consteval RawLineWeights make_raw_line_weights(
    const std::array<std::uint16_t, kRawPatternCount>& raw_to_orbit) {
  RawLineWeights result{};
  for (std::size_t pattern = 0; pattern < result.size(); ++pattern) {
    const std::size_t orbit = raw_to_orbit[pattern];
    for (std::size_t knot = 0; knot < result[pattern].size(); ++knot) {
      result[pattern][knot] = frozen_pattern_gain_model::kLineWeights
          [knot * frozen_pattern_gain_model::kReversalOrbitCount + orbit];
    }
  }
  return result;
}

inline constexpr auto kScoringLines = make_scoring_lines();
inline constexpr auto kPatternOffsets = make_pattern_offsets();
inline constexpr auto kRawToReversalOrbit = make_raw_to_reversal_orbit();
inline constexpr auto kCellLineContributions = make_cell_line_contributions(kScoringLines);
inline constexpr auto kRawLineWeights = make_raw_line_weights(kRawToReversalOrbit);

static_assert(kScoringLines.size() == kScoringLineCount);
static_assert(kPatternOffsets[7] + 2187 == kRawPatternCount);
static_assert(*std::max_element(kRawToReversalOrbit.begin(), kRawToReversalOrbit.end()) + 1 ==
              frozen_pattern_gain_model::kReversalOrbitCount);
static_assert(frozen_pattern_gain_model::kFractionalBits == kPatternGainFractionalBits);
static_assert(frozen_pattern_gain_model::kScale == kPatternGainScale);
static_assert(frozen_pattern_gain_model::kLineWeights.size() ==
              frozen_pattern_gain_model::kLineKnots.size() *
                  frozen_pattern_gain_model::kReversalOrbitCount);
static_assert(frozen_pattern_gain_model::kGainWeights.size() ==
              frozen_pattern_gain_model::kGainKnots.size() *
                  frozen_pattern_gain_model::kGainFeatureCount);

template <std::uint16_t Digit>
void accumulate_line_codes(std::array<std::uint16_t, kScoringLineCount>& codes,
                           Bitboard pieces) noexcept {
  static_assert(Digit == 1 || Digit == 2);
  while (pieces != 0) {
    const int cell = std::countr_zero(pieces);
    pieces &= pieces - Bitboard{1};
    const CellLineContributions& contributions = kCellLineContributions[cell];
    for (const LineContribution contribution : contributions.values) {
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
  std::array<std::int64_t, frozen_pattern_gain_model::kGainFeatureCount> features{};
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

template <std::size_t KnotCount>
[[nodiscard]] std::int64_t interpolate(const std::array<std::int64_t, KnotCount>& values,
                                       const std::array<std::uint8_t, KnotCount>& knots,
                                       int ply) noexcept {
  static_assert(KnotCount >= 2);
  assert(ply >= knots.front() && ply <= knots.back());
  std::size_t upper = 1;
  while (ply > knots[upper]) {
    ++upper;
  }
  const std::size_t lower = upper - 1;
  const int left = knots[lower];
  const int right = knots[upper];
  const std::int64_t numerator = static_cast<std::int64_t>(right - ply) * values[lower] +
                                 static_cast<std::int64_t>(ply - left) * values[upper];
  return rounded_divide(numerator, right - left);
}

[[nodiscard]] std::int64_t line_residual(const Position& position) noexcept {
  std::array<std::int64_t, frozen_pattern_gain_model::kLineKnots.size()> sums{};
  const Player own = position.side_to_move();
  const Bitboard player_one = position.board().bits(Player::kOne);
  const Bitboard player_two = position.board().bits(Player::kTwo);
  const Bitboard own_pieces = own == Player::kOne ? player_one : player_two;
  const Bitboard opponent_pieces = own == Player::kOne ? player_two : player_one;
  std::array<std::uint16_t, kScoringLineCount> codes{};
  accumulate_line_codes<1>(codes, own_pieces);
  accumulate_line_codes<2>(codes, opponent_pieces);
  for (std::size_t line_index = 0; line_index < kScoringLines.size(); ++line_index) {
    const ScoringLine& line = kScoringLines[line_index];
    const std::size_t pattern = kPatternOffsets[line.length] + codes[line_index];
    const auto& weights = kRawLineWeights[pattern];
    for (std::size_t knot = 0; knot < sums.size(); ++knot) {
      sums[knot] += weights[knot];
    }
  }
  return interpolate(sums, frozen_pattern_gain_model::kLineKnots, position.ply());
}

[[nodiscard]] std::int64_t gain_residual(
    const std::array<std::int64_t, frozen_pattern_gain_model::kGainFeatureCount>& features,
    int ply) noexcept {
  std::array<std::int64_t, frozen_pattern_gain_model::kGainKnots.size()> sums{};
  for (std::size_t knot = 0; knot < sums.size(); ++knot) {
    for (std::size_t feature = 0; feature < features.size(); ++feature) {
      sums[knot] += features[feature] *
                    frozen_pattern_gain_model::kGainWeights
                        [knot * frozen_pattern_gain_model::kGainFeatureCount + feature];
    }
  }
  return interpolate(sums, frozen_pattern_gain_model::kGainKnots, ply);
}

}  // namespace

FixedEvaluation evaluate_pattern_gain_fixed(const Position& position) noexcept {
  const Bitboard legal_moves = position.legal_moves();
  return evaluate_pattern_gain_fixed(position, legal_moves, std::popcount(legal_moves));
}

FixedEvaluation evaluate_pattern_gain_fixed(const Position& position, Bitboard legal_moves,
                                            int empty_count) noexcept {
  assert(legal_moves == position.legal_moves());
  assert(empty_count == std::popcount(legal_moves));
  if (empty_count <= 2) {
    return static_cast<FixedEvaluation>(
        evaluate_two_ply_closure(position, legal_moves, empty_count) * kPatternGainScale);
  }

  const GainEvaluation gains = evaluate_gains(position, legal_moves, empty_count);
  const int ply = position.ply();
  const std::int64_t value = static_cast<std::int64_t>(gains.closure_value) * kPatternGainScale +
                             frozen_pattern_gain_model::kIntercept[static_cast<std::size_t>(ply)] +
                             line_residual(position) + gain_residual(gains.features, ply);
  assert(value >= std::numeric_limits<FixedEvaluation>::min());
  assert(value <= std::numeric_limits<FixedEvaluation>::max());
  return static_cast<FixedEvaluation>(value);
}

Score round_pattern_gain_evaluation(FixedEvaluation value) noexcept {
  return static_cast<Score>(rounded_divide(value, kPatternGainScale));
}

}  // namespace poe2::minimax
