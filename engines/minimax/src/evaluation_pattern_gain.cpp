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

struct ScoringLine {
  std::array<std::uint8_t, kBoardSize> cells{};
  std::uint8_t length = 0;
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

inline constexpr auto kScoringLines = make_scoring_lines();
inline constexpr auto kPatternOffsets = make_pattern_offsets();
inline constexpr auto kRawToReversalOrbit = make_raw_to_reversal_orbit();

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

template <std::size_t Size>
void insert_descending(std::array<Score, Size>& values, Score value) noexcept {
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

[[nodiscard]] GainEvaluation evaluate_gains(const Position& position, Bitboard legal_moves,
                                            int empty_count) noexcept {
  assert(empty_count > 2);
  std::array<Score, kCellCount> own_gains{};
  std::array<Score, kCellCount> opponent_gains{};
  std::array<Score, 4> own_top{};
  std::array<Score, 4> opponent_top{};
  std::array<std::int64_t, 4> own_counts{};
  std::array<std::int64_t, 4> opponent_counts{};
  constexpr std::array<Score, 4> kThresholds{{2, 4, 8, 16}};

  const Player player = position.side_to_move();
  int opponent_best_index = -1;
  int opponent_best_count = 0;
  Bitboard moves = legal_moves;
  while (moves != 0) {
    const int move_index = std::countr_zero(moves);
    moves &= moves - Bitboard{1};
    const ScoreByPlayer gains = position.score_gains_unchecked(move_index);
    const Score own = player == Player::kOne ? gains.player_one : gains.player_two;
    const Score reply = player == Player::kOne ? gains.player_two : gains.player_one;
    own_gains[static_cast<std::size_t>(move_index)] = own;
    opponent_gains[static_cast<std::size_t>(move_index)] = reply;
    insert_descending(own_top, own);
    insert_descending(opponent_top, reply);
    for (std::size_t threshold = 0; threshold < kThresholds.size(); ++threshold) {
      own_counts[threshold] += own >= kThresholds[threshold] ? 1 : 0;
      opponent_counts[threshold] += reply >= kThresholds[threshold] ? 1 : 0;
    }
  }

  moves = legal_moves;
  bool contested_best = false;
  while (moves != 0) {
    const int move_index = std::countr_zero(moves);
    moves &= moves - Bitboard{1};
    const std::size_t index = static_cast<std::size_t>(move_index);
    if (opponent_gains[index] == opponent_top[0]) {
      opponent_best_index = move_index;
      ++opponent_best_count;
    }
    contested_best = contested_best ||
                     (own_gains[index] == own_top[0] && opponent_gains[index] == opponent_top[0]);
  }

  const bool unique_opponent_best = opponent_best_count == 1;
  Score best_pair_value = std::numeric_limits<Score>::lowest();
  Score selected_reply = opponent_top[0];
  moves = legal_moves;
  while (moves != 0) {
    const int move_index = std::countr_zero(moves);
    moves &= moves - Bitboard{1};
    const std::size_t index = static_cast<std::size_t>(move_index);
    const Score reply = unique_opponent_best && move_index == opponent_best_index ? opponent_top[1]
                                                                                  : opponent_top[0];
    const Score candidate = own_gains[index] - reply;
    if (candidate > best_pair_value) {
      best_pair_value = candidate;
      selected_reply = reply;
    } else if (candidate == best_pair_value) {
      // A row-major tie-break would make the denial feature orientation dependent. Choosing the
      // largest reply is equivalent to the minimum denial among all closure-optimal squares and
      // is D4 invariant.
      selected_reply = std::max(selected_reply, reply);
    }
  }

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
  for (const std::int64_t value : own_counts) {
    result.features[feature++] = value;
  }
  for (const std::int64_t value : opponent_counts) {
    result.features[feature++] = value;
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
  const Bitboard player_one = position.board().bits(Player::kOne);
  const Bitboard player_two = position.board().bits(Player::kTwo);
  const Player side_to_move = position.side_to_move();

  for (const ScoringLine& line : kScoringLines) {
    std::uint16_t code = 0;
    std::uint16_t place = 1;
    for (std::uint8_t offset = 0; offset < line.length; ++offset) {
      const Bitboard bit = Bitboard{1} << line.cells[offset];
      std::uint16_t digit = 0;
      if ((player_one & bit) != 0) {
        digit = side_to_move == Player::kOne ? 1 : 2;
      } else if ((player_two & bit) != 0) {
        digit = side_to_move == Player::kTwo ? 1 : 2;
      }
      code = static_cast<std::uint16_t>(code + digit * place);
      place = static_cast<std::uint16_t>(place * 3);
    }
    const std::size_t pattern = kPatternOffsets[line.length] + code;
    const std::size_t orbit = kRawToReversalOrbit[pattern];
    for (std::size_t knot = 0; knot < sums.size(); ++knot) {
      sums[knot] += frozen_pattern_gain_model::kLineWeights
          [knot * frozen_pattern_gain_model::kReversalOrbitCount + orbit];
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
