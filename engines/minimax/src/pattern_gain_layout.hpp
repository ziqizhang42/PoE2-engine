#ifndef POE2_MINIMAX_PATTERN_GAIN_LAYOUT_HPP
#define POE2_MINIMAX_PATTERN_GAIN_LAYOUT_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "poe2/board.hpp"
#include "poe2/minimax/net.hpp"

namespace poe2::minimax::pattern_gain {

inline constexpr std::size_t kScoringLineCount = 36;
inline constexpr std::size_t kMaximumLinesPerCell = 4;

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

[[nodiscard]] consteval std::array<std::uint16_t, kPatternGainRawPatternCount>
make_raw_to_reversal_orbit() {
  constexpr std::uint16_t kUnassignedOrbit = std::numeric_limits<std::uint16_t>::max();
  std::array<std::uint16_t, kPatternGainRawPatternCount> result{};
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

inline constexpr auto kScoringLines = make_scoring_lines();
inline constexpr auto kPatternOffsets = make_pattern_offsets();
inline constexpr auto kRawToReversalOrbit = make_raw_to_reversal_orbit();
inline constexpr auto kCellLineContributions = make_cell_line_contributions(kScoringLines);

static_assert(kScoringLines.size() == kScoringLineCount);
static_assert(kPatternOffsets[7] + 2187 == kPatternGainRawPatternCount);
static_assert(*std::max_element(kRawToReversalOrbit.begin(), kRawToReversalOrbit.end()) + 1 ==
              kPatternGainOrbitCount);

}  // namespace poe2::minimax::pattern_gain

#endif  // POE2_MINIMAX_PATTERN_GAIN_LAYOUT_HPP
