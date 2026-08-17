#include <algorithm>
#include <array>
#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "poe2/minimax/evaluation.hpp"
#include "poe2/symmetry.hpp"

namespace {

struct ClosureOracleResult {
  poe2::Score value = 0;
  poe2::Bitboard optimal_moves = 0;
  std::array<poe2::Score, poe2::kCellCount> move_values{};
};

[[nodiscard]] poe2::Score value_for(const poe2::Position& position, poe2::Player perspective) {
  const poe2::Score value = poe2::minimax::evaluate(position);
  return perspective == position.side_to_move() ? value : -value;
}

[[nodiscard]] poe2::Score required_score_gain(const poe2::Position& position, poe2::Player player,
                                              poe2::Square square) {
  const std::optional<poe2::Score> gain = position.score_gain(player, square);
  REQUIRE(gain.has_value());
  return gain.value_or(0);
}

[[nodiscard]] ClosureOracleResult explicit_two_ply_oracle(const poe2::Position& root) {
  ClosureOracleResult result{
      .value = poe2::minimax::evaluate(root),
  };
  const poe2::Player perspective = root.side_to_move();
  poe2::Bitboard first_moves = root.legal_moves();
  if (first_moves == 0) {
    return result;
  }

  result.value = std::numeric_limits<poe2::Score>::lowest();
  poe2::Position position = root;
  while (first_moves != 0) {
    const int first_index = std::countr_zero(first_moves);
    first_moves &= first_moves - poe2::Bitboard{1};
    const poe2::Square first_square = poe2::square_from_index(first_index);
    poe2::MoveUndo first_undo;
    REQUIRE(position.make_move(first_square, first_undo));

    poe2::Score first_value = value_for(position, perspective);
    poe2::Bitboard replies = position.legal_moves();
    if (replies != 0) {
      first_value = std::numeric_limits<poe2::Score>::max();
      while (replies != 0) {
        const int reply_index = std::countr_zero(replies);
        replies &= replies - poe2::Bitboard{1};
        poe2::MoveUndo reply_undo;
        REQUIRE(position.make_move(poe2::square_from_index(reply_index), reply_undo));
        first_value = std::min(first_value, value_for(position, perspective));
        position.unmake_move(reply_undo);
      }
    }
    position.unmake_move(first_undo);

    result.move_values[static_cast<std::size_t>(first_index)] = first_value;
    if (first_value > result.value) {
      result.value = first_value;
      result.optimal_moves = poe2::square_bit(first_square);
    } else if (first_value == result.value) {
      result.optimal_moves |= poe2::square_bit(first_square);
    }
  }

  return result;
}

[[nodiscard]] ClosureOracleResult marginal_formula_oracle(const poe2::Position& position) {
  ClosureOracleResult result{
      .value = poe2::minimax::evaluate(position),
  };
  const poe2::Player player = position.side_to_move();
  const poe2::Player reply_player = poe2::opponent(player);
  const poe2::Bitboard legal_moves = position.legal_moves();
  if (legal_moves == 0) {
    return result;
  }

  result.value = std::numeric_limits<poe2::Score>::lowest();
  poe2::Bitboard first_moves = legal_moves;
  while (first_moves != 0) {
    const int first_index = std::countr_zero(first_moves);
    first_moves &= first_moves - poe2::Bitboard{1};
    const poe2::Square first_square = poe2::square_from_index(first_index);
    const poe2::Score own_gain = required_score_gain(position, player, first_square);

    poe2::Score first_value = result.move_values[static_cast<std::size_t>(first_index)] =
        poe2::minimax::evaluate(position) + 2 * (own_gain - 1);
    poe2::Bitboard replies = legal_moves & ~poe2::square_bit(first_square);
    if (replies != 0) {
      poe2::Score best_reply_gain = std::numeric_limits<poe2::Score>::lowest();
      while (replies != 0) {
        const int reply_index = std::countr_zero(replies);
        replies &= replies - poe2::Bitboard{1};
        const poe2::Score reply_gain =
            required_score_gain(position, reply_player, poe2::square_from_index(reply_index));
        best_reply_gain = std::max(best_reply_gain, reply_gain);
      }
      first_value -= 2 * (best_reply_gain - 1);
      result.move_values[static_cast<std::size_t>(first_index)] = first_value;
    }

    if (first_value > result.value) {
      result.value = first_value;
      result.optimal_moves = poe2::square_bit(first_square);
    } else if (first_value == result.value) {
      result.optimal_moves |= poe2::square_bit(first_square);
    }
  }

  return result;
}

void require_closure_matches_oracles(const poe2::Position& position) {
  const poe2::PositionKey key = position.key();
  const poe2::PositionHash hash = position.hash();
  const poe2::ScoreByPlayer scores = position.scores();
  const poe2::Bitboard legal_moves = position.legal_moves();
  const int ply = position.ply();

  const ClosureOracleResult explicit_result = explicit_two_ply_oracle(position);
  const ClosureOracleResult formula_result = marginal_formula_oracle(position);
  CAPTURE(position.ply(), position.side_to_move(), position.board().empty_count());
  REQUIRE(poe2::minimax::evaluate_two_ply_closure(position) == explicit_result.value);
  REQUIRE(formula_result.value == explicit_result.value);
  REQUIRE(formula_result.optimal_moves == explicit_result.optimal_moves);

  poe2::Bitboard moves = legal_moves;
  while (moves != 0) {
    const int move_index = std::countr_zero(moves);
    moves &= moves - poe2::Bitboard{1};
    REQUIRE(formula_result.move_values[static_cast<std::size_t>(move_index)] ==
            explicit_result.move_values[static_cast<std::size_t>(move_index)]);
  }

  REQUIRE(position.key() == key);
  REQUIRE(position.hash() == hash);
  REQUIRE(position.scores().player_one == scores.player_one);
  REQUIRE(position.scores().player_two == scores.player_two);
  REQUIRE(position.legal_moves() == legal_moves);
  REQUIRE(position.ply() == ply);
}

[[nodiscard]] poe2::Position position_from_prefix(
    const std::array<int, poe2::kCellCount>& order, int ply,
    poe2::Symmetry symmetry = poe2::Symmetry::kIdentity) {
  poe2::Position position;
  for (int index = 0; index < ply; ++index) {
    const poe2::Square square =
        poe2::transform_square(symmetry, poe2::square_from_index(order[index]));
    REQUIRE(position.play(square));
  }
  return position;
}

[[nodiscard]] poe2::Position position_from_history(
    const std::vector<poe2::Square>& history, poe2::Symmetry symmetry = poe2::Symmetry::kIdentity) {
  poe2::Position position;
  for (const poe2::Square square : history) {
    REQUIRE(position.play(poe2::transform_square(symmetry, square)));
  }
  return position;
}

}  // namespace

TEST_CASE("minimax evaluation normalizes final piece counts with the exact handicap",
          "[minimax][evaluation]") {
  poe2::Position position;

  REQUIRE(position.side_to_move() == poe2::Player::kOne);
  REQUIRE(poe2::minimax::evaluate(position) == 2 - poe2::kPlayerTwoHandicapHalfPoints);

  REQUIRE(position.play({0, 0}));
  REQUIRE(position.side_to_move() == poe2::Player::kTwo);
  REQUIRE(poe2::minimax::evaluate(position) == poe2::kPlayerTwoHandicapHalfPoints - 2);

  REQUIRE(position.play({6, 6}));
  REQUIRE(position.side_to_move() == poe2::Player::kOne);
  REQUIRE(poe2::minimax::evaluate(position) == 2 - poe2::kPlayerTwoHandicapHalfPoints);
}

TEST_CASE("final-count normalization is exactly the expected parity correction",
          "[minimax][evaluation]") {
  std::array<int, poe2::kCellCount> move_order{};
  std::iota(move_order.begin(), move_order.end(), 0);
  std::mt19937_64 generator{UINT64_C(0xbab1ec0ffee)};  // NOLINT
  std::shuffle(move_order.begin(), move_order.end(), generator);

  for (int ply = 0; ply <= poe2::kCellCount; ++ply) {
    const poe2::Position position = position_from_prefix(move_order, ply);
    const poe2::Score player_one_advantage =
        2 * (position.score(poe2::Player::kOne) - position.score(poe2::Player::kTwo)) -
        poe2::kPlayerTwoHandicapHalfPoints;
    const poe2::Score realized_value = position.side_to_move() == poe2::Player::kOne
                                           ? player_one_advantage
                                           : -player_one_advantage;
    const poe2::Score parity_correction = position.side_to_move() == poe2::Player::kOne ? 2 : 0;

    CAPTURE(ply, position.side_to_move());
    REQUIRE(poe2::minimax::evaluate(position) == realized_value + parity_correction);
  }
}

TEST_CASE("minimax evaluation is invariant under every board symmetry",
          "[minimax][evaluation][symmetry]") {
  const std::vector<poe2::Square> history{
      {0, 1}, {6, 5}, {2, 4}, {4, 2}, {1, 6}, {5, 0}, {3, 3}, {0, 4}, {6, 2},
  };
  poe2::Position base;
  for (const poe2::Square square : history) {
    REQUIRE(base.play(square));
  }
  const poe2::Score expected = poe2::minimax::evaluate(base);

  for (const poe2::Symmetry symmetry : poe2::kAllSymmetries) {
    poe2::Position equivalent;
    for (const poe2::Square square : history) {
      REQUIRE(equivalent.play(poe2::transform_square(symmetry, square)));
    }
    REQUIRE(poe2::minimax::evaluate(equivalent) == expected);
  }
}

TEST_CASE("two-ply closure handles zero one and two remaining squares exactly",
          "[minimax][evaluation][closure]") {
  std::array<int, poe2::kCellCount> row_major{};
  std::iota(row_major.begin(), row_major.end(), 0);

  for (const int empty_count : {0, 1, 2}) {
    const poe2::Position position = position_from_prefix(row_major, poe2::kCellCount - empty_count);
    REQUIRE(position.board().empty_count() == empty_count);
    require_closure_matches_oracles(position);
  }
}

TEST_CASE("two-ply closure matches explicit minimax throughout a fixed-seed D4 corpus",
          "[minimax][evaluation][closure][symmetry]") {
  std::array<int, poe2::kCellCount> move_order{};
  std::iota(move_order.begin(), move_order.end(), 0);
  // A fixed seed makes this oracle corpus exactly reproducible.
  std::mt19937_64 generator{UINT64_C(0x5deece66d4b59a73)};  // NOLINT
  std::shuffle(move_order.begin(), move_order.end(), generator);
  constexpr std::array<int, 14> kPlySamples{{0, 1, 2, 3, 7, 12, 19, 25, 31, 38, 44, 47, 48, 49}};

  for (const int ply : kPlySamples) {
    const poe2::Position base = position_from_prefix(move_order, ply);
    const ClosureOracleResult base_result = explicit_two_ply_oracle(base);
    require_closure_matches_oracles(base);

    for (const poe2::Symmetry symmetry : poe2::kAllSymmetries) {
      const poe2::Position equivalent = position_from_prefix(move_order, ply, symmetry);
      require_closure_matches_oracles(equivalent);
      REQUIRE(poe2::minimax::evaluate_two_ply_closure(equivalent) == base_result.value);
      REQUIRE(explicit_two_ply_oracle(equivalent).optimal_moves ==
              poe2::transform_bitboard(symmetry, base_result.optimal_moves));
    }
  }
}

TEST_CASE("two-ply closure covers bridges forks crossings and singleton corrections",
          "[minimax][evaluation][closure]") {
  const std::vector<std::vector<poe2::Square>> histories{
      {
          {3, 0},
          {0, 0},
          {3, 1},
          {0, 2},
          {3, 3},
          {0, 4},
          {3, 4},
          {0, 6},
          {3, 5},
      },
      {
          {3, 2},
          {0, 0},
          {3, 4},
          {0, 1},
          {2, 3},
          {0, 5},
          {4, 3},
          {0, 6},
          {2, 2},
          {1, 0},
          {4, 4},
          {1, 6},
          {2, 4},
          {5, 0},
          {4, 2},
          {5, 6},
      },
      {
          {0, 0},
          {6, 6},
          {1, 1},
          {5, 5},
          {2, 2},
          {4, 4},
          {0, 6},
          {6, 0},
          {1, 5},
          {5, 1},
          {2, 4},
          {4, 2},
      },
  };

  for (const std::vector<poe2::Square>& history : histories) {
    for (const poe2::Symmetry symmetry : poe2::kAllSymmetries) {
      require_closure_matches_oracles(position_from_history(history, symmetry));
    }
  }
}

TEST_CASE("two-ply closure handles unique tied and contested reply maxima",
          "[minimax][evaluation][closure]") {
  bool found_unique_reply_best = false;
  bool found_tied_reply_best = false;
  bool found_contested_best = false;
  bool found_own_gain_trap = false;

  for (std::uint64_t seed = 1; seed <= 12; ++seed) {
    std::array<int, poe2::kCellCount> move_order{};
    std::iota(move_order.begin(), move_order.end(), 0);
    std::mt19937_64 generator{seed};
    std::shuffle(move_order.begin(), move_order.end(), generator);

    for (int ply = 0; ply <= poe2::kCellCount - 2; ++ply) {
      const poe2::Position position = position_from_prefix(move_order, ply);
      const poe2::Player player = position.side_to_move();
      const poe2::Player reply_player = poe2::opponent(player);
      const ClosureOracleResult formula = marginal_formula_oracle(position);
      poe2::Score best_reply_gain = std::numeric_limits<poe2::Score>::lowest();
      poe2::Score best_own_gain = std::numeric_limits<poe2::Score>::lowest();
      poe2::Bitboard best_reply_squares = 0;
      poe2::Bitboard best_own_squares = 0;

      poe2::Bitboard moves = position.legal_moves();
      while (moves != 0) {
        const int move_index = std::countr_zero(moves);
        moves &= moves - poe2::Bitboard{1};
        const poe2::Square square = poe2::square_from_index(move_index);
        const poe2::Score reply_gain = required_score_gain(position, reply_player, square);
        const poe2::Score own_gain = required_score_gain(position, player, square);
        if (reply_gain > best_reply_gain) {
          best_reply_gain = reply_gain;
          best_reply_squares = poe2::square_bit(square);
        } else if (reply_gain == best_reply_gain) {
          best_reply_squares |= poe2::square_bit(square);
        }
        if (own_gain > best_own_gain) {
          best_own_gain = own_gain;
          best_own_squares = poe2::square_bit(square);
        } else if (own_gain == best_own_gain) {
          best_own_squares |= poe2::square_bit(square);
        }
      }

      const bool unique_reply_best = std::has_single_bit(best_reply_squares);
      found_unique_reply_best |= unique_reply_best;
      found_tied_reply_best |= std::popcount(best_reply_squares) > 1;
      found_contested_best |=
          unique_reply_best && (formula.optimal_moves & best_reply_squares) != 0;
      found_own_gain_trap |= (formula.optimal_moves & best_own_squares) == 0;
    }
  }

  REQUIRE(found_unique_reply_best);
  REQUIRE(found_tied_reply_best);
  REQUIRE(found_contested_best);
  REQUIRE(found_own_gain_trap);
}
