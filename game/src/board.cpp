#include "poe2/board.hpp"

#include <array>
#include <bit>
#include <cassert>
#include <limits>

namespace poe2 {

namespace {

struct Direction {
  int row_delta = 0;
  int col_delta = 0;
};

struct Run {
  int length = 0;
  Bitboard bits = 0;
};

struct PlayerScoreState {
  Score total = 0;
  Bitboard pieces_in_lines = 0;
};

struct ScoreUpdate {
  Score delta = 0;
  Bitboard pieces_in_lines = 0;
};

struct LineCoordinate {
  std::uint8_t line = 0;
  std::uint8_t offset = 0;
};

constexpr std::array<Direction, 4> kLineDirections{{
    {0, 1},
    {1, 0},
    {1, 1},
    {1, -1},
}};
constexpr std::size_t kLineCount = 2 * kBoardSize - 1;
constexpr std::size_t kLinePatternCount = std::size_t{1} << kBoardSize;
constexpr std::size_t kLineStateCount = 2187;  // 3^7 states: empty, unscored, or scored per cell.
constexpr int kCachedLineDeltaShift = 56;
using LineStates = std::array<std::array<std::uint16_t, kLineCount>, kLineDirections.size()>;

constexpr std::array<std::uint16_t, kBoardSize> kLineStateIncrements = []() consteval {
  std::array<std::uint16_t, kBoardSize> increments{};
  std::uint16_t value = 1;
  for (std::uint16_t& increment : increments) {
    increment = value;
    value = static_cast<std::uint16_t>(value * 3);
  }
  return increments;
}();
static_assert(static_cast<std::size_t>(kLineStateIncrements.back()) * 3 == kLineStateCount);

[[nodiscard]] constexpr PositionHash mix64(PositionHash value) noexcept {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

[[nodiscard]] constexpr std::array<std::array<PositionHash, kCellCount>, 2>
make_zobrist_piece_hashes() noexcept {
  std::array<std::array<PositionHash, kCellCount>, 2> hashes{};

  for (int player = 0; player < 2; ++player) {
    for (int index = 0; index < kCellCount; ++index) {
      const PositionHash key =
          (static_cast<PositionHash>(player) << 32) | static_cast<PositionHash>(index);
      hashes[player][index] = mix64(0x6a09e667f3bcc909ULL ^ key);
    }
  }

  return hashes;
}

constexpr std::array<std::array<PositionHash, kCellCount>, 2> kZobristPieceHashes =
    make_zobrist_piece_hashes();
constexpr PositionHash kZobristSideToMoveHash = mix64(0xbb67ae8584caa73bULL);

[[nodiscard]] constexpr Square step(Square square, Direction direction) noexcept {
  return Square{square.row + direction.row_delta, square.col + direction.col_delta};
}

[[nodiscard]] constexpr Square previous(Square square, Direction direction) noexcept {
  return Square{square.row - direction.row_delta, square.col - direction.col_delta};
}

[[nodiscard]] bool contains(Bitboard bits, Square square) noexcept {
  return is_valid(square) && (bits & square_bit(square)) != 0;
}

[[nodiscard]] constexpr Score line_score(int length) noexcept { return Score{1} << (length - 1); }

[[nodiscard]] constexpr Score line_contribution(int length) noexcept {
  return length >= 2 ? line_score(length) : 0;
}

static_assert(kCellCount < kCachedLineDeltaShift);
static_assert(line_score(kBoardSize) < (Bitboard{1} << (64 - kCachedLineDeltaShift)));

[[nodiscard]] constexpr LineCoordinate line_coordinate(Square square,
                                                       std::size_t direction) noexcept {
  if (direction == 0) {
    return LineCoordinate{.line = static_cast<std::uint8_t>(square.row),
                          .offset = static_cast<std::uint8_t>(square.col)};
  }
  if (direction == 1) {
    return LineCoordinate{.line = static_cast<std::uint8_t>(square.col),
                          .offset = static_cast<std::uint8_t>(square.row)};
  }
  if (direction == 2) {
    return LineCoordinate{
        .line = static_cast<std::uint8_t>(square.col - square.row + kBoardSize - 1),
        .offset = static_cast<std::uint8_t>(square.row < square.col ? square.row : square.col),
    };
  }

  const int sum = square.row + square.col;
  const int first_row = sum < kBoardSize ? 0 : sum - kBoardSize + 1;
  return LineCoordinate{
      .line = static_cast<std::uint8_t>(sum),
      .offset = static_cast<std::uint8_t>(square.row - first_row),
  };
}

[[nodiscard]] constexpr auto make_line_coordinates() noexcept {
  std::array<std::array<LineCoordinate, kLineDirections.size()>, kCellCount> coordinates{};
  for (int square_index_value = 0; square_index_value < kCellCount; ++square_index_value) {
    const Square square = square_from_index(square_index_value);
    for (std::size_t direction = 0; direction < kLineDirections.size(); ++direction) {
      coordinates[square_index_value][direction] = line_coordinate(square, direction);
    }
  }
  return coordinates;
}

[[nodiscard]] constexpr auto make_line_score_updates() noexcept {
  std::array<std::array<std::array<Bitboard, kLinePatternCount>, kLineDirections.size()>,
             kCellCount>
      updates{};

  for (int square_index_value = 0; square_index_value < kCellCount; ++square_index_value) {
    const Square square = square_from_index(square_index_value);
    for (std::size_t direction_index = 0; direction_index < kLineDirections.size();
         ++direction_index) {
      const Direction direction = kLineDirections[direction_index];
      const LineCoordinate coordinate = line_coordinate(square, direction_index);

      for (std::size_t pattern = 0; pattern < kLinePatternCount; ++pattern) {
        int backward_length = 0;
        for (int offset = static_cast<int>(coordinate.offset) - 1;
             offset >= 0 && (pattern & (std::size_t{1} << offset)) != 0; --offset) {
          ++backward_length;
        }

        int forward_length = 0;
        for (int offset = static_cast<int>(coordinate.offset) + 1;
             offset < kBoardSize && (pattern & (std::size_t{1} << offset)) != 0; ++offset) {
          ++forward_length;
        }

        const int merged_length = backward_length + 1 + forward_length;
        Score delta = 0;
        Bitboard pieces_in_line = 0;
        if (merged_length >= 2) {
          delta = line_score(merged_length) - line_contribution(backward_length) -
                  line_contribution(forward_length);
          for (int offset = static_cast<int>(coordinate.offset) - backward_length;
               offset <= static_cast<int>(coordinate.offset) + forward_length; ++offset) {
            const int distance = offset - static_cast<int>(coordinate.offset);
            const Square line_square{
                square.row + distance * direction.row_delta,
                square.col + distance * direction.col_delta,
            };
            pieces_in_line |= square_bit(line_square);
          }
        }

        // Board bits occupy the low 49 bits; keep the directional score delta in the top byte.
        updates[square_index_value][direction_index][pattern] =
            pieces_in_line | (static_cast<Bitboard>(delta) << kCachedLineDeltaShift);
      }
    }
  }

  return updates;
}

inline constexpr auto kLineCoordinates = make_line_coordinates();
const auto kLineScoreUpdates = make_line_score_updates();

[[nodiscard]] constexpr auto make_line_gain_updates() noexcept {
  struct DecodedState {
    std::uint8_t occupancy = 0;
    std::uint8_t scored = 0;
  };
  std::array<DecodedState, kLineStateCount> decoded_states{};
  for (std::size_t state = 0; state < kLineStateCount; ++state) {
    std::size_t remaining = state;
    for (int offset = 0; offset < kBoardSize; ++offset) {
      const std::size_t cell = remaining % 3;
      remaining /= 3;
      if (cell != 0) {
        decoded_states[state].occupancy |= static_cast<std::uint8_t>(std::uint8_t{1} << offset);
      }
      if (cell == 2) {
        decoded_states[state].scored |= static_cast<std::uint8_t>(std::uint8_t{1} << offset);
      }
    }
  }

  std::array<std::array<std::int8_t, kLineStateCount>, kBoardSize> updates{};

  for (int move_offset = 0; move_offset < kBoardSize; ++move_offset) {
    const std::uint8_t move_bit = static_cast<std::uint8_t>(std::uint8_t{1} << move_offset);
    for (std::size_t state = 0; state < kLineStateCount; ++state) {
      const std::uint8_t occupancy = decoded_states[state].occupancy;
      int backward_length = 0;
      for (int offset = move_offset - 1;
           offset >= 0 && (occupancy & (std::size_t{1} << offset)) != 0; --offset) {
        ++backward_length;
      }

      int forward_length = 0;
      for (int offset = move_offset + 1;
           offset < kBoardSize && (occupancy & (std::size_t{1} << offset)) != 0; ++offset) {
        ++forward_length;
      }

      const int merged_length = backward_length + 1 + forward_length;
      Score delta = 0;
      std::uint8_t existing_run = 0;
      if (merged_length >= 2) {
        delta = line_score(merged_length) - line_contribution(backward_length) -
                line_contribution(forward_length);
        const std::uint8_t run = static_cast<std::uint8_t>(((std::uint16_t{1} << merged_length) - 1)
                                                           << (move_offset - backward_length));
        existing_run = static_cast<std::uint8_t>(run & ~move_bit);
      }

      const int gain =
          delta -
          std::popcount(static_cast<std::uint8_t>(existing_run & ~decoded_states[state].scored));
      assert(gain >= std::numeric_limits<std::int8_t>::min());
      assert(gain <= std::numeric_limits<std::int8_t>::max());
      updates[move_offset][state] = static_cast<std::int8_t>(gain);
    }
  }

  return updates;
}

const auto kLineGainUpdates = make_line_gain_updates();

[[nodiscard]] constexpr auto make_line_state_occupancies() noexcept {
  std::array<std::uint8_t, kLineStateCount> occupancies{};
  for (std::size_t state = 0; state < kLineStateCount; ++state) {
    std::size_t remaining = state;
    for (int offset = 0; offset < kBoardSize; ++offset) {
      if (remaining % 3 != 0) {
        occupancies[state] |= static_cast<std::uint8_t>(std::uint8_t{1} << offset);
      }
      remaining /= 3;
    }
  }
  return occupancies;
}

inline constexpr auto kLineStateOccupancies = make_line_state_occupancies();

[[nodiscard]] Run scan_run(Bitboard pieces, Square start, Direction direction) noexcept {
  Run run;

  for (Square current = start; contains(pieces, current); current = step(current, direction)) {
    ++run.length;
    run.bits |= square_bit(current);
  }

  return run;
}

[[nodiscard]] Score& mutable_score_for_player(ScoreByPlayer& scores, Player player) noexcept {
  return player == Player::kOne ? scores.player_one : scores.player_two;
}

[[nodiscard]] Score score_for_player(ScoreByPlayer scores, Player player) noexcept {
  return player == Player::kOne ? scores.player_one : scores.player_two;
}

[[nodiscard]] PlayerScoreState score_state(const Board& board, Player player) noexcept {
  const Bitboard pieces = board.bits(player);
  PlayerScoreState state;

  for (int row = 0; row < kBoardSize; ++row) {
    for (int col = 0; col < kBoardSize; ++col) {
      const Square start{row, col};
      if (!contains(pieces, start)) {
        continue;
      }

      for (const Direction direction : kLineDirections) {
        if (contains(pieces, previous(start, direction))) {
          continue;
        }

        const Run run = scan_run(pieces, start, direction);
        if (run.length >= 2) {
          state.total += line_score(run.length);
          state.pieces_in_lines |= run.bits;
        }
      }
    }
  }

  state.total += std::popcount(pieces & ~state.pieces_in_lines);
  return state;
}

[[nodiscard]] ScoreUpdate score_update_for_move(Bitboard existing_pieces_in_lines, int move_index,
                                                const LineStates& states) noexcept {
  ScoreUpdate update;

  for (std::size_t direction = 0; direction < kLineDirections.size(); ++direction) {
    const LineCoordinate coordinate = kLineCoordinates[move_index][direction];
    const std::size_t occupancy = kLineStateOccupancies[states[direction][coordinate.line]];
    const Bitboard cached = kLineScoreUpdates[move_index][direction][occupancy];
    update.delta += static_cast<Score>(cached >> kCachedLineDeltaShift);
    update.pieces_in_lines |= cached & kBoardMask;
  }

  // Every cached run contains the new move bit, while all its other bits are existing pieces.
  // Runs through the move intersect only at that bit, so count it once and cancel it with the
  // singleton point. This also covers the no-run case, where the popcount is zero.
  update.delta += 1 - std::popcount(update.pieces_in_lines & ~existing_pieces_in_lines);

  return update;
}

[[nodiscard]] Score score_gain_for_move(int move_index, const LineStates& states) noexcept {
  Score gain = 0;
  for (std::size_t direction = 0; direction < kLineDirections.size(); ++direction) {
    const LineCoordinate coordinate = kLineCoordinates[move_index][direction];
    gain += kLineGainUpdates[coordinate.offset][states[direction][coordinate.line]];
  }
  return gain == 0 ? 1 : gain;
}

}  // namespace

Bitboard Board::bits(Player player) const noexcept { return pieces_[player_index(player)]; }

Bitboard Board::occupied() const noexcept { return pieces_[0] | pieces_[1]; }

Bitboard Board::empty_squares() const noexcept { return kBoardMask & ~occupied(); }

Cell Board::cell_at(Square square) const noexcept {
  if (!is_valid(square)) {
    return Cell::kEmpty;
  }

  const Bitboard bit = square_bit(square);
  if ((pieces_[0] & bit) != 0) {
    return Cell::kPlayerOne;
  }
  if ((pieces_[1] & bit) != 0) {
    return Cell::kPlayerTwo;
  }
  return Cell::kEmpty;
}

bool Board::can_place(Square square) const noexcept {
  return is_valid(square) && (occupied() & square_bit(square)) == 0;
}

bool Board::is_empty(Square square) const noexcept {
  return is_valid(square) && cell_at(square) == Cell::kEmpty;
}

bool Board::is_full() const noexcept { return empty_squares() == 0; }

int Board::piece_count() const noexcept { return std::popcount(occupied()); }

int Board::empty_count() const noexcept { return std::popcount(empty_squares()); }

bool Board::place(Player player, Square square) noexcept {
  if (!can_place(square)) {
    return false;
  }

  place_unchecked(player, square_index(square));
  return true;
}

bool Board::remove(Player player, Square square) noexcept {
  if (!is_valid(square)) {
    return false;
  }

  const int move_index = square_index(square);
  if ((pieces_[player_index(player)] & (Bitboard{1} << move_index)) == 0) {
    return false;
  }

  remove_unchecked(player, move_index);
  return true;
}

void Board::place_unchecked(Player player, int move_index) noexcept {
  assert(move_index >= 0 && move_index < kCellCount);
  const Bitboard bit = Bitboard{1} << move_index;
  assert((occupied() & bit) == 0);
  pieces_[player_index(player)] |= bit;
}

void Board::remove_unchecked(Player player, int move_index) noexcept {
  assert(move_index >= 0 && move_index < kCellCount);
  const Bitboard bit = Bitboard{1} << move_index;
  Bitboard& pieces = pieces_[player_index(player)];
  assert((pieces & bit) != 0);
  pieces &= ~bit;
}

const Board& Position::board() const noexcept { return board_; }

Player Position::side_to_move() const noexcept { return side_to_move_; }

int Position::ply() const noexcept { return ply_; }

Bitboard Position::legal_moves() const noexcept { return board_.empty_squares(); }

Score Position::score(Player player) const noexcept { return score_for_player(scores_, player); }

std::optional<Score> Position::score_gain(Player player, Square square) const noexcept {
  if (!board_.can_place(square)) {
    return std::nullopt;
  }

  return score_gain_unchecked(player, square);
}

Score Position::score_gain_unchecked(Player player, Square square) const noexcept {
  assert(board_.can_place(square));
  return score_gain_unchecked(player, square_index(square));
}

Score Position::score_gain_unchecked(Player player, int move_index) const noexcept {
  assert(move_index >= 0 && move_index < kCellCount);
  assert((board_.occupied() & (Bitboard{1} << move_index)) == 0);
  const int index = player_index(player);
  return score_gain_for_move(move_index, line_states_[index]);
}

[[gnu::always_inline]] ScoreByPlayer Position::score_gains_unchecked(
    int move_index) const noexcept {
  assert(move_index >= 0 && move_index < kCellCount);
  assert((board_.occupied() & (Bitboard{1} << move_index)) == 0);

  return ScoreByPlayer{
      .player_one = score_gain_for_move(move_index, line_states_[0]),
      .player_two = score_gain_for_move(move_index, line_states_[1]),
  };
}

ScoreByPlayer Position::scores() const noexcept { return scores_; }

PositionKey Position::key() const noexcept {
  return make_position_key(board_.bits(Player::kOne), board_.bits(Player::kTwo), side_to_move_);
}

PositionHash Position::hash() const noexcept { return hash_; }

bool Position::is_full() const noexcept { return board_.is_full(); }

bool Position::play(Square square) noexcept {
  MoveUndo undo;
  return make_move(square, undo);
}

bool Position::make_move(Square square, MoveUndo& undo) noexcept {
  if (!board_.can_place(square)) {
    return false;
  }

  make_move_unchecked(square_index(square), undo);
  return true;
}

void Position::make_move_unchecked(int move_index, MoveUndo& undo) noexcept {
  assert(move_index >= 0 && move_index < kCellCount);
  assert((board_.occupied() & (Bitboard{1} << move_index)) == 0);

  const Player player = side_to_move_;
  const int index = player_index(player);
  const ScoreUpdate update =
      score_update_for_move(pieces_in_lines_[index], move_index, line_states_[index]);
  undo = MoveUndo{
      .player = player,
      .move_index = static_cast<std::uint8_t>(move_index),
      .previous_hash = hash_,
      .previous_score = score_for_player(scores_, player),
      .previous_pieces_in_lines = pieces_in_lines_[index],
  };

  board_.place_unchecked(player, move_index);

  for (std::size_t direction = 0; direction < kLineDirections.size(); ++direction) {
    const LineCoordinate coordinate = kLineCoordinates[move_index][direction];
    line_states_[index][direction][coordinate.line] = static_cast<std::uint16_t>(
        line_states_[index][direction][coordinate.line] + kLineStateIncrements[coordinate.offset]);
  }

  Bitboard newly_scored = update.pieces_in_lines & ~pieces_in_lines_[index];
  while (newly_scored != 0) {
    const int scored_index = std::countr_zero(newly_scored);
    newly_scored &= newly_scored - Bitboard{1};
    for (std::size_t direction = 0; direction < kLineDirections.size(); ++direction) {
      const LineCoordinate coordinate = kLineCoordinates[scored_index][direction];
      line_states_[index][direction][coordinate.line] =
          static_cast<std::uint16_t>(line_states_[index][direction][coordinate.line] +
                                     kLineStateIncrements[coordinate.offset]);
    }
  }

  mutable_score_for_player(scores_, player) += update.delta;
  pieces_in_lines_[index] |= update.pieces_in_lines;
  hash_ = update_position_hash(hash_, player, move_index);
  side_to_move_ = opponent(side_to_move_);
  ++ply_;
}

void Position::unmake_move(const MoveUndo& undo) noexcept {
  assert(ply_ > 0);
  assert(side_to_move_ == opponent(undo.player));

  const int index = player_index(undo.player);
  const int move_index = undo.move_index;
  board_.remove_unchecked(undo.player, move_index);
  for (std::size_t direction = 0; direction < kLineDirections.size(); ++direction) {
    const LineCoordinate coordinate = kLineCoordinates[move_index][direction];
    line_states_[index][direction][coordinate.line] = static_cast<std::uint16_t>(
        line_states_[index][direction][coordinate.line] - kLineStateIncrements[coordinate.offset]);
  }
  Bitboard newly_scored = pieces_in_lines_[index] & ~undo.previous_pieces_in_lines;
  while (newly_scored != 0) {
    const int scored_index = std::countr_zero(newly_scored);
    newly_scored &= newly_scored - Bitboard{1};
    for (std::size_t direction = 0; direction < kLineDirections.size(); ++direction) {
      const LineCoordinate coordinate = kLineCoordinates[scored_index][direction];
      line_states_[index][direction][coordinate.line] =
          static_cast<std::uint16_t>(line_states_[index][direction][coordinate.line] -
                                     kLineStateIncrements[coordinate.offset]);
    }
  }
  mutable_score_for_player(scores_, undo.player) = undo.previous_score;
  pieces_in_lines_[index] = undo.previous_pieces_in_lines;
  hash_ = undo.previous_hash;
  side_to_move_ = undo.player;
  --ply_;
}

Score score(const Board& board, Player player) noexcept { return score_state(board, player).total; }

ScoreByPlayer score(const Board& board) noexcept {
  return ScoreByPlayer{
      .player_one = score(board, Player::kOne),
      .player_two = score(board, Player::kTwo),
  };
}

PositionHash position_key_hash(PositionKey key) noexcept {
  PositionHash hash = 0;

  for (const Player player : {Player::kOne, Player::kTwo}) {
    Bitboard bits = position_key_bits(key, player);
    while (bits != 0) {
      const int index = std::countr_zero(bits);
      hash ^= kZobristPieceHashes[player_index(player)][index];
      bits &= bits - Bitboard{1};
    }
  }

  if (position_key_side_to_move(key) == Player::kTwo) {
    hash ^= kZobristSideToMoveHash;
  }

  return hash;
}

PositionHash update_position_hash(PositionHash hash, Player player, Square square) noexcept {
  assert(is_valid(square));
  return update_position_hash(hash, player, square_index(square));
}

PositionHash update_position_hash(PositionHash hash, Player player, int move_index) noexcept {
  assert(move_index >= 0 && move_index < kCellCount);
  return hash ^ kZobristPieceHashes[player_index(player)][move_index] ^ kZobristSideToMoveHash;
}

Player leader_after_handicap(ScoreByPlayer scores) noexcept {
  const int player_one_half_points = scores.player_one * 2;
  const int player_two_half_points = scores.player_two * 2 + kPlayerTwoHandicapHalfPoints;

  return player_one_half_points > player_two_half_points ? Player::kOne : Player::kTwo;
}

std::optional<GameResult> result_if_full(const Board& board) noexcept {
  if (!board.is_full()) {
    return std::nullopt;
  }

  const ScoreByPlayer scores = score(board);
  return GameResult{
      .scores = scores,
      .winner = leader_after_handicap(scores),
  };
}

std::optional<GameResult> result_if_full(const Position& position) noexcept {
  if (!position.is_full()) {
    return std::nullopt;
  }

  const ScoreByPlayer scores = position.scores();
  return GameResult{
      .scores = scores,
      .winner = leader_after_handicap(scores),
  };
}

std::optional<Player> winner_if_full(const Board& board) noexcept {
  const std::optional<GameResult> result = result_if_full(board);
  if (!result.has_value()) {
    return std::nullopt;
  }

  return result->winner;
}

std::optional<Player> winner_if_full(const Position& position) noexcept {
  const std::optional<GameResult> result = result_if_full(position);
  if (!result.has_value()) {
    return std::nullopt;
  }

  return result->winner;
}

}  // namespace poe2
