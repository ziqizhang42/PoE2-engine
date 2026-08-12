#include "poe2/symmetry.hpp"

#include <bit>
#include <cassert>

namespace poe2 {

namespace {

[[nodiscard]] constexpr bool position_key_less(PositionKey lhs, PositionKey rhs) noexcept {
  return lhs.low < rhs.low || (lhs.low == rhs.low && lhs.high < rhs.high);
}

[[nodiscard]] constexpr std::size_t symmetry_index(Symmetry symmetry) noexcept {
  return static_cast<std::size_t>(symmetry);
}

[[nodiscard]] CanonicalPositionView canonical_view_from_orientations(
    const std::array<std::array<Bitboard, 2>, kAllSymmetries.size()>& pieces,
    const std::array<PositionHash, kAllSymmetries.size()>& hashes, Player side_to_move) noexcept {
  PositionKey canonical_key = make_position_key(pieces[0][0], pieces[0][1], side_to_move);
  std::size_t canonical_index = 0;

  for (std::size_t index = 1; index < kAllSymmetries.size(); ++index) {
    const PositionKey candidate =
        make_position_key(pieces[index][0], pieces[index][1], side_to_move);
    if (position_key_less(candidate, canonical_key)) {
      canonical_key = candidate;
      canonical_index = index;
    }
  }

  const Symmetry transform = kAllSymmetries[canonical_index];
  return CanonicalPositionView{
      .key = canonical_key,
      .hash = hashes[canonical_index],
      .transform = transform,
      .inverse_transform = inverse_symmetry(transform),
  };
}

}  // namespace

Bitboard transform_bitboard(Symmetry symmetry, Bitboard bits) noexcept {
  Bitboard transformed = 0;
  bits &= kBoardMask;

  while (bits != 0) {
    const int index = std::countr_zero(bits);
    transformed |= square_bit(transform_square(symmetry, square_from_index(index)));
    bits &= bits - Bitboard{1};
  }

  return transformed;
}

PositionKey transform_position_key(Symmetry symmetry, PositionKey key) noexcept {
  return make_position_key(transform_bitboard(symmetry, position_key_bits(key, Player::kOne)),
                           transform_bitboard(symmetry, position_key_bits(key, Player::kTwo)),
                           position_key_side_to_move(key));
}

CanonicalPositionKey canonicalize_position_key(PositionKey key) noexcept {
  PositionKey canonical_key = transform_position_key(Symmetry::kIdentity, key);
  Symmetry canonical_transform = Symmetry::kIdentity;

  for (const Symmetry symmetry : kAllSymmetries) {
    const PositionKey transformed = transform_position_key(symmetry, key);
    if (position_key_less(transformed, canonical_key)) {
      canonical_key = transformed;
      canonical_transform = symmetry;
    }
  }

  return CanonicalPositionKey{
      .key = canonical_key,
      .transform = canonical_transform,
      .inverse_transform = inverse_symmetry(canonical_transform),
  };
}

PositionSymmetryTracker::PositionSymmetryTracker(PositionKey key) noexcept
    : side_to_move_(position_key_side_to_move(key)) {
  for (const Symmetry symmetry : kAllSymmetries) {
    const std::size_t index = symmetry_index(symmetry);
    const PositionKey transformed = transform_position_key(symmetry, key);
    pieces_[index][0] = position_key_bits(transformed, Player::kOne);
    pieces_[index][1] = position_key_bits(transformed, Player::kTwo);
    hashes_[index] = position_key_hash(transformed);
  }
}

Player PositionSymmetryTracker::side_to_move() const noexcept { return side_to_move_; }

PositionKey PositionSymmetryTracker::key(Symmetry symmetry) const noexcept {
  const std::size_t index = symmetry_index(symmetry);
  return make_position_key(pieces_[index][0], pieces_[index][1], side_to_move_);
}

PositionHash PositionSymmetryTracker::hash(Symmetry symmetry) const noexcept {
  return hashes_[symmetry_index(symmetry)];
}

CanonicalPositionView PositionSymmetryTracker::canonical_view() const noexcept {
  return canonical_view_from_orientations(pieces_, hashes_, side_to_move_);
}

CanonicalPositionView PositionSymmetryTracker::preview_move(Square square) const noexcept {
  assert(is_valid(square));
  return preview_move(square_index(square));
}

CanonicalPositionView PositionSymmetryTracker::preview_move(int move_index) const noexcept {
  assert(move_index >= 0 && move_index < kCellCount);
  const Bitboard live_move = Bitboard{1} << move_index;
  assert((pieces_[0][0] & live_move) == 0);
  assert((pieces_[0][1] & live_move) == 0);

  const int player = player_index(side_to_move_);
  PositionKey canonical_key =
      make_position_key(pieces_[0][0] | (player == 0 ? live_move : 0),
                        pieces_[0][1] | (player == 1 ? live_move : 0), opponent(side_to_move_));
  PositionHash canonical_hash = update_position_hash(hashes_[0], side_to_move_, move_index);
  std::size_t canonical_index = 0;

  for (std::size_t index = 1; index < kAllSymmetries.size(); ++index) {
    const Bitboard move_bit = transformed_move_bits[index][move_index];
    const PositionKey candidate = make_position_key(
        pieces_[index][0] | (player == 0 ? move_bit : 0),
        pieces_[index][1] | (player == 1 ? move_bit : 0), opponent(side_to_move_));
    if (position_key_less(candidate, canonical_key)) {
      canonical_key = candidate;
      canonical_hash =
          update_position_hash(hashes_[index], side_to_move_, std::countr_zero(move_bit));
      canonical_index = index;
    }
  }

  const Symmetry transform = kAllSymmetries[canonical_index];
  return CanonicalPositionView{
      .key = canonical_key,
      .hash = canonical_hash,
      .transform = transform,
      .inverse_transform = inverse_symmetry(transform),
  };
}

std::uint8_t PositionSymmetryTracker::stabilizer_mask() const noexcept {
  std::uint8_t mask = 1;

  for (std::size_t symmetry = 1; symmetry < kAllSymmetries.size(); ++symmetry) {
    if (pieces_[symmetry] == pieces_[0]) {
      mask |= static_cast<std::uint8_t>(std::uint8_t{1} << symmetry);
    }
  }

  return mask;
}

Bitboard PositionSymmetryTracker::transformed_move_orbit(Square square) const noexcept {
  assert(is_valid(square));
  const std::uint8_t mask = stabilizer_mask();
  const int index = square_index(square);
  if (mask == 1) {
    return Bitboard{1} << index;
  }
  Bitboard orbit = Bitboard{1} << index;

  for (std::size_t symmetry = 1; symmetry < kAllSymmetries.size(); ++symmetry) {
    if ((mask & (std::uint8_t{1} << symmetry)) != 0) {
      orbit |= transformed_move_bits[symmetry][index];
    }
  }

  return orbit;
}

bool PositionSymmetryTracker::make_move(Square square) noexcept {
  if (!is_valid(square)) {
    return false;
  }
  return make_move(square_index(square));
}

bool PositionSymmetryTracker::make_move(int move_index) noexcept {
  if (move_index < 0 || move_index >= kCellCount) {
    return false;
  }
  const Bitboard live_move = Bitboard{1} << move_index;
  if (((pieces_[0][0] | pieces_[0][1]) & live_move) != 0) {
    return false;
  }

  const int player = player_index(side_to_move_);
  pieces_[0][player] |= live_move;
  hashes_[0] = update_position_hash(hashes_[0], side_to_move_, move_index);
  for (std::size_t index = 1; index < kAllSymmetries.size(); ++index) {
    const Bitboard move_bit = transformed_move_bits[index][move_index];
    pieces_[index][player] |= move_bit;
    hashes_[index] =
        update_position_hash(hashes_[index], side_to_move_, std::countr_zero(move_bit));
  }
  side_to_move_ = opponent(side_to_move_);
  return true;
}

void PositionSymmetryTracker::unmake_move(Square square) noexcept {
  assert(is_valid(square));
  unmake_move(square_index(square));
}

void PositionSymmetryTracker::unmake_move(int move_index) noexcept {
  assert(move_index >= 0 && move_index < kCellCount);
  const Player player = opponent(side_to_move_);
  const int player_array_index = player_index(player);
  const Bitboard live_move = Bitboard{1} << move_index;

  assert((pieces_[0][player_array_index] & live_move) != 0);
  pieces_[0][player_array_index] ^= live_move;
  hashes_[0] = update_position_hash(hashes_[0], player, move_index);
  for (std::size_t index = 1; index < kAllSymmetries.size(); ++index) {
    const Bitboard move_bit = transformed_move_bits[index][move_index];
    assert((pieces_[index][player_array_index] & move_bit) != 0);
    pieces_[index][player_array_index] ^= move_bit;
    hashes_[index] = update_position_hash(hashes_[index], player, std::countr_zero(move_bit));
  }
  side_to_move_ = player;
}

}  // namespace poe2
