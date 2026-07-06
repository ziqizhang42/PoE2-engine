#include "poe2/symmetry.hpp"

#include <bit>

namespace poe2 {

namespace {

[[nodiscard]] constexpr bool position_key_less(PositionKey lhs, PositionKey rhs) noexcept {
  return lhs.low < rhs.low || (lhs.low == rhs.low && lhs.high < rhs.high);
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

}  // namespace poe2
