#ifndef POE2_SYMMETRY_HPP
#define POE2_SYMMETRY_HPP

#include <array>
#include <cstdint>

#include "poe2/board.hpp"

namespace poe2 {

enum class Symmetry : std::uint8_t {
  kIdentity = 0,
  kRotate90,
  kRotate180,
  kRotate270,
  kReflectHorizontal,
  kReflectVertical,
  kReflectMainDiagonal,
  kReflectAntiDiagonal,
};

inline constexpr std::array<Symmetry, 8> kAllSymmetries{{
    Symmetry::kIdentity,
    Symmetry::kRotate90,
    Symmetry::kRotate180,
    Symmetry::kRotate270,
    Symmetry::kReflectHorizontal,
    Symmetry::kReflectVertical,
    Symmetry::kReflectMainDiagonal,
    Symmetry::kReflectAntiDiagonal,
}};

struct CanonicalPositionKey {
  PositionKey key;
  Symmetry transform = Symmetry::kIdentity;
  Symmetry inverse_transform = Symmetry::kIdentity;
};

struct CanonicalPositionView {
  PositionKey key;
  PositionHash hash = 0;
  Symmetry transform = Symmetry::kIdentity;
  Symmetry inverse_transform = Symmetry::kIdentity;

  friend constexpr bool operator==(const CanonicalPositionView&,
                                   const CanonicalPositionView&) = default;
};

[[nodiscard]] constexpr Square transform_square(Symmetry symmetry, Square square) noexcept {
  constexpr int last = kBoardSize - 1;

  switch (symmetry) {
    case Symmetry::kIdentity:
      return square;
    case Symmetry::kRotate90:
      return Square{square.col, last - square.row};
    case Symmetry::kRotate180:
      return Square{last - square.row, last - square.col};
    case Symmetry::kRotate270:
      return Square{last - square.col, square.row};
    case Symmetry::kReflectHorizontal:
      return Square{last - square.row, square.col};
    case Symmetry::kReflectVertical:
      return Square{square.row, last - square.col};
    case Symmetry::kReflectMainDiagonal:
      return Square{square.col, square.row};
    case Symmetry::kReflectAntiDiagonal:
      return Square{last - square.col, last - square.row};
  }

  return square;
}

[[nodiscard]] constexpr Symmetry inverse_symmetry(Symmetry symmetry) noexcept {
  switch (symmetry) {
    case Symmetry::kIdentity:
      return Symmetry::kIdentity;
    case Symmetry::kRotate90:
      return Symmetry::kRotate270;
    case Symmetry::kRotate180:
      return Symmetry::kRotate180;
    case Symmetry::kRotate270:
      return Symmetry::kRotate90;
    case Symmetry::kReflectHorizontal:
    case Symmetry::kReflectVertical:
    case Symmetry::kReflectMainDiagonal:
    case Symmetry::kReflectAntiDiagonal:
      return symmetry;
  }

  return Symmetry::kIdentity;
}

inline constexpr std::array<std::array<Bitboard, kCellCount>, kAllSymmetries.size()>
    transformed_move_bits = []() consteval {
      std::array<std::array<Bitboard, kCellCount>, kAllSymmetries.size()> result{};

      for (std::size_t symmetry_index = 0; symmetry_index < kAllSymmetries.size();
           ++symmetry_index) {
        for (int square_index_value = 0; square_index_value < kCellCount; ++square_index_value) {
          const Square square = square_from_index(square_index_value);
          result[symmetry_index][square_index_value] =
              square_bit(transform_square(kAllSymmetries[symmetry_index], square));
        }
      }

      return result;
    }();

class PositionSymmetryTracker final {
 public:
  explicit PositionSymmetryTracker(PositionKey key) noexcept;

  [[nodiscard]] Player side_to_move() const noexcept;
  [[nodiscard]] PositionKey key(Symmetry symmetry = Symmetry::kIdentity) const noexcept;
  [[nodiscard]] PositionHash hash(Symmetry symmetry = Symmetry::kIdentity) const noexcept;
  [[nodiscard]] CanonicalPositionView canonical_view() const noexcept;
  [[nodiscard]] CanonicalPositionView preview_move(Square square) const noexcept;
  [[nodiscard]] std::uint8_t stabilizer_mask() const noexcept;
  [[nodiscard]] Bitboard transformed_move_orbit(Square square) const noexcept;

  [[nodiscard]] bool make_move(Square square) noexcept;
  void unmake_move(Square square) noexcept;

 private:
  std::array<std::array<Bitboard, 2>, kAllSymmetries.size()> pieces_{};
  std::array<PositionHash, kAllSymmetries.size()> hashes_{};
  Player side_to_move_ = Player::kOne;
};

[[nodiscard]] Bitboard transform_bitboard(Symmetry symmetry, Bitboard bits) noexcept;
[[nodiscard]] PositionKey transform_position_key(Symmetry symmetry, PositionKey key) noexcept;
[[nodiscard]] CanonicalPositionKey canonicalize_position_key(PositionKey key) noexcept;

}  // namespace poe2

#endif  // POE2_SYMMETRY_HPP
