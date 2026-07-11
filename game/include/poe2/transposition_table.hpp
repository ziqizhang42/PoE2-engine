#ifndef POE2_TRANSPOSITION_TABLE_HPP
#define POE2_TRANSPOSITION_TABLE_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "poe2/board.hpp"

namespace poe2 {

enum class TranspositionBound : std::uint8_t {
  kExact = 0,
  kLower,
  kUpper,
};

struct TranspositionValue {
  Score score = 0;
  int depth = 0;
  TranspositionBound bound = TranspositionBound::kExact;
  std::optional<Square> best_move;
};

struct TranspositionEntry {
  PositionKey key;
  PositionHash hash = 0;
  TranspositionValue value;
};

class TranspositionTable final {
 public:
  explicit TranspositionTable(std::size_t capacity = 0);

  void resize(std::size_t capacity);
  void resize_bytes(std::size_t byte_budget);
  void clear() noexcept;

  [[nodiscard]] std::size_t capacity() const noexcept;
  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] std::size_t storage_bytes() const noexcept;
  [[nodiscard]] bool empty() const noexcept;

  [[nodiscard]] bool store(PositionKey key, TranspositionValue value);
  [[nodiscard]] bool store(PositionKey key, PositionHash hash, TranspositionValue value);
  [[nodiscard]] bool store(const Position& position, TranspositionValue value);
  [[nodiscard]] std::optional<TranspositionEntry> probe(PositionKey key) const noexcept;
  [[nodiscard]] std::optional<TranspositionEntry> probe(PositionKey key,
                                                        PositionHash hash) const noexcept;
  [[nodiscard]] std::optional<TranspositionEntry> probe(const Position& position) const noexcept;

 private:
  static constexpr std::size_t kWays = 2;
  static constexpr std::uint8_t kNoBestMove = std::numeric_limits<std::uint8_t>::max();
  static constexpr std::uint8_t kOccupiedMask = std::uint8_t{1} << 7;
  static constexpr std::uint8_t kBoundMask = 0x03;

  struct PackedEntry {
    PositionKey key;
    Score score = 0;
    std::uint8_t depth = 0;
    std::uint8_t best_move_index = kNoBestMove;
    std::uint8_t flags = 0;
  };

  struct alignas(64) Bucket {
    std::array<PackedEntry, kWays> slots{};
    std::uint8_t next_victim = 0;
  };

  static_assert(sizeof(PackedEntry) == 24);
  static_assert(sizeof(Bucket) == 64);

  [[nodiscard]] static bool occupied(const PackedEntry& entry) noexcept;
  [[nodiscard]] static TranspositionBound bound(const PackedEntry& entry) noexcept;
  [[nodiscard]] static TranspositionValue value(const PackedEntry& entry) noexcept;
  static void write(PackedEntry& entry, PositionKey key, TranspositionValue value) noexcept;
  [[nodiscard]] std::size_t active_ways() const noexcept;

  std::vector<Bucket> buckets_;
  std::size_t capacity_ = 0;
  std::size_t size_ = 0;
};

}  // namespace poe2

#endif  // POE2_TRANSPOSITION_TABLE_HPP
