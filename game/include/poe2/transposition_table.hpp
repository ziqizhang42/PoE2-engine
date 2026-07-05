#ifndef POE2_TRANSPOSITION_TABLE_HPP
#define POE2_TRANSPOSITION_TABLE_HPP

#include <cstddef>
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
  void clear() noexcept;

  [[nodiscard]] std::size_t capacity() const noexcept;
  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] bool empty() const noexcept;

  void store(PositionKey key, TranspositionValue value);
  void store(const Position& position, TranspositionValue value);
  [[nodiscard]] std::optional<TranspositionEntry> probe(PositionKey key) const noexcept;
  [[nodiscard]] std::optional<TranspositionEntry> probe(const Position& position) const noexcept;

 private:
  void store(PositionKey key, PositionHash hash, TranspositionValue value);
  [[nodiscard]] std::optional<TranspositionEntry> probe(PositionKey key,
                                                        PositionHash hash) const noexcept;

  struct Bucket {
    bool occupied = false;
    TranspositionEntry entry;
  };

  std::vector<Bucket> buckets_;
  std::size_t size_ = 0;
};

}  // namespace poe2

#endif  // POE2_TRANSPOSITION_TABLE_HPP
