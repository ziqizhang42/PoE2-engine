#include "poe2/transposition_table.hpp"

#include <algorithm>
#include <bit>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <utility>

namespace poe2 {

namespace {

[[nodiscard]] std::size_t normalized_slot_capacity(std::size_t capacity) {
  if (capacity <= 1) {
    return capacity;
  }

  constexpr std::size_t kMaxPowerOfTwo = std::size_t{1}
                                         << (std::numeric_limits<std::size_t>::digits - 1);
  if (capacity > kMaxPowerOfTwo) {
    throw std::length_error("transposition table capacity is too large");
  }

  return std::bit_ceil(capacity);
}

[[nodiscard]] std::size_t bucket_index(PositionHash hash, std::size_t capacity) noexcept {
  assert(capacity > 0);
  assert(std::has_single_bit(capacity));
  return static_cast<std::size_t>(hash & static_cast<PositionHash>(capacity - 1));
}

}  // namespace

TranspositionTable::TranspositionTable(std::size_t capacity) { resize(capacity); }

void TranspositionTable::resize(std::size_t capacity) {
  const std::size_t slot_capacity = normalized_slot_capacity(capacity);
  const std::size_t bucket_count = (slot_capacity + kWays - 1) / kWays;
  std::vector<Bucket> buckets(bucket_count);
  buckets_ = std::move(buckets);
  capacity_ = slot_capacity;
  size_ = 0;
}

void TranspositionTable::resize_bytes(std::size_t byte_budget) {
  const std::size_t maximum_bucket_count = byte_budget / sizeof(Bucket);
  const std::size_t bucket_count =
      maximum_bucket_count == 0 ? 0 : std::bit_floor(maximum_bucket_count);
  buckets_ = std::vector<Bucket>(bucket_count);
  capacity_ = bucket_count * kWays;
  size_ = 0;
}

void TranspositionTable::clear() noexcept {
  for (Bucket& bucket : buckets_) {
    for (PackedEntry& entry : bucket.slots) {
      entry.flags = 0;
    }
    bucket.next_victim = 0;
  }
  size_ = 0;
}

std::size_t TranspositionTable::capacity() const noexcept { return capacity_; }

std::size_t TranspositionTable::size() const noexcept { return size_; }

std::size_t TranspositionTable::storage_bytes() const noexcept {
  return buckets_.size() * sizeof(Bucket);
}

bool TranspositionTable::empty() const noexcept { return size_ == 0; }

bool TranspositionTable::store(PositionKey key, TranspositionValue value) {
  if (buckets_.empty()) {
    return false;
  }

  const PositionHash hash = position_key_hash(key);
  return store(key, hash, value);
}

bool TranspositionTable::store(const Position& position, TranspositionValue value) {
  return store(position.key(), position.hash(), value);
}

bool TranspositionTable::store(PositionKey key, PositionHash hash, TranspositionValue value) {
  if (buckets_.empty()) {
    return false;
  }

  value.depth =
      std::clamp(value.depth, 0, static_cast<int>(std::numeric_limits<std::uint8_t>::max()));
  Bucket& bucket = buckets_[bucket_index(hash, buckets_.size())];
  const std::size_t ways = active_ways();

  for (std::size_t index = 0; index < ways; ++index) {
    PackedEntry& entry = bucket.slots[index];
    if (!occupied(entry) || entry.key != key) {
      continue;
    }

    const TranspositionValue current = TranspositionTable::value(entry);
    if (current.depth > value.depth ||
        (current.depth == value.depth && current.bound == TranspositionBound::kExact &&
         value.bound != TranspositionBound::kExact)) {
      return false;
    }

    write(entry, key, value);
    return true;
  }

  for (std::size_t index = 0; index < ways; ++index) {
    PackedEntry& entry = bucket.slots[index];
    if (!occupied(entry)) {
      write(entry, key, value);
      ++size_;
      return true;
    }
  }

  const auto eligible = [&value](const PackedEntry& entry) noexcept {
    const int entry_depth = static_cast<int>(entry.depth);
    if (entry_depth > value.depth) {
      return false;
    }
    return entry_depth != value.depth ||
           TranspositionTable::bound(entry) != TranspositionBound::kExact ||
           value.bound == TranspositionBound::kExact;
  };

  const bool first_eligible = eligible(bucket.slots[0]);
  const bool second_eligible = ways == kWays && eligible(bucket.slots[1]);
  if (!first_eligible && !second_eligible) {
    return false;
  }

  std::size_t victim = first_eligible ? 0 : 1;
  if (first_eligible && second_eligible) {
    const PackedEntry& first = bucket.slots[0];
    const PackedEntry& second = bucket.slots[1];
    if (first.depth != second.depth) {
      victim = first.depth < second.depth ? 0 : 1;
    } else if (bound(first) != bound(second) && (bound(first) == TranspositionBound::kExact ||
                                                 bound(second) == TranspositionBound::kExact)) {
      victim = bound(first) == TranspositionBound::kExact ? 1 : 0;
    } else {
      victim = bucket.next_victim;
      bucket.next_victim ^= 1;
    }
  }

  write(bucket.slots[victim], key, value);
  return true;
}

std::optional<TranspositionEntry> TranspositionTable::probe(PositionKey key) const noexcept {
  if (buckets_.empty()) {
    return std::nullopt;
  }

  const PositionHash hash = position_key_hash(key);
  return probe(key, hash);
}

std::optional<TranspositionEntry> TranspositionTable::probe(
    const Position& position) const noexcept {
  return probe(position.key(), position.hash());
}

std::optional<TranspositionEntry> TranspositionTable::probe(PositionKey key,
                                                            PositionHash hash) const noexcept {
  if (buckets_.empty()) {
    return std::nullopt;
  }

  const Bucket& bucket = buckets_[bucket_index(hash, buckets_.size())];
  const std::size_t ways = active_ways();
  for (std::size_t index = 0; index < ways; ++index) {
    const PackedEntry& entry = bucket.slots[index];
    if (occupied(entry) && entry.key == key) {
      return TranspositionEntry{
          .key = entry.key,
          .hash = hash,
          .value = value(entry),
      };
    }
  }

  return std::nullopt;
}

bool TranspositionTable::occupied(const PackedEntry& entry) noexcept {
  return (entry.flags & kOccupiedMask) != 0;
}

TranspositionBound TranspositionTable::bound(const PackedEntry& entry) noexcept {
  return static_cast<TranspositionBound>(entry.flags & kBoundMask);
}

TranspositionValue TranspositionTable::value(const PackedEntry& entry) noexcept {
  std::optional<Square> best_move;
  if (entry.best_move_index < kCellCount) {
    best_move = square_from_index(entry.best_move_index);
  }

  return TranspositionValue{
      .score = entry.score,
      .depth = entry.depth,
      .bound = bound(entry),
      .best_move = best_move,
  };
}

void TranspositionTable::write(PackedEntry& entry, PositionKey key,
                               TranspositionValue value) noexcept {
  entry.key = key;
  entry.score = value.score;
  entry.depth = static_cast<std::uint8_t>(value.depth);
  entry.best_move_index = value.best_move.has_value() && is_valid(*value.best_move)
                              ? static_cast<std::uint8_t>(square_index(*value.best_move))
                              : kNoBestMove;
  entry.flags = kOccupiedMask | static_cast<std::uint8_t>(value.bound);
}

std::size_t TranspositionTable::active_ways() const noexcept { return capacity_ == 1 ? 1 : kWays; }

}  // namespace poe2
