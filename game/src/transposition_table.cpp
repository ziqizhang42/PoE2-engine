#include "poe2/transposition_table.hpp"

#include <bit>
#include <cassert>
#include <limits>
#include <stdexcept>

namespace poe2 {

namespace {

[[nodiscard]] std::size_t normalized_capacity(std::size_t capacity) {
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

TranspositionTable::TranspositionTable(std::size_t capacity)
    : buckets_(normalized_capacity(capacity)) {}

void TranspositionTable::resize(std::size_t capacity) {
  const std::size_t bucket_count = normalized_capacity(capacity);
  buckets_.clear();
  buckets_.resize(bucket_count);
  size_ = 0;
}

void TranspositionTable::clear() noexcept {
  for (Bucket& bucket : buckets_) {
    bucket.occupied = false;
  }
  size_ = 0;
}

std::size_t TranspositionTable::capacity() const noexcept { return buckets_.size(); }

std::size_t TranspositionTable::size() const noexcept { return size_; }

bool TranspositionTable::empty() const noexcept { return size_ == 0; }

void TranspositionTable::store(PositionKey key, TranspositionValue value) {
  if (buckets_.empty()) {
    return;
  }

  const PositionHash hash = position_key_hash(key);
  store(key, hash, value);
}

void TranspositionTable::store(const Position& position, TranspositionValue value) {
  store(position.key(), position.hash(), value);
}

void TranspositionTable::store(PositionKey key, PositionHash hash, TranspositionValue value) {
  if (buckets_.empty()) {
    return;
  }

  Bucket& bucket = buckets_[bucket_index(hash, buckets_.size())];
  if (bucket.occupied && bucket.entry.key != key && bucket.entry.value.depth > value.depth) {
    return;
  }

  if (!bucket.occupied) {
    ++size_;
  }

  bucket.occupied = true;
  bucket.entry = TranspositionEntry{
      .key = key,
      .hash = hash,
      .value = value,
  };
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
  if (!bucket.occupied || bucket.entry.hash != hash || bucket.entry.key != key) {
    return std::nullopt;
  }

  return bucket.entry;
}

}  // namespace poe2
