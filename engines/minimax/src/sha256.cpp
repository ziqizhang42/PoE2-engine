#include "poe2/minimax/sha256.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace poe2::minimax::labeling {

namespace {

constexpr std::array<std::uint32_t, 64> kRoundConstants{{
    UINT32_C(0x428a2f98), UINT32_C(0x71374491), UINT32_C(0xb5c0fbcf), UINT32_C(0xe9b5dba5),
    UINT32_C(0x3956c25b), UINT32_C(0x59f111f1), UINT32_C(0x923f82a4), UINT32_C(0xab1c5ed5),
    UINT32_C(0xd807aa98), UINT32_C(0x12835b01), UINT32_C(0x243185be), UINT32_C(0x550c7dc3),
    UINT32_C(0x72be5d74), UINT32_C(0x80deb1fe), UINT32_C(0x9bdc06a7), UINT32_C(0xc19bf174),
    UINT32_C(0xe49b69c1), UINT32_C(0xefbe4786), UINT32_C(0x0fc19dc6), UINT32_C(0x240ca1cc),
    UINT32_C(0x2de92c6f), UINT32_C(0x4a7484aa), UINT32_C(0x5cb0a9dc), UINT32_C(0x76f988da),
    UINT32_C(0x983e5152), UINT32_C(0xa831c66d), UINT32_C(0xb00327c8), UINT32_C(0xbf597fc7),
    UINT32_C(0xc6e00bf3), UINT32_C(0xd5a79147), UINT32_C(0x06ca6351), UINT32_C(0x14292967),
    UINT32_C(0x27b70a85), UINT32_C(0x2e1b2138), UINT32_C(0x4d2c6dfc), UINT32_C(0x53380d13),
    UINT32_C(0x650a7354), UINT32_C(0x766a0abb), UINT32_C(0x81c2c92e), UINT32_C(0x92722c85),
    UINT32_C(0xa2bfe8a1), UINT32_C(0xa81a664b), UINT32_C(0xc24b8b70), UINT32_C(0xc76c51a3),
    UINT32_C(0xd192e819), UINT32_C(0xd6990624), UINT32_C(0xf40e3585), UINT32_C(0x106aa070),
    UINT32_C(0x19a4c116), UINT32_C(0x1e376c08), UINT32_C(0x2748774c), UINT32_C(0x34b0bcb5),
    UINT32_C(0x391c0cb3), UINT32_C(0x4ed8aa4a), UINT32_C(0x5b9cca4f), UINT32_C(0x682e6ff3),
    UINT32_C(0x748f82ee), UINT32_C(0x78a5636f), UINT32_C(0x84c87814), UINT32_C(0x8cc70208),
    UINT32_C(0x90befffa), UINT32_C(0xa4506ceb), UINT32_C(0xbef9a3f7), UINT32_C(0xc67178f2),
}};

[[nodiscard]] constexpr std::uint32_t read_big_endian(const std::uint8_t* bytes) noexcept {
  return (static_cast<std::uint32_t>(bytes[0]) << 24) |
         (static_cast<std::uint32_t>(bytes[1]) << 16) |
         (static_cast<std::uint32_t>(bytes[2]) << 8) | static_cast<std::uint32_t>(bytes[3]);
}

void transform(std::array<std::uint32_t, 8>& state, const std::uint8_t* block) noexcept {
  std::array<std::uint32_t, 64> words{};
  for (std::size_t index = 0; index < 16; ++index) {
    words[index] = read_big_endian(block + index * 4);
  }
  for (std::size_t index = 16; index < words.size(); ++index) {
    const std::uint32_t first = std::rotr(words[index - 15], 7) ^ std::rotr(words[index - 15], 18) ^
                                (words[index - 15] >> 3);
    const std::uint32_t second = std::rotr(words[index - 2], 17) ^ std::rotr(words[index - 2], 19) ^
                                 (words[index - 2] >> 10);
    words[index] = words[index - 16] + first + words[index - 7] + second;
  }

  std::uint32_t a = state[0];
  std::uint32_t b = state[1];
  std::uint32_t c = state[2];
  std::uint32_t d = state[3];
  std::uint32_t e = state[4];
  std::uint32_t f = state[5];
  std::uint32_t g = state[6];
  std::uint32_t h = state[7];

  for (std::size_t index = 0; index < words.size(); ++index) {
    const std::uint32_t choose = (e & f) ^ (~e & g);
    const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
    const std::uint32_t sum_one = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
    const std::uint32_t sum_zero = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
    const std::uint32_t temporary_one =
        h + sum_one + choose + kRoundConstants[index] + words[index];
    const std::uint32_t temporary_two = sum_zero + majority;
    h = g;
    g = f;
    f = e;
    e = d + temporary_one;
    d = c;
    c = b;
    b = a;
    a = temporary_one + temporary_two;
  }

  state[0] += a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
  state[4] += e;
  state[5] += f;
  state[6] += g;
  state[7] += h;
}

}  // namespace

Sha256Digest sha256(std::span<const std::uint8_t> bytes) {
  if (bytes.size() > std::numeric_limits<std::uint64_t>::max() / 8) {
    throw std::length_error{"SHA-256 input is too large"};
  }

  std::array<std::uint32_t, 8> state{{
      UINT32_C(0x6a09e667),
      UINT32_C(0xbb67ae85),
      UINT32_C(0x3c6ef372),
      UINT32_C(0xa54ff53a),
      UINT32_C(0x510e527f),
      UINT32_C(0x9b05688c),
      UINT32_C(0x1f83d9ab),
      UINT32_C(0x5be0cd19),
  }};

  std::size_t offset = 0;
  while (bytes.size() - offset >= 64) {
    transform(state, bytes.data() + offset);
    offset += 64;
  }

  std::array<std::uint8_t, 128> tail{};
  const std::size_t remainder = bytes.size() - offset;
  if (remainder > 0) {
    std::copy_n(bytes.data() + offset, remainder, tail.data());
  }
  tail[remainder] = UINT8_C(0x80);
  const std::size_t tail_size = remainder < 56 ? 64 : 128;
  const std::uint64_t bit_length = static_cast<std::uint64_t>(bytes.size()) * 8;
  for (std::size_t index = 0; index < 8; ++index) {
    tail[tail_size - 1 - index] = static_cast<std::uint8_t>(bit_length >> (index * 8));
  }
  transform(state, tail.data());
  if (tail_size == 128) {
    transform(state, tail.data() + 64);
  }

  Sha256Digest digest{};
  for (std::size_t word = 0; word < state.size(); ++word) {
    for (std::size_t byte = 0; byte < 4; ++byte) {
      digest[word * 4 + byte] =
          static_cast<std::uint8_t>(state[word] >> (24 - static_cast<int>(byte) * 8));
    }
  }
  return digest;
}

Sha256Digest sha256(std::string_view bytes) {
  return sha256(std::span{reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()});
}

std::string sha256_text(const Sha256Digest& digest) {
  constexpr std::array<char, 16> kHex{
      {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'}};
  std::string text;
  text.reserve(digest.size() * 2);
  for (const std::uint8_t byte : digest) {
    text.push_back(kHex[byte >> 4]);
    text.push_back(kHex[byte & UINT8_C(0x0f)]);
  }
  return text;
}

}  // namespace poe2::minimax::labeling
