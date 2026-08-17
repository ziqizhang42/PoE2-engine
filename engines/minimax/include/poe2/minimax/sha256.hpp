#ifndef POE2_MINIMAX_SHA256_HPP
#define POE2_MINIMAX_SHA256_HPP

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace poe2::minimax::labeling {

using Sha256Digest = std::array<std::uint8_t, 32>;

[[nodiscard]] Sha256Digest sha256(std::span<const std::uint8_t> bytes);
[[nodiscard]] Sha256Digest sha256(std::string_view bytes);
[[nodiscard]] std::string sha256_text(const Sha256Digest& digest);

}  // namespace poe2::minimax::labeling

#endif  // POE2_MINIMAX_SHA256_HPP
