#ifndef POE2_MINIMAX_OPTIONS_HPP
#define POE2_MINIMAX_OPTIONS_HPP

#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "poe2/minimax/search.hpp"

namespace poe2::minimax {

[[nodiscard]] std::optional<SearchOptions> parse_search_options(
    std::span<const std::string_view> arguments, std::string& error);

}  // namespace poe2::minimax

#endif  // POE2_MINIMAX_OPTIONS_HPP
