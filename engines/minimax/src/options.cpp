#include "poe2/minimax/options.hpp"

#include <charconv>
#include <limits>

namespace poe2::minimax {

namespace {

[[nodiscard]] std::optional<std::size_t> parse_hash_bytes(std::string_view text,
                                                          std::string& error) {
  std::size_t hash_megabytes = 0;
  const char* begin = text.data();
  const char* end = begin + text.size();
  const auto result = std::from_chars(begin, end, hash_megabytes);

  if (text.empty() || result.ec != std::errc{} || result.ptr != end) {
    error = "malformed --hash-mb value: ";
    error += text;
    return std::nullopt;
  }
  if (hash_megabytes > std::numeric_limits<std::size_t>::max() / kMebibyte) {
    error = "--hash-mb value is too large: ";
    error += text;
    return std::nullopt;
  }

  return hash_megabytes * kMebibyte;
}

}  // namespace

std::optional<SearchOptions> parse_search_options(std::span<const std::string_view> arguments,
                                                  std::string& error) {
  SearchOptions options;
  error.clear();

  for (std::size_t index = 0; index < arguments.size(); ++index) {
    const std::string_view argument = arguments[index];
    if (argument == "--no-symmetry") {
      options.use_symmetry = false;
      continue;
    }
    if (argument == "--evaluator") {
      ++index;
      if (index == arguments.size()) {
        error = "missing value for --evaluator";
        return std::nullopt;
      }
      if (arguments[index] == "static") {
        options.evaluator = Evaluator::kStatic;
      } else if (arguments[index] == "two-ply-closure") {
        options.evaluator = Evaluator::kTwoPlyClosure;
      } else if (arguments[index] == "pattern-gain") {
        options.evaluator = Evaluator::kPatternGain;
      } else {
        error = "--evaluator must be static, two-ply-closure, or pattern-gain: ";
        error += arguments[index];
        return std::nullopt;
      }
      continue;
    }
    if (argument == "--hash-mb") {
      ++index;
      if (index == arguments.size()) {
        error = "missing value for --hash-mb";
        return std::nullopt;
      }

      const std::optional<std::size_t> hash_bytes = parse_hash_bytes(arguments[index], error);
      if (!hash_bytes.has_value()) {
        return std::nullopt;
      }
      options.hash_bytes = *hash_bytes;
      continue;
    }

    error = "unknown argument: ";
    error += argument;
    return std::nullopt;
  }

  return options;
}

}  // namespace poe2::minimax
