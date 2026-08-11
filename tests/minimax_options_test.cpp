#include <catch2/catch_test_macros.hpp>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "poe2/minimax/options.hpp"

namespace {

[[nodiscard]] std::optional<poe2::minimax::SearchOptions> parse(
    std::initializer_list<std::string_view> arguments, std::string& error) {
  const std::vector<std::string_view> values{arguments};
  return poe2::minimax::parse_search_options(values, error);
}

}  // namespace

TEST_CASE("minimax command-line options have stable defaults", "[minimax][cli]") {
  std::string error;
  const std::optional<poe2::minimax::SearchOptions> options = parse({}, error);

  if (!options.has_value()) {
    FAIL("default minimax options should parse");
    return;
  }
  const poe2::minimax::SearchOptions value = *options;
  REQUIRE(value == poe2::minimax::SearchOptions{});
  REQUIRE(value.hash_bytes == 64 * poe2::minimax::kMebibyte);
  REQUIRE(value.use_symmetry);
  REQUIRE(value.use_two_ply_closure);
  REQUIRE(error.empty());
}

TEST_CASE("minimax command-line options allow disabling search features", "[minimax][cli]") {
  std::string error;
  const std::optional<poe2::minimax::SearchOptions> options =
      parse({"--hash-mb", "0", "--no-symmetry", "--no-two-ply-closure"}, error);

  if (!options.has_value()) {
    FAIL("explicit minimax options should parse");
    return;
  }
  const poe2::minimax::SearchOptions value = *options;
  REQUIRE(value.hash_bytes == 0);
  REQUIRE_FALSE(value.use_symmetry);
  REQUIRE_FALSE(value.use_two_ply_closure);
  REQUIRE(error.empty());
}

TEST_CASE("minimax command-line options reject malformed and missing hash sizes",
          "[minimax][cli]") {
  std::string error;

  REQUIRE_FALSE(parse({"--hash-mb"}, error).has_value());
  REQUIRE(error == "missing value for --hash-mb");

  REQUIRE_FALSE(parse({"--hash-mb", "twelve"}, error).has_value());
  REQUIRE(error == "malformed --hash-mb value: twelve");

  REQUIRE_FALSE(parse({"--hash-mb", "-1"}, error).has_value());
  REQUIRE(error == "malformed --hash-mb value: -1");
}

TEST_CASE("minimax command-line options reject overflow and unknown arguments", "[minimax][cli]") {
  std::string error;
  const std::string overflow = std::to_string(std::numeric_limits<std::size_t>::max());

  REQUIRE_FALSE(parse({"--hash-mb", overflow}, error).has_value());
  REQUIRE(error == "--hash-mb value is too large: " + overflow);

  REQUIRE_FALSE(parse({"--mystery"}, error).has_value());
  REQUIRE(error == "unknown argument: --mystery");
}
