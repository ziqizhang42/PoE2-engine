#include <catch2/catch_test_macros.hpp>
#include <string_view>

#include "poe2/version.hpp"

TEST_CASE("smoke test links the game library", "[smoke]") {
  REQUIRE(std::string_view{poe2::kProjectName} == "PoE2");
  REQUIRE(poe2::rules_version() == 1);
}
