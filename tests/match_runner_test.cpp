#include "poe2/match_runner.hpp"

#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
#include <string_view>

TEST_CASE("opening book parser accepts comments and normalizes moves", "[match_runner]") {
  constexpr std::string_view kBook =
      "# generated\n"
      "\n"
      "A1 b1 c1 d1 # trailing comment\n"
      "g7 f7 e7 d7\n";

  const poe2::match_runner::OpeningBook book =
      poe2::match_runner::parse_opening_book_text("inline", kBook);

  REQUIRE(book.path == "inline");
  REQUIRE(book.lines.size() == 2);
  REQUIRE(book.lines[0].line_number == 3);
  REQUIRE(book.lines[0].text == "a1 b1 c1 d1");
  REQUIRE(book.lines[1].line_number == 4);
  REQUIRE(book.lines[1].text == "g7 f7 e7 d7");
}

TEST_CASE("opening book parser rejects illegal prefixes", "[match_runner]") {
  REQUIRE_THROWS_AS(poe2::match_runner::parse_opening_book_text("inline", "a1 a1\n"),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(poe2::match_runner::parse_opening_book_text("inline", "z9\n"),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(poe2::match_runner::parse_opening_book_text("inline", "# empty\n"),
                    std::invalid_argument);
}
