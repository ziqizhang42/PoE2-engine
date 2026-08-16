#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "poe2/match_runner.hpp"
#include "poe2/move.hpp"

namespace poe2::match_runner {

namespace {

[[nodiscard]] std::string trim_copy(std::string_view text) {
  const std::size_t first = text.find_first_not_of(" \t\r\n");
  if (first == std::string_view::npos) {
    return {};
  }

  const std::size_t last = text.find_last_not_of(" \t\r\n");
  return std::string{text.substr(first, last - first + 1)};
}

[[nodiscard]] std::string line_location(std::string_view path, int line_number) {
  std::ostringstream output;
  if (!path.empty()) {
    output << path << ':';
  }
  output << line_number;
  return output.str();
}

[[nodiscard]] std::string strip_comment(std::string_view line) {
  const std::size_t comment = line.find('#');
  if (comment != std::string_view::npos) {
    line = line.substr(0, comment);
  }

  return trim_copy(line);
}

[[nodiscard]] std::optional<OpeningLine> parse_opening_line(std::string_view path, int line_number,
                                                            std::string_view line) {
  const std::string opening_text = strip_comment(line);
  if (opening_text.empty()) {
    return std::nullopt;
  }

  if (opening_text == "startpos") {
    return OpeningLine{
        .line_number = line_number,
        .text = "startpos",
    };
  }

  Position position;
  std::vector<std::string> moves;
  std::istringstream input{opening_text};
  std::string token;
  while (input >> token) {
    const std::optional<Move> parsed = parse_move(token);
    if (!parsed.has_value()) {
      throw std::invalid_argument{line_location(path, line_number) +
                                  " malformed opening move: " + token};
    }

    const std::string formatted = format_move(*parsed);
    const MoveResult result = apply_move(position, *parsed);
    if (!result.accepted) {
      const std::string_view error =
          result.error.has_value() ? move_error_name(*result.error) : "unknown";
      throw std::invalid_argument{line_location(path, line_number) + " illegal opening move " +
                                  formatted + ": " + std::string{error}};
    }
    if (result.game_result.has_value()) {
      throw std::invalid_argument{line_location(path, line_number) +
                                  " opening line reaches a terminal position"};
    }

    moves.push_back(formatted);
  }

  const std::string normalized_text = format_opening_moves(moves);
  return OpeningLine{
      .line_number = line_number,
      .moves = std::move(moves),
      .text = normalized_text,
  };
}

}  // namespace

std::string format_opening_moves(const std::vector<std::string>& moves) {
  std::string text;
  for (std::size_t index = 0; index < moves.size(); ++index) {
    if (index != 0) {
      text += ' ';
    }
    text += moves[index];
  }
  return text;
}

OpeningBook parse_opening_book_text(std::string_view path, std::string_view text) {
  OpeningBook book{
      .path = std::string{path},
  };

  std::istringstream input{std::string{text}};
  std::string line;
  int line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (std::optional<OpeningLine> opening = parse_opening_line(path, line_number, line);
        opening.has_value()) {
      book.lines.push_back(std::move(*opening));
    }
  }

  if (book.lines.empty()) {
    throw std::invalid_argument{"opening book has no opening lines: " + std::string{path}};
  }

  return book;
}

OpeningBook load_opening_book(std::string_view path) {
  std::ifstream input{std::string{path}};
  if (!input) {
    throw std::invalid_argument{"failed to open opening book: " + std::string{path}};
  }

  std::ostringstream text;
  text << input.rdbuf();
  return parse_opening_book_text(path, text.str());
}

}  // namespace poe2::match_runner
