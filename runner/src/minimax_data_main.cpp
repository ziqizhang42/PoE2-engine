#include <charconv>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "poe2/match_runner.hpp"
#include "poe2/minimax/labeling.hpp"
#include "poe2/move.hpp"

namespace {

namespace fs = std::filesystem;

struct CommandOptions {
  fs::path input_path;
  fs::path output_path;
  fs::path manifest_path;
  poe2::minimax::labeling::LabelMode mode = poe2::minimax::labeling::LabelMode::kExact;
  std::uint64_t node_limit = 0;
  std::size_t hash_bytes = poe2::minimax::kDefaultHashBytes;
  std::size_t workers = 1;
  bool has_mode = false;
  bool require_all = false;
};

void print_usage(std::ostream& output) {
  output << "usage:\n"
         << "  poe2_minimax_data labels --input <opening-book> --output <labels.bin>\n"
         << "                            --manifest <labels.json> --mode exact|teacher\n"
         << "                            --nodes <n> [--hash-mb <n>] [--workers <n>]\n"
         << "                            [--require-all]\n";
}

[[nodiscard]] std::optional<std::uint64_t> parse_positive_u64(std::string_view text) {
  std::uint64_t value = 0;
  const char* const begin = text.data();
  const char* const end = begin + text.size();
  const auto result = std::from_chars(begin, end, value);
  if (text.empty() || result.ec != std::errc{} || result.ptr != end || value == 0) {
    return std::nullopt;
  }
  return value;
}

[[nodiscard]] CommandOptions parse_options(int argc, char* argv[]) {
  if (argc < 2 || std::string_view{argv[1]} != "labels") {
    throw std::invalid_argument{"expected labels subcommand"};
  }

  CommandOptions options;
  for (int index = 2; index < argc; ++index) {
    const std::string_view argument{argv[index]};
    if (argument == "--require-all") {
      options.require_all = true;
      continue;
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument{"missing value for " + std::string{argument}};
    }
    const std::string_view value{argv[++index]};
    if (argument == "--input") {
      options.input_path = value;
    } else if (argument == "--output") {
      options.output_path = value;
    } else if (argument == "--manifest") {
      options.manifest_path = value;
    } else if (argument == "--mode") {
      options.has_mode = true;
      if (value == "exact") {
        options.mode = poe2::minimax::labeling::LabelMode::kExact;
      } else if (value == "teacher") {
        options.mode = poe2::minimax::labeling::LabelMode::kTeacher;
      } else {
        throw std::invalid_argument{"--mode must be exact or teacher"};
      }
    } else if (argument == "--nodes") {
      const std::optional<std::uint64_t> nodes = parse_positive_u64(value);
      if (!nodes.has_value()) {
        throw std::invalid_argument{"--nodes requires a positive integer"};
      }
      options.node_limit = *nodes;
    } else if (argument == "--hash-mb") {
      const std::optional<std::uint64_t> hash_megabytes = parse_positive_u64(value);
      if (!hash_megabytes.has_value() ||
          *hash_megabytes > std::numeric_limits<std::size_t>::max() / poe2::minimax::kMebibyte) {
        throw std::invalid_argument{"--hash-mb requires a representable positive integer"};
      }
      options.hash_bytes = static_cast<std::size_t>(*hash_megabytes) * poe2::minimax::kMebibyte;
    } else if (argument == "--workers") {
      const std::optional<std::uint64_t> workers = parse_positive_u64(value);
      if (!workers.has_value() || *workers > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument{"--workers requires a representable positive integer"};
      }
      options.workers = static_cast<std::size_t>(*workers);
    } else {
      throw std::invalid_argument{"unknown argument: " + std::string{argument}};
    }
  }

  if (options.input_path.empty()) {
    throw std::invalid_argument{"missing --input"};
  }
  if (options.output_path.empty()) {
    throw std::invalid_argument{"missing --output"};
  }
  if (options.manifest_path.empty()) {
    throw std::invalid_argument{"missing --manifest"};
  }
  if (!options.has_mode) {
    throw std::invalid_argument{"missing --mode"};
  }
  if (options.node_limit == 0) {
    throw std::invalid_argument{"missing --nodes"};
  }
  return options;
}

[[nodiscard]] std::string read_file(const fs::path& path) {
  std::ifstream input{path, std::ios::binary};
  if (!input) {
    throw std::runtime_error{"failed to read " + path.string()};
  }
  return std::string{std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
}

[[nodiscard]] std::vector<poe2::minimax::labeling::LabelInput> make_label_inputs(
    const poe2::match_runner::OpeningBook& book) {
  std::vector<poe2::minimax::labeling::LabelInput> inputs;
  inputs.reserve(book.lines.size());
  for (const poe2::match_runner::OpeningLine& line : book.lines) {
    if (line.line_number <= 0 ||
        static_cast<std::uint64_t>(line.line_number) > std::numeric_limits<std::uint32_t>::max()) {
      throw std::runtime_error{"opening line number is out of range"};
    }

    poe2::Position position;
    for (const std::string& move_text : line.moves) {
      const std::optional<poe2::Move> move = poe2::parse_move(move_text);
      if (!move.has_value() || !position.play(move->square)) {
        throw std::logic_error{"validated opening could not be reconstructed"};
      }
    }
    inputs.push_back(poe2::minimax::labeling::LabelInput{
        .position = position,
        .source_id = poe2::minimax::labeling::stable_digest(line.text),
        .source_line = static_cast<std::uint32_t>(line.line_number),
    });
  }
  return inputs;
}

int run(int argc, char* argv[]) {
  if (argc == 2 && std::string_view{argv[1]} == "--help") {
    print_usage(std::cout);
    return 0;
  }

  const CommandOptions options = parse_options(argc, argv);
  const std::string source_text = read_file(options.input_path);
  const poe2::match_runner::OpeningBook book =
      poe2::match_runner::parse_opening_book_text(options.input_path.string(), source_text);
  const std::vector<poe2::minimax::labeling::LabelInput> inputs = make_label_inputs(book);
  const poe2::minimax::labeling::LabelDataset dataset = poe2::minimax::labeling::generate_labels(
      inputs,
      poe2::minimax::labeling::LabelingOptions{
          .mode = options.mode,
          .node_limit = options.node_limit,
          .hash_bytes = options.hash_bytes,
          .workers = options.workers,
          .require_all = options.require_all,
      },
      options.input_path.string(), poe2::minimax::labeling::stable_digest(source_text));
  poe2::minimax::labeling::write_dataset(dataset, options.output_path, options.manifest_path);

  std::cout << "label_dataset mode=" << poe2::minimax::labeling::label_mode_name(options.mode)
            << " inputs=" << dataset.input_count << " records=" << dataset.records.size()
            << " unsolved=" << dataset.unsolved_source_lines.size()
            << " nodes=" << options.node_limit << " workers_requested=" << options.workers
            << " workers_used=" << dataset.workers_used
            << " output=" << options.output_path.string()
            << " manifest=" << options.manifest_path.string() << '\n';
  return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "fatal " << error.what() << '\n';
  } catch (...) {
    std::cerr << "fatal unknown\n";
  }
  print_usage(std::cerr);
  return 1;
}
