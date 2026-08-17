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
  fs::path output_directory;
  std::string corpus_id;
  poe2::minimax::labeling::LabelMode mode = poe2::minimax::labeling::LabelMode::kExact;
  std::uint64_t node_limit = 0;
  std::size_t hash_bytes = poe2::minimax::kDefaultHashBytes;
  std::size_t workers = 1;
  std::size_t progress_interval = 100;
  std::uint32_t shard_index = 0;
  std::uint32_t shard_count = 1;
  bool has_mode = false;
  bool require_all = false;
};

void print_usage(std::ostream& output) {
  output << "usage:\n"
         << "  poe2_minimax_data labels --input <opening-book> --output-dir <directory>\n"
         << "                            --corpus-id <id> --mode exact|teacher\n"
         << "                            --nodes <n> [--hash-mb <n>] [--workers <n>]\n"
         << "                            [--shard-index <zero-based>] [--shard-count <n>]\n"
         << "                            [--progress-every <n>] [--require-all]\n";
}

[[nodiscard]] std::optional<std::uint64_t> parse_u64(std::string_view text) {
  std::uint64_t value = 0;
  const char* const begin = text.data();
  const char* const end = begin + text.size();
  const auto result = std::from_chars(begin, end, value);
  if (text.empty() || result.ec != std::errc{} || result.ptr != end) {
    return std::nullopt;
  }
  return value;
}

[[nodiscard]] std::optional<std::uint64_t> parse_positive_u64(std::string_view text) {
  const std::optional<std::uint64_t> value = parse_u64(text);
  return value.has_value() && *value > 0 ? value : std::nullopt;
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
    } else if (argument == "--output-dir") {
      options.output_directory = value;
    } else if (argument == "--corpus-id") {
      options.corpus_id = value;
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
    } else if (argument == "--progress-every") {
      const std::optional<std::uint64_t> interval = parse_positive_u64(value);
      if (!interval.has_value() || *interval > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument{"--progress-every requires a representable positive integer"};
      }
      options.progress_interval = static_cast<std::size_t>(*interval);
    } else if (argument == "--shard-index") {
      const std::optional<std::uint64_t> shard_index = parse_u64(value);
      if (!shard_index.has_value() || *shard_index > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument{"--shard-index requires a representable nonnegative integer"};
      }
      options.shard_index = static_cast<std::uint32_t>(*shard_index);
    } else if (argument == "--shard-count") {
      const std::optional<std::uint64_t> shard_count = parse_positive_u64(value);
      if (!shard_count.has_value() || *shard_count > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument{"--shard-count requires a representable positive integer"};
      }
      options.shard_count = static_cast<std::uint32_t>(*shard_count);
    } else {
      throw std::invalid_argument{"unknown argument: " + std::string{argument}};
    }
  }

  if (options.input_path.empty()) {
    throw std::invalid_argument{"missing --input"};
  }
  if (options.output_directory.empty()) {
    throw std::invalid_argument{"missing --output-dir"};
  }
  if (options.corpus_id.empty()) {
    throw std::invalid_argument{"missing --corpus-id"};
  }
  if (!options.has_mode) {
    throw std::invalid_argument{"missing --mode"};
  }
  if (options.node_limit == 0) {
    throw std::invalid_argument{"missing --nodes"};
  }
  if (options.shard_index >= options.shard_count) {
    throw std::invalid_argument{"--shard-index must be less than --shard-count"};
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
  if (book.lines.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error{"opening book contains too many positions"};
  }
  std::vector<poe2::minimax::labeling::LabelInput> inputs;
  inputs.reserve(book.lines.size());
  for (std::size_t ordinal = 0; ordinal < book.lines.size(); ++ordinal) {
    const poe2::match_runner::OpeningLine& line = book.lines[ordinal];
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
    const std::uint64_t source_id = poe2::minimax::labeling::stable_digest(line.text);
    inputs.push_back(poe2::minimax::labeling::LabelInput{
        .position = position,
        .source_id = source_id,
        .family_id = source_id,
        .trajectory_id = source_id,
        .source_line = static_cast<std::uint32_t>(line.line_number),
        .source_ordinal = static_cast<std::uint32_t>(ordinal),
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
  poe2::minimax::labeling::DatasetOutput output =
      poe2::minimax::labeling::reserve_dataset_output(options.output_directory);
  const poe2::minimax::labeling::LabelDataset dataset = poe2::minimax::labeling::generate_labels(
      inputs,
      poe2::minimax::labeling::LabelingOptions{
          .mode = options.mode,
          .node_limit = options.node_limit,
          .hash_bytes = options.hash_bytes,
          .workers = options.workers,
          .require_all = options.require_all,
      },
      poe2::minimax::labeling::LabelSource{
          .corpus_id = options.corpus_id,
          .source_name = options.input_path.filename().string(),
          .source_digest = poe2::minimax::labeling::sha256(source_text),
          .shard_index = options.shard_index,
          .shard_count = options.shard_count,
      },
      [](const poe2::minimax::labeling::LabelProgress& progress) {
        std::cerr << "label_progress completed=" << progress.completed
                  << " total=" << progress.total << " records=" << progress.records
                  << " unsolved=" << progress.unsolved << '\n';
      },
      options.progress_interval);
  poe2::minimax::labeling::write_dataset(output, dataset);

  std::cout << "label_dataset mode=" << poe2::minimax::labeling::label_mode_name(options.mode)
            << " inputs=" << dataset.input_count << " records=" << dataset.records.size()
            << " unsolved=" << dataset.unsolved_source_lines.size()
            << " nodes=" << options.node_limit << " workers_requested=" << options.workers
            << " workers_used=" << dataset.workers_used << " shard=" << options.shard_index << '/'
            << options.shard_count << " output_dir=" << output.directory().string() << '\n';
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
