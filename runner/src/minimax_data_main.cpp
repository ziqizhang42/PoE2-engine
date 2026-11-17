#include <algorithm>
#include <charconv>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "poe2/match_runner.hpp"
#include "poe2/minimax/feature_data.hpp"
#include "poe2/minimax/labeling.hpp"
#include "poe2/minimax/net.hpp"
#include "poe2/minimax/position_source.hpp"
#include "poe2/move.hpp"

namespace {

namespace fs = std::filesystem;
namespace labeling = poe2::minimax::labeling;
namespace feature_data = poe2::minimax::feature_data;
namespace position_source = poe2::minimax::position_source;

struct LabelCommandOptions {
  fs::path input_path;
  fs::path output_directory;
  std::string corpus_id;
  labeling::LabelMode mode = labeling::LabelMode::kExact;
  std::uint64_t node_limit = 0;
  std::size_t hash_bytes = poe2::minimax::kDefaultHashBytes;
  std::size_t workers = 1;
  std::size_t progress_interval = 100;
  std::uint32_t shard_index = 0;
  std::uint32_t shard_count = 1;
  std::optional<std::uint32_t> source_shard;
  bool has_mode = false;
  bool has_opening_shard = false;
  bool require_all = false;
  poe2::minimax::Evaluator evaluator = poe2::minimax::Evaluator::kTwoPlyClosure;
  std::shared_ptr<const poe2::minimax::Net> net = poe2::minimax::default_net();
  bool has_net = false;
};

struct SourceCommandOptions {
  fs::path output_directory;
  position_source::PositionSourceOptions source;
  std::uint64_t progress_interval = 100;
  bool has_seed = false;
  std::optional<std::uint64_t> trajectory_begin;
  std::optional<std::uint64_t> trajectory_end;
  fs::path chunk_directory;
  bool has_net = false;
};

struct FeatureCommandOptions {
  fs::path source_directory;
  fs::path label_directory;
  fs::path output_directory;
};

void print_usage(std::ostream& output) {
  output << "usage:\n"
         << "  poe2_minimax_data source --output-dir <directory> --corpus-id <id>\n"
         << "                            --seed <n> --trajectories <n>\n"
         << "                            [--samples-per-trajectory <1..8>]\n"
         << "                            [--shards <n>] [--workers <n>]\n"
         << "                            [--search-nodes <n>] [--search-hash-mb <n>]\n"
         << "                            [--noise-percent <0..100>]\n"
         << "                            [--random-weight <n>] [--greedy-weight <n>]\n"
         << "                            [--opponent-weight <n>] [--search-weight <n>]\n"
         << "                            [--progress-every <n>]\n"
         << "                            [--evaluator two-ply-closure|pattern-gain]\n"
         << "                            [--net <file>]\n"
         << "\n"
         << "  poe2_minimax_data source-chunk <source options> --output-dir <chunk-directory>\n"
         << "                            --trajectory-begin <n> --trajectory-end <n>\n"
         << "  poe2_minimax_data source-assemble <source options> --output-dir <directory>\n"
         << "                            --chunks <directory-of-chunks>\n"
         << "\n"
         << "  poe2_minimax_data features --source <position-source-directory>\n"
         << "                              --labels <label-corpus-directory>\n"
         << "                              --output-dir <directory>\n"
         << "\n"
         << "  poe2_minimax_data labels --input <opening-book-or-source-corpus>\n"
         << "                            --output-dir <directory> --mode exact|teacher\n"
         << "                            --nodes <n> [--hash-mb <n>] [--workers <n>]\n"
         << "                            [--corpus-id <id>] [--source-shard <zero-based>]\n"
         << "                            [--shard-index <zero-based>] [--shard-count <n>]\n"
         << "                            [--progress-every <n>] [--require-all]\n"
         << "                            [--evaluator two-ply-closure|pattern-gain]\n"
         << "                            [--net <file>]\n";
}

[[nodiscard]] FeatureCommandOptions parse_feature_options(int argc, char* argv[]) {
  FeatureCommandOptions options;
  for (int index = 2; index < argc; ++index) {
    const std::string_view argument{argv[index]};
    if (index + 1 >= argc) {
      throw std::invalid_argument{"missing value for " + std::string{argument}};
    }
    const std::string_view value{argv[++index]};
    if (argument == "--source") {
      options.source_directory = value;
    } else if (argument == "--labels") {
      options.label_directory = value;
    } else if (argument == "--output-dir") {
      options.output_directory = value;
    } else {
      throw std::invalid_argument{"unknown argument: " + std::string{argument}};
    }
  }
  if (options.source_directory.empty()) {
    throw std::invalid_argument{"missing --source"};
  }
  if (options.label_directory.empty()) {
    throw std::invalid_argument{"missing --labels"};
  }
  if (options.output_directory.empty()) {
    throw std::invalid_argument{"missing --output-dir"};
  }
  return options;
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

[[nodiscard]] std::uint64_t require_u64(std::string_view value, std::string_view option) {
  const std::optional<std::uint64_t> parsed = parse_u64(value);
  if (!parsed.has_value()) {
    throw std::invalid_argument{std::string{option} + " requires a nonnegative integer"};
  }
  return *parsed;
}

[[nodiscard]] std::uint64_t require_positive_u64(std::string_view value, std::string_view option) {
  const std::optional<std::uint64_t> parsed = parse_positive_u64(value);
  if (!parsed.has_value()) {
    throw std::invalid_argument{std::string{option} + " requires a positive integer"};
  }
  return *parsed;
}

template <typename Integer>
[[nodiscard]] Integer checked_integer(std::uint64_t value, std::string_view option) {
  if (value > std::numeric_limits<Integer>::max()) {
    throw std::invalid_argument{std::string{option} + " is too large"};
  }
  return static_cast<Integer>(value);
}

[[nodiscard]] poe2::minimax::Evaluator parse_data_evaluator(std::string_view value) {
  if (value == "two-ply-closure") {
    return poe2::minimax::Evaluator::kTwoPlyClosure;
  }
  if (value == "pattern-gain") {
    return poe2::minimax::Evaluator::kPatternGain;
  }
  throw std::invalid_argument{"--evaluator must be two-ply-closure or pattern-gain"};
}

[[nodiscard]] std::size_t checked_mebibytes(std::uint64_t value, std::string_view option) {
  if (value > std::numeric_limits<std::size_t>::max() / poe2::minimax::kMebibyte) {
    throw std::invalid_argument{std::string{option} + " is too large"};
  }
  return static_cast<std::size_t>(value) * poe2::minimax::kMebibyte;
}

[[nodiscard]] LabelCommandOptions parse_label_options(int argc, char* argv[]) {
  LabelCommandOptions options;
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
        options.mode = labeling::LabelMode::kExact;
      } else if (value == "teacher") {
        options.mode = labeling::LabelMode::kTeacher;
      } else {
        throw std::invalid_argument{"--mode must be exact or teacher"};
      }
    } else if (argument == "--nodes") {
      options.node_limit = require_positive_u64(value, argument);
    } else if (argument == "--hash-mb") {
      options.hash_bytes = checked_mebibytes(require_positive_u64(value, argument), argument);
    } else if (argument == "--workers") {
      options.workers =
          checked_integer<std::size_t>(require_positive_u64(value, argument), argument);
    } else if (argument == "--progress-every") {
      options.progress_interval =
          checked_integer<std::size_t>(require_positive_u64(value, argument), argument);
    } else if (argument == "--shard-index") {
      options.has_opening_shard = true;
      options.shard_index = checked_integer<std::uint32_t>(require_u64(value, argument), argument);
    } else if (argument == "--shard-count") {
      options.has_opening_shard = true;
      options.shard_count =
          checked_integer<std::uint32_t>(require_positive_u64(value, argument), argument);
    } else if (argument == "--source-shard") {
      options.source_shard = checked_integer<std::uint32_t>(require_u64(value, argument), argument);
    } else if (argument == "--evaluator") {
      options.evaluator = parse_data_evaluator(value);
    } else if (argument == "--net") {
      options.net = poe2::minimax::load_net_file(fs::path{value});
      options.has_net = true;
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
  if (!options.has_mode) {
    throw std::invalid_argument{"missing --mode"};
  }
  if (options.node_limit == 0) {
    throw std::invalid_argument{"missing --nodes"};
  }
  if (options.shard_index >= options.shard_count) {
    throw std::invalid_argument{"--shard-index must be less than --shard-count"};
  }
  if (options.has_net && options.evaluator != poe2::minimax::Evaluator::kPatternGain) {
    throw std::invalid_argument{"--net needs --evaluator pattern-gain"};
  }
  return options;
}

[[nodiscard]] SourceCommandOptions parse_source_options(int argc, char* argv[]) {
  SourceCommandOptions options;
  for (int index = 2; index < argc; ++index) {
    const std::string_view argument{argv[index]};
    if (index + 1 >= argc) {
      throw std::invalid_argument{"missing value for " + std::string{argument}};
    }
    const std::string_view value{argv[++index]};
    if (argument == "--output-dir") {
      options.output_directory = value;
    } else if (argument == "--corpus-id") {
      options.source.corpus_id = value;
    } else if (argument == "--seed") {
      options.source.seed = require_u64(value, argument);
      options.has_seed = true;
    } else if (argument == "--trajectories") {
      options.source.trajectory_count = require_positive_u64(value, argument);
    } else if (argument == "--samples" || argument == "--samples-per-trajectory") {
      options.source.samples_per_trajectory =
          checked_integer<std::uint16_t>(require_positive_u64(value, argument), argument);
    } else if (argument == "--shards") {
      options.source.shard_count =
          checked_integer<std::uint32_t>(require_positive_u64(value, argument), argument);
    } else if (argument == "--workers") {
      options.source.workers =
          checked_integer<std::size_t>(require_positive_u64(value, argument), argument);
    } else if (argument == "--search-nodes") {
      options.source.search_nodes = require_positive_u64(value, argument);
    } else if (argument == "--search-hash-mb") {
      options.source.search_hash_bytes =
          checked_mebibytes(require_positive_u64(value, argument), argument);
    } else if (argument == "--noise-percent") {
      options.source.noise_percent =
          checked_integer<std::uint8_t>(require_u64(value, argument), argument);
    } else if (argument == "--random-weight") {
      options.source.policy_weights.random =
          checked_integer<std::uint32_t>(require_u64(value, argument), argument);
    } else if (argument == "--greedy-weight") {
      options.source.policy_weights.immediate_gain =
          checked_integer<std::uint32_t>(require_u64(value, argument), argument);
    } else if (argument == "--opponent-weight") {
      options.source.policy_weights.opponent_aware =
          checked_integer<std::uint32_t>(require_u64(value, argument), argument);
    } else if (argument == "--search-weight") {
      options.source.policy_weights.noisy_search =
          checked_integer<std::uint32_t>(require_u64(value, argument), argument);
    } else if (argument == "--progress-every") {
      options.progress_interval = require_positive_u64(value, argument);
    } else if (argument == "--evaluator") {
      options.source.evaluator = parse_data_evaluator(value);
    } else if (argument == "--net") {
      options.source.net = poe2::minimax::load_net_file(fs::path{value});
      options.has_net = true;
    } else if (argument == "--trajectory-begin") {
      options.trajectory_begin = require_u64(value, argument);
    } else if (argument == "--trajectory-end") {
      options.trajectory_end = require_positive_u64(value, argument);
    } else if (argument == "--chunks") {
      options.chunk_directory = value;
    } else {
      throw std::invalid_argument{"unknown argument: " + std::string{argument}};
    }
  }
  if (options.output_directory.empty()) {
    throw std::invalid_argument{"missing --output-dir"};
  }
  if (options.source.corpus_id.empty()) {
    throw std::invalid_argument{"missing --corpus-id"};
  }
  if (!options.has_seed) {
    throw std::invalid_argument{"missing --seed"};
  }
  if (options.source.trajectory_count == 0) {
    throw std::invalid_argument{"missing --trajectories"};
  }
  if (options.has_net && options.source.evaluator != poe2::minimax::Evaluator::kPatternGain) {
    throw std::invalid_argument{"--net needs --evaluator pattern-gain"};
  }
  return options;
}

[[nodiscard]] std::string read_file(const fs::path& path) {
  std::ifstream input{path, std::ios::binary};
  if (!input) {
    throw std::runtime_error{"failed to read " + path.string()};
  }
  return {std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
}

[[nodiscard]] std::vector<labeling::LabelInput> make_label_inputs(
    const poe2::match_runner::OpeningBook& book) {
  if (book.lines.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error{"opening book contains too many positions"};
  }
  std::vector<labeling::LabelInput> inputs;
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
    const std::uint64_t source_id = labeling::stable_digest(line.text);
    inputs.push_back(labeling::LabelInput{
        .position = position,
        .source_id = source_id,
        .family_id = source_id,
        .trajectory_id = source_id,
        .trajectory_index = ordinal,
        .source_line = static_cast<std::uint32_t>(line.line_number),
        .source_ordinal = static_cast<std::uint32_t>(ordinal),
        .split = labeling::DatasetSplit::kTrain,
    });
  }
  return inputs;
}

int run_labels(const LabelCommandOptions& options) {
  std::vector<labeling::LabelInput> inputs;
  labeling::LabelSource source;
  if (fs::is_directory(options.input_path)) {
    if (!options.source_shard.has_value()) {
      throw std::invalid_argument{"a position-source directory requires --source-shard"};
    }
    if (options.has_opening_shard) {
      throw std::invalid_argument{"--shard-index/--shard-count apply only to opening books"};
    }
    position_source::ReadSourceShard source_shard =
        position_source::read_position_source_shard(options.input_path, *options.source_shard);
    if (!options.corpus_id.empty() && options.corpus_id != source_shard.source.corpus_id) {
      throw std::invalid_argument{"--corpus-id differs from the position source"};
    }
    source = std::move(source_shard.source);
    inputs = std::move(source_shard.inputs);
  } else {
    if (options.source_shard.has_value()) {
      throw std::invalid_argument{"--source-shard requires a position-source directory"};
    }
    if (options.corpus_id.empty()) {
      throw std::invalid_argument{"an opening-book input requires --corpus-id"};
    }
    const std::string source_text = read_file(options.input_path);
    const poe2::match_runner::OpeningBook book =
        poe2::match_runner::parse_opening_book_text(options.input_path.string(), source_text);
    inputs = make_label_inputs(book);
    source = labeling::LabelSource{
        .corpus_id = options.corpus_id,
        .source_name = options.input_path.filename().string(),
        .source_digest = labeling::sha256(source_text),
        .shard_index = options.shard_index,
        .shard_count = options.shard_count,
    };
  }

  labeling::DatasetOutput output = labeling::reserve_dataset_output(options.output_directory);
  const labeling::LabelDataset dataset = labeling::generate_labels(
      inputs,
      labeling::LabelingOptions{
          .mode = options.mode,
          .node_limit = options.node_limit,
          .hash_bytes = options.hash_bytes,
          .workers = options.workers,
          .require_all = options.require_all,
          .evaluator = options.evaluator,
          .net = options.net,
      },
      std::move(source),
      [](const labeling::LabelProgress& progress) {
        std::cerr << "label_progress completed=" << progress.completed
                  << " total=" << progress.total << " records=" << progress.records
                  << " unsolved=" << progress.unsolved << '\n';
      },
      options.progress_interval);
  labeling::write_dataset(output, dataset);

  std::cout << "label_dataset mode=" << labeling::label_mode_name(options.mode)
            << " inputs=" << dataset.input_count << " records=" << dataset.records.size()
            << " unsolved=" << dataset.unsolved_source_lines.size()
            << " nodes=" << options.node_limit << " workers_requested=" << options.workers
            << " workers_used=" << dataset.workers_used << " shard=" << dataset.source.shard_index
            << '/' << dataset.source.shard_count << " output_dir=" << output.directory().string()
            << '\n';
  return 0;
}

void require_source_mode(const SourceCommandOptions& options, bool chunk, bool assemble) {
  if (options.trajectory_begin.has_value() != chunk ||
      options.trajectory_end.has_value() != chunk) {
    throw std::invalid_argument{
        chunk ? "source-chunk requires --trajectory-begin and --trajectory-end"
              : "--trajectory-begin and --trajectory-end need source-chunk"};
  }
  if (options.chunk_directory.empty() == assemble) {
    throw std::invalid_argument{assemble ? "source-assemble requires --chunks"
                                         : "--chunks needs source-assemble"};
  }
}

int run_source_chunk(const SourceCommandOptions& options) {
  require_source_mode(options, true, false);
  const std::optional<std::uint64_t> begin = options.trajectory_begin;
  const std::optional<std::uint64_t> end = options.trajectory_end;
  if (!begin.has_value() || !end.has_value()) {
    throw std::invalid_argument{"source-chunk requires --trajectory-begin and --trajectory-end"};
  }
  position_source::reserve_source_chunk_output(options.output_directory);
  const position_source::PositionSourceChunk chunk =
      position_source::generate_position_source_chunk(
          options.source, *begin, *end,
          [](const position_source::PositionSourceProgress& progress) {
            std::cerr << "source_progress completed_trajectories="
                      << progress.completed_trajectories
                      << " total_trajectories=" << progress.total_trajectories << '\n';
          },
          options.progress_interval);
  position_source::write_position_source_chunk(options.output_directory, chunk);
  std::cout << "source_chunk trajectories=" << chunk.trajectory_begin << '-' << chunk.trajectory_end
            << " records=" << chunk.records.size()
            << " output_dir=" << options.output_directory.string() << '\n';
  return 0;
}

int run_source_assemble(const SourceCommandOptions& options) {
  require_source_mode(options, false, true);
  std::vector<position_source::PositionSourceChunk> chunks;
  for (const fs::directory_entry& entry : fs::directory_iterator(options.chunk_directory)) {
    chunks.push_back(position_source::read_position_source_chunk(entry.path(), options.source.net));
  }
  std::ranges::sort(chunks, {}, &position_source::PositionSourceChunk::trajectory_begin);
  const position_source::PositionSourceCorpus corpus =
      position_source::assemble_position_source(options.source, chunks);
  position_source::SourceOutput output =
      position_source::reserve_source_output(options.output_directory);
  position_source::write_position_source(output, corpus);
  std::cout << "position_source trajectories=" << options.source.trajectory_count
            << " records=" << corpus.records.size() << " shards=" << options.source.shard_count
            << " duplicates=" << corpus.duplicate_positions << " chunks=" << chunks.size()
            << " output_dir=" << output.directory().string() << '\n';
  return 0;
}

int run_source(const SourceCommandOptions& options) {
  require_source_mode(options, false, false);
  position_source::SourceOutput output =
      position_source::reserve_source_output(options.output_directory);
  const position_source::PositionSourceCorpus corpus = position_source::generate_position_source(
      options.source,
      [](const position_source::PositionSourceProgress& progress) {
        std::cerr << "source_progress completed_trajectories=" << progress.completed_trajectories
                  << " total_trajectories=" << progress.total_trajectories << '\n';
      },
      options.progress_interval);
  position_source::write_position_source(output, corpus);
  std::cout << "position_source trajectories=" << options.source.trajectory_count
            << " records=" << corpus.records.size() << " shards=" << options.source.shard_count
            << " duplicates=" << corpus.duplicate_positions
            << " workers_requested=" << options.source.workers
            << " workers_used=" << corpus.workers_used
            << " output_dir=" << output.directory().string() << '\n';
  return 0;
}

int run_features(const FeatureCommandOptions& options) {
  const feature_data::FeatureDataset dataset =
      feature_data::build_feature_dataset(options.source_directory, options.label_directory);
  feature_data::write_feature_dataset(options.output_directory, dataset);
  std::cout << "feature_dataset source_records=" << dataset.source_record_count
            << " records=" << dataset.records.size()
            << " duplicates=" << dataset.duplicate_record_count
            << " duplicate_groups=" << dataset.duplicate_group_count
            << " varying_label_groups=" << dataset.varying_label_group_count
            << " output_dir=" << options.output_directory.string() << '\n';
  return 0;
}

int run(int argc, char* argv[]) {
  if (argc == 2 && std::string_view{argv[1]} == "--help") {
    print_usage(std::cout);
    return 0;
  }
  if (argc < 2) {
    throw std::invalid_argument{"missing subcommand"};
  }
  const std::string_view command{argv[1]};
  if (command == "labels") {
    return run_labels(parse_label_options(argc, argv));
  }
  if (command == "source") {
    return run_source(parse_source_options(argc, argv));
  }
  if (command == "source-chunk") {
    return run_source_chunk(parse_source_options(argc, argv));
  }
  if (command == "source-assemble") {
    return run_source_assemble(parse_source_options(argc, argv));
  }
  if (command == "features") {
    return run_features(parse_feature_options(argc, argv));
  }
  throw std::invalid_argument{"expected source, labels, or features subcommand"};
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
