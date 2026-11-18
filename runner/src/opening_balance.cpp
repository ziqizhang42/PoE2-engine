#include "poe2/opening_balance.hpp"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <exception>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

#include "engine_process.hpp"
#include "poe2/engine_stdio.hpp"

namespace poe2::match_runner {

namespace {

using detail::EngineProcess;
using detail::LineReadResult;
using detail::LineReadStatus;

constexpr std::chrono::milliseconds kEngineLineTimeout{60000};
constexpr std::string_view kCorrectionPrefix = "correction_";

[[nodiscard]] std::vector<std::string> split_words(std::string_view text) {
  std::istringstream input{std::string{text}};
  std::vector<std::string> words;
  std::string word;
  while (input >> word) {
    words.push_back(std::move(word));
  }
  return words;
}

[[nodiscard]] std::vector<std::string> split_csv_row(std::string_view row) {
  if (row.find('"') != std::string_view::npos) {
    throw std::runtime_error{"quoted games.csv fields are not supported: " + std::string{row}};
  }
  std::vector<std::string> fields;
  std::size_t start = 0;
  while (true) {
    const std::size_t comma = row.find(',', start);
    fields.emplace_back(row.substr(start, comma - start));
    if (comma == std::string_view::npos) {
      return fields;
    }
    start = comma + 1;
  }
}

template <typename Number>
[[nodiscard]] Number parse_number(std::string_view text, std::string_view what) {
  Number value{};
  const char* begin = text.data();
  const char* end = begin + text.size();
  const auto result = std::from_chars(begin, end, value);
  if (result.ec != std::errc{} || result.ptr != end) {
    throw std::invalid_argument{"malformed " + std::string{what} + ": " + std::string{text}};
  }
  return value;
}

[[nodiscard]] std::string fixed_text(double value, int decimals) {
  std::ostringstream output;
  output << std::fixed << std::setprecision(decimals) << value;
  return output.str();
}

[[nodiscard]] std::string signed_text(double value, int decimals) {
  std::ostringstream output;
  output << std::showpos << std::fixed << std::setprecision(decimals) << value;
  return output.str();
}

[[nodiscard]] std::string percent_text(int part, int whole) {
  if (whole == 0) {
    return "n/a";
  }
  return fixed_text(100.0 * part / whole, 1) + '%';
}

[[nodiscard]] std::string shell_quote(std::string_view text) {
  std::string quoted = "'";
  for (const char ch : text) {
    if (ch == '\'') {
      quoted += "'\\''";
    } else {
      quoted += ch;
    }
  }
  quoted += "'";
  return quoted;
}

[[nodiscard]] std::string plies_list_text(const std::vector<int>& plies) {
  std::string text;
  for (const int value : plies) {
    if (!text.empty()) {
      text += ',';
    }
    text += std::to_string(value);
  }
  return text;
}

struct ErrorSummary {
  double mean = 0.0;
  double standard_error = 0.0;
  double sd = 0.0;
};

[[nodiscard]] ErrorSummary summarize_errors(const std::vector<double>& errors) {
  ErrorSummary summary;
  if (errors.empty()) {
    return summary;
  }
  const auto count = static_cast<double>(errors.size());
  double total = 0.0;
  for (const double error : errors) {
    total += error;
  }
  summary.mean = total / count;
  double squares = 0.0;
  for (const double error : errors) {
    squares += (error - summary.mean) * (error - summary.mean);
  }
  summary.sd = std::sqrt(squares / count);
  if (errors.size() > 1) {
    summary.standard_error = std::sqrt(squares / (count - 1.0) / count);
  }
  return summary;
}

void expect_line(EngineProcess& engine, std::string_view expected) {
  while (true) {
    const LineReadResult line = engine.read_line(kEngineLineTimeout);
    if (line.status != LineReadStatus::kLine) {
      throw std::runtime_error{"engine stopped answering while waiting for " +
                               std::string{expected}};
    }
    if (line.line == expected) {
      return;
    }
  }
}

[[nodiscard]] int search_score(EngineProcess& engine, std::uint64_t nodes,
                               const std::vector<std::string>& moves) {
  std::string position = "position startpos";
  if (!moves.empty()) {
    position += " moves " + format_opening_moves(moves);
  }
  if (!engine.write_line(engine_stdio::kCommandNewGame) || !engine.write_line(position) ||
      !engine.write_line("go nodes " + std::to_string(nodes))) {
    throw std::runtime_error{"could not send a search to the engine"};
  }

  std::optional<int> score;
  while (true) {
    const LineReadResult line = engine.read_line(kEngineLineTimeout);
    if (line.status != LineReadStatus::kLine) {
      throw std::runtime_error{"engine stopped answering during a search of: " + position};
    }
    if (line.line.starts_with(engine_stdio::kResponseBestMove)) {
      break;
    }
    if (line.line.starts_with("info error")) {
      throw std::runtime_error{"engine rejected " + position + ": " + line.line};
    }
    if (!line.line.starts_with("info ")) {
      continue;
    }
    const std::vector<std::string> words = split_words(line.line);
    const auto found = std::ranges::find(words, "score");
    if (found != words.end() && std::next(found) != words.end()) {
      score = parse_number<int>(*std::next(found), "search score");
    }
  }
  if (!score.has_value()) {
    throw std::runtime_error{"engine reported no score for: " + position};
  }
  return *score;
}

// The low bit of an FNV-1a hash is a fixed parity of the input's low bits; the top bit depends on
// every input bit.
[[nodiscard]] bool in_lower_half(std::uint64_t hash) noexcept { return (hash >> 63U) == 0U; }

}  // namespace

std::string_view partition_name(BookPartition partition) noexcept {
  return partition == BookPartition::kDevelopment ? "development" : "holdout";
}

double first_player_points(int score, int plies) noexcept {
  const int first_player_score = (plies % 2 == 0) ? score : -score;
  return first_player_score / 2.0;
}

std::optional<double> LeadCorrection::predict(int plies, double points) const {
  const auto found = by_plies.find(plies);
  if (found == by_plies.end()) {
    return std::nullopt;
  }
  return found->second.intercept + (found->second.slope * points);
}

LeadCorrection fit_lead_correction(const std::vector<LeadSample>& samples, int min_openings) {
  std::map<int, std::vector<const LeadSample*>> by_plies;
  for (const LeadSample& sample : samples) {
    by_plies[sample.plies].push_back(&sample);
  }

  LeadCorrection correction;
  for (const auto& [plies, group] : by_plies) {
    if (static_cast<int>(group.size()) < min_openings) {
      continue;
    }
    const auto count = static_cast<double>(group.size());
    double points_total = 0.0;
    double lead_total = 0.0;
    for (const LeadSample* sample : group) {
      points_total += sample->points;
      lead_total += sample->lead;
    }
    const double mean_points = points_total / count;
    const double mean_lead = lead_total / count;
    double cross = 0.0;
    double squares = 0.0;
    for (const LeadSample* sample : group) {
      cross += (sample->points - mean_points) * (sample->lead - mean_lead);
      squares += (sample->points - mean_points) * (sample->points - mean_points);
    }
    if (squares <= 0.0) {
      continue;
    }
    const double slope = cross / squares;
    correction.by_plies[plies] = LengthCorrection{
        .intercept = mean_lead - (slope * mean_points),
        .slope = slope,
        .openings = static_cast<int>(group.size()),
    };
  }
  return correction;
}

std::string format_correction_record(const CorrectionRecord& record,
                                     const std::vector<std::string>& notes) {
  std::ostringstream output;
  output << "# poe2 opening lead correction\n"
         << "# predicted first-player lead after the handicap = intercept + slope * points,\n"
         << "# where points is the first player's search score in points;\n"
         << "# correction_<plies>=<intercept> <slope> <openings fitted>\n";
  for (const std::string& note : notes) {
    output << "# " << note << '\n';
  }
  output << "engine_version=" << record.engine.version << '\n'
         << "net_name=" << record.engine.net_name << '\n'
         << "net_id=" << record.engine.net_id << '\n'
         << "net_sha256=" << record.engine.net_sha256 << '\n'
         << "nodes=" << record.search.nodes << '\n'
         << "hash_mb=" << record.search.hash_mb << '\n';
  for (const auto& [plies, length] : record.correction.by_plies) {
    output << kCorrectionPrefix << plies << '=' << fixed_text(length.intercept, 6) << ' '
           << fixed_text(length.slope, 6) << ' ' << length.openings << '\n';
  }
  return output.str();
}

CorrectionRecord parse_correction_record(std::string_view text) {
  CorrectionRecord record;
  bool has_nodes = false;
  bool has_hash = false;
  std::istringstream input{std::string{text}};
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line.starts_with('#')) {
      continue;
    }
    const std::size_t equals = line.find('=');
    if (equals == std::string::npos) {
      throw std::invalid_argument{"malformed correction line: " + line};
    }
    const std::string key = line.substr(0, equals);
    const std::string value = line.substr(equals + 1);
    if (key == "engine_version") {
      record.engine.version = value;
    } else if (key == "net_name") {
      record.engine.net_name = value;
    } else if (key == "net_id") {
      record.engine.net_id = value;
    } else if (key == "net_sha256") {
      record.engine.net_sha256 = value;
    } else if (key == "nodes") {
      record.search.nodes = parse_number<std::uint64_t>(value, "nodes");
      has_nodes = true;
    } else if (key == "hash_mb") {
      record.search.hash_mb = parse_number<int>(value, "hash_mb");
      has_hash = true;
    } else if (key.starts_with(kCorrectionPrefix)) {
      const int plies =
          parse_number<int>(std::string_view{key}.substr(kCorrectionPrefix.size()), "plies");
      const std::vector<std::string> fields = split_words(value);
      if (fields.size() != 3) {
        throw std::invalid_argument{"malformed correction line: " + line};
      }
      record.correction.by_plies[plies] = LengthCorrection{
          .intercept = parse_number<double>(fields[0], "intercept"),
          .slope = parse_number<double>(fields[1], "slope"),
          .openings = parse_number<int>(fields[2], "openings"),
      };
    } else {
      throw std::invalid_argument{"unknown correction key: " + key};
    }
  }
  if (record.engine.net_sha256.empty() || !has_nodes || !has_hash ||
      record.correction.by_plies.empty()) {
    throw std::invalid_argument{
        "correction record needs net_sha256, nodes, hash_mb, and constants"};
  }
  return record;
}

double OpeningRecord::mean_lead() const noexcept {
  return games == 0 ? 0.0 : static_cast<double>(lead_half_points) / 2.0 / games;
}

std::vector<OpeningRecord> collect_opening_records(
    const std::vector<std::filesystem::path>& games_files) {
  std::vector<OpeningRecord> records;
  std::unordered_map<std::string, std::size_t> index_by_position;
  for (const std::filesystem::path& path : games_files) {
    std::ifstream input{path};
    if (!input) {
      throw std::runtime_error{"failed to read " + path.string()};
    }
    std::string header;
    std::getline(input, header);
    const std::vector<std::string> columns = split_csv_row(header);
    const auto column = [&](std::string_view name) {
      const auto found = std::ranges::find(columns, name);
      if (found == columns.end()) {
        throw std::runtime_error{path.string() + " has no column " + std::string{name}};
      }
      return static_cast<std::size_t>(found - columns.begin());
    };
    const std::size_t opening_column = column("opening_moves");
    const std::size_t winner_column = column("winner");
    const std::size_t reason_column = column("reason");
    const std::size_t player_one_column = column("player_one_score");
    const std::size_t player_two_column = column("player_two_score");

    std::map<std::size_t, std::vector<std::string>> winners_by_opening;
    std::string row;
    while (std::getline(input, row)) {
      if (row.empty()) {
        continue;
      }
      const std::vector<std::string> fields = split_csv_row(row);
      if (fields.size() != columns.size()) {
        throw std::runtime_error{path.string() + " has a malformed row: " + row};
      }
      const std::string& opening = fields[opening_column];
      if (fields[reason_column] != "normal" || opening.empty() || opening == "startpos") {
        continue;
      }
      const std::vector<std::string> moves = split_words(opening);
      const PositionKey key = canonical_opening_key(moves);
      const auto [found, inserted] =
          index_by_position.try_emplace(canonical_key_text(key), records.size());
      if (inserted) {
        records.push_back(OpeningRecord{.moves = moves, .canonical_key = key});
      }
      OpeningRecord& record = records[found->second];
      const int lead_half_points =
          (2 * (parse_number<int>(fields[player_one_column], "player_one_score") -
                parse_number<int>(fields[player_two_column], "player_two_score"))) -
          kPlayerTwoHandicapHalfPoints;
      ++record.games;
      record.lead_half_points += lead_half_points;
      record.first_player_wins += lead_half_points > 0 ? 1 : 0;
      winners_by_opening[found->second].push_back(fields[winner_column]);
    }
    for (const auto& [index, winners] : winners_by_opening) {
      if (winners.size() != 2) {
        continue;
      }
      OpeningRecord& record = records[index];
      ++record.pairs;
      record.outright_pairs += winners[0] == winners[1] && winners[0] != "none" ? 1 : 0;
    }
  }

  std::ranges::sort(records, [](const OpeningRecord& lhs, const OpeningRecord& rhs) {
    if (lhs.canonical_key.low != rhs.canonical_key.low) {
      return lhs.canonical_key.low < rhs.canonical_key.low;
    }
    return lhs.canonical_key.high < rhs.canonical_key.high;
  });
  return records;
}

HeldOutCheck check_held_out(const std::vector<OpeningRecord>& records,
                            const std::vector<int>& scores, double threshold, int min_openings) {
  if (records.size() != scores.size()) {
    throw std::invalid_argument{"held-out check needs one score per opening"};
  }
  const auto in_fit = [](const OpeningRecord& record) {
    return in_lower_half(hash_opening(0, record.plies(), record.canonical_key));
  };

  HeldOutCheck check;
  std::vector<LeadSample> samples;
  for (std::size_t index = 0; index < records.size(); ++index) {
    const OpeningRecord& record = records[index];
    if (in_fit(record)) {
      samples.push_back(LeadSample{
          .plies = record.plies(),
          .points = first_player_points(scores[index], record.plies()),
          .lead = record.mean_lead(),
      });
    }
  }
  check.fit_openings = static_cast<int>(samples.size());
  const LeadCorrection correction = fit_lead_correction(samples, min_openings);

  struct LengthTotals {
    std::vector<double> errors;
    double points = 0.0;
    double predicted = 0.0;
    double lead = 0.0;
    int kept = 0;
  };
  std::map<int, LengthTotals> totals;
  std::vector<double> errors;
  for (std::size_t index = 0; index < records.size(); ++index) {
    const OpeningRecord& record = records[index];
    if (in_fit(record)) {
      continue;
    }
    const double points = first_player_points(scores[index], record.plies());
    const std::optional<double> predicted = correction.predict(record.plies(), points);
    if (!predicted.has_value()) {
      continue;
    }
    ++check.held_out_openings;
    check.pairs += record.pairs;
    check.outright_pairs += record.outright_pairs;
    const double error = *predicted - record.mean_lead();
    errors.push_back(error);
    LengthTotals& length = totals[record.plies()];
    length.errors.push_back(error);
    length.points += points;
    length.predicted += *predicted;
    length.lead += record.mean_lead();
    if (std::abs(*predicted) < threshold) {
      ++length.kept;
      ++check.kept_openings;
      check.kept_pairs += record.pairs;
      check.kept_outright_pairs += record.outright_pairs;
      check.kept_games += record.games;
      check.kept_first_player_wins += record.first_player_wins;
    }
    if (std::abs(points) < threshold) {
      ++check.uncorrected_kept_openings;
      check.uncorrected_kept_games += record.games;
      check.uncorrected_kept_first_player_wins += record.first_player_wins;
    }
  }

  const ErrorSummary overall = summarize_errors(errors);
  check.mean_error = overall.mean;
  check.error_standard_error = overall.standard_error;
  check.residual_sd = overall.sd;
  for (const auto& [plies, length] : totals) {
    const auto count = static_cast<double>(length.errors.size());
    const ErrorSummary summary = summarize_errors(length.errors);
    check.lengths.push_back(HeldOutLength{
        .plies = plies,
        .openings = static_cast<int>(length.errors.size()),
        .mean_points = length.points / count,
        .mean_predicted = length.predicted / count,
        .mean_lead = length.lead / count,
        .mean_error = summary.mean,
        .error_standard_error = summary.standard_error,
        .kept = length.kept,
    });
  }
  return check;
}

std::vector<std::string> describe_held_out(const HeldOutCheck& check, double threshold) {
  const std::string limit = fixed_text(threshold, 2);
  std::vector<std::string> lines;
  lines.push_back("held-out check: fitted on " + std::to_string(check.fit_openings) +
                  " openings, tested on " + std::to_string(check.held_out_openings) +
                  " (split by canonical position hash); leads in points after the handicap");
  for (const HeldOutLength& length : check.lengths) {
    lines.push_back(
        "  " + std::to_string(length.plies) + " plies: openings " +
        std::to_string(length.openings) + ", search " + signed_text(length.mean_points, 2) +
        ", corrected " + signed_text(length.mean_predicted, 2) + ", games " +
        signed_text(length.mean_lead, 2) + ", corrected minus games " +
        signed_text(length.mean_error, 2) + " (se " + fixed_text(length.error_standard_error, 2) +
        "), kept " + percent_text(length.kept, length.openings));
  }
  lines.push_back("  all: corrected minus games " + signed_text(check.mean_error, 2) + " (se " +
                  fixed_text(check.error_standard_error, 2) + "), residual sd " +
                  fixed_text(check.residual_sd, 2));
  lines.push_back("  kept |corrected| < " + limit + ": " +
                  percent_text(check.kept_openings, check.held_out_openings) +
                  " of openings; outright pairs " +
                  percent_text(check.kept_outright_pairs, check.kept_pairs) + " of " +
                  std::to_string(check.kept_pairs) +
                  " (every opening: " + percent_text(check.outright_pairs, check.pairs) + " of " +
                  std::to_string(check.pairs) + "); first player wins " +
                  percent_text(check.kept_first_player_wins, check.kept_games) + " of " +
                  std::to_string(check.kept_games) + " games");
  lines.push_back(
      "  uncorrected |search| < " + limit + ": " +
      percent_text(check.uncorrected_kept_openings, check.held_out_openings) +
      " of openings; first player wins " +
      percent_text(check.uncorrected_kept_first_player_wins, check.uncorrected_kept_games) +
      " of " + std::to_string(check.uncorrected_kept_games) + " games");
  return lines;
}

BookPartition assign_partition(std::uint64_t seed, const GeneratedOpening& opening,
                               const std::unordered_map<std::string, BookPartition>& previous) {
  if (const auto found = previous.find(canonical_key_text(opening.canonical_key));
      found != previous.end()) {
    return found->second;
  }
  return in_lower_half(hash_opening(seed, opening.plies, opening.canonical_key))
             ? BookPartition::kDevelopment
             : BookPartition::kHoldout;
}

QuotaFiller::QuotaFiller(std::optional<int> quota, double threshold)
    : quota_(quota), threshold_(threshold) {}

void QuotaFiller::offer(SelectedOpening candidate, BookPartition partition) {
  PartitionSelection& selection =
      partition == BookPartition::kDevelopment ? development_ : holdout_;
  if (quota_.has_value() && static_cast<int>(selection.kept.size()) >= *quota_) {
    return;
  }
  ++selection.candidates;
  if (std::abs(candidate.predicted_lead) < threshold_) {
    selection.kept.push_back(std::move(candidate));
  }
}

bool QuotaFiller::complete() const noexcept {
  return quota_.has_value() && static_cast<int>(development_.kept.size()) >= *quota_ &&
         static_cast<int>(holdout_.kept.size()) >= *quota_;
}

const PartitionSelection& QuotaFiller::selection(BookPartition partition) const noexcept {
  return partition == BookPartition::kDevelopment ? development_ : holdout_;
}

std::string format_balanced_book(const BalancedBookSettings& settings, BookPartition partition,
                                 const std::map<int, QuotaFiller>& lengths) {
  std::size_t count = 0;
  for (const auto& [plies, filler] : lengths) {
    count += filler.selection(partition).kept.size();
  }

  std::ostringstream output;
  output << "# poe2 opening suite\n"
         << "# generator=search-balanced-corpus\n"
         << "# partition=" << partition_name(partition) << '\n'
         << "# seed=" << settings.seed << '\n'
         << "# plies=" << plies_list_text(settings.plies) << '\n'
         << "# max_score_gap=" << settings.max_score_gap << '\n'
         << "# engine_version=" << settings.correction.engine.version << '\n'
         << "# net=" << settings.correction.engine.net_name << ' '
         << settings.correction.engine.net_id << ' ' << settings.correction.engine.net_sha256
         << '\n'
         << "# nodes=" << settings.correction.search.nodes << '\n'
         << "# hash_mb=" << settings.correction.search.hash_mb << '\n'
         << "# correction=" << settings.correction_file.path << ' '
         << settings.correction_file.digest << '\n';
  for (const auto& [plies, length] : settings.correction.correction.by_plies) {
    output << "# correction_" << plies << '=' << fixed_text(length.intercept, 6) << ' '
           << fixed_text(length.slope, 6) << '\n';
  }
  output << "# threshold=" << fixed_text(settings.threshold, 2) << '\n'
         << "# per_length=" << settings.per_length << '\n';
  const auto write_source = [&](std::string_view name, const std::optional<BookSource>& source) {
    if (source.has_value()) {
      output << "# " << name << '=' << source->path << ' ' << source->digest << '\n';
    }
  };
  write_source("previous_development", settings.previous_development);
  write_source("previous_holdout", settings.previous_holdout);
  output << "# count=" << count << '\n';
  for (const auto& [plies, filler] : lengths) {
    const PartitionSelection& selection = filler.selection(partition);
    const auto inherited = std::ranges::count_if(
        selection.kept, [](const SelectedOpening& opening) { return opening.inherited; });
    output << "# depth_count_" << plies << '=' << selection.kept.size() << " candidates_" << plies
           << '=' << selection.candidates << " inherited_" << plies << '=' << inherited << '\n';
  }
  for (const auto& [plies, filler] : lengths) {
    for (const SelectedOpening& selected : filler.selection(partition).kept) {
      output << format_opening_moves(selected.opening.moves) << "  # score=" << selected.score
             << " lead=" << signed_text(selected.predicted_lead, 2) << '\n';
    }
  }
  return output.str();
}

std::string balance_engine_command(const std::string& engine_path, const BalanceSearch& search,
                                   const std::string& net_path) {
  std::string command = shell_quote(engine_path) + " --hash-mb " + std::to_string(search.hash_mb);
  if (!net_path.empty()) {
    command += " --net " + shell_quote(net_path);
  }
  return command;
}

std::vector<int> search_opening_scores(const std::string& engine_command, std::uint64_t nodes,
                                       const std::vector<std::vector<std::string>>& openings,
                                       int workers) {
  std::signal(SIGPIPE, SIG_IGN);
  std::vector<int> scores(openings.size(), 0);
  std::atomic<std::size_t> next{0};
  std::atomic<bool> failed{false};
  std::exception_ptr failure;
  std::mutex failure_mutex;

  const auto work = [&]() {
    try {
      EngineProcess engine{engine_command};
      engine.start();
      if (!engine.write_line(engine_stdio::kCommandPoe2)) {
        throw std::runtime_error{"could not start engine: " + engine_command};
      }
      expect_line(engine, engine_stdio::kResponsePoe2Ok);
      while (!failed.load()) {
        const std::size_t index = next.fetch_add(1);
        if (index >= openings.size()) {
          break;
        }
        scores[index] = search_score(engine, nodes, openings[index]);
      }
      static_cast<void>(engine.write_line(engine_stdio::kCommandQuit));
    } catch (...) {
      const std::scoped_lock lock{failure_mutex};
      if (!failure) {
        failure = std::current_exception();
      }
      failed.store(true);
    }
  };

  const int thread_count =
      std::max(1, std::min(workers, static_cast<int>(std::max<std::size_t>(openings.size(), 1))));
  std::vector<std::thread> threads;
  threads.reserve(static_cast<std::size_t>(thread_count));
  for (int worker = 0; worker < thread_count; ++worker) {
    threads.emplace_back(work);
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
  if (failure) {
    std::rethrow_exception(failure);
  }
  return scores;
}

}  // namespace poe2::match_runner
