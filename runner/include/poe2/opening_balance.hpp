#ifndef POE2_OPENING_BALANCE_HPP
#define POE2_OPENING_BALANCE_HPP

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "poe2/match_runner.hpp"
#include "poe2/opening_generation.hpp"

namespace poe2::match_runner {

enum class BookPartition : std::uint8_t {
  kDevelopment,
  kHoldout,
};

[[nodiscard]] std::string_view partition_name(BookPartition partition) noexcept;

// A search score is in half-points from the side to move after the opening and already includes
// the handicap. Returns points from the first player's side.
[[nodiscard]] double first_player_points(int score, int plies) noexcept;

struct LengthCorrection {
  double intercept = 0.0;
  double slope = 0.0;
  int openings = 0;
};

// Predicted first-player lead after the handicap, in points: intercept + slope * points, fitted
// separately for each opening length because the search's bias depends on it.
struct LeadCorrection {
  std::map<int, LengthCorrection> by_plies;

  [[nodiscard]] std::optional<double> predict(int plies, double points) const;
};

struct LeadSample {
  int plies = 0;
  double points = 0.0;
  double lead = 0.0;
};

// Least squares of lead on points for every length with at least min_openings samples.
[[nodiscard]] LeadCorrection fit_lead_correction(const std::vector<LeadSample>& samples,
                                                 int min_openings);

struct BalanceSearch {
  std::uint64_t nodes = 200000;
  int hash_mb = 16;
};

// What a correction was fitted for: the corrected scores are only meaningful from the same net,
// node budget, and table size.
struct CorrectionRecord {
  EngineIdentity engine;
  BalanceSearch search;
  LeadCorrection correction;
};

// Constants are written with six decimals, and books are always selected with the parsed
// values, so a book depends only on the file's text.
[[nodiscard]] std::string format_correction_record(const CorrectionRecord& record,
                                                   const std::vector<std::string>& notes);
[[nodiscard]] CorrectionRecord parse_correction_record(std::string_view text);

// Game results of one opening across match runs.
struct OpeningRecord {
  std::vector<std::string> moves;
  PositionKey canonical_key;
  int games = 0;
  long long lead_half_points = 0;
  int first_player_wins = 0;
  int pairs = 0;
  int outright_pairs = 0;

  [[nodiscard]] int plies() const noexcept { return static_cast<int>(moves.size()); }
  [[nodiscard]] double mean_lead() const noexcept;
};

// Reads each run's games.csv. Only normal games from a non-empty opening count; a pair is the
// two normal games of one opening in one run, and it is outright when one engine wins both.
// Openings equal under symmetry are merged.
[[nodiscard]] std::vector<OpeningRecord> collect_opening_records(
    const std::vector<std::filesystem::path>& games_files);

struct HeldOutLength {
  int plies = 0;
  int openings = 0;
  double mean_points = 0.0;
  double mean_predicted = 0.0;
  double mean_lead = 0.0;
  double mean_error = 0.0;
  double error_standard_error = 0.0;
  int kept = 0;
};

struct HeldOutCheck {
  int fit_openings = 0;
  int held_out_openings = 0;
  int kept_openings = 0;
  int pairs = 0;
  int outright_pairs = 0;
  int kept_pairs = 0;
  int kept_outright_pairs = 0;
  int kept_games = 0;
  int kept_first_player_wins = 0;
  int uncorrected_kept_openings = 0;
  int uncorrected_kept_games = 0;
  int uncorrected_kept_first_player_wins = 0;
  double mean_error = 0.0;
  double error_standard_error = 0.0;
  double residual_sd = 0.0;
  std::vector<HeldOutLength> lengths;
};

// Fits on the openings whose canonical position hash is in the lower half and tests the selection
// rule on the rest. scores[i] is the search score of records[i].
[[nodiscard]] HeldOutCheck check_held_out(const std::vector<OpeningRecord>& records,
                                          const std::vector<int>& scores, double threshold,
                                          int min_openings);
[[nodiscard]] std::vector<std::string> describe_held_out(const HeldOutCheck& check,
                                                         double threshold);

// An opening found in a previous book keeps that book's partition, so a new holdout never
// contains a position the old development book exposed, and the reverse. Others are split by
// hash.
[[nodiscard]] BookPartition assign_partition(
    std::uint64_t seed, const GeneratedOpening& opening,
    const std::unordered_map<std::string, BookPartition>& previous);

struct SelectedOpening {
  GeneratedOpening opening;
  int score = 0;
  double predicted_lead = 0.0;
  bool inherited = false;
};

struct PartitionSelection {
  int candidates = 0;
  std::vector<SelectedOpening> kept;
};

// Walks candidates of one length in stream order. Without a quota it keeps every passing one;
// with a quota it stops counting for a partition once that partition is full, so the result
// depends only on the order of the stream.
class QuotaFiller final {
 public:
  QuotaFiller(std::optional<int> quota, double threshold);

  void offer(SelectedOpening candidate, BookPartition partition);
  [[nodiscard]] bool complete() const noexcept;
  [[nodiscard]] const PartitionSelection& selection(BookPartition partition) const noexcept;

 private:
  std::optional<int> quota_;
  double threshold_ = 0.0;
  PartitionSelection development_;
  PartitionSelection holdout_;
};

struct BookSource {
  std::string path;
  std::string digest;
};

struct BalancedBookSettings {
  std::uint64_t seed = 0;
  std::vector<int> plies;
  int max_score_gap = 0;
  double threshold = 0.0;
  int per_length = 0;
  CorrectionRecord correction;
  BookSource correction_file;
  std::optional<BookSource> previous_development;
  std::optional<BookSource> previous_holdout;
};

// Header comment lines record every input; each opening line carries its search score and
// predicted lead as a trailing comment.
[[nodiscard]] std::string format_balanced_book(const BalancedBookSettings& settings,
                                               BookPartition partition,
                                               const std::map<int, QuotaFiller>& lengths);

[[nodiscard]] std::string balance_engine_command(const std::string& engine_path,
                                                 const BalanceSearch& search,
                                                 const std::string& net_path);

// Fixed-node search of each opening after clearing the engine's table. Workers take openings in
// any order; each score depends only on its opening, so the result does not depend on workers.
[[nodiscard]] std::vector<int> search_opening_scores(
    const std::string& engine_command, std::uint64_t nodes,
    const std::vector<std::vector<std::string>>& openings, int workers);

}  // namespace poe2::match_runner

#endif  // POE2_OPENING_BALANCE_HPP
