#include "poe2/opening_balance.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "poe2/match_runner.hpp"
#include "poe2/opening_generation.hpp"

namespace {

using poe2::match_runner::BookPartition;

[[nodiscard]] std::vector<std::string> words(std::string_view text) {
  std::istringstream input{std::string{text}};
  std::vector<std::string> result;
  std::string word;
  while (input >> word) {
    result.push_back(word);
  }
  return result;
}

[[nodiscard]] poe2::match_runner::GeneratedOpening opening(std::string_view moves) {
  const std::vector<std::string> list = words(moves);
  return poe2::match_runner::GeneratedOpening{
      .plies = static_cast<int>(list.size()),
      .moves = list,
      .canonical_key = poe2::match_runner::canonical_opening_key(list),
  };
}

[[nodiscard]] poe2::match_runner::SelectedOpening scored(std::string_view moves, double lead) {
  return poe2::match_runner::SelectedOpening{.opening = opening(moves), .predicted_lead = lead};
}

[[nodiscard]] std::filesystem::path write_games(const std::filesystem::path& path,
                                                std::string_view rows) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output{path};
  output << "game,engine_one_player,opening_slot,opening_line,opening_moves,winner,reason,plies,"
            "player_one_score,player_two_score\n"
         << rows;
  return path;
}

}  // namespace

TEST_CASE("opening positions are compared up to the board's symmetries", "[opening_balance]") {
  const poe2::PositionKey key = poe2::match_runner::canonical_opening_key(words("a1 b2"));
  REQUIRE(poe2::match_runner::canonical_opening_key(words("g7 f6")) == key);
  REQUIRE(poe2::match_runner::canonical_opening_key(words("a7 b6")) == key);
  REQUIRE(poe2::match_runner::canonical_opening_key(words("g1 f2")) == key);
  REQUIRE_FALSE(poe2::match_runner::canonical_opening_key(words("b2 a1")) == key);
  REQUIRE_THROWS_AS(poe2::match_runner::canonical_opening_key(words("a1 a1")),
                    std::invalid_argument);

  std::unordered_set<std::string> seen;
  std::mt19937_64 rng{3};  // NOLINT(bugprone-random-generator-seed)
  int generated = 0;
  for (int attempt = 0; attempt < 20000; ++attempt) {
    if (const auto line = poe2::match_runner::generate_random_opening(rng, 2, 4, seen);
        line.has_value()) {
      ++generated;
    }
  }
  REQUIRE(generated == 315);
  REQUIRE(poe2::match_runner::generate_systematic_two_ply_openings().size() == 315);
}

TEST_CASE("search scores are read from the first player's side", "[opening_balance]") {
  REQUIRE(poe2::match_runner::first_player_points(7, 4) == Catch::Approx(3.5));
  REQUIRE(poe2::match_runner::first_player_points(7, 3) == Catch::Approx(-3.5));
}

TEST_CASE("the lead correction is a least-squares line for each length", "[opening_balance]") {
  std::vector<poe2::match_runner::LeadSample> samples;
  for (int index = 0; index < 10; ++index) {
    const double points = index - 3.0;
    samples.push_back({.plies = 4, .points = points, .lead = -2.5 + (0.9 * points)});
    samples.push_back({.plies = 6,
                       .points = points,
                       .lead = 1.0 + (0.5 * points) + (index % 2 == 0 ? 0.25 : -0.25)});
  }
  samples.push_back({.plies = 8, .points = 1.0, .lead = 1.0});

  const poe2::match_runner::LeadCorrection correction =
      poe2::match_runner::fit_lead_correction(samples, 5);
  REQUIRE(correction.by_plies.size() == 2);
  REQUIRE(correction.by_plies.at(4).intercept == Catch::Approx(-2.5));
  REQUIRE(correction.by_plies.at(4).slope == Catch::Approx(0.9));
  REQUIRE(correction.by_plies.at(4).openings == 10);
  REQUIRE(correction.by_plies.at(6).slope == Catch::Approx(0.5 - (1.25 / 82.5)));
  REQUIRE(correction.predict(4, 3.0).value_or(0.0) == Catch::Approx(0.2));
  REQUIRE_FALSE(correction.predict(8, 1.0).has_value());
}

TEST_CASE("a correction record round-trips through its text", "[opening_balance]") {
  poe2::match_runner::CorrectionRecord record{
      .engine = {.version = "0.3.2",
                 .net_name = "run1/g002",
                 .net_id = "232a3cc37caf",
                 .net_sha256 = std::string(64, 'a')},
      .search = {.nodes = 1234, .hash_mb = 2},
  };
  record.correction.by_plies[2] = {.intercept = -3.1000004, .slope = 0.8894471, .openings = 135};
  record.correction.by_plies[14] = {.intercept = 0.25, .slope = 1.0, .openings = 30};

  const std::string text = poe2::match_runner::format_correction_record(record, {"note = 1"});
  REQUIRE(text.find("\n# note = 1\n") != std::string::npos);
  REQUIRE(text.find("\ncorrection_2=-3.100000 0.889447 135\n") != std::string::npos);
  const poe2::match_runner::CorrectionRecord parsed =
      poe2::match_runner::parse_correction_record(text);
  REQUIRE(parsed.engine.net_sha256 == record.engine.net_sha256);
  REQUIRE(parsed.engine.version == "0.3.2");
  REQUIRE(parsed.search.nodes == 1234);
  REQUIRE(parsed.search.hash_mb == 2);
  REQUIRE(parsed.correction.by_plies.at(2).intercept == -3.1);
  REQUIRE(parsed.correction.by_plies.at(2).slope == 0.889447);
  REQUIRE(parsed.correction.by_plies.at(14).openings == 30);
  REQUIRE(poe2::match_runner::format_correction_record(parsed, {"note = 1"}) == text);

  REQUIRE_THROWS_AS(poe2::match_runner::parse_correction_record(text + "threshold=2\n"),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(poe2::match_runner::parse_correction_record("net_sha256=x\nnodes=1\n"),
                    std::invalid_argument);
}

TEST_CASE("game records merge symmetric openings and count outright pairs", "[opening_balance]") {
  const std::filesystem::path root = "opening-balance-test";
  const std::filesystem::path first = write_games(root / "first" / "games.csv",
                                                  "1,p1,1,1,a1 b2,engine_one,normal,49,100,90\n"
                                                  "2,p2,1,1,a1 b2,engine_two,normal,49,100,90\n"
                                                  "3,p1,2,2,c3 d4,engine_one,normal,49,100,90\n"
                                                  "4,p2,2,2,c3 d4,engine_one,normal,49,90,95\n"
                                                  "5,p1,3,3,e5 f6,none,timeout,9,10,10\n"
                                                  "6,p2,3,3,e5 f6,engine_two,normal,49,99,90\n");
  const std::filesystem::path second = write_games(root / "second" / "games.csv",
                                                   "1,p1,1,1,g7 f6,engine_two,normal,49,90,90\n"
                                                   "2,p2,1,1,g7 f6,engine_two,normal,49,100,90\n");

  const std::vector<poe2::match_runner::OpeningRecord> records =
      poe2::match_runner::collect_opening_records({first, second});
  REQUIRE(records.size() == 3);
  std::map<std::string, const poe2::match_runner::OpeningRecord*> by_first_move;
  for (const poe2::match_runner::OpeningRecord& record : records) {
    by_first_move[record.moves.front()] = &record;
  }
  const poe2::match_runner::OpeningRecord& merged = *by_first_move.at("a1");
  REQUIRE(merged.games == 4);
  REQUIRE(merged.pairs == 2);
  REQUIRE(merged.outright_pairs == 1);
  REQUIRE(merged.first_player_wins == 3);
  REQUIRE(merged.mean_lead() == Catch::Approx(((4.5 * 3) - 5.5) / 4.0));
  REQUIRE(by_first_move.at("c3")->outright_pairs == 1);
  REQUIRE(by_first_move.at("e5")->games == 1);
  REQUIRE(by_first_move.at("e5")->pairs == 0);
  std::filesystem::remove_all(root);
}

TEST_CASE("the held-out check fits on one half and tests on the other", "[opening_balance]") {
  std::vector<poe2::match_runner::OpeningRecord> records;
  std::vector<int> scores;
  for (const poe2::match_runner::GeneratedOpening& line :
       poe2::match_runner::generate_systematic_two_ply_openings()) {
    const int score = static_cast<int>(records.size() % 13) - 6;
    records.push_back(poe2::match_runner::OpeningRecord{
        .moves = line.moves,
        .canonical_key = line.canonical_key,
        .games = 2,
        .lead_half_points = 2LL * (score - 4),
        .first_player_wins = score > 4 ? 2 : 0,
        .pairs = 1,
        .outright_pairs = (score >= 3 && score <= 5) ? 0 : 1,
    });
    scores.push_back(score);
  }

  const poe2::match_runner::HeldOutCheck check =
      poe2::match_runner::check_held_out(records, scores, 1.0, 30);
  REQUIRE(check.fit_openings + check.held_out_openings == 315);
  REQUIRE(check.fit_openings > 100);
  REQUIRE(check.held_out_openings > 100);
  REQUIRE(check.mean_error == Catch::Approx(0.0).margin(1e-9));
  REQUIRE(check.kept_openings > 0);
  REQUIRE(check.kept_outright_pairs == 0);
  REQUIRE(check.uncorrected_kept_openings > 0);
  REQUIRE(poe2::match_runner::describe_held_out(check, 1.0).size() == 5);
}

TEST_CASE("an opening from a previous book keeps that partition", "[opening_balance]") {
  const poe2::match_runner::GeneratedOpening line = opening("a1 b2 c3 d4");
  const poe2::match_runner::GeneratedOpening mirrored = opening("g1 f2 e3 d4");
  REQUIRE(line.canonical_key == mirrored.canonical_key);

  int development = 0;
  for (std::uint64_t seed = 1; seed <= 64; ++seed) {
    const BookPartition partition = poe2::match_runner::assign_partition(seed, line, {});
    REQUIRE(poe2::match_runner::assign_partition(seed, mirrored, {}) == partition);
    development += partition == BookPartition::kDevelopment ? 1 : 0;
  }
  REQUIRE(development > 16);
  REQUIRE(development < 48);

  const std::unordered_map<std::string, BookPartition> previous{
      {poe2::match_runner::canonical_key_text(line.canonical_key), BookPartition::kHoldout}};
  for (std::uint64_t seed = 1; seed <= 64; ++seed) {
    REQUIRE(poe2::match_runner::assign_partition(seed, mirrored, previous) ==
            BookPartition::kHoldout);
  }
}

TEST_CASE("a quota stops counting a partition once it is full", "[opening_balance]") {
  poe2::match_runner::QuotaFiller filler{2, 2.0};
  filler.offer(scored("a1 b1", 1.99), BookPartition::kDevelopment);
  filler.offer(scored("a1 c1", -2.0), BookPartition::kDevelopment);
  filler.offer(scored("a1 d1", -1.0), BookPartition::kHoldout);
  filler.offer(scored("a1 e1", 0.0), BookPartition::kDevelopment);
  filler.offer(scored("a1 f1", 0.0), BookPartition::kDevelopment);
  REQUIRE_FALSE(filler.complete());
  filler.offer(scored("a1 g1", 3.0), BookPartition::kHoldout);
  filler.offer(scored("b1 c1", 0.5), BookPartition::kHoldout);
  REQUIRE(filler.complete());
  filler.offer(scored("b1 d1", 0.5), BookPartition::kHoldout);

  const poe2::match_runner::PartitionSelection& development =
      filler.selection(BookPartition::kDevelopment);
  REQUIRE(development.candidates == 3);
  REQUIRE(development.kept.size() == 2);
  REQUIRE(development.kept[0].opening.moves == words("a1 b1"));
  REQUIRE(development.kept[1].opening.moves == words("a1 e1"));
  const poe2::match_runner::PartitionSelection& holdout = filler.selection(BookPartition::kHoldout);
  REQUIRE(holdout.candidates == 3);
  REQUIRE(holdout.kept.size() == 2);
  REQUIRE(holdout.kept[1].opening.moves == words("b1 c1"));

  poe2::match_runner::QuotaFiller unlimited{std::nullopt, 2.0};
  for (int index = 0; index < 5; ++index) {
    unlimited.offer(scored("a1 b1", index - 2.0), BookPartition::kHoldout);
  }
  REQUIRE_FALSE(unlimited.complete());
  REQUIRE(unlimited.selection(BookPartition::kHoldout).kept.size() == 3);
}

TEST_CASE("a balanced book records its inputs and loads as an opening book", "[opening_balance]") {
  poe2::match_runner::BalancedBookSettings settings{
      .seed = 5,
      .plies = {2, 4},
      .max_score_gap = 4,
      .threshold = 2.0,
      .per_length = 1,
      .correction_file = {.path = "correction.txt", .digest = "fnv1a64:0"},
      .previous_holdout = poe2::match_runner::BookSource{.path = "old.txt", .digest = "fnv1a64:1"},
  };
  settings.correction.engine = {
      .version = "0.3.2", .net_name = "run1/g002", .net_id = "id", .net_sha256 = "sha"};
  settings.correction.correction.by_plies[2] = {.intercept = -3.0, .slope = 0.9, .openings = 40};
  settings.correction.correction.by_plies[4] = {.intercept = -2.5, .slope = 0.9, .openings = 40};

  std::map<int, poe2::match_runner::QuotaFiller> lengths;
  lengths.try_emplace(2, std::nullopt, 2.0)
      .first->second.offer(scored("a1 b2", -0.25), BookPartition::kDevelopment);
  auto& deeper = lengths.try_emplace(4, 1, 2.0).first->second;
  poe2::match_runner::SelectedOpening inherited = scored("a1 b2 c3 d4", 1.5);
  inherited.score = 9;
  inherited.inherited = true;
  deeper.offer(scored("a1 b2 c3 e4", 2.5), BookPartition::kDevelopment);
  deeper.offer(inherited, BookPartition::kDevelopment);

  const std::string text =
      poe2::match_runner::format_balanced_book(settings, BookPartition::kDevelopment, lengths);
  for (const std::string_view expected :
       {"# generator=search-balanced-corpus\n", "# partition=development\n", "# seed=5\n",
        "# net=run1/g002 id sha\n", "# nodes=200000\n", "# hash_mb=16\n",
        "# correction=correction.txt fnv1a64:0\n", "# correction_2=-3.000000 0.900000\n",
        "# threshold=2.00\n", "# per_length=1\n", "# previous_holdout=old.txt fnv1a64:1\n",
        "# count=2\n", "# depth_count_2=1 candidates_2=1 inherited_2=0\n",
        "# depth_count_4=1 candidates_4=2 inherited_4=1\n", "\na1 b2  # score=0 lead=-0.25\n",
        "\na1 b2 c3 d4  # score=9 lead=+1.50\n"}) {
    INFO(expected);
    REQUIRE(text.find(expected) != std::string::npos);
  }
  REQUIRE(text.find("previous_development") == std::string::npos);

  const poe2::match_runner::OpeningBook book =
      poe2::match_runner::parse_opening_book_text("balanced", text);
  REQUIRE(book.lines.size() == 2);
  REQUIRE(book.lines[1].text == "a1 b2 c3 d4");
}

TEST_CASE("committed balanced books satisfy their selection invariants", "[opening_balance]") {
  const std::filesystem::path root = std::filesystem::path{POE2_SOURCE_DIR} / "eval" / "openings";
  const auto canonical_positions = [](const poe2::match_runner::OpeningBook& book) {
    std::unordered_set<std::string> positions;
    for (const poe2::match_runner::OpeningLine& line : book.lines) {
      if (!line.moves.empty()) {
        REQUIRE(positions
                    .insert(poe2::match_runner::canonical_key_text(
                        poe2::match_runner::canonical_opening_key(line.moves)))
                    .second);
      }
    }
    return positions;
  };
  const auto load = [&](std::string_view name) {
    return poe2::match_runner::load_opening_book((root / name).string());
  };
  const std::unordered_set<std::string> old_development =
      canonical_positions(load("development.txt"));
  const std::unordered_set<std::string> old_holdout = canonical_positions(load("holdout.txt"));

  REQUIRE(poe2::match_runner::file_digest((root / "development-v2.txt").string()) ==
          "fnv1a64:7d1344d4c57abc5e");
  REQUIRE(poe2::match_runner::file_digest((root / "holdout-v2.txt").string()) ==
          "fnv1a64:b4beb9fe1c802219");

  std::unordered_set<std::string> seen;
  for (const auto& [name, two_ply, forbidden] :
       {std::tuple{"development-v2.txt", 54, &old_holdout},
        std::tuple{"holdout-v2.txt", 64, &old_development}}) {
    INFO(name);
    const poe2::match_runner::OpeningBook book = load(name);
    std::map<int, int> lengths;
    for (const std::string& position : canonical_positions(book)) {
      REQUIRE(seen.insert(position).second);
      REQUIRE_FALSE(forbidden->contains(position));
    }
    for (const poe2::match_runner::OpeningLine& line : book.lines) {
      ++lengths[static_cast<int>(line.moves.size())];
    }
    REQUIRE(lengths.at(2) == two_ply);
    for (const int plies : {4, 6, 8, 10, 12, 14}) {
      REQUIRE(lengths.at(plies) == 657);
    }
    REQUIRE(lengths.size() == 7);

    std::ifstream input{root / name};
    std::string text;
    int checked = 0;
    while (std::getline(input, text)) {
      const std::size_t lead = text.find("  # score=");
      if (text.starts_with('#') || lead == std::string::npos) {
        continue;
      }
      const std::size_t value = text.find("lead=", lead);
      REQUIRE(value != std::string::npos);
      REQUIRE(std::abs(std::stod(text.substr(value + 5))) < 2.0);
      ++checked;
    }
    REQUIRE(checked == static_cast<int>(book.lines.size()));
  }
}
