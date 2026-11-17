#include <algorithm>
#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "poe2/engine_stdio.hpp"
#include "poe2/minimax/engine.hpp"
#include "poe2/minimax/evaluation.hpp"
#include "poe2/minimax/net.hpp"
#include "poe2/minimax/options.hpp"
#include "poe2/minimax/search.hpp"
#include "poe2/move.hpp"

namespace fs = std::filesystem;
using Catch::Matchers::ContainsSubstring;

namespace {

[[nodiscard]] fs::path net_fixtures() {
  return fs::path{POE2_SOURCE_DIR} / "tests" / "fixtures" / "nets";
}

[[nodiscard]] std::vector<std::uint8_t> read_bytes(const fs::path& path) {
  std::ifstream input{path, std::ios::binary};
  REQUIRE(input.good());
  return {std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
}

[[nodiscard]] fs::path reference_net_path() {
  return net_fixtures() / std::string{poe2::minimax::kPatternGainDesign} / "reference.poe2net";
}

[[nodiscard]] poe2::Position position_from_bitboards(poe2::Bitboard player_one,
                                                     poe2::Bitboard player_two) {
  poe2::Position position;
  const int ply = std::popcount(player_one | player_two);
  for (int move = 0; move < ply; ++move) {
    poe2::Bitboard& remaining = move % 2 == 0 ? player_one : player_two;
    REQUIRE(remaining != 0);
    const int index = std::countr_zero(remaining);
    remaining &= remaining - 1;
    REQUIRE(position.play(poe2::square_from_index(index)));
  }
  return position;
}

[[nodiscard]] poe2::minimax::PatternGainParameters reference_parameters() {
  return poe2::minimax::load_net_file(reference_net_path())->parameters();
}

void require_rejected(const std::vector<std::uint8_t>& bytes, std::string_view reason) {
  try {
    static_cast<void>(poe2::minimax::parse_net(bytes));
    FAIL("net was accepted");
  } catch (const poe2::minimax::NetError& error) {
    REQUIRE_THAT(error.what(), ContainsSubstring(std::string{reason}));
  }
}

[[nodiscard]] std::vector<std::uint8_t> valid_net_bytes(
    const poe2::minimax::NetMetadata& metadata = {{"name", "test/net"}}) {
  return poe2::minimax::serialize_net(poe2::minimax::kPatternGainDesign, metadata,
                                      reference_parameters());
}

}  // namespace

TEST_CASE("every shipped net design still evaluates its reference positions exactly",
          "[minimax][net][reference]") {
  std::vector<std::string> fixture_designs;
  for (const fs::directory_entry& entry : fs::directory_iterator(net_fixtures())) {
    fixture_designs.push_back(entry.path().filename().string());
  }
  std::ranges::sort(fixture_designs);
  std::vector<std::string> supported;
  for (const std::string_view design : poe2::minimax::supported_net_designs()) {
    supported.emplace_back(design);
  }
  std::ranges::sort(supported);
  REQUIRE(fixture_designs == supported);

  for (const std::string& design : fixture_designs) {
    const std::shared_ptr<const poe2::minimax::Net> net =
        poe2::minimax::load_net_file(net_fixtures() / design / "reference.poe2net");
    REQUIRE(net->design() == design);
    std::ifstream positions{net_fixtures() / design / "positions.txt"};
    std::string line;
    std::size_t checked = 0;
    while (std::getline(positions, line)) {
      if (line.empty() || line.front() == '#') {
        continue;
      }
      std::istringstream fields{line};
      std::string player_one;
      std::string player_two;
      poe2::minimax::FixedEvaluation expected = 0;
      REQUIRE(static_cast<bool>(fields >> player_one >> player_two >> expected));
      const poe2::Position position = position_from_bitboards(std::stoull(player_one, nullptr, 16),
                                                              std::stoull(player_two, nullptr, 16));
      REQUIRE(poe2::minimax::evaluate_pattern_gain_fixed(position, net->pattern_gain()) ==
              expected);
      ++checked;
    }
    REQUIRE(checked >= 1000);
  }
}

TEST_CASE("net files carry their name, identity, and canonical bytes", "[minimax][net][format]") {
  const std::vector<std::uint8_t> bytes = read_bytes(reference_net_path());
  const std::shared_ptr<const poe2::minimax::Net> net = poe2::minimax::parse_net(bytes);
  REQUIRE(net->name() == "run1/g000");
  REQUIRE(net->metadata_value("header_sha256").size() == 64);
  REQUIRE(net->id() == "633309adb322");
  REQUIRE(net->sha256().size() == 64);
  REQUIRE(net->id() != net->sha256().substr(0, 12));
  REQUIRE(poe2::minimax::serialize_net(net->design(), net->metadata(), net->parameters()) == bytes);

  const std::shared_ptr<const poe2::minimax::Net> unnamed =
      poe2::minimax::parse_net(valid_net_bytes());
  REQUIRE(unnamed->name() == "test/net");
  REQUIRE(unnamed->id() == unnamed->sha256().substr(0, 12));
}

TEST_CASE("damaged, foreign, and unsupported net files are refused with a reason",
          "[minimax][net][format]") {
  const std::vector<std::uint8_t> valid = valid_net_bytes();
  REQUIRE_NOTHROW(poe2::minimax::parse_net(valid));

  std::vector<std::uint8_t> foreign = valid;
  foreign[0] = 'X';
  require_rejected(foreign, "not a PoE2 net file");

  require_rejected(std::vector<std::uint8_t>(valid.begin(), valid.begin() + 40), "truncated");

  std::vector<std::uint8_t> damaged = valid;
  damaged[200] ^= 1;
  require_rejected(damaged, "integrity check");

  std::vector<std::uint8_t> future = valid;
  future[8] = 2;
  require_rejected(future, "container format 2");

  require_rejected(poe2::minimax::serialize_net("pattern-gain-v9", {{"name", "future/net"}},
                                                reference_parameters()),
                   "net design \"pattern-gain-v9\" is not supported by this engine (supported: "
                   "pattern-gain-v1)");

  require_rejected(valid_net_bytes({{"teacher", "x"}}), "has no name");
  require_rejected(valid_net_bytes({{"name", "x"}, {"corpus", "y"}}), "ascending order");
  require_rejected(valid_net_bytes({{"name", "bad\tvalue"}}), "malformed");
  require_rejected(valid_net_bytes({{"header_sha256", "abc"}, {"name", "x"}}), "header_sha256");

  poe2::minimax::PatternGainParameters knots = reference_parameters();
  knots.gain_knots = {0, 10, 24, 36, 49};
  require_rejected(
      poe2::minimax::serialize_net(poe2::minimax::kPatternGainDesign, {{"name", "x"}}, knots),
      "net gain knots must be 0, 12, 24, 36, 49 in design pattern-gain-v1");
  poe2::minimax::PatternGainParameters intercept = reference_parameters();
  intercept.intercept[3] = (1 << 24) + 1;
  require_rejected(
      poe2::minimax::serialize_net(poe2::minimax::kPatternGainDesign, {{"name", "x"}}, intercept),
      "intercepts");

  const fs::path missing = fs::temp_directory_path() / "poe2-missing-net.poe2net";
  try {
    static_cast<void>(poe2::minimax::load_net_file(missing));
    FAIL("missing net was accepted");
  } catch (const poe2::minimax::NetError& error) {
    REQUIRE_THAT(error.what(), ContainsSubstring(missing.string()));
  }
}

TEST_CASE("search with a file-loaded net matches the same net built in", "[minimax][net][search]") {
  const std::shared_ptr<const poe2::minimax::Net> file =
      poe2::minimax::load_net_file(reference_net_path());
  poe2::Position position;
  for (const int index : {24, 0, 23, 48, 32}) {
    REQUIRE(position.play(poe2::square_from_index(index)));
  }
  const poe2::engine::EngineLimits limits{.nodes = 50'000};
  poe2::minimax::Search built_in{poe2::minimax::SearchOptions{.hash_bytes = 0}};
  poe2::minimax::Search loaded{poe2::minimax::SearchOptions{.hash_bytes = 0, .net = file}};
  const poe2::engine::EngineResult expected = built_in.run(position, limits, {});
  const poe2::engine::EngineResult actual = loaded.run(position, limits, {});
  REQUIRE(actual.best_move == expected.best_move);
  REQUIRE(actual.score == expected.score);
  REQUIRE(actual.nodes == expected.nodes);
  REQUIRE(&loaded.net() == file.get());
}

TEST_CASE("the engine reports its version, designs, and net", "[minimax][net][identity]") {
  const std::shared_ptr<const poe2::minimax::Net> file =
      poe2::minimax::load_net_file(reference_net_path());
  const std::vector<std::string> identity = poe2::minimax::engine_identity(*file);
  REQUIRE(identity.size() == 3);
  REQUIRE(identity[0] == "version " + std::string{POE2_PROJECT_VERSION});
  REQUIRE(identity[1] == "designs pattern-gain-v1");
  REQUIRE(identity[2] == "net run1/g000 633309adb322 " + std::string{file->sha256()});

  poe2::minimax::MinimaxEngine engine{poe2::minimax::SearchOptions{.hash_bytes = 0, .net = file}};
  std::istringstream input{"poe2\nquit\n"};
  std::ostringstream output;
  REQUIRE(poe2::engine_stdio::run_engine_stdio("minimax", engine, input, output) == 0);
  REQUIRE(output.str() == "id name minimax\nid " + identity[0] + "\nid " + identity[1] + "\nid " +
                              identity[2] + "\npoe2ok\n");
}

TEST_CASE("the --net option loads a file or explains why it cannot", "[minimax][net][cli]") {
  std::string error;
  const std::string path = reference_net_path().string();
  const std::vector<std::string_view> arguments{"--net", path};
  const std::optional<poe2::minimax::SearchOptions> options =
      poe2::minimax::parse_search_options(arguments, error);
  if (!options.has_value()) {
    FAIL("--net with a valid file should parse: " + error);
    return;
  }
  REQUIRE(options->net != nullptr);
  REQUIRE(options->net->name() == "run1/g000");

  const std::vector<std::string_view> missing_value{"--net"};
  REQUIRE_FALSE(poe2::minimax::parse_search_options(missing_value, error).has_value());
  REQUIRE(error == "missing value for --net");

  const std::vector<std::string_view> missing_file{"--net", "/nonexistent/net.poe2net"};
  REQUIRE_FALSE(poe2::minimax::parse_search_options(missing_file, error).has_value());
  REQUIRE_THAT(error, ContainsSubstring("could not read net file /nonexistent/net.poe2net"));
}
