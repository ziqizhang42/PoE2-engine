#ifndef POE2_MINIMAX_SEARCH_HPP
#define POE2_MINIMAX_SEARCH_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "poe2/engine.hpp"
#include "poe2/minimax/net.hpp"
#include "poe2/transposition_table.hpp"

namespace poe2::minimax {

inline constexpr std::size_t kMebibyte = std::size_t{1024} * 1024;
inline constexpr std::size_t kDefaultHashBytes = std::size_t{64} * kMebibyte;

enum class Evaluator : std::uint8_t {
  kStatic,
  kTwoPlyClosure,
  kPatternGain,
};

[[nodiscard]] std::string_view evaluator_name(Evaluator evaluator) noexcept;

struct SearchOptions {
  std::size_t hash_bytes = kDefaultHashBytes;
  bool use_symmetry = true;
  Evaluator evaluator = Evaluator::kPatternGain;
  // The pattern-gain net; null selects the built-in net.
  std::shared_ptr<const Net> net;

  friend bool operator==(const SearchOptions&, const SearchOptions&) = default;
};

struct LeafEvaluator {
  Evaluator kind = Evaluator::kPatternGain;
  const PatternGainTables* tables = nullptr;
};

struct AnalysisLine {
  int rank = 0;
  Move move;
  std::vector<Move> equivalent_moves;
  Score score = 0;
  std::vector<Move> principal_variation;

  friend bool operator==(const AnalysisLine&, const AnalysisLine&) = default;
};

struct AnalysisResult {
  std::vector<AnalysisLine> lines;
  int completed_depth = 0;
  std::uint64_t nodes = 0;

  friend bool operator==(const AnalysisResult&, const AnalysisResult&) = default;
};

using AnalysisProgressSink = std::function<void(const AnalysisResult&)>;

class Search final {
 public:
  explicit Search(const SearchOptions& options = {});

  void new_game() noexcept;
  [[nodiscard]] engine::EngineResult run(const Position& position,
                                         const engine::EngineLimits& limits,
                                         const engine::InfoSink& info);
  [[nodiscard]] AnalysisResult analyze(const Position& position, const engine::EngineLimits& limits,
                                       int multi_pv = 1, const AnalysisProgressSink& progress = {});
  [[nodiscard]] int terminal_depth(const Position& position) const noexcept;
  [[nodiscard]] const TranspositionTable& transposition_table() const noexcept;
  [[nodiscard]] const Net& net() const noexcept;

 private:
  void prepare_history(const Position& position) noexcept;
  [[nodiscard]] LeafEvaluator leaf_evaluator() const noexcept;

  bool use_symmetry_ = true;
  Evaluator evaluator_ = Evaluator::kPatternGain;
  std::shared_ptr<const Net> net_;
  TranspositionTable table_;
  std::array<std::array<std::int16_t, kCellCount>, 2> history_scores_{};
  std::optional<PositionKey> history_root_;
};

}  // namespace poe2::minimax

#endif  // POE2_MINIMAX_SEARCH_HPP
