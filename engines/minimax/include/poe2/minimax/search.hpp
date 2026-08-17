#ifndef POE2_MINIMAX_SEARCH_HPP
#define POE2_MINIMAX_SEARCH_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include "poe2/engine.hpp"
#include "poe2/transposition_table.hpp"

namespace poe2::minimax {

inline constexpr std::size_t kMebibyte = std::size_t{1024} * 1024;
inline constexpr std::size_t kDefaultHashBytes = std::size_t{64} * kMebibyte;

struct SearchOptions {
  std::size_t hash_bytes = kDefaultHashBytes;
  bool use_symmetry = true;
  bool use_two_ply_closure = true;

  friend constexpr bool operator==(const SearchOptions&, const SearchOptions&) = default;
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
  explicit Search(SearchOptions options = {});

  void new_game() noexcept;
  [[nodiscard]] engine::EngineResult run(const Position& position,
                                         const engine::EngineLimits& limits,
                                         const engine::InfoSink& info);
  [[nodiscard]] AnalysisResult analyze(const Position& position, const engine::EngineLimits& limits,
                                       int multi_pv = 1, const AnalysisProgressSink& progress = {});
  [[nodiscard]] const TranspositionTable& transposition_table() const noexcept;

 private:
  void prepare_history(const Position& position) noexcept;

  bool use_symmetry_ = true;
  bool use_two_ply_closure_ = true;
  TranspositionTable table_;
  std::array<std::array<std::int16_t, kCellCount>, 2> history_scores_{};
  std::optional<PositionKey> history_root_;
};

}  // namespace poe2::minimax

#endif  // POE2_MINIMAX_SEARCH_HPP
