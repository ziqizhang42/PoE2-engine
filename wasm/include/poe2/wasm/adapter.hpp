#ifndef POE2_WASM_ADAPTER_HPP
#define POE2_WASM_ADAPTER_HPP

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "poe2/minimax/search.hpp"
#include "poe2/move.hpp"

namespace poe2::wasm {

inline constexpr int kApiVersion = 1;
inline constexpr std::size_t kDefaultHashBytes = std::size_t{16} * minimax::kMebibyte;

enum class AnalysisErrorCode : std::uint8_t {
  kInvalidRequest = 0,
  kInvalidHistory,
  kMalformedHistoryMove,
  kIllegalHistoryMove,
  kInvalidSearchTime,
  kInvalidMaxDepth,
  kInvalidMultiPv,
  kTerminalPosition,
  kSearchIncomplete,
};

enum class AnalysisErrorField : std::uint8_t {
  kRequest = 0,
  kMoves,
  kSearchTimeMs,
  kMaxDepth,
  kMultiPv,
};

struct AnalysisRequest {
  std::vector<std::string> moves;
  int search_time_ms = 0;
  std::optional<int> max_depth;
  int multi_pv = 1;
};

struct AnalysisError {
  AnalysisErrorCode code = AnalysisErrorCode::kInvalidRequest;
  std::string message;
  std::optional<AnalysisErrorField> field;
  std::optional<std::size_t> move_index;
  std::optional<std::string> move;
  std::optional<MoveError> reason;
};

struct AnalysisLine {
  int rank = 0;
  std::string move;
  std::vector<std::string> equivalent_moves;
  Score evaluation_half_points = 0;
  std::vector<std::string> principal_variation;

  friend bool operator==(const AnalysisLine&, const AnalysisLine&) = default;
};

struct AnalysisSuccess {
  std::string best_move;
  Score evaluation_half_points = 0;
  std::vector<std::string> principal_variation;
  std::vector<AnalysisLine> lines;
  int completed_depth = 0;
  std::uint64_t nodes = 0;

  friend bool operator==(const AnalysisSuccess&, const AnalysisSuccess&) = default;
};

using AnalysisResponse = std::variant<AnalysisSuccess, AnalysisError>;
using AnalysisProgressSink = std::function<void(const AnalysisSuccess&)>;

class Analyzer final {
 public:
  explicit Analyzer(std::size_t hash_bytes = kDefaultHashBytes);

  [[nodiscard]] AnalysisResponse analyze(const AnalysisRequest& request,
                                         const AnalysisProgressSink& progress = {});
  [[nodiscard]] std::size_t hash_storage_bytes() const noexcept;

 private:
  minimax::Search search_;
};

[[nodiscard]] std::string_view analysis_error_code_name(AnalysisErrorCode code) noexcept;
[[nodiscard]] std::string_view analysis_error_field_name(AnalysisErrorField field) noexcept;
[[nodiscard]] std::string_view engine_version() noexcept;

}  // namespace poe2::wasm

#endif  // POE2_WASM_ADAPTER_HPP
