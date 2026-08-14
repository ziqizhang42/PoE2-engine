#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <climits>
#include <cstddef>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "poe2/wasm/adapter.hpp"

namespace poe2::wasm {

namespace {

using emscripten::val;

class BrowserAnalyzer final {
 public:
  [[nodiscard]] val analyze(val request_value, val progress_callback);

 private:
  Analyzer analyzer_;
};

void set_versions(val& object) {
  object.set("engineVersion", std::string{engine_version()});
  object.set("apiVersion", kApiVersion);
}

[[nodiscard]] val error_value(const AnalysisError& error) {
  val detail = val::object();
  detail.set("code", std::string{analysis_error_code_name(error.code)});
  detail.set("message", error.message);
  if (error.field.has_value()) {
    detail.set("field", std::string{analysis_error_field_name(*error.field)});
  }
  if (error.move_index.has_value()) {
    detail.set("moveIndex", static_cast<double>(*error.move_index));
  }
  if (error.move.has_value()) {
    detail.set("move", *error.move);
  }
  if (error.reason.has_value()) {
    detail.set("reason", std::string{move_error_name(*error.reason)});
  }

  val response = val::object();
  response.set("ok", false);
  response.set("error", detail);
  set_versions(response);
  return response;
}

[[nodiscard]] val error_value(AnalysisErrorCode code, std::string message,
                              AnalysisErrorField field) {
  return error_value(AnalysisError{
      .code = code,
      .message = std::move(message),
      .field = field,
  });
}

[[nodiscard]] val move_array_value(const std::vector<std::string>& moves) {
  val value = val::array();
  for (std::size_t index = 0; index < moves.size(); ++index) {
    value.set(index, moves[index]);
  }
  return value;
}

[[nodiscard]] val analysis_line_value(const AnalysisLine& line) {
  val value = val::object();
  value.set("rank", line.rank);
  value.set("move", line.move);
  value.set("equivalentMoves", move_array_value(line.equivalent_moves));
  value.set("evaluationHalfPoints", line.evaluation_half_points);
  value.set("principalVariation", move_array_value(line.principal_variation));
  return value;
}

[[nodiscard]] val success_value(const AnalysisSuccess& success) {
  val lines = val::array();
  for (std::size_t index = 0; index < success.lines.size(); ++index) {
    lines.set(index, analysis_line_value(success.lines[index]));
  }

  val response = val::object();
  response.set("ok", true);
  response.set("bestMove", success.best_move);
  response.set("evaluationHalfPoints", success.evaluation_half_points);
  response.set("principalVariation", move_array_value(success.principal_variation));
  response.set("lines", lines);
  response.set("completedDepth", success.completed_depth);
  response.set("nodes", static_cast<double>(success.nodes));
  set_versions(response);
  return response;
}

[[nodiscard]] bool is_type(const val& value, const std::string& expected) {
  return value.typeOf().as<std::string>() == expected;
}

[[nodiscard]] bool is_safe_integer(const val& value) {
  return val::global("Number").call<bool>("isSafeInteger", value);
}

val BrowserAnalyzer::analyze(val request_value, val progress_callback) {
  if (request_value.isNull() || request_value.isUndefined() || !is_type(request_value, "object") ||
      val::global("Array").call<bool>("isArray", request_value)) {
    return error_value(AnalysisErrorCode::kInvalidRequest,
                       "request must be an analysis request object", AnalysisErrorField::kRequest);
  }

  const val moves_value = request_value["moves"];
  if (!val::global("Array").call<bool>("isArray", moves_value)) {
    return error_value(AnalysisErrorCode::kInvalidHistory, "moves must be an array",
                       AnalysisErrorField::kMoves);
  }

  const unsigned int move_count = moves_value["length"].as<unsigned int>();
  if (move_count > static_cast<unsigned int>(kCellCount)) {
    return error_value(AnalysisErrorCode::kInvalidHistory,
                       "moves cannot contain more than 49 entries", AnalysisErrorField::kMoves);
  }

  std::vector<std::string> moves;
  moves.reserve(move_count);
  for (unsigned int index = 0; index < move_count; ++index) {
    const val move_value = moves_value[index];
    if (!is_type(move_value, "string")) {
      AnalysisError error{
          .code = AnalysisErrorCode::kMalformedHistoryMove,
          .message = "each history move must be an a1-g7 string",
          .field = AnalysisErrorField::kMoves,
          .move_index = index,
      };
      return error_value(error);
    }
    moves.push_back(move_value.as<std::string>());
  }

  const val search_time_value = request_value["searchTimeMs"];
  if (!is_safe_integer(search_time_value)) {
    return error_value(AnalysisErrorCode::kInvalidSearchTime,
                       "searchTimeMs must be a positive integer",
                       AnalysisErrorField::kSearchTimeMs);
  }
  const double search_time = search_time_value.as<double>();
  if (search_time <= 0 || search_time > INT_MAX) {
    return error_value(AnalysisErrorCode::kInvalidSearchTime,
                       "searchTimeMs must be between 1 and 2147483647",
                       AnalysisErrorField::kSearchTimeMs);
  }

  std::optional<int> max_depth;
  if (request_value.hasOwnProperty("maxDepth")) {
    const val max_depth_value = request_value["maxDepth"];
    if (!max_depth_value.isUndefined()) {
      if (!is_safe_integer(max_depth_value)) {
        return error_value(AnalysisErrorCode::kInvalidMaxDepth,
                           "maxDepth must be an integer from 1 through 49",
                           AnalysisErrorField::kMaxDepth);
      }
      const double depth = max_depth_value.as<double>();
      if (depth <= 0 || depth > kCellCount) {
        return error_value(AnalysisErrorCode::kInvalidMaxDepth,
                           "maxDepth must be an integer from 1 through 49",
                           AnalysisErrorField::kMaxDepth);
      }
      max_depth = static_cast<int>(depth);
    }
  }

  int multi_pv = 1;
  if (request_value.hasOwnProperty("multiPv")) {
    const val multi_pv_value = request_value["multiPv"];
    if (!multi_pv_value.isUndefined()) {
      if (!is_safe_integer(multi_pv_value)) {
        return error_value(AnalysisErrorCode::kInvalidMultiPv,
                           "multiPv must be an integer from 1 through 5",
                           AnalysisErrorField::kMultiPv);
      }
      const double count = multi_pv_value.as<double>();
      if (count < 1 || count > 5) {
        return error_value(AnalysisErrorCode::kInvalidMultiPv,
                           "multiPv must be an integer from 1 through 5",
                           AnalysisErrorField::kMultiPv);
      }
      multi_pv = static_cast<int>(count);
    }
  }

  AnalysisProgressSink progress;
  if (is_type(progress_callback, "function")) {
    progress = [callback = std::move(progress_callback)](const AnalysisSuccess& update) {
      callback(success_value(update));
    };
  }

  const AnalysisResponse response = analyzer_.analyze(
      AnalysisRequest{
          .moves = std::move(moves),
          .search_time_ms = static_cast<int>(search_time),
          .max_depth = max_depth,
          .multi_pv = multi_pv,
      },
      progress);
  if (const auto* success = std::get_if<AnalysisSuccess>(&response); success != nullptr) {
    return success_value(*success);
  }
  return error_value(std::get<AnalysisError>(response));
}

}  // namespace

}  // namespace poe2::wasm

EMSCRIPTEN_BINDINGS(poe2_engine_wasm) {
  emscripten::class_<poe2::wasm::BrowserAnalyzer>("BrowserAnalyzer")
      .constructor<>()
      .function("analyze", &poe2::wasm::BrowserAnalyzer::analyze);
}
