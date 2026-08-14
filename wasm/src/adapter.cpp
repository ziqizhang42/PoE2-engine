#include "poe2/wasm/adapter.hpp"

#include <chrono>
#include <utility>

#include "poe2/wasm/version.hpp"

namespace poe2::wasm {

namespace {

[[nodiscard]] AnalysisError make_error(AnalysisErrorCode code, std::string message,
                                       std::optional<AnalysisErrorField> field = std::nullopt) {
  return AnalysisError{
      .code = code,
      .message = std::move(message),
      .field = field,
  };
}

[[nodiscard]] AnalysisError malformed_move_error(std::size_t index, const std::string& move) {
  AnalysisError error =
      make_error(AnalysisErrorCode::kMalformedHistoryMove,
                 "history move is not valid a1-g7 notation", AnalysisErrorField::kMoves);
  error.move_index = index;
  error.move = move;
  return error;
}

[[nodiscard]] AnalysisError illegal_move_error(std::size_t index, const std::string& move,
                                               std::optional<MoveError> reason) {
  AnalysisError error = make_error(AnalysisErrorCode::kIllegalHistoryMove,
                                   "history move is illegal in the reconstructed position",
                                   AnalysisErrorField::kMoves);
  error.move_index = index;
  error.move = move;
  error.reason = reason;
  return error;
}

[[nodiscard]] std::vector<std::string> format_moves(const std::vector<Move>& moves) {
  std::vector<std::string> formatted;
  formatted.reserve(moves.size());
  for (const Move move : moves) {
    formatted.push_back(format_move(move));
  }
  return formatted;
}

[[nodiscard]] AnalysisSuccess make_success(const minimax::AnalysisResult& result,
                                           Player root_player) {
  AnalysisSuccess success{
      .completed_depth = result.completed_depth,
      .nodes = result.nodes,
  };
  success.lines.reserve(result.lines.size());
  for (const minimax::AnalysisLine& source : result.lines) {
    success.lines.push_back(AnalysisLine{
        .rank = source.rank,
        .move = format_move(source.move),
        .equivalent_moves = format_moves(source.equivalent_moves),
        .evaluation_half_points = root_player == Player::kOne ? source.score : -source.score,
        .principal_variation = format_moves(source.principal_variation),
    });
  }

  if (!success.lines.empty()) {
    const AnalysisLine& best = success.lines.front();
    success.best_move = best.move;
    success.evaluation_half_points = best.evaluation_half_points;
    success.principal_variation = best.principal_variation;
  }
  return success;
}

}  // namespace

Analyzer::Analyzer(std::size_t hash_bytes)
    : search_(minimax::SearchOptions{
          .hash_bytes = hash_bytes,
          .use_symmetry = true,
          .use_two_ply_closure = true,
      }) {}

AnalysisResponse Analyzer::analyze(const AnalysisRequest& request,
                                   const AnalysisProgressSink& progress) {
  if (request.moves.size() > static_cast<std::size_t>(kCellCount)) {
    return make_error(AnalysisErrorCode::kInvalidHistory,
                      "moves cannot contain more than 49 entries", AnalysisErrorField::kMoves);
  }
  if (request.search_time_ms <= 0) {
    return make_error(AnalysisErrorCode::kInvalidSearchTime,
                      "searchTimeMs must be a positive integer", AnalysisErrorField::kSearchTimeMs);
  }
  if (request.max_depth.has_value() &&
      (*request.max_depth <= 0 || *request.max_depth > kCellCount)) {
    return make_error(AnalysisErrorCode::kInvalidMaxDepth,
                      "maxDepth must be an integer from 1 through 49",
                      AnalysisErrorField::kMaxDepth);
  }
  if (request.multi_pv < 1 || request.multi_pv > 5) {
    return make_error(AnalysisErrorCode::kInvalidMultiPv,
                      "multiPv must be an integer from 1 through 5", AnalysisErrorField::kMultiPv);
  }

  Position position;
  for (std::size_t index = 0; index < request.moves.size(); ++index) {
    const std::string& move_text = request.moves[index];
    const std::optional<Move> move = parse_move(move_text);
    if (!move.has_value()) {
      return malformed_move_error(index, move_text);
    }

    const MoveResult move_result = apply_move(position, *move);
    if (!move_result.accepted) {
      return illegal_move_error(index, move_text, move_result.error);
    }
  }

  if (position.is_full()) {
    return make_error(AnalysisErrorCode::kTerminalPosition,
                      "the reconstructed position has no legal move", AnalysisErrorField::kMoves);
  }

  const Player root_player = position.side_to_move();
  const engine::EngineLimits limits{
      .depth = request.max_depth,
      .move_time = std::chrono::milliseconds{request.search_time_ms},
  };
  minimax::AnalysisProgressSink search_progress;
  if (progress) {
    search_progress = [&progress, root_player](const minimax::AnalysisResult& update) {
      progress(make_success(update, root_player));
    };
  }
  const minimax::AnalysisResult result =
      search_.analyze(position, limits, request.multi_pv, search_progress);
  AnalysisSuccess success = make_success(result, root_player);
  if (success.lines.empty() || success.completed_depth <= 0) {
    return make_error(AnalysisErrorCode::kSearchIncomplete,
                      "search time expired before depth 1 completed");
  }

  if (success.best_move.empty() || success.principal_variation.empty()) {
    return make_error(AnalysisErrorCode::kSearchIncomplete,
                      "the completed search did not return a legal principal variation");
  }
  return success;
}

std::size_t Analyzer::hash_storage_bytes() const noexcept {
  return search_.transposition_table().storage_bytes();
}

std::string_view analysis_error_code_name(AnalysisErrorCode code) noexcept {
  switch (code) {
    case AnalysisErrorCode::kInvalidRequest:
      return "invalid_request";
    case AnalysisErrorCode::kInvalidHistory:
      return "invalid_history";
    case AnalysisErrorCode::kMalformedHistoryMove:
      return "malformed_history_move";
    case AnalysisErrorCode::kIllegalHistoryMove:
      return "illegal_history_move";
    case AnalysisErrorCode::kInvalidSearchTime:
      return "invalid_search_time";
    case AnalysisErrorCode::kInvalidMaxDepth:
      return "invalid_max_depth";
    case AnalysisErrorCode::kInvalidMultiPv:
      return "invalid_multi_pv";
    case AnalysisErrorCode::kTerminalPosition:
      return "terminal_position";
    case AnalysisErrorCode::kSearchIncomplete:
      return "search_incomplete";
  }
  return "invalid_request";
}

std::string_view analysis_error_field_name(AnalysisErrorField field) noexcept {
  switch (field) {
    case AnalysisErrorField::kRequest:
      return "request";
    case AnalysisErrorField::kMoves:
      return "moves";
    case AnalysisErrorField::kSearchTimeMs:
      return "searchTimeMs";
    case AnalysisErrorField::kMaxDepth:
      return "maxDepth";
    case AnalysisErrorField::kMultiPv:
      return "multiPv";
  }
  return "request";
}

std::string_view engine_version() noexcept { return kEngineVersion; }

}  // namespace poe2::wasm
