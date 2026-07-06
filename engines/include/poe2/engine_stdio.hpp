#ifndef POE2_ENGINE_STDIO_HPP
#define POE2_ENGINE_STDIO_HPP

#include <iosfwd>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "poe2/engine.hpp"
#include "poe2/move.hpp"

namespace poe2::engine_stdio {

inline constexpr std::string_view kCommandPoe2 = "poe2";
inline constexpr std::string_view kCommandIsReady = "isready";
inline constexpr std::string_view kCommandNewGame = "newgame";
inline constexpr std::string_view kCommandPosition = "position";
inline constexpr std::string_view kCommandGo = "go";
inline constexpr std::string_view kCommandQuit = "quit";
inline constexpr std::string_view kCommandExit = "exit";

inline constexpr std::string_view kResponsePoe2Ok = "poe2ok";
inline constexpr std::string_view kResponseReadyOk = "readyok";
inline constexpr std::string_view kResponseBestMove = "bestmove";
inline constexpr std::string_view kNoMove = "none";

using EngineLimits = poe2::engine::EngineLimits;
using EngineResult = poe2::engine::EngineResult;
using Engine = poe2::engine::Engine;
using InfoSink = poe2::engine::InfoSink;

[[nodiscard]] bool is_command(std::string_view line, std::string_view command) noexcept;
[[nodiscard]] bool is_position_command(std::string_view line) noexcept;

[[nodiscard]] std::string format_position_command(std::span<const std::string> moves);
[[nodiscard]] std::string format_go_command(const EngineLimits& limits);
[[nodiscard]] std::string format_bestmove(Move move);
[[nodiscard]] std::string format_no_move();
[[nodiscard]] std::optional<std::string> format_info_result(const EngineResult& result);
[[nodiscard]] std::optional<std::string> parse_bestmove_text(std::string_view line);
[[nodiscard]] std::optional<EngineLimits> parse_go_limits(std::string_view command,
                                                          std::ostream& output);

[[nodiscard]] bool set_position_from_command(std::string_view command, Position& position,
                                             std::ostream& output);

int run_engine_stdio(std::string_view name, Engine& engine);
int run_engine_stdio(std::string_view name, Engine& engine, std::istream& input,
                     std::ostream& output);

}  // namespace poe2::engine_stdio

#endif  // POE2_ENGINE_STDIO_HPP
