#include <exception>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "poe2/engine_stdio.hpp"
#include "poe2/minimax/engine.hpp"
#include "poe2/minimax/options.hpp"

namespace {

int run(int argc, char* argv[]) {
  std::vector<std::string_view> arguments;
  arguments.reserve(static_cast<std::size_t>(argc > 1 ? argc - 1 : 0));
  for (int index = 1; index < argc; ++index) {
    arguments.emplace_back(argv[index]);
  }

  std::string error;
  const std::optional<poe2::minimax::SearchOptions> options =
      poe2::minimax::parse_search_options(arguments, error);
  if (!options.has_value()) {
    std::cerr << "error " << error << '\n';
    return 2;
  }

  poe2::minimax::MinimaxEngine engine{*options};
  return poe2::engine_stdio::run_engine_stdio("minimax", engine);
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "fatal " << error.what() << '\n';
  } catch (...) {
    std::cerr << "fatal unknown\n";
  }

  return 1;
}
