#include <exception>
#include <iostream>

#include "poe2/engine_stdio.hpp"
#include "poe2/minimax/engine.hpp"

namespace {

int run() {
  poe2::minimax::MinimaxEngine engine;
  return poe2::engine_stdio::run_engine_stdio("minimax", engine);
}

}  // namespace

int main() {
  try {
    return run();
  } catch (const std::exception& error) {
    std::cerr << "fatal " << error.what() << '\n';
  } catch (...) {
    std::cerr << "fatal unknown\n";
  }

  return 1;
}
