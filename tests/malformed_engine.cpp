#include <iostream>
#include <string>

int main() {
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line == "isready") {
      std::cout << "readyok\n" << std::flush;
    } else if (line.starts_with("go")) {
      std::cout << "bestmove z9\n" << std::flush;
    } else if (line == "quit") {
      break;
    }
  }
  return 0;
}
