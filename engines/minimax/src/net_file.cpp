#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "poe2/minimax/net.hpp"

namespace poe2::minimax {

namespace {

constexpr std::uintmax_t kMaximumNetFileBytes = std::uintmax_t{1} << 20;

}  // namespace

std::shared_ptr<const Net> load_net_file(const std::filesystem::path& path) {
  const std::string label = "net file " + path.string();
  std::error_code error;
  const std::uintmax_t size = std::filesystem::file_size(path, error);
  if (error) {
    throw NetError{"could not read " + label + ": " + error.message()};
  }
  if (size > kMaximumNetFileBytes) {
    throw NetError{label + " is larger than any supported net"};
  }
  std::ifstream input{path, std::ios::binary};
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  if (!input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size)) ||
      input.peek() != std::ifstream::traits_type::eof()) {
    throw NetError{"could not read " + label};
  }
  std::expected<std::shared_ptr<const Net>, std::string> net = try_parse_net(bytes);
  if (!net.has_value()) {
    throw NetError{label + ": " + net.error()};
  }
  return std::move(*net);
}

}  // namespace poe2::minimax
