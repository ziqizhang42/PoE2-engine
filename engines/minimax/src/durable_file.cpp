#include "poe2/minimax/durable_file.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>

namespace poe2::minimax {

namespace fs = std::filesystem;

void sync_path(const fs::path& path) {
  const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) {
    throw std::system_error{errno, std::generic_category(), "failed to open " + path.string()};
  }
  const int result = ::fsync(descriptor);
  const int error = errno;
  ::close(descriptor);
  if (result != 0) {
    throw std::system_error{error, std::generic_category(), "failed to sync " + path.string()};
  }
}

void commit_marker(const fs::path& incomplete, const fs::path& complete,
                   std::string_view contents) {
  const fs::path directory = complete.has_parent_path() ? complete.parent_path() : fs::path{"."};
  sync_path(directory);
  {
    std::ofstream output{incomplete, std::ios::binary | std::ios::trunc};
    if (!output) {
      throw std::runtime_error{"failed to open " + incomplete.string()};
    }
    output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    output.close();
    if (!output) {
      throw std::runtime_error{"failed to write " + incomplete.string()};
    }
  }
  sync_path(incomplete);
  fs::rename(incomplete, complete);
  sync_path(directory);
  sync_path(directory.has_parent_path() ? directory.parent_path() : fs::path{"."});
}

}  // namespace poe2::minimax
