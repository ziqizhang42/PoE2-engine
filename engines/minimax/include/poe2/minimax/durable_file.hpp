#ifndef POE2_MINIMAX_DURABLE_FILE_HPP
#define POE2_MINIMAX_DURABLE_FILE_HPP

#include <filesystem>
#include <string_view>

namespace poe2::minimax {

// A data artifact's COMPLETE marker must not reach the disk before the files it vouches for;
// otherwise a power loss can leave a truncated artifact that looks finished.

// Flushes a file's contents, or a directory's entries, to the storage device.
void sync_path(const std::filesystem::path& path);

// Writes the final marker contents into INCOMPLETE and renames it to COMPLETE, after syncing
// the directory so every file already renamed into it is on disk first.
void commit_marker(const std::filesystem::path& incomplete, const std::filesystem::path& complete,
                   std::string_view contents);

}  // namespace poe2::minimax

#endif  // POE2_MINIMAX_DURABLE_FILE_HPP
