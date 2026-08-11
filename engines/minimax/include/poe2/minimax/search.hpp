#ifndef POE2_MINIMAX_SEARCH_HPP
#define POE2_MINIMAX_SEARCH_HPP

#include <cstddef>

#include "poe2/engine.hpp"
#include "poe2/transposition_table.hpp"

namespace poe2::minimax {

inline constexpr std::size_t kMebibyte = std::size_t{1024} * 1024;
inline constexpr std::size_t kDefaultHashBytes = std::size_t{64} * kMebibyte;

struct SearchOptions {
  std::size_t hash_bytes = kDefaultHashBytes;
  bool use_symmetry = true;
  bool use_two_ply_closure = true;

  friend constexpr bool operator==(const SearchOptions&, const SearchOptions&) = default;
};

class Search final {
 public:
  explicit Search(SearchOptions options = {});

  void new_game() noexcept;
  [[nodiscard]] engine::EngineResult run(const Position& position,
                                         const engine::EngineLimits& limits,
                                         const engine::InfoSink& info);
  [[nodiscard]] const TranspositionTable& transposition_table() const noexcept;

 private:
  bool use_symmetry_ = true;
  bool use_two_ply_closure_ = true;
  TranspositionTable table_;
};

}  // namespace poe2::minimax

#endif  // POE2_MINIMAX_SEARCH_HPP
