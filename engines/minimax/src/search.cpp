#include "poe2/minimax/search.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <limits>
#include <optional>
#include <sstream>
#include <utility>

#include "poe2/minimax/evaluation.hpp"
#include "poe2/move.hpp"
#include "poe2/symmetry.hpp"

namespace poe2::minimax {

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kChildKeySetCapacity = 64;
constexpr std::size_t kPlyCount = static_cast<std::size_t>(kCellCount) + 1;
constexpr std::uint64_t kDeadlineCheckInterval = 256;
constexpr Score kSearchInfinity = std::numeric_limits<Score>::max();
using HistoryScores = std::array<std::array<std::uint64_t, kCellCount>, 2>;
// On a 7x7 board, one placement can join two length-three runs in each of four directions:
// 4 * (64 - 4 - 4) = 224 points.
constexpr Score kMaximumMoveScoreGain = 224;
static_assert(std::has_single_bit(kChildKeySetCapacity));
static_assert(kChildKeySetCapacity > static_cast<std::size_t>(kCellCount));
static_assert(std::has_single_bit(kDeadlineCheckInterval));
static_assert(kBoardSize == 7);
static_assert(kCellCount <= std::numeric_limits<std::uint8_t>::max());
static_assert(kMaximumMoveScoreGain <= std::numeric_limits<std::uint8_t>::max());

class SearchState final {
 public:
  SearchState(std::chrono::milliseconds move_time, TranspositionTable& table,
              HistoryScores& history_scores)
      : deadline_(Clock::now() + move_time), table_(table), history_scores_(history_scores) {}

  [[nodiscard]] bool enter_node() noexcept {
    // A steady-clock query at every node is measurable work in this very small-node search.
    // Sampling still bounds deadline overshoot to a small fraction of a millisecond at normal
    // search rates, while keeping the hot path to an increment and a predictable branch.
    if ((nodes_ & (kDeadlineCheckInterval - 1)) == 0 && Clock::now() >= deadline_) {
      return false;
    }

    ++nodes_;
    return true;
  }

  [[nodiscard]] std::optional<TranspositionEntry> probe(PositionKey key,
                                                        PositionHash hash) noexcept {
    ++tt_probes_;
    std::optional<TranspositionEntry> entry = table_.probe(key, hash);
    if (entry.has_value()) {
      ++tt_hits_;
    }
    return entry;
  }

  void record_tt_cutoff() noexcept { ++tt_cutoffs_; }
  void record_alpha_beta_cutoff(Player player, int move_index, int depth) noexcept {
    assert(move_index >= 0 && move_index < kCellCount);
    assert(depth > 0);
    ++alpha_beta_cutoffs_;
    ++history_updates_;

    const std::uint64_t unsigned_depth = static_cast<std::uint64_t>(depth);
    const std::uint64_t bonus = unsigned_depth * unsigned_depth;
    std::uint64_t& score = history_scores_[static_cast<std::size_t>(player_index(player))]
                                          [static_cast<std::size_t>(move_index)];
    score += std::min(bonus, std::numeric_limits<std::uint64_t>::max() - score);
  }
  void record_score_gain_evaluations(std::uint64_t count) noexcept {
    score_gain_evaluations_ += count;
  }
  void record_static_evaluation() noexcept { ++static_evaluations_; }
  void record_closure_evaluation(std::uint64_t gain_queries) noexcept {
    ++closure_evaluations_;
    closure_gain_queries_ += gain_queries;
  }

  void store(PositionKey key, PositionHash hash, TranspositionValue value) {
    if (table_.store(key, hash, value)) {
      ++tt_stores_;
    }
  }

  void record_symmetry_prunes(std::uint64_t count) noexcept { symmetry_prunes_ += count; }

  void clear_principal_variation(int ply) noexcept {
    principal_variation_lengths_[static_cast<std::size_t>(ply)] = 0;
  }

  void prepend_principal_variation(int ply, Move move) noexcept {
    const std::size_t index = static_cast<std::size_t>(ply);
    const std::size_t child_index = index + 1;
    assert(child_index < kPlyCount);
    const std::uint8_t child_length = principal_variation_lengths_[child_index];

    principal_variations_[index][0] = move;
    for (std::uint8_t offset = 0; offset < child_length; ++offset) {
      principal_variations_[index][static_cast<std::size_t>(offset) + 1] =
          principal_variations_[child_index][offset];
    }
    principal_variation_lengths_[index] = static_cast<std::uint8_t>(child_length + 1);
  }

  [[nodiscard]] const std::array<Move, kCellCount>& principal_variation(int ply) const noexcept {
    return principal_variations_[static_cast<std::size_t>(ply)];
  }

  [[nodiscard]] std::size_t principal_variation_length(int ply) const noexcept {
    return principal_variation_lengths_[static_cast<std::size_t>(ply)];
  }

  [[nodiscard]] std::uint64_t nodes() const noexcept { return nodes_; }
  [[nodiscard]] std::uint64_t tt_probes() const noexcept { return tt_probes_; }
  [[nodiscard]] std::uint64_t tt_hits() const noexcept { return tt_hits_; }
  [[nodiscard]] std::uint64_t tt_cutoffs() const noexcept { return tt_cutoffs_; }
  [[nodiscard]] std::uint64_t alpha_beta_cutoffs() const noexcept { return alpha_beta_cutoffs_; }
  [[nodiscard]] std::uint64_t tt_stores() const noexcept { return tt_stores_; }
  [[nodiscard]] std::uint64_t symmetry_prunes() const noexcept { return symmetry_prunes_; }
  [[nodiscard]] std::uint64_t score_gain_evaluations() const noexcept {
    return score_gain_evaluations_;
  }
  [[nodiscard]] std::uint64_t static_evaluations() const noexcept { return static_evaluations_; }
  [[nodiscard]] std::uint64_t closure_evaluations() const noexcept { return closure_evaluations_; }
  [[nodiscard]] std::uint64_t closure_gain_queries() const noexcept {
    return closure_gain_queries_;
  }
  [[nodiscard]] std::uint64_t history_score(Player player, int move_index) const noexcept {
    assert(move_index >= 0 && move_index < kCellCount);
    return history_scores_[static_cast<std::size_t>(player_index(player))]
                          [static_cast<std::size_t>(move_index)];
  }
  [[nodiscard]] std::uint64_t history_updates() const noexcept { return history_updates_; }
  [[nodiscard]] std::uint64_t maximum_history_score() const noexcept {
    std::uint64_t maximum = 0;
    for (const auto& player_scores : history_scores_) {
      for (const std::uint64_t score : player_scores) {
        maximum = std::max(maximum, score);
      }
    }
    return maximum;
  }

 private:
  Clock::time_point deadline_;
  TranspositionTable& table_;
  HistoryScores& history_scores_;
  std::array<std::array<Move, kCellCount>, kPlyCount> principal_variations_{};
  std::array<std::uint8_t, kPlyCount> principal_variation_lengths_{};
  std::uint64_t nodes_ = 0;
  std::uint64_t tt_probes_ = 0;
  std::uint64_t tt_hits_ = 0;
  std::uint64_t tt_cutoffs_ = 0;
  std::uint64_t alpha_beta_cutoffs_ = 0;
  std::uint64_t tt_stores_ = 0;
  std::uint64_t symmetry_prunes_ = 0;
  std::uint64_t score_gain_evaluations_ = 0;
  std::uint64_t static_evaluations_ = 0;
  std::uint64_t closure_evaluations_ = 0;
  std::uint64_t closure_gain_queries_ = 0;
  std::uint64_t history_updates_ = 0;
};

class ChildKeySet final {
 public:
  void clear() noexcept { occupied_ = 0; }

  [[nodiscard]] bool insert(PositionKey key, PositionHash hash) noexcept {
    std::size_t index = static_cast<std::size_t>(hash) & (kChildKeySetCapacity - 1);

    for (std::size_t probe = 0; probe < kChildKeySetCapacity; ++probe) {
      const std::uint64_t occupied_bit = std::uint64_t{1} << index;
      if ((occupied_ & occupied_bit) == 0) {
        occupied_ |= occupied_bit;
        keys_[index] = key;
        return true;
      }
      if (keys_[index] == key) {
        return false;
      }
      index = (index + 1) & (kChildKeySetCapacity - 1);
    }

    assert(false && "child-key set cannot fill with at most 49 successors");
    return false;
  }

 private:
  std::array<PositionKey, kChildKeySetCapacity> keys_{};
  std::uint64_t occupied_ = 0;
};

class IdentityPolicy final {
 public:
  static constexpr bool kUsesSymmetry = false;

  explicit IdentityPolicy(const Position&) noexcept {}

  [[nodiscard]] static CanonicalPositionView current_view(const Position& position) noexcept {
    return CanonicalPositionView{
        .key = position.key(),
        .hash = position.hash(),
    };
  }

  [[nodiscard]] static CanonicalPositionView preview_move(const CanonicalPositionView& current,
                                                          int move_index) noexcept {
    const Player player = position_key_side_to_move(current.key);
    Bitboard player_one = position_key_bits(current.key, Player::kOne);
    Bitboard player_two = position_key_bits(current.key, Player::kTwo);
    if (player == Player::kOne) {
      player_one |= Bitboard{1} << move_index;
    } else {
      player_two |= Bitboard{1} << move_index;
    }

    return CanonicalPositionView{
        .key = make_position_key(player_one, player_two, opponent(player)),
        .hash = update_position_hash(current.hash, player, move_index),
    };
  }

  static void start_node(int) noexcept {}
  [[nodiscard]] static Bitboard move_orbit(int, int move_index) noexcept {
    return Bitboard{1} << move_index;
  }
};

class D4Policy final {
 public:
  static constexpr bool kUsesSymmetry = true;

  explicit D4Policy(const Position& position) noexcept : tracker_(position.key()) {
    assert(tracker_.hash() == position.hash());
  }

  [[nodiscard]] CanonicalPositionView current_view(const Position&) const noexcept {
    return tracker_.canonical_view();
  }

  [[nodiscard]] CanonicalPositionView preview_move(const CanonicalPositionView&,
                                                   int move_index) const noexcept {
    return tracker_.preview_move(move_index);
  }

  void start_node(int ply) noexcept {
    const std::size_t index = static_cast<std::size_t>(ply);
    child_keys_[index].clear();
    stabilizer_masks_[index] = tracker_.stabilizer_mask();
  }

  [[nodiscard]] Bitboard move_orbit(int ply, int move_index) const noexcept {
    const std::uint8_t stabilizer_mask = stabilizer_masks_[static_cast<std::size_t>(ply)];
    if (stabilizer_mask == 1) {
      return Bitboard{1} << move_index;
    }
    Bitboard orbit = Bitboard{1} << move_index;

    for (std::size_t symmetry = 1; symmetry < kAllSymmetries.size(); ++symmetry) {
      if ((stabilizer_mask & (std::uint8_t{1} << symmetry)) != 0) {
        orbit |= transformed_move_bits[symmetry][move_index];
      }
    }
    return orbit;
  }

  [[nodiscard]] bool remember_child(int ply, const CanonicalPositionView& child) noexcept {
    return child_keys_[static_cast<std::size_t>(ply)].insert(child.key, child.hash);
  }

  void make_move(int move_index) noexcept {
    const bool made_move = tracker_.make_move(move_index);
    assert(made_move);
    (void)made_move;
  }

  void unmake_move(int move_index) noexcept { tracker_.unmake_move(move_index); }

 private:
  PositionSymmetryTracker tracker_;
  std::array<ChildKeySet, kPlyCount> child_keys_{};
  std::array<std::uint8_t, kPlyCount> stabilizer_masks_{};
};

struct NodeResult {
  Score value = 0;
  std::optional<Move> best_move;
};

struct FixedPrincipalVariation {
  std::array<Move, kCellCount> moves{};
  std::size_t size = 0;
};

struct RootMoveGroup {
  PositionKey canonical_key;
  Move representative;
  Bitboard moves = 0;
};

[[nodiscard]] std::vector<RootMoveGroup> root_move_groups(const Position& position) {
  std::vector<RootMoveGroup> groups;
  groups.reserve(static_cast<std::size_t>(position.board().empty_count()));

  PositionSymmetryTracker tracker{position.key()};
  Bitboard moves = position.legal_moves();
  while (moves != 0) {
    const int move_index = std::countr_zero(moves);
    moves &= moves - Bitboard{1};
    const CanonicalPositionView child = tracker.preview_move(move_index);
    const auto existing = std::find_if(
        groups.begin(), groups.end(),
        [&child](const RootMoveGroup& group) noexcept { return group.canonical_key == child.key; });
    if (existing != groups.end()) {
      existing->moves |= Bitboard{1} << move_index;
      continue;
    }

    groups.push_back(RootMoveGroup{
        .canonical_key = child.key,
        .representative = Move{.square = square_from_index(move_index)},
        .moves = Bitboard{1} << move_index,
    });
  }

  return groups;
}

[[nodiscard]] const RootMoveGroup* find_root_move_group(const std::vector<RootMoveGroup>& groups,
                                                        Move move) noexcept {
  const Bitboard move_bit = square_bit(move.square);
  const auto group = std::find_if(groups.begin(), groups.end(),
                                  [move_bit](const RootMoveGroup& candidate) noexcept {
                                    return (candidate.moves & move_bit) != 0;
                                  });
  return group == groups.end() ? nullptr : &*group;
}

class ScoreGainMovePicker final {
 public:
  template <typename Policy>
  ScoreGainMovePicker(const Position& position, Bitboard moves, int ply, const Policy& policy,
                      SearchState& state) {
    Bitboard unselected_moves = moves;
    while (unselected_moves != 0) {
      const int move_index = std::countr_zero(unselected_moves);
      moves_[size_++].move_index = static_cast<std::uint8_t>(move_index);
      unselected_moves &= ~policy.move_orbit(ply, move_index);
    }

    if (size_ <= 1) {
      return;
    }

    const Player player = position.side_to_move();
    for (std::size_t index = 0; index < size_; ++index) {
      const Score score_gain = position.score_gain_unchecked(player, moves_[index].move_index);
      assert(score_gain > 0 && score_gain <= kMaximumMoveScoreGain);
      moves_[index].score_gain = static_cast<std::uint8_t>(score_gain);
      moves_[index].history_score = state.history_score(player, moves_[index].move_index);
    }
    state.record_score_gain_evaluations(size_);

    // Moves arrive in increasing index order, which is already the final tie-break order. A
    // stable insertion sort therefore preserves row-major order when gain and history both tie.
    for (std::size_t index = 1; index < size_; ++index) {
      const ScoredMove candidate = moves_[index];
      std::size_t insertion = index;
      while (insertion > 0 && (moves_[insertion - 1].score_gain < candidate.score_gain ||
                               (moves_[insertion - 1].score_gain == candidate.score_gain &&
                                moves_[insertion - 1].history_score < candidate.history_score))) {
        moves_[insertion] = moves_[insertion - 1];
        --insertion;
      }
      moves_[insertion] = candidate;
    }
  }

  [[nodiscard]] std::optional<int> next(Bitboard remaining_moves) noexcept {
    while (next_ < size_) {
      const int move_index = moves_[next_++].move_index;
      if ((remaining_moves & (Bitboard{1} << move_index)) != 0) {
        return move_index;
      }
    }
    return std::nullopt;
  }

 private:
  struct ScoredMove {
    std::uint8_t move_index = 0;
    std::uint8_t score_gain = 0;
    std::uint64_t history_score = 0;
  };

  std::array<ScoredMove, kCellCount> moves_{};
  std::size_t size_ = 0;
  std::size_t next_ = 0;
};

struct EmptyPositionView {};

template <typename Policy, bool UseTable>
using SearchPositionView =
    std::conditional_t<UseTable || Policy::kUsesSymmetry, CanonicalPositionView, EmptyPositionView>;

[[nodiscard]] std::optional<Move> remap_cached_move(const TranspositionEntry& entry,
                                                    const CanonicalPositionView& view,
                                                    Bitboard legal_moves) noexcept {
  if (!entry.value.best_move.has_value()) {
    return std::nullopt;
  }

  const Square live_square = transform_square(view.inverse_transform, *entry.value.best_move);
  if ((legal_moves & square_bit(live_square)) == 0) {
    return std::nullopt;
  }
  return Move{.square = live_square};
}

void store_result(SearchState& state, const CanonicalPositionView& view, int depth, Score value,
                  TranspositionBound bound, std::optional<Move> best_move) {
  std::optional<Square> canonical_best_move;
  if (best_move.has_value()) {
    canonical_best_move = transform_square(view.transform, best_move->square);
  }

  state.store(view.key, view.hash,
              TranspositionValue{
                  .score = value,
                  .depth = depth,
                  .bound = bound,
                  .best_move = canonical_best_move,
              });
}

template <typename Policy, bool UseTable, bool UseTwoPlyClosure>
[[nodiscard]] std::optional<NodeResult> negamax(Position& position, int depth, int ply, Score alpha,
                                                Score beta, SearchState& state, Policy& policy,
                                                SearchPositionView<Policy, UseTable> view,
                                                int pending_policy_move) {
  if constexpr (!UseTable) {
    state.clear_principal_variation(ply);
  }
  if (!state.enter_node()) {
    return std::nullopt;
  }

  const Score original_alpha = alpha;
  const Score original_beta = beta;
  const Bitboard legal_moves = position.legal_moves();
  std::optional<Move> cached_move;
  if constexpr (UseTable) {
    const std::optional<TranspositionEntry> cached = state.probe(view.key, view.hash);
    if (cached.has_value()) {
      cached_move = remap_cached_move(*cached, view, legal_moves);
      const bool has_required_move = depth == 0 || legal_moves == 0 || cached_move.has_value();
      if (cached->value.depth >= depth) {
        if (cached->value.bound == TranspositionBound::kExact && has_required_move) {
          state.record_tt_cutoff();
          return NodeResult{.value = cached->value.score, .best_move = cached_move};
        }

        if (cached->value.bound == TranspositionBound::kLower && cached->value.score > alpha) {
          alpha = cached->value.score;
        } else if (cached->value.bound == TranspositionBound::kUpper &&
                   cached->value.score < beta) {
          beta = cached->value.score;
        }

        if (alpha >= beta) {
          state.record_tt_cutoff();
          return NodeResult{.value = cached->value.score, .best_move = cached_move};
        }
      }
    }
  }

  if (depth == 0 || legal_moves == 0) {
    state.record_static_evaluation();
    Score value = 0;
    if constexpr (UseTwoPlyClosure) {
      const int empty_count = std::popcount(legal_moves);
      const std::uint64_t gain_queries =
          empty_count == 1 ? 1 : static_cast<std::uint64_t>(2 * empty_count);
      state.record_closure_evaluation(gain_queries);
      value = evaluate_two_ply_closure(position, legal_moves, empty_count);
    } else {
      value = evaluate(position);
    }
    if constexpr (UseTable) {
      store_result(state, view, depth, value, TranspositionBound::kExact, std::nullopt);
    }
    return NodeResult{.value = value};
  }

  // The canonical child view is enough for deadline, TT, and leaf handling, so the symmetry
  // tracker intentionally remains at the parent until this node is known to expand.
  if constexpr (Policy::kUsesSymmetry) {
    if (pending_policy_move >= 0) {
      policy.make_move(pending_policy_move);
    }
  }
  const auto finish_node = [&policy,
                            pending_policy_move](std::optional<NodeResult> result) noexcept {
    if constexpr (Policy::kUsesSymmetry) {
      if (pending_policy_move >= 0) {
        policy.unmake_move(pending_policy_move);
      }
    }
    return result;
  };

  policy.start_node(ply);
  Bitboard remaining_moves = legal_moves;
  const Player player = position.side_to_move();
  NodeResult best;
  bool found_move = false;

  const auto search_move = [&](int move_index) -> bool {
    const Bitboard move_bit = Bitboard{1} << move_index;
    const Bitboard equivalent_moves = remaining_moves & policy.move_orbit(ply, move_index);
    assert((equivalent_moves & move_bit) != 0);
    remaining_moves &= ~equivalent_moves;
    SearchPositionView<Policy, UseTable> child_view;
    if constexpr (UseTable || Policy::kUsesSymmetry) {
      child_view = policy.preview_move(view, move_index);
    }

    if constexpr (Policy::kUsesSymmetry) {
      if (!policy.remember_child(ply, child_view)) {
        state.record_symmetry_prunes(std::popcount(equivalent_moves));
        return true;
      }
      state.record_symmetry_prunes(std::popcount(equivalent_moves) - 1);
    }

    MoveUndo undo;
    position.make_move_unchecked(move_index, undo);

    std::optional<NodeResult> child = negamax<Policy, UseTable, UseTwoPlyClosure>(
        position, depth - 1, ply + 1, -beta, -alpha, state, policy, child_view, move_index);

    position.unmake_move(undo);
    if (!child.has_value()) {
      return false;
    }

    const Score value = -child->value;
    if (!found_move || value > best.value) {
      const Move move{.square = square_from_index(move_index)};
      found_move = true;
      best.value = value;
      best.best_move = move;
      if constexpr (!UseTable) {
        state.prepend_principal_variation(ply, move);
      }
    }
    if (value > alpha) {
      alpha = value;
    }
    if (alpha >= beta) {
      state.record_alpha_beta_cutoff(player, move_index, depth);
    }
    return true;
  };

  if (cached_move.has_value() && !search_move(square_index(cached_move->square))) {
    return finish_node(std::nullopt);
  }

  ScoreGainMovePicker move_picker{position, remaining_moves, ply, policy, state};
  while (remaining_moves != 0 && alpha < beta) {
    const std::optional<int> move_index = move_picker.next(remaining_moves);
    assert(move_index.has_value());
    if (!move_index.has_value() || !search_move(*move_index)) {
      return finish_node(std::nullopt);
    }
  }

  assert(found_move);
  if constexpr (UseTable) {
    TranspositionBound bound = TranspositionBound::kExact;
    if (best.value <= original_alpha) {
      bound = TranspositionBound::kUpper;
    } else if (best.value >= original_beta) {
      bound = TranspositionBound::kLower;
    }
    store_result(state, view, depth, best.value, bound, best.best_move);
  }
  return finish_node(best);
}

template <typename Policy, bool UseTable, bool UseTwoPlyClosure>
[[nodiscard]] std::optional<NodeResult> search_root_groups(
    Position& position, int depth, Bitboard group_representatives, SearchState& state,
    Policy& policy, SearchPositionView<Policy, UseTable> root_view) {
  if constexpr (!UseTable) {
    state.clear_principal_variation(0);
  }
  if (!state.enter_node()) {
    return std::nullopt;
  }

  policy.start_node(0);
  Bitboard remaining_moves = group_representatives;
  Score alpha = -kSearchInfinity;
  NodeResult best;
  bool found_move = false;

  const auto search_move = [&](int move_index) -> bool {
    const Bitboard move_bit = Bitboard{1} << move_index;
    const Bitboard equivalent_moves = remaining_moves & policy.move_orbit(0, move_index);
    assert((equivalent_moves & move_bit) != 0);
    remaining_moves &= ~equivalent_moves;

    SearchPositionView<Policy, UseTable> child_view;
    if constexpr (UseTable || Policy::kUsesSymmetry) {
      child_view = policy.preview_move(root_view, move_index);
    }

    MoveUndo undo;
    position.make_move_unchecked(move_index, undo);
    std::optional<NodeResult> child = negamax<Policy, UseTable, UseTwoPlyClosure>(
        position, depth - 1, 1, -kSearchInfinity, -alpha, state, policy, child_view, move_index);
    position.unmake_move(undo);
    if (!child.has_value()) {
      return false;
    }

    const Score value = -child->value;
    if (!found_move || value > best.value) {
      const Move move{.square = square_from_index(move_index)};
      found_move = true;
      best.value = value;
      best.best_move = move;
      if constexpr (!UseTable) {
        state.prepend_principal_variation(0, move);
      }
    }
    if (value > alpha) {
      alpha = value;
    }
    return true;
  };

  ScoreGainMovePicker move_picker{position, remaining_moves, 0, policy, state};
  while (remaining_moves != 0) {
    const std::optional<int> move_index = move_picker.next(remaining_moves);
    assert(move_index.has_value());
    if (!move_index.has_value() || !search_move(*move_index)) {
      return std::nullopt;
    }
  }

  assert(found_move);
  return best;
}

[[nodiscard]] FixedPrincipalVariation reconstruct_identity_principal_variation(
    const Position& root, int depth, Move best_move, const TranspositionTable& table) {
  FixedPrincipalVariation principal_variation;
  if (depth <= 0) {
    return principal_variation;
  }

  Position position = root;
  if (!position.play(best_move.square)) {
    return principal_variation;
  }
  principal_variation.moves[principal_variation.size++] = best_move;

  for (int remaining_depth = depth - 1; remaining_depth > 0; --remaining_depth) {
    const CanonicalPositionView view = IdentityPolicy::current_view(position);
    const std::optional<TranspositionEntry> cached = table.probe(view.key, view.hash);
    if (!cached.has_value() || cached->value.bound != TranspositionBound::kExact ||
        cached->value.depth < remaining_depth) {
      break;
    }

    const std::optional<Move> move = remap_cached_move(*cached, view, position.legal_moves());
    if (!move.has_value() || !position.play(move->square)) {
      break;
    }
    principal_variation.moves[principal_variation.size++] = *move;
  }

  return principal_variation;
}

[[nodiscard]] FixedPrincipalVariation reconstruct_d4_principal_variation(
    const Position& root, int depth, Move best_move, const TranspositionTable& table) {
  FixedPrincipalVariation principal_variation;
  if (depth <= 0) {
    return principal_variation;
  }

  Position position = root;
  PositionSymmetryTracker tracker{root.key()};
  if (!position.play(best_move.square) || !tracker.make_move(best_move.square)) {
    return principal_variation;
  }
  principal_variation.moves[principal_variation.size++] = best_move;

  for (int remaining_depth = depth - 1; remaining_depth > 0; --remaining_depth) {
    const CanonicalPositionView view = tracker.canonical_view();
    const std::optional<TranspositionEntry> cached = table.probe(view.key, view.hash);
    if (!cached.has_value() || cached->value.bound != TranspositionBound::kExact ||
        cached->value.depth < remaining_depth) {
      break;
    }

    const std::optional<Move> move = remap_cached_move(*cached, view, position.legal_moves());
    if (!move.has_value() || !position.play(move->square) || !tracker.make_move(move->square)) {
      break;
    }
    principal_variation.moves[principal_variation.size++] = *move;
  }

  return principal_variation;
}

template <typename Policy>
[[nodiscard]] FixedPrincipalVariation reconstruct_principal_variation(
    const Position& root, int depth, Move best_move, const TranspositionTable& table) {
  if constexpr (Policy::kUsesSymmetry) {
    return reconstruct_d4_principal_variation(root, depth, best_move, table);
  } else {
    return reconstruct_identity_principal_variation(root, depth, best_move, table);
  }
}

[[nodiscard]] std::vector<Move> moves_in_group(const RootMoveGroup& group) {
  std::vector<Move> moves;
  moves.reserve(static_cast<std::size_t>(std::popcount(group.moves)));
  Bitboard remaining = group.moves;
  while (remaining != 0) {
    const int move_index = std::countr_zero(remaining);
    remaining &= remaining - Bitboard{1};
    moves.push_back(Move{.square = square_from_index(move_index)});
  }
  return moves;
}

[[nodiscard]] AnalysisResult single_analysis_result(const engine::EngineResult& engine_result,
                                                    const std::vector<RootMoveGroup>& groups) {
  AnalysisResult result{
      .completed_depth = engine_result.depth,
      .nodes = engine_result.nodes,
  };
  if (engine_result.depth <= 0 || !engine_result.best_move.has_value() ||
      !engine_result.score.has_value()) {
    return result;
  }

  const RootMoveGroup* group = find_root_move_group(groups, *engine_result.best_move);
  if (group == nullptr) {
    return result;
  }
  result.lines.push_back(AnalysisLine{
      .rank = 1,
      .move = *engine_result.best_move,
      .equivalent_moves = moves_in_group(*group),
      .score = *engine_result.score,
      .principal_variation = engine_result.principal_variation,
  });
  return result;
}

template <typename Policy, bool UseTable>
[[nodiscard]] AnalysisLine make_analysis_line(const Position& root, Move best_move, Score score,
                                              const RootMoveGroup& group, const SearchState& state,
                                              int depth, int rank,
                                              const TranspositionTable& table) {
  FixedPrincipalVariation principal_variation;
  if constexpr (!UseTable) {
    principal_variation.moves = state.principal_variation(0);
    principal_variation.size = state.principal_variation_length(0);
  } else {
    principal_variation = reconstruct_principal_variation<Policy>(root, depth, best_move, table);
  }

  if (principal_variation.size == 0) {
    principal_variation.moves[0] = best_move;
    principal_variation.size = 1;
  }

  AnalysisLine line{
      .rank = rank,
      .move = best_move,
      .equivalent_moves = moves_in_group(group),
      .score = score,
  };
  line.principal_variation.assign(
      principal_variation.moves.begin(),
      principal_variation.moves.begin() + static_cast<std::ptrdiff_t>(principal_variation.size));
  return line;
}

template <typename Policy, bool UseTable>
void commit_iteration(engine::EngineResult& result, const NodeResult& iteration,
                      const SearchState& state, const Position& root, int depth,
                      const TranspositionTable& table) {
  assert(iteration.best_move.has_value());
  if (!iteration.best_move.has_value()) {
    return;
  }

  result.best_move = iteration.best_move;
  result.score = iteration.value;
  result.depth = depth;

  FixedPrincipalVariation principal_variation;
  if constexpr (!UseTable) {
    principal_variation.moves = state.principal_variation(0);
    principal_variation.size = state.principal_variation_length(0);
  } else {
    principal_variation =
        reconstruct_principal_variation<Policy>(root, depth, *iteration.best_move, table);
  }

  if (principal_variation.size == 0) {
    principal_variation.moves[0] = *iteration.best_move;
    principal_variation.size = 1;
  }
  result.principal_variation.assign(
      principal_variation.moves.begin(),
      principal_variation.moves.begin() + static_cast<std::ptrdiff_t>(principal_variation.size));
}

void emit_diagnostics(const engine::InfoSink& info, const SearchState& state,
                      const TranspositionTable& table) {
  if (!info) {
    return;
  }

  std::ostringstream diagnostics;
  diagnostics << "ttprobes " << state.tt_probes() << " tthits " << state.tt_hits() << " ttcutoffs "
              << state.tt_cutoffs() << " abcutoffs " << state.alpha_beta_cutoffs() << " ttstores "
              << state.tt_stores() << " symmetryprunes " << state.symmetry_prunes()
              << " moveorderevals " << state.score_gain_evaluations() << " staticevals "
              << state.static_evaluations() << " closureevals " << state.closure_evaluations()
              << " closuregainqueries " << state.closure_gain_queries() << " gainqueries "
              << state.score_gain_evaluations() + state.closure_gain_queries() << " historyupdates "
              << state.history_updates() << " historymax " << state.maximum_history_score()
              << " hashentries " << table.size() << " hashcapacity " << table.capacity()
              << " hashbytes " << table.storage_bytes();
  info(diagnostics.str());
}

template <typename Policy, bool UseTable, bool UseTwoPlyClosure>
[[nodiscard]] engine::EngineResult run_timed_search(const Position& position,
                                                    std::chrono::milliseconds move_time,
                                                    int maximum_depth, const engine::InfoSink& info,
                                                    TranspositionTable& table,
                                                    HistoryScores& history_scores,
                                                    engine::EngineResult result) {
  SearchState state{move_time, table, history_scores};
  Position search_position = position;
  Policy policy{search_position};
  SearchPositionView<Policy, UseTable> root_view;
  if constexpr (UseTable || Policy::kUsesSymmetry) {
    root_view = policy.current_view(search_position);
  }

  for (int depth = 1; depth <= maximum_depth; ++depth) {
    std::optional<NodeResult> iteration = negamax<Policy, UseTable, UseTwoPlyClosure>(
        search_position, depth, 0, -kSearchInfinity, kSearchInfinity, state, policy, root_view, -1);
    if (!iteration.has_value()) {
      break;
    }

    commit_iteration<Policy, UseTable>(result, *iteration, state, position, depth, table);
  }

  result.nodes = state.nodes();
  emit_diagnostics(info, state, table);
  return result;
}

template <bool UseTwoPlyClosure>
[[nodiscard]] engine::EngineResult run_configured_search(
    const Position& position, std::chrono::milliseconds move_time, int maximum_depth,
    const engine::InfoSink& info, bool use_symmetry, TranspositionTable& table,
    HistoryScores& history_scores, engine::EngineResult result) {
  if (use_symmetry) {
    if (table.capacity() == 0) {
      return run_timed_search<D4Policy, false, UseTwoPlyClosure>(
          position, move_time, maximum_depth, info, table, history_scores, std::move(result));
    }
    return run_timed_search<D4Policy, true, UseTwoPlyClosure>(
        position, move_time, maximum_depth, info, table, history_scores, std::move(result));
  }
  if (table.capacity() == 0) {
    return run_timed_search<IdentityPolicy, false, UseTwoPlyClosure>(
        position, move_time, maximum_depth, info, table, history_scores, std::move(result));
  }
  return run_timed_search<IdentityPolicy, true, UseTwoPlyClosure>(
      position, move_time, maximum_depth, info, table, history_scores, std::move(result));
}

template <typename Policy, bool UseTable, bool UseTwoPlyClosure>
[[nodiscard]] AnalysisResult run_timed_single_analysis(
    const Position& position, std::chrono::milliseconds move_time, int maximum_depth,
    const AnalysisProgressSink& progress, TranspositionTable& table, HistoryScores& history_scores,
    const std::vector<RootMoveGroup>& groups, engine::EngineResult result) {
  SearchState state{move_time, table, history_scores};
  Position search_position = position;
  Policy policy{search_position};
  SearchPositionView<Policy, UseTable> root_view;
  if constexpr (UseTable || Policy::kUsesSymmetry) {
    root_view = policy.current_view(search_position);
  }

  for (int depth = 1; depth <= maximum_depth; ++depth) {
    std::optional<NodeResult> iteration = negamax<Policy, UseTable, UseTwoPlyClosure>(
        search_position, depth, 0, -kSearchInfinity, kSearchInfinity, state, policy, root_view, -1);
    if (!iteration.has_value()) {
      break;
    }

    commit_iteration<Policy, UseTable>(result, *iteration, state, position, depth, table);
    result.nodes = state.nodes();
    progress(single_analysis_result(result, groups));
  }

  result.nodes = state.nodes();
  return single_analysis_result(result, groups);
}

template <bool UseTwoPlyClosure>
[[nodiscard]] AnalysisResult run_configured_single_analysis(
    const Position& position, std::chrono::milliseconds move_time, int maximum_depth,
    const AnalysisProgressSink& progress, bool use_symmetry, TranspositionTable& table,
    HistoryScores& history_scores, const std::vector<RootMoveGroup>& groups,
    engine::EngineResult result) {
  if (use_symmetry) {
    if (table.capacity() == 0) {
      return run_timed_single_analysis<D4Policy, false, UseTwoPlyClosure>(
          position, move_time, maximum_depth, progress, table, history_scores, groups,
          std::move(result));
    }
    return run_timed_single_analysis<D4Policy, true, UseTwoPlyClosure>(
        position, move_time, maximum_depth, progress, table, history_scores, groups,
        std::move(result));
  }
  if (table.capacity() == 0) {
    return run_timed_single_analysis<IdentityPolicy, false, UseTwoPlyClosure>(
        position, move_time, maximum_depth, progress, table, history_scores, groups,
        std::move(result));
  }
  return run_timed_single_analysis<IdentityPolicy, true, UseTwoPlyClosure>(
      position, move_time, maximum_depth, progress, table, history_scores, groups,
      std::move(result));
}

template <typename Policy, bool UseTable, bool UseTwoPlyClosure>
[[nodiscard]] AnalysisResult run_timed_multi_analysis(
    const Position& position, std::chrono::milliseconds move_time, int maximum_depth,
    std::size_t line_count, const AnalysisProgressSink& progress, TranspositionTable& table,
    HistoryScores& history_scores, const std::vector<RootMoveGroup>& groups) {
  SearchState state{move_time, table, history_scores};
  Position search_position = position;
  Policy policy{search_position};
  SearchPositionView<Policy, UseTable> root_view;
  if constexpr (UseTable || Policy::kUsesSymmetry) {
    root_view = policy.current_view(search_position);
  }

  Bitboard all_representatives = 0;
  for (const RootMoveGroup& group : groups) {
    all_representatives |= square_bit(group.representative.square);
  }

  AnalysisResult result;
  for (int depth = 1; depth <= maximum_depth; ++depth) {
    std::vector<AnalysisLine> iteration_lines;
    iteration_lines.reserve(line_count);
    Bitboard remaining_representatives = all_representatives;
    bool iteration_complete = true;

    for (std::size_t rank = 0; rank < line_count; ++rank) {
      const std::optional<NodeResult> line_result =
          search_root_groups<Policy, UseTable, UseTwoPlyClosure>(
              search_position, depth, remaining_representatives, state, policy, root_view);
      if (!line_result.has_value() || !line_result->best_move.has_value()) {
        iteration_complete = false;
        break;
      }

      const Move best_move = *line_result->best_move;
      const RootMoveGroup* group = find_root_move_group(groups, best_move);
      if (group == nullptr) {
        iteration_complete = false;
        break;
      }
      iteration_lines.push_back(
          make_analysis_line<Policy, UseTable>(position, best_move, line_result->value, *group,
                                               state, depth, static_cast<int>(rank) + 1, table));
      remaining_representatives &= ~square_bit(group->representative.square);
    }

    if (!iteration_complete) {
      break;
    }

    result.lines = std::move(iteration_lines);
    result.completed_depth = depth;
    result.nodes = state.nodes();
    if (progress) {
      progress(result);
    }
  }

  result.nodes = state.nodes();
  return result;
}

template <bool UseTwoPlyClosure>
[[nodiscard]] AnalysisResult run_configured_multi_analysis(
    const Position& position, std::chrono::milliseconds move_time, int maximum_depth,
    std::size_t line_count, const AnalysisProgressSink& progress, bool use_symmetry,
    TranspositionTable& table, HistoryScores& history_scores,
    const std::vector<RootMoveGroup>& groups) {
  if (use_symmetry) {
    if (table.capacity() == 0) {
      return run_timed_multi_analysis<D4Policy, false, UseTwoPlyClosure>(
          position, move_time, maximum_depth, line_count, progress, table, history_scores, groups);
    }
    return run_timed_multi_analysis<D4Policy, true, UseTwoPlyClosure>(
        position, move_time, maximum_depth, line_count, progress, table, history_scores, groups);
  }
  if (table.capacity() == 0) {
    return run_timed_multi_analysis<IdentityPolicy, false, UseTwoPlyClosure>(
        position, move_time, maximum_depth, line_count, progress, table, history_scores, groups);
  }
  return run_timed_multi_analysis<IdentityPolicy, true, UseTwoPlyClosure>(
      position, move_time, maximum_depth, line_count, progress, table, history_scores, groups);
}

}  // namespace

Search::Search(SearchOptions options)
    : use_symmetry_(options.use_symmetry), use_two_ply_closure_(options.use_two_ply_closure) {
  table_.resize_bytes(options.hash_bytes);
}

void Search::prepare_history(const Position& position) noexcept {
  const PositionKey root = position.key();
  if (history_root_.has_value() && *history_root_ != root) {
    for (auto& player_scores : history_scores_) {
      for (std::uint64_t& score : player_scores) {
        score /= 2;
      }
    }
  }
  history_root_ = root;
}

void Search::new_game() noexcept {
  table_.clear();
  for (auto& player_scores : history_scores_) {
    player_scores.fill(0);
  }
  history_root_.reset();
}

const TranspositionTable& Search::transposition_table() const noexcept { return table_; }

engine::EngineResult Search::run(const Position& position, const engine::EngineLimits& limits,
                                 const engine::InfoSink& info) {
  const Bitboard legal_moves = position.legal_moves();
  if (legal_moves == 0) {
    if (limits.move_time.has_value() && limits.move_time->count() > 0) {
      SearchState state{*limits.move_time, table_, history_scores_};
      emit_diagnostics(info, state, table_);
    }
    return {};
  }

  engine::EngineResult result{
      .best_move = Move{.square = square_from_index(std::countr_zero(legal_moves))},
  };

  if (!limits.move_time.has_value() || limits.move_time->count() <= 0) {
    if (info) {
      info("error minimax_requires_movetime");
    }
    return result;
  }

  const int empty_count = position.board().empty_count();
  // A closure leaf resolves up to two remaining placements. Keep one explicit root ply so the
  // search still returns a move when two or fewer squares remain.
  const int terminal_depth = use_two_ply_closure_ ? std::max(1, empty_count - 2) : empty_count;
  int maximum_depth = terminal_depth;
  if (limits.depth.has_value()) {
    maximum_depth = std::clamp(*limits.depth, 0, terminal_depth);
  }
  if (maximum_depth > 0) {
    prepare_history(position);
  }

  if (use_two_ply_closure_) {
    return run_configured_search<true>(position, *limits.move_time, maximum_depth, info,
                                       use_symmetry_, table_, history_scores_, std::move(result));
  }
  return run_configured_search<false>(position, *limits.move_time, maximum_depth, info,
                                      use_symmetry_, table_, history_scores_, std::move(result));
}

AnalysisResult Search::analyze(const Position& position, const engine::EngineLimits& limits,
                               int multi_pv, const AnalysisProgressSink& progress) {
  if (multi_pv <= 0) {
    return {};
  }

  const std::vector<RootMoveGroup> groups = root_move_groups(position);
  if (multi_pv == 1 && !progress) {
    return single_analysis_result(run(position, limits, {}), groups);
  }

  const Bitboard legal_moves = position.legal_moves();
  if (legal_moves == 0 || !limits.move_time.has_value() || limits.move_time->count() <= 0) {
    return {};
  }

  const int empty_count = position.board().empty_count();
  const int terminal_depth = use_two_ply_closure_ ? std::max(1, empty_count - 2) : empty_count;
  int maximum_depth = terminal_depth;
  if (limits.depth.has_value()) {
    maximum_depth = std::clamp(*limits.depth, 0, terminal_depth);
  }
  if (maximum_depth > 0) {
    prepare_history(position);
  }

  if (multi_pv == 1) {
    engine::EngineResult result{
        .best_move = Move{.square = square_from_index(std::countr_zero(legal_moves))},
    };
    if (use_two_ply_closure_) {
      return run_configured_single_analysis<true>(position, *limits.move_time, maximum_depth,
                                                  progress, use_symmetry_, table_, history_scores_,
                                                  groups, std::move(result));
    }
    return run_configured_single_analysis<false>(position, *limits.move_time, maximum_depth,
                                                 progress, use_symmetry_, table_, history_scores_,
                                                 groups, std::move(result));
  }

  const std::size_t line_count = std::min(static_cast<std::size_t>(multi_pv), groups.size());
  if (use_two_ply_closure_) {
    return run_configured_multi_analysis<true>(position, *limits.move_time, maximum_depth,
                                               line_count, progress, use_symmetry_, table_,
                                               history_scores_, groups);
  }
  return run_configured_multi_analysis<false>(position, *limits.move_time, maximum_depth,
                                              line_count, progress, use_symmetry_, table_,
                                              history_scores_, groups);
}

}  // namespace poe2::minimax
