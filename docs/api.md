# Game API

The game library exposes one authoritative rules layer and optional search helpers.

## Core Concepts

- `poe2::Position` is the authoritative game state. It owns the board, side to move, cached scores, exact position key, and incremental hash.
- `poe2::Board` is the low-level 7x7 occupancy representation. Prefer `Position` for game play because it also tracks turn order and scores.
- `poe2::Move` is the public move type. Today it wraps a `Square`, but callers should use `Move` at API boundaries so the protocol can grow without changing every caller.
- `poe2::GameResult` contains final scores and winner after a full board.

## Match Runner Usage

Use `poe2/move.hpp` for authoritative move submission.

```cpp
#include "poe2/move.hpp"

poe2::Position position;
const std::optional<poe2::Move> parsed = poe2::parse_move("c4");
if (!parsed.has_value()) {
  // Reject malformed engine output.
}

const poe2::MoveResult result = poe2::apply_move(position, *parsed);
if (!result.accepted) {
  // Reject the move. result.error contains the reason.
}
if (result.game_result.has_value()) {
  // The game ended on this move.
}
```

`apply_move()` validates the move, calls `Position::play()` for accepted moves, and returns a terminal result when the board becomes full.

## Move Coordinates

Text moves use two-character coordinates from `a1` through `g7`.

- File `a` maps to column `0`; file `g` maps to column `6`.
- Rank `1` maps to row `0`; rank `7` maps to row `6`.
- Parsing accepts uppercase files, so `C4` parses as `c4`.
- `format_move()` returns lowercase coordinates and returns an empty string for an invalid square.

## Move Errors

`validate_move()` and `apply_move()` use typed errors:

- `MoveError::kOutOfBounds`: the square is outside the 7x7 board.
- `MoveError::kGameOver`: the board is already full.
- `MoveError::kOccupied`: the square is already occupied.

`move_error_name()` returns stable snake-case names for logs and protocols.

## Engine Usage

Engines can use the same public move type, but hot search code can call the reversible position API directly.

```cpp
poe2::MoveUndo undo;
if (position.make_move(move.square, undo)) {
  // Search child node.
  position.unmake_move(undo);
}
```

Use `poe2/transposition_table.hpp` for search caching. The table stores exact `PositionKey` values and uses `Position::hash()` for fast bucket lookup when probing or storing a live `Position`.
`update_position_hash()` applies the reversible piece-and-side-to-move Zobrist transition used by
`Position`; applying it twice with the same player and square restores the original hash.

## Symmetry Helpers

Use `poe2/symmetry.hpp` when an engine wants to treat rotated or reflected boards as the same search state.

The symmetry API transforms only geometry:

- `transform_square(symmetry, square)` maps a move square into another orientation.
- `transform_bitboard(symmetry, bits)` maps occupied squares in one bitboard.
- `transform_position_key(symmetry, key)` maps both player bitboards and preserves side to move.
- `canonicalize_position_key(key)` transforms the key under all eight symmetries and returns the lexicographically smallest `(low, high)` key, plus the transform used and its inverse.

Canonicalization never swaps players.

### Incremental Symmetry Tracking

Searches that opt into symmetry can construct one `PositionSymmetryTracker` from the live key. It maintains both player bitboards and hashes in all eight orientations as moves are made and unmade.
`preview_move()` computes the child canonical view without changing the tracker:

```cpp
#include "poe2/symmetry.hpp"

poe2::PositionSymmetryTracker symmetry{position.key()};
const poe2::CanonicalPositionView child = symmetry.preview_move(move.square);

poe2::MoveUndo undo;
if (position.make_move(move.square, undo)) {
  if (symmetry.make_move(move.square)) {
    // child.key and child.hash are ready for the recursive search.
    symmetry.unmake_move(move.square);
  }
  position.unmake_move(undo);
}
```

`canonical_view()` returns the current key, hash, and live-to-canonical transform pair.
`stabilizer_mask()` identifies the transforms that leave the current colored position unchanged.
`transformed_move_orbit()` returns every move equivalent to a square under that stabilizer.
The compile-time `transformed_move_bits` table is available to searches that want to reuse a previously computed stabilizer mask.

### Transposition Table

`TranspositionTable` uses aligned two-way buckets. `capacity()` counts usable entry slots, while `storage_bytes()` reports the cache-line allocation. Keys remain authoritative: hashes select a bucket but are not stored in its packed entries.

Callers that already maintain the exact hash can avoid recomputation with the prehashed overloads:

```cpp
const poe2::CanonicalPositionView view = symmetry.canonical_view();
if (const std::optional<poe2::TranspositionEntry> hit = table.probe(view.key, view.hash)) {
  // Use the exact-key hit.
}

const bool accepted = table.store(view.key, view.hash, searched_value);
```

The overloads taking only a key recompute its hash. The `Position` overloads use the live, non-canonical key and its already-maintained hash. `store()` reports whether the replacement policy accepted the write.

### Minimax Command Line

The minimax executable enables D4 symmetry, a 64 MiB transposition table, and the `pattern-gain` evaluator by default:

```bash
build/release/engines/minimax/poe2_minimax \
  [--hash-mb <size>] [--no-symmetry] \
  [--evaluator static|two-ply-closure|pattern-gain]
```

`--hash-mb 0` disables the table. `--no-symmetry` selects the separately compiled identity search path, so it does not construct or update a symmetry tracker.

`static` is the normalized board score. `two-ply-closure` minimizes over the opponent's best reply after each candidate move, then applies the static score. `pattern-gain` adds the frozen line-pattern and marginal-gain-summary residual to that closure value, keeping scale-32 fixed-point scores inside search. It remains exact at terminal positions and with at most two empty squares.
