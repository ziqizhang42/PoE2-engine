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

## Symmetry Helpers

Use `poe2/symmetry.hpp` when an engine wants to treat rotated or reflected boards as the same search state.

The symmetry API transforms only geometry:

- `transform_square(symmetry, square)` maps a move square into another orientation.
- `transform_bitboard(symmetry, bits)` maps occupied squares in one bitboard.
- `transform_position_key(symmetry, key)` maps both player bitboards and preserves side to move.
- `canonicalize_position_key(key)` transforms the key under all eight symmetries and returns the lexicographically smallest `(low, high)` key, plus the transform used and its inverse.

Canonicalization never swaps players.

### Symmetric Transposition Table

The current `TranspositionTable` intentionally stores exactly the key it is given. A symmetry-aware engine can canonicalize before probing or storing:

```cpp
#include "poe2/symmetry.hpp"
#include "poe2/transposition_table.hpp"

const poe2::CanonicalPositionKey canonical =
    poe2::canonicalize_position_key(position.key());

if (const std::optional<poe2::TranspositionEntry> hit = table.probe(canonical.key)) {
  if (hit->value.best_move.has_value()) {
    const poe2::Square move_in_current_position =
        poe2::transform_square(canonical.inverse_transform, *hit->value.best_move);
  }
}
```

When storing a best move, store it in canonical orientation because the entry is keyed by `canonical.key`:

```cpp
poe2::TranspositionValue value = searched_value;
if (value.best_move.has_value()) {
  value.best_move = poe2::transform_square(canonical.transform, *value.best_move);
}

table.store(canonical.key, value);
```

An engine should avoid `table.store(position, value)` and `table.probe(position)` for symmetry-aware use because those overloads use the live position's non-canonical key and hash.

### Child Deduplication

The same helper can also suppress duplicate child nodes. The simplest version canonicalizes each child after making the move and skips children whose canonical key has already been searched from this parent:

```cpp
std::vector<poe2::PositionKey> searched_children;
poe2::Bitboard moves = position.legal_moves();

while (moves != 0) {
  const int move_index = std::countr_zero(moves);
  const poe2::Square move = poe2::square_from_index(move_index);
  moves &= moves - poe2::Bitboard{1};

  poe2::MoveUndo undo;
  if (!position.make_move(move, undo)) {
    continue;
  }

  const poe2::PositionKey child_key =
      poe2::canonicalize_position_key(position.key()).key;

  const bool duplicate =
      std::find(searched_children.begin(), searched_children.end(), child_key) !=
      searched_children.end();
  if (!duplicate) {
    searched_children.push_back(child_key);
    // Search this representative child.
  }

  position.unmake_move(undo);
}
```

This works for both fully symmetric and partially symmetric positions because it compares the canonical result of the actual child position. A later optimized engine can replace the vector with a small hash set or compute parent-stabilizing symmetries first.
