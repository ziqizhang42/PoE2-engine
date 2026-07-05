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
