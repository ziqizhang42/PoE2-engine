## Game Rules

The rules of Powers of Exponent 2 are as follows:

### Setup

- The game is played on a rectangular grid of width 7 and height 7
- The game starts with an empty grid
- **Player 1** starts with a score of 0
- **Player 2** starts with a handicap of 5.5 (effectively player 2 starts with +5 and wins ties).

### Gameplay

- Players alternate turns, with **Player 1** going first
- On each turn, a player places their digit (1 or 2, matching their player number) in an empty cell
- The game continues until the board is completely filled (winner is the player with the higher score)

### Scoring System

The total score is the sum of all individual line scores, calculated as follows:

**Line Score Formula:** For a contiguous line of length X: **2^(X-1)**

Examples:

- Line of 1 piece: 2^0 = 1 point
- Line of 2 pieces: 2^1 = 2 points
- Line of 3 pieces: 2^2 = 4 points
- Line of 4 pieces: 2^3 = 8 points
- Line of 5 pieces: 2^4 = 16 points

### Scoring Rules

1. **Directions:** Lines are counted in all four directions:
   - Horizontal
   - Vertical
   - Diagonal (both directions)

2. **Maximal Lines Only:** Every maximal contiguous line counts. Shorter lines that are subsets of a maximal line in the same direction are not counted separately.

3. **Overlapping Lines:** Lines can overlap as long as they are not subsets of each other. One piece can count for multiple lines in different directions.

4. **Single Pieces:** If a piece does not form part of any line (length ≥ 2), it counts as a line of length 1 (worth 1 point). However, do NOT count a piece as length 1 if it's already part of another line.

5. **Recalculation:** The score is recalculated each turn from scratch based on the current board position (not incremental).

### Winning Condition

When the board fills completely, the player with the higher total score wins.
