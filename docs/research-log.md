# Research log

Experiments that answered a question about what to try next, newest first. Each entry gives
the question, what was measured, and the conclusion drawn from it. Release notes are in
`CHANGELOG.md`; game results are in `eval/results.csv` and `models/`.

## 2026-11-18: do openings chosen for balance make a match say more?

Yes, about 4.6 times as much per game. On the rule-generated books the opening decides most
games: over 41,000 games the first player won 61% and led by 8.1 points before the 5.5
handicap, the same opening gave nearly the same lead in different matches (correlation 0.99),
and one engine won both games of only 7% of pairs. Ways of choosing openings, each tested on
development openings it was not fitted on:

| Openings kept | Share kept | Pairs one engine won outright |
|---|---:|---:|
| All | 100% | 7.2% |
| Score gap centred on its even point (-3 to +1) | 91% | 7.6% |
| Depth-1 search predicts within 2 points | 20% | 23.4% |
| 200,000-node search predicts within 2 points | 20% | 26.1% |
| Two or more earlier games ended within 2 points | 17% | 32.6% |

The raw search over-rates the first player by about 3 points in openings of 2 to 6 plies and
about 1 point in longer ones, so the prediction is the score corrected by a line fitted for
each length. Books built with the 200,000-node rule were then played by the same two engines
as a match on the old development book, 4,000 games each:

| | Old development book | `development-v2` |
|---|---:|---:|
| Pairs one engine won outright | 9.1% | 32.3% |
| First player wins | 60.5% | 51.4% |
| Better engine's score | 52.8% | 60.5% |
| Better engine's margin per game | +0.59 | +0.61 |
| Normalized Elo | 46 | 98 |

Conclusion: a gate on the balanced books should need far fewer games; they are not the
default yet. A score on one book cannot be compared with a score on the other, and the gate's
alternative of 20 normalized Elo is a difference about half as large on the balanced book. An opening's lead correlated
0.94 to 0.96 between neighbouring generations and 0.85 between v0.2 and g002, so refit the
correction and regenerate after a large change. The first player's lead did not move with
three times the thinking time (-0.13 points, standard error 0.07), which puts the even
handicap for engines of this strength near 7.5.

## 2026-11-18: which search changes add strength at a fixed time?

Searching late moves less deeply does; the others tried do not. Each change played the build
with the faster search on the development book at 100 ms a move, with fixed-length matches.

| Change | Games | Score | 95% interval |
|---|---:|---:|---:|
| Moves after the fourth 2 plies shallower | 4,000 | 53.0% | 52.0% to 54.2% |
| The same, a second match | 4,000 | 52.7% | 51.7% to 53.9% |
| The same, at 300 ms a move | 1,000 | 53.4% | 51.7% to 55.6% |
| 1 ply shallower, 2 from the twelfth move | 4,000 | 52.8% | 51.7% to 54.0% |
| 1 ply shallower | 3,000 | 51.8% | 50.7% to 53.1% |
| Order moves by cutoff history before gain | 4,000 | 50.9% | 50.1% to 51.9% |
| Keep the best move of an unfinished iteration | 4,000 | 50.5% | 49.6% to 51.5% |
| Let a new search overwrite older table entries | 4,000 | 50.2% | 49.8% to 50.7% |

A reduced move that beats the best so far is searched again in full. In 1,000-game
screenings, starting reductions at depth 4 instead of 3, reducing after three or six moves
instead of four, and sparing moves that gain 4 or more did not help. The first row then passed
its gate on the holdout book, 204-182 over 386 games (52.8%, 95% interval 50.3% to 56.3%).
Fixed-node and fixed-depth searches are unchanged, so labels are too.

Ordering by history first needs 23% to 25% fewer nodes to reach the same depth, but it changes
fixed-node searches and so labels, and on top of the reductions it scored 50.7% over 2,000
games (49.4% to 52.2%). Conclusion: more reduction was better at every step, so 3 plies is
worth trying. History-first ordering belongs at the start of a run, where labels start fresh.
So does trying the reductions in fixed-node searches: whether they give better labels for the
same nodes is not measured. The check is to label a sample both ways and compare both with
exact results late in the game.

## 2026-11-18: what does a faster search with the same results buy?

A third less labeling time and a small gain in games. Two changes were measured together:
updating the evaluator's inputs move by move instead of recomputing them at every position,
and starting to load a position's table entry before it is needed. An audit measured them
alone at 1.51 and 1.06 times the speed.

| Measurement | Result |
|---|---:|
| Search to 1M nodes, 40 positions, 64 MiB table | 1.56 times as fast |
| Same with a 16 MiB table | 1.58 times as fast |
| Moves, scores, depths, and node counts that differ | 0 |
| Labeling 2,000 positions of g002 at 5M nodes | 188 s to 122 s, labels byte for byte the same |
| Games at 100 ms a move against the build before | 780-740 over 1,520 (51.3%, 95% interval 49.6% to 53.1%) |

Large pages for the table added 1.3% to 1.9% on top, about the size of the measurement
noise, and were left out. Conclusion: at 100 ms a move, 1.56 times the speed is worth about
one game in a hundred. More strength at a fixed time has to come from searching fewer
positions per ply, which changes what the engine plays and needs its own gates.

## 2026-11-17: do the gains of g001 and g002 add up against v0.2?

No. g001 and g002 each played v0.2 for 1,000 games on the same 500 openings.

| Against v0.2 | g001 | g002 |
|---|---:|---:|
| Result | 677-323 | 660-340 |
| Score | 67.7% | 66.0% |
| 95% interval | 64.5% to 71.4% | 62.8% to 69.7% |
| Average score margin per game | +5.37 | +4.93 |

Opening by opening, g002 scored 1.7 percentage points less than g001 (standard error 0.9)
and its margin was 0.43 points a game smaller (standard error 0.13). g002 beats g001 when
the two play each other, but against v0.2 it is no better than g001 and slightly worse by
margin. Conclusion: beating its parent does not show that a generation is stronger against
other opponents. Compare a new generation with an older one as well before counting it as
progress.

## 2026-11-17: does a net taught by g001 beat g001?

Yes, but by much less than g001 beat v0.2. `run1/g002`, whose data is 303,534 positions from
40,000 generated games all labeled by g001, beat g001 204-182 over 386 games (52.8%, 95%
interval 50.3% to 56.3%).

| | g001 against v0.2 | g002 against g001 |
|---|---:|---:|
| Games until the gate decided | 136 | 386 |
| Score | 66.2% | 52.8% |
| Paired openings won both / split / lost both | 22 / 46 / 0 | 12 / 180 / 1 |
| Sealed-test MAE against its own teacher's labels | 3.909 | 3.443 |

g002 fits g001's labels more closely than g001 fit v0.2's, but that does not compare their
strength, because the labels differ. Conclusion: the gain per generation of this design is
shrinking fast, as the teacher comparison of 2026-11-16 predicted. Whether a `g003` would
still pass its gate is open.

## 2026-11-17: does loading the net from a file cost search speed?

No, after two fixes. Measured by fixed-node searches, old and new engines interleaved, with
identical moves, scores, and node counts.

| Build | New against old, paired median |
|---|---:|
| Native, built-in net | 1.000 to 1.002 |
| Native, net from a file | 0.996 to 1.003 |
| Browser package | 1.031 to 1.040 |

- Reading the phase knots from the file made the interpolation's divisions non-constant and
  cost about 0.6% native speed. The knots are now fixed by the design; a net with other knots
  needs a new design.
- Reading weights from loaded tables cost the browser package 1 to 2%. Storing line weights
  by knot and summing only the two knots a position's ply lies between removed that loss
  and more.

## 2026-11-16: is `run1/g001` a better teacher than v0.2?

Slightly, where it can be checked. 1,962 of g001's training positions were relabeled with
g001 as teacher and compared with the labels g001 was trained on, and with exact results
where both searches solved the position.

| | Hand-written evaluator to v0.2 | v0.2 to g001 |
|---|---:|---:|
| Positions solved exactly by both | 40% | 42% |
| Average label change on the rest | 11.2 | 4.6 |
| Predicted winner flips | 18% | 9% |
| Error against exact results at moves 28 to 30 | 8.1 to 1.8 | 1.7 to 1.5 |
| Correct winner at moves 28 to 30 | 92.6% to 97.5% | 96.4% to 98.2% |

Earlier in the game the labels still move by 6 to 7.5 on average, which cannot be checked
against exact results. Conclusion: a `g002` taught by g001 should gain less than g001 did,
and may tie it.

## 2026-11-16: would more training steps or more data have helped g001?

No. g001's best checkpoint was its last step, and its training error was below its
validation error, which suggested both. Measured on g001's data:

| Step limit | Stopped at | Validation MAE | Sealed-test MAE |
|---|---:|---:|---:|
| 2,000 (as trained) | 2,000 | 3.9638 | 3.9131 |
| 20,000 | 3,325 | 3.9630 | 3.9128 |

| Training positions | Validation MAE |
|---:|---:|
| 28,000 | 4.452 |
| 55,000 | 4.148 |
| 111,000 | 4.026 |
| 222,000 (all) | 3.964 |

Each doubling of the data gives about half the gain of the one before, so another doubling
(a two-generation window) would give about 0.03, less if the older data carries a weaker
teacher's labels. Conclusion: this design is close to the best it can do on these labels.
Gains come from better labels or a larger design, not from longer training or more data.

## 2026-11-16: does a net taught by v0.2 beat v0.2?

Yes. `run1/g001`, whose data is 302,842 positions from 40,000 generated games all labeled
by v0.2, beat v0.2 90-46 over 136 games (66.2%, 95% interval 58.3% to 75.7%), winning
both games of 22 paired openings and losing both of none.

## 2026-11-16: does lookup training match dense training?

Yes. On v0.2's 189,246 training positions, validation MAE was 4.4280 dense and 4.4271
lookup, and each method repeats bit for bit on this GPU. Dense on the GPU and dense on the
CPU differ from each other as much as dense and lookup do. Lookup needs about 3.3 KB of host
memory per position against about 21 KB for dense, which stops fitting this machine between
1 and 1.5 million positions (`docs/training-workflow.md` has the full table). Conclusion:
`run1` trains with lookup.

## 2026-11-14: does the training pipeline reproduce v0.2?

Yes. A pipeline reproduction of v0.2 reached sealed-test MAE 4.929 against v0.2's 4.940
and played even with it (49.8% over 1,308 games). Its recipe was retired, and v0.2 became
generation `g000` of `run1`.
