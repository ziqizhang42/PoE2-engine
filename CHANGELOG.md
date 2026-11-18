# Changelog

## Unreleased

### Engine

- The search is about 1.6 times faster and returns the same results. It updates the evaluator's inputs move by move instead of recomputing them at every position, and starts loading a position's table entry before it needs it. Searches to a fixed node count or depth return the same moves, scores, and node counts, every evaluation has the same value, and labels are byte for byte the same. At 100 ms a move it beat the previous build 780-740 over 1,520 games (51.3%, 95% interval 49.6% to 53.1%).
- Timed searches look deeper. At each position the engine searches its first four moves at full depth and tries the rest two plies shallower, searching again in full any move that beats the best so far. Searches to a fixed node count or depth are unchanged, so labels are the same. The depth a timed search reports is that of its first moves; later moves may have been searched less deeply. At 100 ms a move it beat the previous build 204-182 over 386 games (52.8%, 95% interval 50.3% to 56.3%).

### Evaluation

- Balanced opening books `eval/openings/development-v2.txt` and `eval/openings/holdout-v2.txt`, about 4,000 openings each. An opening is kept only when a 200,000-node search, corrected for each opening length from match results, predicts a result within 2 points of even. Between the same two engines over 4,000 games, the share of opening pairs that one engine won outright rose from 9% on the old development book to 32%, and the first player's win rate fell from 60% to 51%. `poe2_runner openings fit-balance` refits the correction and `openings generate-balanced` regenerates the books byte for byte ([docs/evaluation.md](docs/evaluation.md)). Gates still use the old books.

## 0.3.2 - 2026-11-17 · net run1/g002

- The built-in net is `run1/g002` (`232a3cc37caf`), the second generation of `run1`. `run1/g001` taught it: its data is 303,534 positions from 40,000 generated games, all labeled by g001. It beat g001 204-182 over 386 games at 100 ms a move (52.8%, 95% interval 50.3% to 56.3%).
- The engine is the same as in 0.3.1.

## 0.3.1 - 2026-11-17 · net run1/g001

- The built-in net is `run1/g001` (`8e39040c1537`), the first generation of `run1`. v0.2 taught it: its data is 302,842 positions from 40,000 generated games, all labeled by v0.2. It beat v0.2 90-46 over 136 games at 100 ms a move (66.2%, 95% interval 58.3% to 75.7%).
- The engine is the same as in 0.3.0.

## 0.3.0 - 2026-11-17 · net run1/g000

The built-in net is unchanged: `run1/g000` (`633309adb322`) is v0.2.

### Engine

- Nets are files. The engine has one built in and loads another with `--net <file>`, as do the inference and data tools. `poe2_minimax --version` and the protocol handshake report the engine version, the designs it supports, and the net in use ([docs/nets.md](docs/nets.md)).
- 0.3.0 evaluates every position as 0.2.0 does and searches as fast. Over 2,000 games at 100 ms a move against the previous engine, both playing v0.2, it scored 50.1% (95% interval 49.6% to 50.7%).
- The browser package loads a net from bytes with `createEngine({ net })`, reports its version and net as `engine.info`, and searches about 3% faster.
- Removed: the compiled-in model header, the `POE2_MINIMAX_MODEL_HEADER` build option, and `tools/export_minimax_pattern_gain.py`.

### Training

- A run is a chain of generations written as a short recipe. Each generation is taught by its parent and must beat it in a gate before the next one starts ([docs/training-workflow.md](docs/training-workflow.md)).
- Optional: comparisons between any two nets of a recipe, training on several generations' data, and a lookup training method that is faster and needs far less memory.
- A run recovers from a pause, a crash, or a power loss when the same command is rerun, keeps working when the repository is moved, and refuses work that will not fit in memory.
- Every gated net is archived in `models/` with its lineage and training record. `poe2-train handoff --apply` installs a gate winner as the built-in net and raises the last version number.

## 0.2.0 - 2026-08-18

- Add a new `pattern-gain` evaluator which adds learned line-pattern and marginal-gain-summary residuals to the exact two-ply-closure baseline while retaining exact terminal and two-empty evaluation.
- Add deterministic position generation, labeling, feature export, offline training, quantization, and C++/Python parity tooling for the learned model.
- Add descriptive evaluator options: `static`, `two-ply-closure`, and `pattern-gain`.

### Pattern-gain training results

The architecture was selected on the earlier 80k development corpus, then retrained from scratch on the new corpus with seed `20260818`.

| Corpus accounting | Positions |
|---|---:|
| Raw generated | 320,000 |
| Duplicates removed | 16,401 |
| Unique train | 189,246 |
| Unique validation | 74,157 |
| Initial held-out test | 40,196 |
| Earlier-development overlaps excluded from test | 217 |
| Final sealed test | 39,979 |

The selected combined model has 5,293 parameters and uses the checkpoint from step 850. It uses line knots at plies `0/28/49`, gain knots at `0/12/24/36/49`, Huber delta 8, and L2 strength `1e-4`. The shipped fixed-point model uses five fractional bits (scale 32).

| Evaluator | Validation MAE | Final-test MAE | Final-test RMSE | Final-test sign accuracy |
|---|---:|---:|---:|---:|
| Two-ply closure | — | 18.908 | 25.114 | 75.39% |
| Gain-summary control | 7.400 | 7.836 | 10.511 | 90.68% |
| Line-only fallback | 5.383 | — | — | — |
| Combined float model | 4.443 | 4.946 | 6.634 | 94.14% |
| Combined scale-32 model (shipped) | 4.429 | 4.940 | 6.643 | 93.43% |

On the sealed test, the combined float model reduced MAE by 73.84% versus two-ply closure and by 36.88% versus the gain-summary control.

In a paired 100 ms/move gate against the `v0.1` engine baseline (`000053-96a4b598a45e`), the `v0.2` candidate (`000072-1f8162e8fd16`) scored 91.7% (77–7 over 84 games from 42 side-swapped openings). The 95% paired confidence sequence was 78.9%–98.2%, and the normalized-Elo GSPRT stopped with `accept_alt`.

## 0.1.0 - 2026-08-13

- Add the browser WebAssembly engine package.
- Add Multi-PV analysis and progress streaming.
