# Changelog

## Unreleased

- 2026-08-20: A pipeline reproduction of v0.2 (`training/minimax/recipes/run1.toml` at `4198145`) matched it offline, with sealed-test MAE 4.929 vs 4.940.
- 2026-11-14: Engine gates build candidates privately. Every gated model is archived in `models/` with its metrics and listed in `models/index.csv`, and only gate winners can be installed.
- 2026-11-14: The reproduction played even with `000074` (49.8% over 1,308 games, no difference), and its recipe was retired. v0.2 is generation `g000` of the next run.
- 2026-11-14: Datasets can name an archived teacher model, which drives position sampling and labels and is recorded in every data manifest. v0.2 is archived as `g000`, and `run1` trains `g001` from it.
- 2026-11-14: Recipes can be written compactly as shared defaults plus one short entry per generation, which expands into the same datasets, iterations, and fingerprints as the long form. A generation can learn from an earlier generation of the same recipe straight from the run folder once that parent wins its gate, so one `poe2-train run` trains a chain of generations and stops at the first that does not win. `run1.toml` now uses the compact form, with unchanged fingerprints.
- 2026-11-14: Recipes can list comparisons between any two of their models, such as the newest generation against `g000` or a generation that lost to its parent against an older one. A run plays each comparison as soon as both models exist, and `record-comparison` records it in `eval/results.csv` as `training-comparison` and in the new `models/comparisons.csv`, apart from the gates that decide champions and chains.
- 2026-11-14: A generation can train on more than its own data: a weighted window of its ancestors' datasets, copies of their positions relabeled by its parent, and named extras such as a sibling that lost or the imported v0.2 corpus, under one leak rule that keeps the primary's validation and test positions unseen. An optional record cap keeps whole games, and training can expand features in pieces to fit larger sets on the GPU. Nothing changes unless these options are set.
- 2026-11-14: A training run cleans up after a crash or power loss the way it does after a pause. The next command removes whatever the stopped run had started but not finished and then resumes. Data tools, the match runner, and training reports sync their files before marking them complete, a cut-short last line of the event log is dropped instead of blocking the run, and private builds reuse a verified Catch2 checkout already on disk instead of fetching it from GitHub.
- 2026-11-16: A training run no longer depends on where the repository folder lives. Fingerprints and records name paths relative to the repository, and the match runner records them that way, so a finished or unfinished run keeps working after the repository is moved or copied, and a moved private build tree is rebuilt from scratch when it is needed again. This changes the fingerprints of stages that name a path: `run1`'s candidate validation and gate, which have not started. The pilot's candidate validation, recorded with absolute paths, still validates in place.
- 2026-11-16: Pattern training can compute the line term by looking up the weights of each position's 36 line patterns (`method = "lookup"`) instead of building 1,716 dense counts per position. It optimises the same model, repeats bit for bit, and on the v0.2 corpus took 6.2 ms per step instead of 13.1 on the GPU, with a sixth of the GPU memory. It trains two million records in 2.4 GiB of GPU memory, where dense no longer fits this machine. Dense stays the default, and the method is fingerprinted only when used.

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
