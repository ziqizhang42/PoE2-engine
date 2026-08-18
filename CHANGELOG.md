# Changelog

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

## 0.1.0 - 2026-08-13

- Add the browser WebAssembly engine package.
- Add Multi-PV analysis and progress streaming.
