# Research log

Experiments that answered a question about what to try next, newest first. Each entry gives
the question, what was measured, and the conclusion drawn from it. Release notes are in
`CHANGELOG.md`; game results are in `eval/results.csv` and `models/`.

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
