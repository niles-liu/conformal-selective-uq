# Pre-registration: <phase> (<date>)

> **Gate-first.** Fill this out and freeze it (commit, or at minimum timestamp) **before** running
> the method. The point is that the ship/reject statistic and its threshold are committed *prior* to
> seeing the result. Mirror this into a paired `reports/<phase>_results.md` after.

## Question
One sentence: what does this phase test?

## Data / splits
- Task layer(s): 1a (known-ceiling synthetic) / 1b (real public) / which.
- Feature panel + PIT cutoff; calibration/test split; the shift axis (what defines train vs. test).
- Cluster unit for inference (sector / regime / day).

## Baselines (computed first)
- Naive split-conformal (marginal); unconditional CQR; random abstention; oracle abstention (1a only).
- Report their numbers here once known; they set the bar the method must clear.

## Pre-committed gates (numbers frozen)
| Gate | Statistic | Threshold | Ship if |
|---|---|---|---|
| G-PRIMARY (i) useful abstention | retained error/width vs random @ retention X% | Δ = … , X = … | reduction ≥ Δ |
| G-PRIMARY (ii) conditional coverage | per-group coverage vs nominal (target = …%) | τ = … | within ±τ all groups |
| G-SHIFT | worst-group coverage out-of-regime; naive gap first | τ_shift = … ; naive-close margin = … | worst-group within ±τ_shift AND beats naive by margin |
| G-MECHANISM (1a) | confident∩recoverable AUC; width vs noise ρ | θ = … ; θ′ = … | AUC ≥ θ AND ρ ≥ θ′ |
| G-NULL | label-shuffle: selective advantage | within band of random | advantage collapses |

## Seeds / reproducibility
Seeds (≥3): … . Each reported number is the multi-seed mean ± band. Run command: `python src/run_<phase>.py`.

## 4-lens adversarial check (fill in results doc, after)
- **Reproduce**: does every printed number reproduce exactly from the seeded run?
- **Leakage**: calibration never saw test labels? G-NULL collapses? PIT respected? Does the abstention
  score use labels from the fold the conformal layer calibrates on (if so, retained-set coverage is empirical)?
- **Statistics**: bands from clustered/blocked resampling, not row bootstrap? Power adequate per group?
- **Mechanism**: does the result mean what we claim (esp. G-MECHANISM), or a confound?
