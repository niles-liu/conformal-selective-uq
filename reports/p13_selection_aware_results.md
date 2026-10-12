# Results: P13, selection-aware calibration for the kept set (2026-10-12)

> **VERDICT: CONFIRM on the three-seed mean (all three gates pass), with a per-seed caveat.**
> Scoring stocks on cross-fitted training residuals and calibrating split CQR on the kept stocks only
> gives kept-set coverage of **0.896** on the calm test rows (target 0.90), against **0.928** for P2's
> arrangement, at **12% less width**, and the score finds the unrecoverable stocks at least as well
> (AUC 0.845 against 0.813). **But the static version is noisy across seeds** (0.869 / 0.891 /
> 0.927): it is right on average, not on every draw. **The online version on the same kept set is
> stable:** 0.901 to 0.904 on every seed, and 0.835 under the shift, where every static variant
> collapses. Pre-reg and gates: `reports/p13_selection_aware_preregistration.md` (committed before
> the run).

## Results (Layer 1a, seeds {0,1,2}, target 90%; kept set = the lowest-score 70% of test rows)
| variant | kept calm coverage | per seed | width | kept shift coverage | calibration rows |
|---|---|---|---|---|---|
| V0: `u_cal`, calibrate on all (P2's arrangement) | 0.928 +/-0.009 | 0.916 / 0.928 / 0.939 | 6.27 | 0.607 | 23,436 |
| V1: `u_cal`, calibrate on kept | 0.876 +/-0.017 | 0.862 / 0.865 / 0.900 | 5.17 | 0.514 | 16,632 |
| V2all: `u_train`, calibrate on all | 0.931 +/-0.011 | 0.916 / 0.934 / 0.943 | 6.25 | 0.606 | 23,436 |
| **V2: `u_train`, calibrate on kept (proposed)** | **0.896 +/-0.024** | 0.869 / 0.891 / 0.927 | **5.49** | 0.541 | 16,632 |
| V3: `u_half`, calibrate on kept second half of 2021 | 0.884 +/-0.030 | 0.860 / 0.866 / 0.926 | 5.40 | 0.535 | 8,448 |
| V2-ACI: `u_train`, online on the kept set | 0.903 +/-0.001 | 0.901 / 0.904 / 0.904 | 5.56 | **0.835** | 16,632 |

33,132 kept calm test rows per seed (mean). Seed-0 clustered band (sector x month) on V2's kept calm
coverage: 0.869 [0.860, 0.876]. Worst-group |gap| on the kept calm rows (sector, cap_tier,
mom_decile; n >= 50): V0 0.072 (per seed 0.069 / 0.071 / 0.077), V2 0.085 (0.119 / 0.073 / 0.063).

| score | AUC (-u -> recoverable) | retained-MAE reduction vs random at 70% |
|---|---|---|
| `u_cal` (P2) | 0.813 +/-0.034 | 0.1775 (per seed 0.1968 / 0.1739 / 0.1618) |
| `u_train` (cross-fitted training residuals) | 0.845 +/-0.044 | 0.1895 (0.2158 / 0.1821 / 0.1704) |
| `u_half` (first half of 2021) | 0.785 +/-0.046 | 0.1716 (0.1933 / 0.1493 / 0.1723) |

`u_cal` and `u_train` make the same keep or drop call on 79.2% of test rows.

## Pre-registered gates
| Gate | Value | Result |
|---|---|---|
| **G-VALID** V2 kept calm coverage within +/-0.02 of 0.90 | 0.896 (0.004 off) | **PASS** |
| **G-TIGHT** V2 width / V2all width <= 0.95 | 0.878 | **PASS** |
| **G-MECH** `u_train` AUC >= 0.70 and reduction >= 0.119 | 0.845 and 0.1895 | **PASS** |
| G-CAVEAT (report) V0 kept calm coverage - 0.90 | +0.028 | P2's arrangement over-covers the kept set |
| G-REUSE (report) V1 - V2 kept calm coverage | -0.020 | reusing calibration labels for the selection costs 2 points |
| G-SPLIT (report) V3 against V2 | 0.884 vs 0.896, width 5.40 vs 5.49 | splitting the fold works, with half the calibration rows and more spread |
| G-SHIFT (report) kept shift coverage, V2 and V2-ACI | 0.541 and 0.835 | static fails under the shift, as expected; online holds most of it |

Pre-committed verdict rule (G-VALID and G-TIGHT and G-MECH) gives **CONFIRM**.

## What it shows

**1. P2 over-covers the kept set because it calibrates on every stock.** V0 covers the kept calm rows
at 0.928, and V2all, a different label-free kept set still calibrated on every stock, at 0.931. The
radius is partly sized for the noisy stocks the abstention layer then drops. Calibrating on the kept
stocks gives 0.896 at 12% less width (V2 against V2all, same kept rows).

**2. The selection must be label-free, and making it so costs nothing.** V1 calibrates on the kept
stocks but selects them with P2's calibration-fold score, and under-covers by 2 more points (0.876).
The kept set alone moves coverage by only 0.003 (V0 against V2all), so most of that shortfall is
reuse: stocks kept partly for luckily small calibration residuals. Cross-fitting the score on
training residuals removes the reuse and finds the unrecoverable stocks no worse (AUC 0.845 against
0.813; MAE reduction 0.190 against 0.178).

**3. Caveat: static calibration on the kept set is noisy from seed to seed.** V2 covers 0.869, 0.891
and 0.927, and seed 0's clustered band [0.860, 0.876] excludes 0.90. The gate was frozen on the mean,
which is nominal; the spread says the guarantee is only as good as the exchangeability between 2021
and 2023-2024 for the stocks a seed keeps. A likely reason, not tested: for low-noise stocks, error
from a drifting recoverable signal is a larger share of the residual, so drift moves their coverage
more than the full population's (0.901 in P2). The spread is reported, not used to re-label the
verdict.

**4. The online version is the robust form, as in P12b.** Group-conditional ACI on the kept stocks
(V2-ACI) covers 0.901 to 0.904 on every seed at width 5.56 (1% above V2, 11% below calibrating on
every stock), and 0.835 under the shift, where every static variant falls to 0.51 to 0.61.

**Group coverage on the kept set does not improve:** worst-group |gap| 0.085 for V2 against 0.072 for
V0 (seed 0 drives it, at 0.119). Calibrating on the kept stocks fixes marginal coverage, not the
sector mix behind P2's G-PRIMARY (ii) miss.

## 4-lens
- **Reproduce.** Deterministic and seeded; `u_cal` reproduces P2 exactly (AUC 0.813, reductions
  0.1968 / 0.1739 / 0.1618).
- **Leakage.** `u_train` reads only training labels, out of fold over contiguous time blocks; `u_half`
  reads only the first half of 2021, and the conformal step only the second; the threshold uses no
  labels.
- **Statistics.** Three seeds +/-sd, per-seed values shown because the spread is the main caveat;
  seed-0 clustered band; per-group n >= 50.
- **Mechanism.** V2 against V2all isolates calibrating on the kept set (same kept rows); V1 against
  V2 mostly isolates label reuse (their kept sets agree on 79% of rows, and V0 against V2all bounds
  the kept-set effect at 0.003).

## Next
1. Register the online form (V2-ACI) as the primary in a confirm on fresh seeds.
2. Group coverage on the kept set needs its own fix (the sector mix), not more calibration rows.

## Artifacts
`experiments/run_p13.py` (driver, about 2 minutes); `reports/p13_selection_aware_preregistration.md`
(frozen first).
