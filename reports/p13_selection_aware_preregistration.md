# Pre-registration: P13, selection-aware calibration for the kept set (2026-10-12)

> **What P13 asks.** P2's abstention score `u_i` is a per-stock residual scale estimated on the
> 2021 calibration fold, and the conformal layer calibrates on that same fold. Coverage on the kept
> set (the 70% of test rows with the lowest `u_i`) therefore has no finite-sample guarantee: the
> selection reuses calibration labels, and the conformal radius is set on every stock, kept or not.
> P13 asks whether a selection-aware arrangement, which estimates `u_i` without touching the
> calibration labels and then calibrates only on the kept stocks, **(a)** gives nominal coverage on
> the kept set, **(b)** gives narrower intervals there than calibrating on every stock, and **(c)**
> still finds the unrecoverable stocks.
>
> **Why it should work.** If the selection rule is a fixed function of data that neither the
> calibration nor the test labels touch, then the kept calibration rows and the kept test rows are
> exchangeable whenever the full rows are, so split conformal calibrated on the kept rows covers the
> kept test rows at `1 - alpha` (marginally, in finite samples, under exchangeability). That is a
> standard property of split conformal under a fixed selection, not a new result. P13 checks it on
> this substrate and measures what it costs and what it buys.
>
> **What was known when this was written.** P1 to P12 were complete and published, including P2's
> numbers: full-population static CQR covers 0.901 on base-regime test rows; `u_cal` (P2's score)
> has AUC 0.813 against the recoverable mask and cuts retained MAE at 70% by 0.178 against random;
> per-group gaps of ACI on P2's kept set reach 0.06. Not computed before this was frozen: any
> coverage or width of static CQR on a kept set, and anything using `u_train` or `u_half`. The
> substrate and seeds are P2's, so P13 is a new question on known data, not a confirmation on fresh
> data. Gate-first: this file and the driver are committed before the driver is run.

## Substrate (frozen, unchanged from P2)
Layer 1a semi-synthetic target, seeds {0,1,2}, the P2 split: train `report_date < 20210101`,
calibration 2021, test `>= 20220101`. In the test fold 2022 is the shift regime (overlay amplified
x1.8) and 2023-2024 is the base regime. **The gates are evaluated on base-regime test rows**, where
full-population static CQR already covers 0.901 (P2), so exchangeability between calibration and
test is approximately right there. Shift-regime rows are reported, not gated.

## Method (frozen)
Base learners: the frozen LightGBM mean and quantile models (`selective_conformal.learners`), fit
once per seed on the training fold, as in P2.

Abstention scores, all with the frozen shrinkage `K = 20` (`selective.name_uncertainty`):
- **`u_cal`** (P2's score): residuals `|y - mean(x)|` on the calibration fold.
- **`u_train`** (selection-aware, primary): out-of-fold training residuals, cross-fitted over **5
  contiguous time blocks** of training sessions (fit the mean model on four blocks, predict the
  fifth). Contiguous blocks keep the autocorrelated hidden overlay (AR(1), rho = 0.95) from leaking
  into the out-of-fold residuals. No calibration or test label is touched.
- **`u_half`** (selection-aware, secondary): calibration residuals from the first half of 2021
  (`report_date < 20210701`) only.

**Kept set.** P2's label-free threshold: `tau` = the 70% quantile of the score over test rows; a row
(calibration or test) is kept when its stock's score is `<= tau`. The same `tau` applies to both
folds, so both use the same selection function.

**Intervals.** Static split CQR on the frozen quantile bands (`cqr_from_quantiles`, `alpha = 0.10`),
calibrated either on every calibration row ("all") or on the kept calibration rows only ("kept").

### Variants
| Variant | Score | Calibrated on | Role |
|---|---|---|---|
| V0 | `u_cal` | all calibration rows | P2's arrangement (the caveat) |
| V1 | `u_cal` | kept calibration rows | select-then-calibrate that reuses calibration labels for the selection |
| V2all | `u_train` | all calibration rows | same kept set as V2, isolates the effect of calibrating on it |
| **V2** | `u_train` | kept calibration rows | **proposed** selection-aware calibration |
| V3 | `u_half` | kept rows of the second half of 2021 | secondary: selection-aware by splitting the fold |
| V2-ACI | `u_train` | kept calibration rows, online (`aci_from_quantiles`, `mom_decile` groups, gamma 0.05, kept test rows as the stream) | report-only, for the shift regime |

## Pre-committed gates (target 90%, mean over seeds {0,1,2}; kept base-regime test rows unless stated)
| Gate | Statistic | Pass if |
|---|---|---|
| **G-VALID** (primary) | V2 coverage | within **+/-0.02** of 0.90 |
| **G-TIGHT** (primary) | V2 mean width / V2all mean width (same kept rows) | **<= 0.95** |
| **G-MECH** (primary) | `u_train`: AUC(-u -> recoverable) and the retained-MAE reduction against random at 70% (all test rows, computed as in P2) | AUC **>= 0.70** and reduction **>= 0.119** (P2's frozen theta and Delta) |
| G-CAVEAT (report) | V0 coverage - 0.90 | how far P2's arrangement sits from nominal on the kept set |
| G-REUSE (report) | V1 coverage - V2 coverage | what reusing calibration labels for the selection costs |
| G-SPLIT (report) | V3 coverage and width against V2 | the split-fold alternative |
| G-SHIFT (report) | kept shift-regime coverage of V2 and V2-ACI | expected: static fails under the shift, as every static method did in P1/P2 |

Also reported: per-seed values; a seed-0 clustered bootstrap band (sector x month) on V2's kept-base
coverage; worst-group |gap| on the kept base rows for V0 and V2 (sector, cap_tier, mom_decile; n >= 50).

## Verdict rule (pre-committed)
- **CONFIRM:** G-VALID and G-TIGHT and G-MECH. Statement: estimating the abstention score away from
  the calibration labels and calibrating on the kept stocks gives nominal coverage on the kept set,
  with narrower intervals than calibrating on every stock, while the score still finds the
  unrecoverable stocks. P2's kept-set caveat has a working fix that is finite-sample valid under
  exchangeability.
- **QUALIFIED:** G-VALID holds but G-TIGHT or G-MECH fails (valid, but it buys little width, or the
  selection-aware score finds the noisy stocks less well). Report which.
- **REJECT-as-finding:** G-VALID fails. Report by how much and in which direction.

## Threats pre-registered
- **Exchangeability is approximate.** 2021 and 2023-2024 differ in market features. Full-population
  static CQR covers 0.901 on the base rows (P2), which bounds how much of any miss is drift rather
  than selection.
- **Fewer calibration rows.** Calibrating on the kept stocks uses about 70% of the fold (roughly 16k
  rows); the finite-sample term 1/(n+1) is negligible at that size.
- **No retuning.** K, the retention level, alpha, gamma, the block count and the half-year date are
  fixed here and inherited from P2 where P2 fixed them.

## 4-lens (after)
Reproduce (deterministic, seeded; `u_cal`'s AUC must reproduce P2's 0.813) / Leakage (`u_train` and
`u_half` never read a label they are later judged on) / Statistics (3 seeds +/-sd, seed-0 clustered
band, per-group n >= 50) / Mechanism (V2 against V2all isolates calibration; V1 against V2 isolates
label reuse).

## Run
`python experiments/run_p13.py`, then mirror into `reports/p13_selection_aware_results.md`.
