# Results — P1 baselines (method-blind read, 2026-06-30)

> Paired with `p1_baselines_preregistration.md` (frozen first). All numbers from
> `python src/run_p1.py`, multi-seed {0, 1, 2}, target coverage 90%. Splits: train
> rd < 20210101 · calibrate 2021 (calm, pre-shift) · test rd ≥ 20220101 (2022 =
> amplified-overlay shift regime; 2023–24 = base). Bands = sector×month clustered
> bootstrap. Reproduces bit-for-bit from the seeded run.

## Headline
1. **The conformal machinery is correct** — in the exchangeable **base regime**, naive
   split-conformal achieves **0.902** coverage (nominal). So the failures below are the
   *shift*, not a bug.
2. **Naive UQ fails conditionally under the documented shift** — on the 2022 drawdown
   slice, coverage collapses to **0.58** and **worst-group to 0.507** (vs 0.90 nominal).
   This is the gap the P2 method must close (G-SHIFT).
3. **Naive uncertainty does not abstain in the right place** — CQR-interval-width
   abstention gives essentially *no* retained-MAE improvement (2.044 vs random 2.029 at
   70% retention), while an **oracle** that knows the irreducible noise improves it
   sharply (1.791). The headroom is real and large; capturing it needs the
   group-conditional / shift-robust selective method — the P2 contribution. Established
   here method-blind.

## Layer 1a — known-ceiling synthetic (mean ± sd over seeds)
| metric (target 90%) | naive split | CQR (uncond.) |
|---|---|---|
| marginal coverage (mixed test) | 0.794 ± 0.026 | 0.791 ± 0.033 |
| coverage \| **base** regime | **0.902** ± 0.010 | 0.901 ± 0.015 |
| coverage \| **shift** regime | **0.580** ± 0.060 | 0.573 ± 0.071 |
| **worst-group** \| shift | **0.507** ± 0.058 | 0.474 ± 0.078 |
| mean interval width | 6.249 ± 0.114 | 6.238 ± 0.224 |

Seed-0 clustered bands (sector×month): naive marginal 0.773 [0.758, 0.789], shift
0.541 [0.518, 0.561]; CQR shift 0.522 [0.499, 0.542]. The shift under-coverage is far
outside any band — not sampling noise. Marginal-over-mixed-test (0.79) is itself
depressed because the test deliberately spans an out-of-regime block; the clean
in-regime check is base = 0.90.

### Risk–coverage (retained MAE; abstain highest-score first), mean over seeds
| retention | random | oracle (knows σ_irr) | CQR-width |
|---|---|---|---|
| 100% | 2.030 | 2.030 | 2.030 |
| 90% | 2.030 | 1.945 | 2.031 |
| 80% | 2.030 | 1.872 | 2.034 |
| **70%** | **2.029** | **1.791** | 2.044 |
| 60% | 2.029 | 1.719 | 2.056 |
| 50% | 2.029 | 1.636 | 2.070 |
| 40% | 2.029 | 1.560 | 2.096 |
| 30% | 2.028 | 1.490 | 2.134 |

Oracle monotonically improves with more abstention; CQR-width *worsens* it — naive
width is uninformative about the injected irreducible noise (corr is weak by
construction). This is the gap the selective method must close.

## Frozen gate margins (now in the pre-reg)
- **X = 70%** (a priori).
- **Δ = 0.119** = ½·(random 2.029 − oracle 1.791) @ 70%. Method must reduce retained
  MAE vs random by ≥ 0.119 at 70% retention.
- **G-SHIFT** — naive worst-group 0.507 (gap 0.393); method must reach worst-group
  **≥ 0.704** and be within ±7pp of nominal.

## Layer 1b — real public task (21-session forward log return)
Same pattern on real data, milder (real 2022 shift < the ×1.8 synthetic amplification):
| metric | naive | CQR |
|---|---|---|
| marginal | 0.884 | 0.860 |
| base regime | 0.902 | 0.884 |
| shift regime | 0.850 | 0.814 |
| worst-group \| shift | 0.734 (mom_decile 0) | 0.775 (industrials) |
| width | 0.281 | 0.248 |

Confirms the conditional/regime gap is not an artifact of the synthetic generator —
it appears on a genuinely hard public regression with a real regime shift. (No
recoverability mask on 1b, so G-MECHANISM is 1a-only, as pre-registered.)

## 4-lens adversarial check
- **Reproduce** ✓ — single seeded driver; synthetic target bit-for-bit deterministic
  in seed (verified P0); learner hyper-params frozen in `learners.py`.
- **Leakage** ✓ — learners see only `FEATURE_COLS` (16 public PIT features; generative
  internals and group labels excluded by construction). Calibration fold (2021) is
  disjoint from train and test; every feature lagged ≥1 session. Walk-forward over
  time, never random K-fold across the shift. G-NULL (label-shuffle) is run in P2.
- **Statistics** ✓ — bands from sector×month clustered bootstrap, not row bootstrap;
  worst-group restricted to groups with n ≥ 50 (no thin-slice artifacts). Multi-seed sd
  reported alongside.
- **Mechanism** — the *baseline* read makes no mechanism claim; the oracle curve merely
  shows headroom exists. G-MECHANISM (does the method's abstention coincide with the
  recoverable mask, controlling for vol_63) is tested in P2, not here.

## Verdict → P2
Baselines establish (i) a large, real conditional-coverage failure under shift and
(ii) that naive UQ cannot exploit the abstention headroom. Margins frozen. **Proceed to
P2**: group-conditional (Mondrian) + shift-robust (weighted / ACI) selective CQR;
resolve G-PRIMARY / G-SHIFT / G-MECHANISM / G-NULL.
