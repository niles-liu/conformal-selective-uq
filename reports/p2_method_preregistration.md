# Pre-registration — P2 method (2026-06-30)

> **Gate-first.** Written **before** the method is run. The gate *thresholds* are
> already numeric (frozen at P1 from the method-blind baseline read — see
> `p1_baselines_preregistration.md`); this doc freezes the **method specification**
> and every **method-specific knob** so none can be tuned post-hoc to pass a gate.
> Paired with `reports/p2_method_results.md` (filled after the runs).

## What P1 established (the bar to clear)
Naive split-conformal and unconditional CQR hold nominal coverage **in-regime**
(base 0.902) but **fail conditionally under the documented 2022 shift** (coverage
0.580, worst-group **0.507**), and naive interval-width abstention is **uninformative**
about the injected irreducible noise (retained MAE 2.044 @ 70% vs random 2.029) while
an **oracle** that knows `irreducible_sigma` has large headroom (1.791). P2 must close
both gaps with a single, pre-registered, public-feature-only method.

## Method specification (frozen)

All learners see **only** `panel.FEATURE_COLS` (16 public PIT features). Generative
internals and the group labels are never inputs. Base learners + hyper-params are
frozen in `src/learners.py` (no per-phase tuning). Splits identical to P1:
train `rd<20210101` → calibrate `2021` (calm, pre-shift) → test `rd≥20220101`.

1. **Group-conditional (Mondrian) CQR** — `conformal.mondrian_cqr`. CQR intervals with a
   **per-group** conformal quantile `Q_g` (group = momentum decile; see knob freeze),
   global-`Q` fallback for groups with `< MIN_N` calibration points. Targets
   G-PRIMARY(ii) static conditional coverage.
2. **Adaptive Conformal Inference, group-conditional (the headline G-SHIFT method)** —
   `conformal.aci_cqr`. Gibbs–Candès online adaptation (Mondrian variant) layered on the
   CQR conformity scores: per group `g`, an effective miscoverage level `α_{g,t}` is
   carried across **time-ordered test sessions**, updated after each session by
   `α_{g,t+1} = α_{g,t} + γ·(α − err_{g,t})` where `err_{g,t}` is that group's realized
   session miscoverage. The interval at session `t` uses only `α_{g,t}` (carried from the
   past) — the realized `y_t` is revealed **after** the interval is emitted, so this is
   PIT-valid (you predict before the close, then observe). Targets G-SHIFT: it tracks and
   corrects the temporal coverage drift the static methods cannot.
3. **Weighted (covariate-shift) CQR — ablation only** — `conformal.weighted_cqr`,
   Tibshirani et al. 2019 likelihood-ratio weights (calib-vs-test classifier on features).
   Reported as a comparator; **not** the gate-deciding method.

**Selective layer** (`src/selective.py`) — the abstention score is a **per-name calibration
residual scale** `u_i`: the shrunk MAD of name `i`'s residuals on the 2021 calibration fold
(`u_i = (n_i·MAD_i + K·MAD_global)/(n_i + K)`). Rationale (pre-committed): the injected
overlay is **per-name persistent** (λ_i fixed, overlay AR(1) ρ=0.95), so a name's *own*
historical error scale is a public, PIT-legal proxy for its irreducible noise — exactly the
signal raw interval width missed in P1. Abstain on highest `u_i` first; retained-set MAE is
the risk–coverage curve. Comparators: random abstention and the 1a oracle (`irreducible_sigma`).

## A-priori knob freeze (chosen by principle, not by result)
| Knob | Value | Justification (a priori) |
|---|---|---|
| Mondrian / ACI group dimension | **`mom_decile`** | the allocator tilts on momentum, so momentum buckets are its natural conditioning axis — fixed before seeing per-dim coverage |
| ACI step size `γ` | **0.05** | Gibbs–Candès default; not swept |
| Mondrian min group size `MIN_N` | **50** | same threshold the P1 worst-group statistic already uses |
| Name-uncertainty shrinkage `K` | **20** | ~one quarter of a year of calib sessions; standard empirical-Bayes-ish shrink, not swept |
| Target coverage / retention | **90% / 70%** | from P1 |

The **G-SHIFT statistic is held to the P1 standard**: worst-group coverage across **all
three** dimensions (sector / cap_tier / mom_decile), not just the conditioning axis — so the
method is not graded only on what it directly optimizes.

## Pre-committed gates (thresholds frozen at P1; decision rules here)
| Gate | Statistic | Ship if |
|---|---|---|
| **G-PRIMARY (i)** useful abstention | headline-method retained MAE vs **random** @ 70% retention | reduction **≥ Δ = 0.119**, multi-seed clustered band excludes 0 |
| **G-PRIMARY (ii)** conditional coverage | per-group coverage on the **retained** set (all 3 dims, n≥50) | every group within **±τ = 5pp** of 90% |
| **G-SHIFT** | worst-group coverage on the test-drawdown slice (all 3 dims) | worst-group **≥ 0.704** *and* within **±τ_shift = 7pp** of nominal |
| **G-MECHANISM (1a)** | (a) AUC of `−u_i` predicting `recoverable_mask_i`; (b) Spearman(`u_i`, `irreducible_sigma_i`) | **AUC ≥ θ = 0.70** *and* **ρ ≥ θ′ = 0.40**, bands clear; **and** the AUC/ρ **survives partialling out per-name `vol_63`** (the confound check) |
| **G-NULL** | shuffle `y` within date → re-derive `u_i` → G-PRIMARY(i) advantage | retained-MAE advantage over random **collapses into the random band** (no leakage) |

## Adversarial 4-lens check (to fill in results, after)
- **Reproduce** — every number bit-for-bit from the seeded run; γ/K/MIN_N/dim as frozen above.
- **Leakage** — ACI reveals `y_t` only *after* emitting session `t`'s interval; calibration
  fold disjoint from test; `u_i` from calib residuals only; features `known_at ≤ decision`.
- **Statistics** — sector×month clustered bootstrap bands; worst-group n≥50; multi-seed sd.
- **Mechanism** — does `u_i`↔recoverability survive controlling for `vol_63`, or is it a
  volatility confound? (Reported explicitly; the gate requires it to survive.)

## Run
`python src/run_p2.py`  → 1a seeds {0,1,2} (G-PRIMARY/G-SHIFT/G-MECHANISM/G-NULL) + 1b
(G-PRIMARY/G-SHIFT/G-NULL). Mirror into `reports/p2_method_results.md`.
