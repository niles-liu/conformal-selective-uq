# Pre-registration — P9 *selective conformal risk control* in regression (2026-07-01)

> **Gate-first.** The variant spec + gate STRUCTURE are frozen here before the method runs;
> the numeric thresholds are frozen from the **method-blind baseline read**
> (`python src/run_p9.py --baseline`, reported in the "Baselines" block below) and are
> **not** edited after the method is seen. Paired with `reports/p9_riskcontrol_results.md`.
> This is the **open variant SPEC §6 named** ("select-then-calibrate in regression"), scouted
> as SPEC §11 P9 — SCRC (arXiv 2512.12844) is published but stays classification. Not a retune
> of any P1–P7 knob; the frozen interval variants are untouched.

## Question
Can we upgrade the selective layer from **coverage-only** to a **monotone bounded risk** with a
formal **two-λ select-then-calibrate** guarantee — i.e. control a *proper* retained-set risk
(capped miss-magnitude, not just the 0/1 miss rate) on the abstention-retained pool, **held under
the documented shift**, where a single globally-calibrated radius does not?

## The risk (frozen definition — `src/riskcontrol.py`)
Interval `C_λ(x) = [qlo(x) − λ, qhi(x) + λ]` is the frozen CQR base interval (same mean + lo/hi
LightGBM learners, `α=0.10`) with a conformal radius `λ ≥ 0`. Two **monotone (non-increasing in λ),
bounded (B=1)** losses:
- **miscoverage** `L = 1{y ∉ C_λ(x)}` — recovers selective coverage;
- **capped miss-magnitude** `L = clip( dist_outside(y, C_λ(x)) / W0 , 0, 1 )` — the P9 upgrade
  beyond 0/1 coverage (penalises *how far* outside, not merely whether). `W0` = **median base-regime
  CQR width**, frozen from the baseline read (seed-mean), so the loss is unit-free and in `[0,1]`.

The **headline risk is capped miss-magnitude**; miscoverage is reported alongside (it ties the story
to the coverage lineage). Risk = mean loss over the evaluated pool.

## The method (frozen spec — two λ, select-then-calibrate)
- **λ_sel (selection):** retain the fraction **r = 0.70** (a priori, = P2 `X_RET`) of names with the
  lowest per-name uncertainty `u_i` (the frozen `selective.name_uncertainty` score — a **covariate-side**
  rule, so exchangeability is preserved *within* the retained pool). Abstain on the rest.
- **λ_risk (calibration):** the **Conformal Risk Control** radius (Angelopoulos–Bates–Fisch–Lei–
  Schuster 2023) computed **on the retained calibration fold**: smallest λ on a frozen grid with
  `(n/(n+1))·R̂(λ) + B/(n+1) ≤ α_risk`, `n` = retained-calib size. Calibrating on the *selected* fold
  (not the full fold) is the SCRC point: the retained pool's residuals are smaller than the pooled
  set's, so a full-fold radius over-covers/over-widens the retained pool; the select-then-calibrate
  radius is tighter at the same guaranteed risk.

Two variants (both on the SAME selection + CQR base; the only difference is static vs online radius):
| # | Variant (`riskcontrol.*` / `run_p9`) | Radius | Purpose |
|---|---|---|---|
| 1 | **`scrc`** (batch, headline for base) | one `λ_risk` from CRC on retained calib (static) | the finite-sample CRC guarantee holds in the exchangeable **base** regime |
| 2 | **`scrc_aci`** (online, headline for shift) | risk-controlled radius carried over time-ordered sessions: after each session's retained points reveal their loss, `λ ← max(λ + η·(R̂_sess − α_risk), 0)` (widen when the realised session risk exceeds target) | holds the retained risk **under shift** (the static radius cannot — calib is pre-shift) |

`η` (online rate) is frozen at the ACI-family value analogue (**η = γ·W0 = 0.05·W0**, the width-space
step used by `cqr_conditional_aci`); no per-phase tuning. `y_t` is revealed only after session `t`'s
interval is emitted → PIT-valid. Unification: `scrc_aci` with `η=0` (or a stationary stream) reduces to
`scrc`.

## Data / splits (identical to P1–P7)
Layers **1a** (known-ceiling synthetic, the gated layer) + **1b** (real 21d fwd return, external read).
PIT panel; train `report_date<20210101` | calib 2021 (calm) | test `≥20220101` (2022 = amplified-overlay
shift, interleaved with base). Cluster unit for inference: sector × calendar-month (`clustered_bootstrap`).

## Baselines (computed first — method-blind, `run_p9.py --baseline`)
Retained pool = lowest-`u` 70% under a **global** conformal radius (calibrated on the full fold), vs
**random** selection, vs **no selection**. Numbers (mean over seeds {0,1,2} on 1a; seed 0 on 1b):

> **FROZEN FROM BASELINE READ** (`python src/run_p9.py --baseline`, 2026-07-01; 1a seeds {0,1,2}, 1b seed 0):
>
> | pool | cov\|base | cov\|shift | risk\|base | risk\|shift | width |
> |---|---|---|---|---|---|
> | **1a** full (no select) | 0.901 | 0.573 | 0.015 | 0.155 | — |
> | **1a** retained (u, global radius) | 0.928 | 0.607 | **0.010** | **0.132** | 6.218 |
> | **1a** retained (random) | 0.901 | 0.573 | 0.015 | 0.155 | 6.238 |
> | **1b** full | 0.884 | 0.814 | 0.020 | 0.045 | — |
> | **1b** retained (u, global radius) | 0.894 | 0.825 | 0.015 | 0.038 | 0.232 |
> | **1b** retained (random) | 0.884 | 0.814 | 0.020 | 0.045 | 0.248 |
>
> `W0` (median base-regime width) = **6.295 ± 0.182** (1a), **0.228** (1b).
>
> Reads that set the gates: (a) **u-selection is informative** — retained-u risk < random on both
> layers (1a base 0.010 < 0.015, shift 0.132 < 0.155). (b) **The retained pool is over-covered by the
> global radius** (1a base cov 0.928 > 0.90) → select-then-calibrate has room to *tighten* at a
> guaranteed risk. (c) **The severe-1a shift breaks static risk control** (retained shift risk 0.132 ≫
> base 0.010) — the online variant's target; on the mild 1b shift the static radius nearly holds
> (0.038), the familiar 1a-severe/1b-mild split.

## Pre-committed gates (thresholds frozen from the baseline read)
**Frozen targets:** `α_risk = 0.05` (the capped miss-magnitude target; achievable in 1a base — global
gives 0.010 — and clearly binding under 1a shift — 0.132). `τ_risk = 0.03` (shift band → ≤ 0.08).
`margin_shift = 0.03` (online must beat the static-radius shift risk by ≥ this). `η = 0.05·W0` (the
`cqr_conditional_aci` width-space step, per-run `W0`). CRC grid = `linspace(0, 3·Q_global, 121)` per run.

| Gate | Statistic | Ship if |
|---|---|---|
| **G-RISK-BASE (co-primary)** | `scrc` retained capped miss-magnitude risk, base regime, multi-seed (1a) | `≤ α_risk = 0.05` — the batch CRC guarantee holds in the exchangeable regime |
| **G-RISK-SHIFT (co-primary)** | `scrc_aci` retained risk, shift regime, multi-seed (1a) | `≤ α_risk + τ_risk = 0.08` **and** below the static `scrc` shift risk by `≥ margin_shift = 0.03` — online risk control holds under shift where the static radius fails |
| **G-NULL (co-primary)** | label-shuffle within date → risk *advantage of u-selection over random* (mixed test, global radius) | `|advantage| <` ½ the real advantage — collapses (no leakage) |
| **G-EFFICIENCY (report)** | `scrc` (CRC on **selected** calib) retained width vs **full-fold** CRC width at the same `α_risk` (1a) | `scrc` strictly tighter — the select-then-calibrate width win |
| **G-SELECT (report)** | retained-u risk vs retained-random risk (1a) | u-selection risk `<` random — selection is informative (mirrors G-PRIMARY(i)) |

**Verdict rule (pre-committed).** P9 **ships** iff **G-RISK-BASE ∧ G-RISK-SHIFT ∧ G-NULL** pass on 1a.
If `scrc` controls base risk (G-RISK-BASE) and beats the static radius under shift but misses the
absolute `α_risk + τ_risk` band on the deliberately-severe synthetic, the honest verdict is
**QUALIFIED** (online risk control demonstrably helps, but the adversarial-1a horizon wall bounds it —
the same family-wide wall P6/P7 hit), reported as such **with no post-hoc retune** of `r, α_risk, η,
W0, or the grid`. Layer-1b is an external read (reported, not gated). [redacted: refers to a non-public confirmation run]

## Seeds / reproducibility
Seeds **{0,1,2}**, target coverage 90%, `α_risk` as frozen; each number multi-seed mean ± sd; seed-0
clustered band on the headline retained risk. Run: `python src/run_p9.py`. Every number reproduces
bit-for-bit (seeded data, deterministic learners + CRC grid, seeded random-selection draws).

## 4-lens adversarial check (fill in results doc, after)
- **Reproduce** — numbers bit-for-bit from the seeded run; frozen P1–P7 variants untouched.
- **Leakage** — CRC radius fit only on retained *calibration* losses; online `λ` at session `s` uses
  only sessions `<s`; `y_s` revealed after the interval (PIT). Selection uses covariate-side `u_i` only.
  G-NULL collapses.
- **Statistics** — retained-risk band from the sector×month clustered bootstrap, not row bootstrap;
  multi-seed sd; random-selection averaged over draws with its own band.
- **Mechanism** — is the shift risk-control real online adaptation, or does `λ` just saturate the grid?
  Check the realised `λ` path is interior to the grid (moves with the shift, not pinned at the max).
