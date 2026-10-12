# Pre-registration — P6 *alpha-space* group-conditional adaptation on the single-regression shape (2026-06-30)

> **Gate-first.** Written and frozen **before** the new variants run. This is the **fresh
> pre-registered experiment** logged as the single open methods question in
> `p5_archival_note.md` §8 and `p23_adaptive_results.md` §Next — **not** a retune of any frozen P2/P2.1/
> P2.2/P2.3 knob. The five existing variants (`aci_cqr`, `aci_cqr_multi`, `cqr_conditional`,
> `cqr_conditional_aci`, `cqr_conditional_gaci`) are **untouched** and re-run head-to-head for the
> table. Paired with `reports/p6_alphaspace_results.md` (filled after the runs, gates not edited).

## Question
The P5 ladder closed on **one** named design tension (`p5_archival_note.md` §5.3/§8): a **width-space**
additive online offset on a **single fitted** conditional shape (`cqr_conditional_gaci`, P2.3)
**over-covers the base** (1a base 0.966 vs nominal 0.90) when severe-shift and base sessions interleave —
the offset `c` inflates under shift and relaxes only at rate γ·scale per session, so the more-numerous
base sessions inherit the inflated width. The frozen **alpha-space** `aci_cqr` does **not** show this
(base 0.904), because it adapts the *quantile level* (reading the conformity-score quantile function)
rather than adding raw width — but it requires a **per-group empirical-quantile object** (the Mondrian
per-group sparsity, `min_n≥50`, global fallback) that the coupled single-regression shape was built to
avoid.

**P6 tests the reconciliation:** *can the alpha-space (quantile-level) online update be run against the
**single-regression conditional shape** — adapting the level per group, with **no** per-group empirical
quantiles and **no** Mondrian sparsity — and thereby keep `cqr_conditional`'s width win and group-
conditional shift gain **while removing the base over-coverage** that closed the P5 ladder?*

## The variants (frozen spec)

**Conditional quantile-level family (the new object).** Instead of one pinball regression of the CQR
conformity scores `E` on the multi-axis group indicators `φ(x)` at level `1−α` (P2.2's
`cqr_conditional`), fit the **same additive regression at a frozen grid of levels**
`Λ = {0.50, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99}`:
`t_τ(x) = φ(x)ᵀβ_τ`, one `QuantileRegressor(quantile=τ, alpha=0.0, solver="highs", fit_intercept=False)`
per `τ∈Λ`, each fit on the **full** calibration set (drop-first multi-axis design, identical to
`cqr_conditional`). This is the conditional analogue of `aci_cqr`'s per-group empirical quantile
function `Es_grp[g]`, but conditioned on `x` through the regression — **every level's fit uses all
calibration rows + group indicators, so there is no per-group sparsity.** Per point, the predicted
thresholds across `Λ` are **monotone-rearranged** (sorted ascending in level; Chernozhukov–Fernández-Val–
Galichon 2010) to remove quantile crossing, then a per-point effective level reads `t` by **linear
interpolation in level** between the bracketing grid nodes (clamped to `[1−max Λ, 1−min Λ] = [0.01,
0.50]` in miscoverage units).

Two new online variants carry this family with an **alpha-space** (quantile-level) ACI update — the
*exact form and rate* of the frozen `aci_cqr` (`α ← α + γ(α₀ − err)`, **`γ=0.05`**, no width scaling),
the only change being that the per-group quantile is read from the regression family `t_τ(x)` instead of
per-group empirical scores:

| # | Variant (`conformal.*`) | Online correction | analogue of |
|---|---|---|---|
| 6 | `cqr_conditional_qaci` — **global alpha** | one global effective level `α_eff,t`, `α_eff ← clip(α_eff + γ(α − err_t), 0.01, 0.50)`; interval reads `t_{1−α_eff}(x)` from the family | width-space global `cqr_conditional_aci` |
| 7 | `cqr_conditional_gqaci` — **group alpha (headline)** | additive level decomposition `α_eff(x) = clip(α + a + Σ_k b[k,ℓ_k(x)], 0.01, 0.50)`; global `a ← a + γ(α − err_t)`, per-(axis,level) `b[k,ℓ] ← b[k,ℓ] + γ(err_t − err_{k,ℓ,t})` (each group nudged to the **session marginal** `err_t`, so `a` owns the average and `b` the residual — no triple-counting across axes); interval reads `t_{1−α_eff(x)}(x)` | width-space group `cqr_conditional_gaci` (P2.3) |

**The 2×2 (the experimental design).** P2.2/P2.3 give the *width-space* row (global `cond+ACI`, group
`cond+gACI`); P6 adds the *alpha-space* row (global `cond+qACI`, group `cond+gqACI`) on the **same fitted
shape**. The contrast (width vs alpha) × (global vs group) isolates whether the base over-coverage is an
artifact of the **width-space** correction (P6's pre-registered claim) rather than of group-conditionality
or of the conditional shape itself.

**Unification.** With `a≡0, b≡0` the family at the static node `τ=0.90` recovers `cqr_conditional` (P2.2
static). `cqr_conditional_qaci` (global) is the alpha-space analogue of `cqr_conditional_aci`;
`cqr_conditional_gqaci` is the alpha-space analogue of `cqr_conditional_gaci`. Same `α=0.10`, **same
`γ=0.05`**, same splits, same learners, same multi-axis design, same `lo≤hi` guard. The selective layer
`u_i` is **unchanged** (independent of the interval) → G-PRIMARY(i)/G-MECHANISM/G-NULL carry over
identically.

**Anti-windup (faithful to `aci_cqr`).** `aci_cqr` carries the **clamped level itself** as state, so it
cannot wind up; the width-space variants floor `c,d` at `−scale` for the same reason. The alpha-space
analogue therefore carries the offsets clamped to the level range: `a` and each `b[k,ℓ]` are clamped to
`[1−maxΛ−α, 1−minΛ−α] = [−0.09, 0.40]` after each update, and `α_eff` is re-clamped to `[0.01,0.50]`.
Without this the offset would accumulate during a long shift and relax only slowly — reintroducing in
alpha-space the very windup the experiment is testing whether alpha-space avoids. This is part of the
frozen spec (a definitional faithfulness choice, fixed before any run), not a tunable.

**New knobs frozen here:** the level grid `Λ`, monotone rearrangement + linear-in-level interpolation,
the level/offset clamps above, and the level-space additive decomposition `a + Σb`. `γ` is **not** new
(=0.05, identical to `aci_cqr`). **No per-group learning-rate tuning; no post-hoc grid edit.**

## Data / splits (identical to P1–P2.3)
- Layers: **1a** known-ceiling semi-synthetic (the gated layer) + **1b** real 21d forward return
  (external read). PIT panel; train `report_date<20210101` | calib 2021 | test `≥20220101`; shift =
  2022 drawdown sessions interleaved with base in the test stream.
- Cluster unit for inference: sector × calendar-month (clustered bootstrap, `eval.clustered_bootstrap`).

## Baselines (computed first, in the same table)
The full frozen lineage is re-run head-to-head (`aci_cqr`, `aci_cqr_multi`, `cqr_conditional`,
`cqr_conditional_aci`, `cqr_conditional_gaci`). The bars P6's headline `cqr_conditional_gqaci` must clear
are the **P2.3 values it is built to improve**: 1a worst-group|shift floor **0.704** (gACI 0.651, MISS),
1a base nominal **0.90±0.05** (gACI **0.966, MISS** — the target of this experiment), width **< union**
(gACI 8.35 < 9.08, PASS).

## Pre-committed gates (numbers frozen)
| Gate | Statistic | Ship the headline (`cqr_conditional_gqaci`) if |
|---|---|---|
| **G-BASE-CALIB (co-primary — the point of P6)** | marginal **base** coverage, multi-seed (1a) | within **±τ=0.05** of 0.90 — i.e. alpha-space adaptation removes the width-space base over-coverage (gACI 0.966) the P5 tension named |
| **G-SHIFT-COND (co-primary)** | worst-group\|shift **+** shift-marginal, multi-seed (1a) | worst-group\|shift **≥ WG_FLOOR=0.704** **and** shift-marginal within **±τ_shift=0.07** of 0.90 |
| **G-EFFICIENCY (co-primary)** | mean width vs **union**, multi-seed (1a) | **strictly tighter than the P2.1 union** — the width win survives going alpha-space-online |
| **G-ALPHA-FIX (report — the mechanism diagnostic)** | base coverage `cqr_conditional_gqaci` vs `cqr_conditional_gaci`, multi-seed (1a) | **prediction:** alpha-space base is **closer to nominal** than width-space (\|gqACI base −0.90\| < \|gACI base −0.90\|). Directional mechanism confirmation of the §5.3 diagnosis; **reported, not a ship gate** |
| **G-CALIB-MULTI (report)** | per-axis SIGNED worst gap, **full base** + retained-70% (1a) | full-base two-sided ±τ on all 3 axes is the calibration check; the retained-set sector residual stays the **known P2.2 selective×conditional interaction** — reported, **NOT a ship-blocker** (selective layer unmodified) |
| **G-PRIMARY(i) / G-NULL** | selective retained-MAE + shuffle | unchanged (selective `u_i` independent of the interval) |

**Verdict rule (pre-committed).** P6 **ships** iff **G-BASE-CALIB passes AND G-SHIFT-COND passes AND
G-EFFICIENCY passes** on 1a — exactly the three-way combination no prior variant delivered at once, and
the specific combination the P5 tension said was unreached (the width-space family had EFFICIENCY but
failed BASE-CALIB and SHIFT-COND). If the headline clears **G-BASE-CALIB + G-EFFICIENCY** but still
misses the worst-group floor on the deliberately-severe synthetic, the honest verdict is
**QUALIFIED** — *the named design tension is resolved (alpha-space fixes the over-coverage) but the
adversarial-1a worst-group floor remains horizon-bounded* — reported as such, **with no retuning of any
frozen knob (γ, Λ, the clamp) to manufacture a pass.** Layer-1b is an external read (reported, not
gated). [redacted: refers to a non-public confirmation run]

## Seeds / reproducibility
Seeds **{0,1,2}**, target 90%, each number multi-seed mean ± sd; seed-0 clustered band on the headline
base coverage. Run: `python src/run_p6.py` → 1a (gated) + 1b (read). Mirror into
`reports/p6_alphaspace_results.md`. Every number reproduces bit-for-bit from the seeded driver.

## 4-lens adversarial check (fill in results doc, after)
- **Reproduce** — numbers bit-for-bit from the seeded run; the frozen five unchanged from P2.3?
- **Leakage** — the level family is fit only on calibration `E` + group indicators; online `a,b,α_eff`
  at session `s` use only sessions `<s`; `y_s` revealed after the interval is emitted (PIT). G-NULL
  (shared selective layer) still collapses?
- **Statistics** — base-coverage band from the clustered (sector×month) bootstrap, not row bootstrap;
  worst-group floor read with the same `min_n=50` as every prior phase.
- **Mechanism** — does G-ALPHA-FIX actually show alpha-space pulling base toward nominal (the §5.3
  claim), or is any base improvement a width side-effect / a grid-clamp artifact? Check the realised
  `α_eff` range is interior to the clamp (not pinned at 0.01/0.50, which would mean the grid, not the
  adaptation, set the width).
