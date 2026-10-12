# Pre-registration — P7 *cross-axis level combination* for alpha-space group adaptation (2026-06-30)

> **Gate-first.** Written and frozen **before** the new variants run. A **fresh pre-registered
> experiment** following the P6 reversal finding (`p6_alphaspace_results.md` §Next #3), **not** a retune
> of any frozen knob (γ, the level grid Λ, the clamps, ρ are all fixed here before any run). The seven
> prior variants are **untouched** and re-run head-to-head. Paired with `reports/p7_levelcomb_results.md`.

## The finding this targets
P6 established that **alpha-space** adaptation on the single coupled shape resolves the P5 width-space
base over-coverage (base 0.966→0.900) at the lineage's lowest width. But it surfaced a **reversal**: on
the adversarial 1a the *group* alpha variant did **not** beat the *global* one on worst-group|shift
(`cond+qACI` global **0.683** → `cond+gqACI` group **0.655**), and the **worst group changed axis**
(qACI worst = sector=tech; gqACI worst = mom_decile=0.0). That axis-change is the signature of
**cross-axis additive cancellation**: P6's `cqr_conditional_gqaci` combines the three axes' per-group
level offsets **additively** (`α_eff = α + a + Σ_k b[k,ℓ_k(x)]`), so for a point that is *easy on two
axes but hard on one*, the easy axes' tightening offsets (`b>0`) cancel the hard axis's widening one
(`b<0`), and the point never gets the width its hard axis demands.

**P7 tests two candidate fixes, both on the same fitted level family, both leaving the group-offset
*update* untouched (only the combination / rate changes):**

## The variants (frozen spec)
| # | Variant (`conformal.*`) | Change vs P6 `cqr_conditional_gqaci` | role |
|---|---|---|---|
| 8 | `cqr_conditional_gqaci_max` — **cross-axis MAX (headline)** | combine axes by the **most-conservative (widest)** axis, not the sum: `α_eff(x)=clip(α+a+min_k b[k,ℓ_k(x)], 0.01, 0.50)`. Removes cross-axis cancellation — a point is as wide as its worst axis demands. The level-space analogue of the P2.1 *union*, on the **single** fitted shape (no per-group sparsity). | primary |
| 9 | `cqr_conditional_gqaci_asym` — **asymmetric rate (secondary)** | additive combination **unchanged**; the per-group offset updates **slower to tighten than to widen** — a tightening step (`b` increasing) is scaled by **`ρ=0.5`**, a widening step is full-rate. (Adaptive-conformal rationale: under-coverage is the costly direction; `ρ=0.5` = the canonical "halve the rate when removing the safety margin", not tuned.) | report |

Both reuse P6's level family `t_τ(x)` (frozen grid `Λ`, monotone rearrangement, linear-in-level
interpolation), the same `α=0.10`, **same `γ=0.05`**, the same anti-windup offset clamp `[−0.09,0.40]`,
the same global `a` update, the same per-group `b` *update rule* (toward the session marginal `err_t`),
the same splits/learners/design. **`min`-vs-`sum` combination (variant 8) and `ρ` (variant 9) are the
only new knobs, frozen here.** The P6 `cqr_conditional_gqaci` (additive, symmetric) is the direct
control — variant 8 changes only the combination, variant 9 only the rate, so each isolates one
hypothesis. The six prior frozen variants are byte-identical and re-run for the table.

## Data / splits (identical to P1–P6)
1a known-ceiling synthetic (gated) + 1b real 21d forward return (external read); PIT panel; train
`<20210101` | calib 2021 | test `≥20220101`; severe 2022 drawdown interleaved with base; cluster unit
sector×calendar-month.

## Baselines (computed first, same table)
The full lineage is re-run head-to-head. The bars the headline `cqr_conditional_gqaci_max` must clear:
the **P6 alpha-space pair it is built to improve** — global `cond+qACI` worst-group|shift **0.683**
(the value the additive group term failed to beat) and additive group `cond+gqACI` **0.655** — and the
standing project bars (base nominal 0.90±0.05; width < union 9.076; worst-group floor 0.704).

## Pre-committed gates (numbers frozen)
| Gate | Statistic | Ship the headline (`cqr_conditional_gqaci_max`) if |
|---|---|---|
| **G-GROUP-HELPS (primary — the hypothesis)** | worst-group\|shift, multi-seed (1a), `gqaci_max` vs the two P6 alpha variants **in the same run** | `gqaci_max` worst-group\|shift **> both** `cond+qACI` (global, ~0.683) **and** `cond+gqACI` (additive, ~0.655) — i.e. removing cross-axis cancellation makes the group term **add value over the global offset**, reversing the P6 finding |
| **G-BASE-CALIB (co-primary)** | marginal base coverage, multi-seed (1a) | within **±τ=0.05** of 0.90 — the max combination must **not** reintroduce over-coverage |
| **G-EFFICIENCY (co-primary)** | mean width vs **union**, multi-seed (1a) | **strictly tighter than the P2.1 union** (9.076) — the max combination is wider than additive, must still stay under the union |
| **G-SHIFT-COND (report)** | worst-group\|shift + shift-marginal, multi-seed (1a) | worst-group ≥ **0.704** floor **and** shift within **±0.07** — the project SHIP bar; reported (whether the combination fix also clears the horizon wall) |
| **G-CALIB-MULTI / G-PRIMARY(i) / G-NULL** | per-axis gaps, selective retained-MAE, shuffle | reported as in P6; selective layer unmodified |

**Verdict rule (pre-committed).** P7's *direction* ships iff **G-GROUP-HELPS AND G-BASE-CALIB AND
G-EFFICIENCY** pass on 1a (the cross-axis cancellation is the cause, and removing it makes group beat
global without breaking base or efficiency). A **full SHIP** [redacted: refers to a non-public confirmation run] additionally requires
**G-SHIFT-COND** (the 0.704 floor). If `gqaci_max` clears G-GROUP-HELPS + the two co-primaries but
misses the floor, the honest verdict is **QUALIFIED** — *cross-axis cancellation explained the P6
reversal and the union-in-level-space fixes it, but the adversarial-1a worst-group floor remains the
family horizon wall.* No frozen knob (γ, Λ, clamps, ρ, the min combination) tuned post-hoc to
manufacture a pass. The asymmetric-rate arm (variant 9) is **reported** alongside, not separately gated.
[redacted: refers to a non-public confirmation run]

## Seeds / reproducibility
Seeds **{0,1,2}**, target 90%, multi-seed mean ± sd; seed-0 clustered band on the headline base. Run:
`python src/run_p7.py` → 1a (gated) + 1b (read). The six prior frozen variants must reproduce the P6
table bit-for-bit (a regression check that P7 left them untouched). Mirror into
`reports/p7_levelcomb_results.md` without editing the gates.

## 4-lens adversarial check (fill in results doc, after)
- **Reproduce** — `gqaci_max`/`gqaci_asym` bit-for-bit on re-run; the seven prior variants reproduce P6.
- **Leakage** — level family fit only on calibration `E` + group indicators; online `a,b,α_eff` at
  session `s` use only sessions `<s`; `y_s` revealed after the interval (PIT). G-NULL still collapses.
- **Statistics** — base band from the clustered (sector×month) bootstrap; worst-group `min_n=50`;
  multi-seed mean±sd; the G-GROUP-HELPS comparison is per-seed paired (same data, same fold).
- **Mechanism** — does `gqaci_max` actually relocate width onto the previously-cancelled hard-on-one-axis
  points (check the worst group reverts to/stabilises a sensible axis, and per-axis full-base gaps stay
  two-sided), or is any worst-group gain just uniform widening (which G-EFFICIENCY / G-BASE-CALIB would
  catch)?
