# Pre-registration — P2.3 *per-group adaptive* conditional conformal (2026-06-30)

> **Gate-first.** Written **before** the new variant is run. The frozen `aci_cqr`,
> `aci_cqr_multi` (P2.1 union), `cqr_conditional` and `cqr_conditional_aci` (P2.2) are
> **untouched** — this is a fifth, *additive* variant tested head-to-head, never a retune.
> Paired with `reports/p23_adaptive_results.md` (filled after the runs).

## The qualified item this targets
P2.2 (`p22_conditional_results.md`) shipped the **width** win (calibrated additive threshold,
−32% vs the union) but ended **QUALIFIED** on two residuals, one of which P2.3 is built to fix:

- **Shift (the target).** The *static* conditional method is calibrated **in-distribution** by
  design, so it **collapses under the drawdown** (worst-group|shift **0.433**, marginal 0.564).
  The reported P2.2 hybrid `cqr_conditional_aci` adds a single **global** online offset — it lifts
  the *margin* (0.564 → 0.702) but a **global offset cannot repair group-specific shift**, so
  worst-group stays **0.599 < the 0.704 floor**. That is the precise, diagnosed gap.
- **Retained-set sector (NOT targeted — pre-committed).** P2.2's retained-70% gate flagged sector
  (+0.078) while the method held two-sided ±τ on **all three axes on the full base set**. That was
  diagnosed as a **selective×conditional interaction**: downstream abstention keeps the easier
  low-`u` names within a sector, so the *retained subset* over-covers — a property of the selective
  layer, which P2.3 **does not modify**. P2.3 is **not** expected to move it, and per the rule below
  it is **reported, not held against P2.3**.

**Diagnosis (pre-committed):** the shift fix must be **group-conditional**, not global — the
conditional *shape* (P2.2's calibrated additive threshold, the width win) carried online with a
**per-group** adaptive correction (the shift fix `aci_cqr`/`aci_cqr_multi` gave, but on the
calibrated shape rather than per-group empirical quantiles or a union).

## The variant (frozen spec)
**Group-adaptive conditional CQR** (`conformal.cqr_conditional_gaci`) — `cqr_conditional`'s static
shape `t(x)=φ(x)ᵀβ` (the frozen pinball-regression construction, reused unchanged) carried online
with an **additive miscoverage decomposition** over time-ordered sessions:

1. Static shape `t(x)` exactly as `cqr_conditional` (drop-first group-indicator design, pinball QR
   at `1−α`, `solver="highs"`, no L1). **Reused, not re-fit.**
2. Two online additive corrections, both initialised 0, carried across sessions, both updated in
   **width-space** (the analogue `cqr_conditional_aci` uses; `scale = median|t|`):
   - a **global** offset `c` — `c ← max(c + γ·scale·(err_t − α), −scale)` (under-coverage widens);
   - a **per-(axis, level) deviation** offset `d[k,ℓ]` over **full** group levels (every level,
     incl. the reference, initialised 0) — `d[k,ℓ] ← max(d[k,ℓ] + γ·scale·(err_{k,ℓ,t} − err_t), −scale)`.
     Each group is nudged toward the **realised session marginal** `err_t`, so `c` owns the average
     drift and `d[k,ℓ]` owns only each group's *residual* — no triple-counting of the marginal
     across the three axes.
3. Per-point correction `Δ(x) = c + Σ_k d[k, ℓ_k(x)]` (additive across axes, matching the linear
   class). Interval `[qlo − t − Δ, qhi + t + Δ]` (guard `lo ≤ hi`). At session `s` only `c, d`
   carried from sessions `< s` are used; `y_s` is revealed after the interval is emitted → **PIT-valid**.

**Unification (the point).** `Δ≡0` recovers `cqr_conditional` (P2.2 static); `d≡0` recovers
`cqr_conditional_aci` (P2.2 global hybrid); the per-group `d` is the `aci_cqr`/`aci_cqr_multi` shift
fix, applied additively to the calibrated shape instead of via per-group quantiles or a union.

**New knobs (frozen here):** the `c + Σd` decomposition, full-level online dummies, group target =
session marginal `err_t`, floors at `−scale`. Everything else (`α=0.10`, **`γ=0.05` shared** by `c`
and `d`, the splits, the learners, the static-shape knobs) is **identical to P2/P2.1/P2.2**. No
per-group learning-rate tuning.

## Pre-committed gates (decision rules)
| Gate | Statistic | Ship the variant if |
|---|---|---|
| **G-SHIFT-COND (primary)** | `cqr_conditional_gaci` worst-group\|shift + shift-marginal, multi-seed (1a) | worst-group\|shift **≥ WG_FLOOR=0.704** (the floor the static method 0.433 and the **global**-offset hybrid 0.599 both missed) **and** shift-marginal within **±τ_shift=0.07** of 90% — i.e. the group-conditional correction repairs group-specific shift a global offset cannot |
| **G-EFFICIENCY (co-primary)** | mean width, gACI vs **union**, multi-seed (1a) | **strictly tighter than the P2.1 union** — the P2.2 width win must survive adding online adaptation |
| **G-BASE-CALIB (co-primary)** | marginal base coverage, gACI, multi-seed (1a) | base within **±τ=0.05** of 90% — online adaptation must **not** break the in-distribution calibration the static shape has |
| **G-CALIB-MULTI (report)** | per-axis two-sided gap, **full base** + retained-70% (1a) | **full base** two-sided ±τ on all 3 axes is the calibration target and is reported as the headline calibration check. The **retained-set sector residual** is reported as the **known P2.2 selective×conditional interaction** — pre-committed as **NOT a ship-blocker for P2.3** (the selective layer is unmodified here); do **not** claim P2.3 fixes it |
| **G-PRIMARY (i) / G-NULL** | selective retained-MAE + shuffle | unchanged (the selective score `u_i` is independent of the conformal interval) |

**Verdict rule (pre-committed).** P2.3 **ships** iff **G-SHIFT-COND passes** (worst-group\|shift ≥
0.704 and shift-marginal within ±τ_shift) **and G-EFFICIENCY passes** (tighter than the union)
**and G-BASE-CALIB passes** (base within ±τ). That is exactly what no prior variant delivered at
once: calibrated in-distribution (`cqr_conditional`) **and** tighter than the union (`cqr_conditional`)
**and** group-conditionally shift-robust (`aci_cqr_multi`, but that one over-covers). The
retained-set sector residual is **reported**, not gated (justification above, stated up front). If
gACI does **not** clear the worst-group floor under shift, P2.3 is **QUALIFIED/REJECT**, reported
honestly — no retuning of any frozen knob to manufacture a pass. [redacted: refers to a non-public confirmation run]

## Run
`python src/run_p23.py` → 1a seeds {0,1,2} + 1b; head-to-head single-axis ACI / multi-axis union /
`cqr_conditional` (static) / `cqr_conditional_aci` (global hybrid) / **`cqr_conditional_gaci`
(P2.3)**. Mirror into `reports/p23_adaptive_results.md`. [redacted: refers to a non-public confirmation run]
