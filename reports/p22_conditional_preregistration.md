# Pre-registration — P2.2 *calibrated* multi-group coverage (2026-06-30)

> **Gate-first.** Written **before** the new variant is run. The frozen single-axis
> `aci_cqr` (dim=`mom_decile`) and the P2.1 union (`aci_cqr_multi`) are **untouched** — this
> is a third, *additive* variant tested head-to-head, never a retune. Paired with
> `reports/p22_conditional_results.md` (filled after the runs).

## The qualified item this targets
P2.1 logged a clean **one-sided** win and a **two-sided** miss:
- The multi-axis **union** `[min_k lo_k, max_k hi_k]` removes worst-group *under*-coverage on all
  axes (G-SHIFT improves, dominates single-axis) but does so by **over-covering** — on the
  already-calibrated 1a synthetic every axis lands **> 90% + τ** on the retained set and intervals
  widen **~12%** (1a) [redacted: refers to a non-public confirmation run]. So *calibrated* (two-sided ±τ)
  multi-group coverage was **not** achieved. [`p21_multiaxis_results.md` §G-COND-MULTI]

**Diagnosis (pre-committed):** the union over-covers *by construction* — taking the max/min of
three independently-calibrated intervals stacks three one-sided guarantees, so the result is
≥ each and strictly wider. The blunt instrument is the union, not the multi-axis goal. The fix is
to compute a **single coupled threshold per point** over the joint group class, calibrated *once*,
instead of unioning three separately-calibrated intervals.

## The variant (frozen spec)
**Conditional-conformal CQR via a group-indicator linear class** (`conformal.cqr_conditional`) —
the Gibbs–Cherian–Candès (2023) "Conformal Prediction with Conditional Guarantees" linear-class
construction, instantiated with the multi-axis group indicators as the basis:

1. Shared CQR pieces (frozen `_cqr_pieces`): mean + lo/hi quantile learners, calibration
   conformity scores `E = max(qlo−y, y−qhi)`, test-set `qlo/qhi`.
2. Build a feature map `φ(x)` = intercept + **drop-first dummy** indicators for each of the 3
   axes (sector / cap_tier / mom_decile), concatenated. A point's threshold is
   `t(x) = φ(x)ᵀβ` — an **additive** combination of its axis memberships (sector effect +
   cap effect + momentum effect + intercept), not a max over three intervals.
3. Fit `β` by **pinball (quantile) regression** of `E` on `φ` at level `1−α = 0.90`
   (`sklearn.linear_model.QuantileRegressor`, `alpha=0` no L1 shrinkage, `solver="highs"`).
   The first-order conditions of the pinball loss enforce the 90% quantile constraint **per basis
   function** → each group's coverage is targeted at ≈ 90% (two-sided), not ≥ 90% (one-sided).
4. Interval `[qlo − t(x), qhi + t(x)]` (guard `lo ≤ hi`). Unseen test levels → their dummy = 0
   (pooled into the intercept + remaining axes — **graceful pooling, no hard `min_n` fallback**,
   the structural advantage over Mondrian).

**New knobs (frozen here):** quantile level `1−α = 0.90`, QR L1 `alpha = 0.0`, `solver = "highs"`,
drop-first dummy encoding. Everything else (`α=0.10`, K=20 selective, the splits, the learners)
is **identical to P2/P2.1**. No online adaptation in the base method (a `cqr_conditional` + ACI
hybrid is reported as a secondary item, not a gate).

## Pre-committed gates (decision rules)
| Gate | Statistic | Ship the variant if |
|---|---|---|
| **G-CALIB-MULTI (primary)** | retained-70% conditional coverage on **all 3 axes** + **marginal base** coverage, multi-seed (1a) | every axis within **±τ = 5pp** of 90% **two-sided** (so NOT the union's over-cover miss) **and** marginal base within ±τ — i.e. it achieves *calibrated* multi-group coverage where the union over-covered |
| **G-EFFICIENCY (co-primary)** | mean interval width, conditional vs **union**, multi-seed (1a) | conditional **strictly tighter than the P2.1 union** — recovering the width the union spent on over-coverage |
| **G-SHIFT (report, lighter gate)** | conditional worst-group + shift coverage vs single-axis / union (1a) | report where the *static* conditional method lands under shift; it is calibrated-in-distribution by design, so any shift under-coverage is the documented limitation that motivates the reported `cqr_conditional+ACI` hybrid. **Not** a ship-blocker. |
| **G-PRIMARY (i) / G-NULL (re-confirm)** | selective retained-MAE + shuffle | unchanged from P2 (the selective score `u_i` is independent of the conformal interval) |

**Verdict rule (pre-committed).** P2.2 **ships** iff **G-CALIB-MULTI passes on 1a** (all 3 axes
within ±τ two-sided **and** marginal base not over-covering past 90%+τ) **and G-EFFICIENCY passes**
(strictly tighter than the P2.1 union). That is the precise property the union sacrificed:
*calibrated* multi-group coverage at lower width. If the conditional method instead under-covers a
group on base (the opposite failure), report it honestly as the bias/variance cost of the
finite-dim class on rare groups and do **not** declare a clean fix. [redacted: refers to a non-public confirmation run]

## Run
`python src/run_p22.py` → 1a seeds {0,1,2} + 1b; head-to-head single-axis ACI / multi-axis union /
**conditional-conformal** (+ a `cqr_conditional+ACI` shift read). Mirror into
`reports/p22_conditional_results.md`. [redacted: refers to a non-public confirmation run]
