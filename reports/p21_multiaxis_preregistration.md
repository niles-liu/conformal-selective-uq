# Pre-registration — P2.1 multi-axis conditional coverage (2026-06-30)

> **Gate-first.** Written **before** the multi-axis variant is run. The frozen single-axis
> method (`aci_cqr`, dim=`mom_decile`) and all P2 knobs are **untouched** — this is an
> *additive* variant tested head-to-head, never a retune. Paired with
> `reports/p21_multiaxis_results.md` (filled after the runs).

## The qualified item this targets
[redacted: refers to a non-public confirmation run]
- **P2 (1a):** retained-set conditional coverage held within τ=5pp on the **conditioned** axes
  (cap_tier 0.034, mom_decile 0.032) but grazed ~1pp over on the **un-conditioned sector** axis
  (0.060). [`p2_method_results.md` §G-PRIMARY(ii)]
- [redacted: refers to a non-public confirmation run]

**Diagnosis (pre-committed):** single-axis ACI controls only the axis it conditions on. The
principled fix is to condition on **all three axes at once**, not to re-tune which single axis.

## The variant (frozen spec)
**Multi-axis group-conditional ACI** (`conformal.aci_cqr_multi`): run an independent
group-conditional ACI on **each** of the 3 axes (sector / cap_tier / mom_decile) over the shared
CQR conformity scores, then emit the per-point **union** interval `[min_k lo_k, max_k hi_k]`.
Because the union is ⊇ each axis's interval, its coverage dominates each axis's per-group coverage
→ worst-group is controlled across **all** axes simultaneously (Gibbs–Cherian–Candès 2023
conditional-guarantee spirit). **Cost:** wider intervals — an efficiency / over-coverage tradeoff,
reported explicitly. All other knobs (γ=0.05, MIN_N=50, K=20, α=0.10) **identical to P2**.

## Pre-committed gates (decision rules)
| Gate | Statistic | Ship the variant if |
|---|---|---|
| **G-COND-MULTI (primary)** | retained-set conditional coverage (G-PRIMARY ii) on **all 3 axes**, multi-seed | every axis (incl. **sector**) within **±τ = 5pp** of 90% — i.e. it fixes the un-conditioned axes **without** breaking the conditioned ones |
| **G-SHIFT (re-confirm)** | multi-axis ACI worst-group coverage on the drawdown slice, all 3 axes | worst-group **≥ 0.704** and shift coverage within **±τ_shift = 7pp** — and **≥ the single-axis ACI's** worst-group |
| **Efficiency (report, not gate)** | mean interval width, multi-axis vs single-axis ACI; marginal base/shift coverage | report the width inflation and any over-coverage (two-sided: over-covering by > τ is itself a G-COND-MULTI miss) |
| **G-PRIMARY (i) / G-NULL (re-confirm)** | selective retained-MAE + shuffle | unchanged from P2 (the selective score `u_i` is independent of the conformal interval) |

**Verdict rule (pre-committed).** P2.1 **ships** iff G-COND-MULTI passes on 1a (all axes within
±τ) **and** G-SHIFT holds — i.e. the multi-axis variant genuinely closes the un-conditioned-axis
gap rather than trading under-coverage for over-coverage. If it over-corrects (marginal coverage
or any group > 90%+τ), report that honestly as the conservative-union tradeoff and do **not**
declare a clean fix. [redacted: refers to a non-public confirmation run]

## Run
`python src/run_p21.py` → 1a seeds {0,1,2} + 1b; head-to-head single-axis vs multi-axis ACI.
Mirror into `reports/p21_multiaxis_results.md`. [redacted: refers to a non-public confirmation run]
