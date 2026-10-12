# Results — P12: learned regime axis (regime × conformal), 2026-07-14

> **Corrections (Oct 2026).** Checked against the pre-registration and the driver; the body is unchanged
> apart from marked redactions.
> - **Split.** The pre-registration describes a time-ordered 50% / 20% / 30% split; the driver uses
>   P2's date split (train before 2021, calibration 2021, test from 2022: 629 / 252 / 753 sessions).
>   Every number below comes from the date split, and all three regimes still occur in the
>   calibration year on every seed. P12b's pre-registration states the split correctly.
> - **Reproduced** from a fresh clone in October 2026 (pandas 3.0, numpy 2.5, LightGBM 4.7): every
>   number in the table, the band, the recovery and the G-NULL line.

> **VERDICT: REJECT-as-finding (pre-registered union gate) + a post-hoc boundary result.** On a
> substrate where a recurring regime genuinely exists, a **learned regime axis is redundant for a
> conservative (union) group-conditional remedy** — the union already absorbs the regime effect via its
> per-group quantiles (proposed −0.007 over regime-blind static; a **shuffled** regime axis matches the
> learned one). BUT a **post-hoc** test on the *tight calibrated* remedy (added after the union REJECT,
> so hypothesis-generating, not confirmatory) shows the seam **does** appear there: a learned regime axis
> lifts worst-regime **+0.068**, and the learned regime captures the **entire** oracle (true-regime) lift.
> **The seam joins only where the base remedy is not already over-conservative** — and only for the
> coverage-relevant regime distinction (stress-vs-not), which the learned state recovers even though its
> 3-class accuracy is poor. Pre-reg + gates: `reports/p12_regime_axis_preregistration.md`.

## Results (multi-regime synthetic, seeds {0,1,2}, target 90%; eval on TRUE regime, trigger on LEARNED)
| variant | marg | worst-regime | worst-grp | width | per-regime [calm/norm/stress] |
|---|---|---|---|---|---|
| naive (marginal floor) | 0.861 | 0.724 | 0.833 | 11.86 | [0.949/0.931/**0.724**] |
| **static union** (Camp B, regime-blind) | 0.933 | **0.902** | 0.919 | 14.20 | [0.953/0.946/0.902] |
| regime-only ACI (Camp A) | 0.898 | 0.846 | 0.866 | 13.28 | [0.926/0.924/0.846] |
| **PROPOSED union + learned regime** | 0.939 | **0.909** | 0.924 | 14.65 | [0.959/0.950/0.909] |
| oracle union + true regime | 0.947 | 0.938 | 0.933 | 14.92 | [0.955/0.947/0.938] |
| threshold union (ablation) | 0.941 | 0.915 | 0.927 | 14.67 | [0.955/0.957/0.915] |
| **CALIB static** (tight, Camp B) | 0.853 | **0.722** | 0.795 | 11.82 | [0.937/0.922/**0.722**] |
| **CALIB + learned regime** | 0.861 | **0.790** | 0.809 | 13.61 | [0.872/0.919/0.807] |
| CALIB + true regime (oracle) | 0.852 | 0.790 | 0.784 | 13.31 | [0.829/0.809/0.918] |

Seed-0 clustered band, proposed worst-regime (stress): 0.905 [0.895, 0.913]. Regime recovery
(3-class balanced acc vs true): **0.50** (per-seed 0.51/0.51/0.48). G-NULL shuffled-regime worst-regime:
**0.909** (= proposed).

## Pre-registered gates (union remedy — the amended primary)
| Gate | Value | Result |
|---|---|---|
| G-SEAM proposed − static wr ≥ +0.03 | +0.007 | **MISS** |
| G-NOMINAL proposed wr ≥ 0.85 | 0.909 | PASS |
| G-WORSTGROUP proposed wg ≥ static − 0.02 | 0.924 vs 0.919 | PASS |
| G-EFF width ≤ 1.10× static | 14.65 vs 15.62 cap | PASS |
| G-RECOVER 3-class bal-acc ≥ 0.70 | 0.50 | **MISS** |
| G-NULL shuffled collapses | 0.909 | PASS (shuffled = learned) |

Pre-registered verdict rule → **REJECT-as-finding** (G-SEAM fails on the union).

## The finding — a boundary, on two axes

**1. The conservative union remedy absorbs regime; the regime axis is redundant.** In stress the overlay
is amplified → conformity scores inflate → the union's per-group empirical quantiles widen
*automatically*, without any regime label. Static union already covers stress at 0.902. Adding a learned
regime axis buys +0.007 — and a **shuffled** regime axis buys the same (G-NULL = 0.909), so the
"lift" is just "another axis widens the union," not regime information. The 3-class recovery is only 0.50
(the detector cleanly separates *stress* — disp 4.3 vs 2.1 — but confuses calm vs normal, which is
coverage-irrelevant). Clean redundancy result.

**2. (POST-HOC, hypothesis-generating) The tight calibrated remedy is where the seam lives.** The
calibrated node (Gibbs-Cherian-Candès, the T2 *theorem* node) is tuned tight, so it does **not** absorb
regime: CALIB static under-covers stress at **0.722** (like naive). Adding the learned regime axis
reallocates conservatism toward the detected-hard regime → stress 0.722 → 0.807, **worst-regime +0.068**,
and this **equals the true-regime oracle's lift** (learned 0.790 = oracle 0.790) — i.e. the learned state
captures the full coverage-relevant signal despite 0.50 3-class accuracy, because what matters is
detecting stress, which it does. ⚠️ This arm was added **after** the union REJECT, so it is
**exploratory**, not pre-registered; the honest next step is a fresh pre-registered calibrated-seam
confirm (deferred — see below).

**Boundary condition (the contribution):** the regime × conformal seam pays only when **(a)** the regime
is identifiable [redacted: refers to a non-public confirmation run] **and (b)** the
base group-conditional remedy is not already over-conservative (the union self-absorbs regime; the tight
calibrated node does not). This maps *why* the literature's two camps stay un-joined on this problem
class, which is a sharper result than a blanket CONFIRM would have been.

## 4-lens
- **Reproduce** — deterministic, fully seeded; jump model = coordinate descent from fixed k-means++ init.
- **Leakage** — regime detector uses only `disp_lag` (yesterday's dispersion, `known_at ≤ t`); test
  labels via causal forward filter with frozen train+calib centroids; conformal never sees `y_t`.
- **Statistics** — 3 seeds ±sd; seed-0 clustered bootstrap band; worst-regime n ≥ 50.
- **Mechanism** — G-NULL (shuffled = learned on the union → redundancy); oracle = learned on the
  calibrated arm (the learned state carries the coverage-relevant signal); per-regime reallocation
  visible in the calibrated rows (stress ↑, calm ↓).

## Honest scope / amendments (before any coverage was seen unless noted)
- **Remedy switched to the union** (`aci_cqr_multi`) from the calibrated 9-level `gqaci_max` for
  tractability (the alpha-space `QuantileRegressor` level-family is ~100× slower/axis-set) — amended
  before results. The calibrated **node** (`cqr_conditional`, 1 LP) was added later; see the ⚠️ above.
- GBM CQR pieces fit **once per seed** and shared across axis-sets (identical fits; ~50× cheaper).

## Deferred (honest next steps)
1. **Fresh pre-registered calibrated-seam confirm** — register G-SEAM on the calibrated node (and the
   full `gqaci_max`), then run; the +0.068 above is exploratory.
2. **POGO parameter-free arm** (`pogo.py`) on the calibrated family — the no-tuning-knob payoff (T2).
3. **A K=2 (stress/not) detector** — 3-class recovery is coverage-irrelevantly poor; a 2-state detector
   should raise G-RECOVER without changing the coverage conclusion.

## Artifacts
`src/synthetic_regime.py` (multi-regime generator), `src/regime.py` (jump model + causal forward
filter), `src/pogo.py` (parameter-free arm, unused pending the calibrated confirm), `src/run_p12.py`
(driver). [redacted: refers to a non-public confirmation run]
