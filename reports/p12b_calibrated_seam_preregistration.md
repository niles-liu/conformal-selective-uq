# Pre-registration — P12b: calibrated-node regime seam, confirmatory run, 2026-07-14

> **What P12b is.** P12 ran the regime × conformal seam on the **union** remedy (`aci_cqr_multi`)
> and returned **REJECT-as-finding**: the conservative union self-absorbs regime (a *shuffled* regime
> axis matched the learned one, G-NULL). A **post-hoc** probe on the *tight calibrated* node
> (`cqr_conditional`, the Gibbs-Cherian-Candès theorem node) then showed the seam **does** appear there
> — learned regime lifted worst-regime **+0.068**, equal to the true-regime oracle. That arm was added
> **after** the union REJECT, so it is exploratory / hypothesis-generating, **not** confirmatory.
>
> **P12b turns that exploratory result into a pre-registered confirm.** The gates below are frozen
> *before* the confirmatory run and are evaluated on **fresh held-out seeds {3,4,5}** — three
> independent regime realizations of the Layer-1b substrate that did **not** inform the +0.068
> hypothesis (which came entirely from {0,1,2}). This is a true out-of-sample confirmation: the
> synthetic substrate lets us draw genuinely blind confirmation data, [redacted: refers to a non-public confirmation run].
> Exploratory seeds {0,1,2} are reported alongside for continuity but do **not** decide any gate.

## Hypothesis (registered, directional — from the P12 exploratory result)
On the **tight calibrated node** `cqr_conditional` (one pinball regression of the CQR conformity
scores on the multi-axis group-indicator design at 1−α; Gibbs-Cherian-Candès 2023), which is
deliberately tuned tight and therefore under-covers the stress regime (does **not** self-absorb regime
the way the conservative union does), **adding a learned dynamic regime axis reallocates conservatism
toward the detected-hard regime and lifts worst-regime coverage**, and that lift is **regime-specific**
(a shuffled regime axis does not reproduce it) and **captures most of the true-regime oracle headroom**.

## Substrate (frozen — `synthetic_regime.py`, unchanged from P12)
Layer 1b multi-regime substrate, K=3 latent Markov regime over sessions (`P_STAY=0.96`, dwell ~28,
`REGIME_AMP=[0.6,1.0,2.4]`). Regime learnable from the PIT observable `disp_lag` (cross-sectional
dispersion of yesterday's target). **Confirmation seeds {3,4,5}** — fresh independent realizations,
generated 2026-07-14, face validity confirmed (dwell 28–30, ~55 switches, stress disp ~4.1–4.25 vs
calm ~2.1; all three regimes present in every split → the calibration window is NOT one regime).
Time-ordered split (`run_p2._split`): train `<20210101`, calib `<20220101` (`CALIB_END`), test `>=`.
Eval on the **true** `regime_state`; the method triggers on the **learned** regime (anti-circularity).

## Method (frozen — no knob touched)
- **Regime detector** (`regime.py`, unchanged): K=3 Statistical Jump Model on standardized session
  `{disp_lag, disp_ma}`; centroids fit offline on train+calib (identified); test labels by causal
  forward filter with frozen centroids; λ persistence-selected on `LAM_GRID=(10,25,50,100,200)`.
- **Primary remedy = the calibrated node** `cqr_conditional` (1 HiGHS pinball solve per axis-set),
  verbatim P2.2. The learned regime state (0/1/2, stress-ordered) is **added** as a 4th axis to
  `GROUP_COLS = [sector, cap_tier, mom_decile]`; existing axes unchanged.
- **Secondary remedy = the parameter-free POGO arm** `pogo.cqr_conditional_gqaci_max_pogo` (`ETA0=0.10`
  fixed, no per-dataset γ) on the full cross-axis-max level family, plus the tuned-γ
  `cqr_conditional_gqaci_max` (γ=0.05) as the match reference. CQR pieces fit once per seed and shared.

### Variants (calibrated node, primary)
| Variant | Axes | Role |
|---|---|---|
| **cal_static** | 3 static | regime-blind tight baseline to beat on worst-regime |
| **cal_proposed** | 3 static + **learned** regime | the seam |
| cal_oracle | 3 static + **true** regime | identifiability upper bound |
| **cal_shuffled** | 3 static + **shuffled** learned regime | G-NULL-CALIB placebo (must NOT lift) |

### Variants (POGO / online family, secondary)
| Variant | Object | Role |
|---|---|---|
| pogo_static | parameter-free, 3 static | regime-blind base for the POGO seam |
| pogo_proposed | parameter-free, 3 + learned | POGO seam (G-POGO-SEAM) |
| gqaci_max_proposed | tuned γ=0.05, 3 + learned | tuned reference (G-POGO-MATCH) |

## Pre-committed gates (target 90%, **decided on held-out seeds {3,4,5}**)
| Gate | Statistic | Confirm if |
|---|---|---|
| **G-SEAM-CALIB** (primary) | cal_proposed worst-regime − cal_static worst-regime | **≥ +0.03** |
| **G-NULL-CALIB** (primary, mechanism) | cal_shuffled worst-regime − cal_static worst-regime | **< +0.03** (the lift is regime-specific, NOT "just another axis" — this is what separates the calibrated node from the union, where shuffled = learned) |
| **G-ORACLE-CAPTURE** (mechanism) | (cal_proposed lift) / (cal_oracle lift) | **≥ 0.60** (the learned state carries most of the coverage-relevant signal) |
| **G-EFF-CALIB** | cal_proposed mean width / cal_static mean width | **≤ 1.20×** (reallocation, not blanket widening) |
| G-POGO-SEAM (secondary) | pogo_proposed worst-regime − pogo_static worst-regime | ≥ +0.03 (seam holds on the full online cross-axis-max object) |
| G-POGO-MATCH (secondary, T2) | \|pogo_proposed − gqaci_max_proposed\| worst-regime | ≤ 0.03 (parameter-free matches tuned γ **without** a per-dataset knob) |

## Verdict rule (pre-committed)
- **CONFIRM (calibrated seam joins):** G-SEAM-CALIB ∧ G-NULL-CALIB ∧ G-ORACLE-CAPTURE ∧ G-EFF-CALIB on
  {3,4,5}. Statement: on a learnable-regime substrate, adding a learned dynamic regime axis to the
  *tight* group-conditional (Gibbs-Cherian-Candès) node recovers worst-regime coverage the regime-blind
  node misses — a **regime-specific** lift (shuffled placebo does not reproduce it) that captures the
  true-regime oracle headroom, at a reallocation-scale width cost. This is the increment [redacted: refers to a non-public confirmation run],
  confirmed out-of-sample where the regime exists — and it is sharp about
  *where* the seam lives (the tight node, not the self-absorbing conservative union: the P12 boundary).
- **REJECT-as-finding:** G-SEAM-CALIB fails on {3,4,5} → the exploratory +0.068 did not replicate
  out-of-sample; the calibrated seam is not robust. Honest negative.
- **QUALIFIED:** G-SEAM-CALIB holds but G-NULL-CALIB fails (shuffled also lifts → the calibrated node
  also just "adds an axis," same redundancy as the union — the seam is not regime-specific) OR
  G-EFF-CALIB fails (the lift is blanket widening, not reallocation) OR G-ORACLE-CAPTURE fails (learned
  regime recovers too little of the true signal). Report which.
- **POGO (secondary, report-only):** G-POGO-SEAM ∧ G-POGO-MATCH corroborate that the seam survives on
  the full online object and that the parameter-free member matches the tuned γ. Does not gate the
  primary verdict.

## Threats pre-registered (methodology — mirrors P12)
- **No re-use of the exploratory data.** Gates decided on {3,4,5} only; {0,1,2} reported for continuity.
- **Leakage.** Detector uses only `disp_lag` (`known_at ≤ t`); test labels causal forward filter with
  frozen train+calib centroids; conformal never sees `y_t`.
- **Mechanism.** G-NULL-CALIB (shuffled placebo) is the primary mechanism gate — it is exactly the test
  that killed the union's apparent lift; requiring it to *pass* here (shuffled does NOT lift) is what
  makes the calibrated seam a real regime effect rather than the union's redundancy artifact.
- **Not a free lunch.** G-EFF-CALIB guards blanket widening; the calibrated node is a *tight* node
  (it under-covers by construction) — the claim is *relative* worst-regime recovery, not hitting nominal.
- **Honesty (T2).** The POGO arm is a parameter-free *member* of the family POGO (2606.00419) proves; we
  show the no-tuning payoff empirically, we do **not** re-derive the theorem.

## 4-lens (after)
Reproduce (deterministic, seeded) · Leakage (forward-filter causality assert) · Statistics (3 held-out
seeds ±sd, seed-3 clustered bootstrap sector×month band, min_n=50) · Mechanism (G-NULL-CALIB shuffled
placebo + oracle capture: is the lift from the regime, and is it regime-specific on the tight node?).

## Run
`python src/synthetic_regime.py --seed {3,4,5}` (done) then `python src/run_p12b.py`
→ mirror into `reports/p12b_calibrated_seam_results.md`.
