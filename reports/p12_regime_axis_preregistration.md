# Pre-registration — P12: learned regime axis (regime × conformal), 2026-07-14

> **What P12 asks.** [redacted: refers to a non-public confirmation run] P12 asks the question [redacted: refers to a non-public confirmation run], on a substrate where the confound is removed: **when a stable, recurring latent regime
> genuinely exists, does adding a LEARNED regime axis to the group-conditional remedy
> (`cqr_conditional_gqaci_max`) recover per-regime / worst-regime coverage that a regime-blind
> (static-axes-only) method misses — at comparable width, and with the regime actually recovered?**
>
> This joins the un-joined literature seam (scan: [redacted: points to a non-public document]):
> Camp A (learned-regime × conformal) proves only **marginal** coverage; Camp B (group-conditional)
> covers only **static** analyst axes. P12 = a learned dynamic regime axis driving the Camp-B remedy.
>
> Gate-first: frozen before running the method. [redacted: refers to a non-public confirmation run] §T2 theory
> (static base node = Gibbs-Cherian-Candès *theorem*; online cross-axis = *procedure*; POGO = the
> parameter-free upgrade) is in [redacted: points to a non-public document] and governs the honest contribution framing here.

## Substrate (frozen apparatus — `synthetic_regime.py`, Layer 1b)
Layer 1a's single shift window is replaced by a **K=3 latent regime Markov chain over sessions**
(`P_STAY=0.96`, mean dwell ~28 sessions, ~57 switches over 1634 sessions), each state scaling the
irreducible overlay term by `REGIME_AMP = [calm 0.6, normal 1.0, stress 2.4]`. All recoverable /
overlay / base structure identical to Layer 1a. The regime is **learnable from a PIT observable**:
`disp_lag` = cross-sectional dispersion of *yesterday's* realised target (seed-0 separation:
calm 2.12 / normal 2.47 / stress 4.30). True `regime_state` persisted = ground truth. Seeds {0,1,2}
(each a fresh regime realisation → the multi-seed bands). Time-ordered split **train 50% / calib 20%
/ test 30%**; because regimes recur, every block contains all three states (the calibration window is
NOT a single regime [redacted: refers to a non-public confirmation run]).

## Method (frozen)
- **Learned regime axis** (`regime.py`): a K=3 Statistical Jump Model on the standardized session
  detector series {`disp_lag`, `disp_ma`}. Centroids fit offline on train+calib (identified — all
  regimes present); test labels by a **causal forward filter** with frozen centroids (`known_at ≤ t`).
  `λ` persistence-selected (grid {10,25,50,100,200}, smallest λ with switch-rate ≤ 5% of sessions;
  frozen value recorded). The learned state (0/1/2, stress-ordered by centroid) is appended as a 4th
  axis to `GROUP_COLS = [sector, cap_tier, mom_decile]`.
- **Conformal** = the frozen P7 `cqr_conditional_gqaci_max` (γ=0.05, `_QACI_GRID`, clamps) — verbatim,
  no knob touched; the regime label is *added* as an axis, existing axes unchanged.

### Variants run head-to-head
| Variant | Axes | Camp | Role |
|---|---|---|---|
| **naive split** | — | 0 | marginal floor (must under-cover stress) |
| **static gqaci-max** | 3 static | B | regime-blind baseline to beat on worst-regime |
| **regime-marginal ACI** | regime only (single-axis `aci_cqr`) | A | learned-regime but marginal (must miss worst-group) |
| **PROPOSED gqaci-max + learned regime** | 3 static + learned regime | A×B | the seam |
| oracle regime | 3 static + **true** regime | — | upper bound (identifiability loss) |
| threshold regime | 3 static + ordinal `disp` threshold | — | cheap-trigger ablation |
| POGO-style proposed | 3 static + learned regime, **no γ** (`pogo.py`) | — | secondary: no-tuning payoff |

Eval: per-regime coverage on the **true** regime slices, **worst-regime** = min over 3 regimes (n≥50),
worst-group over the 3 static axes, marginal coverage, mean width. Multi-seed mean±sd; seed-0
clustered bootstrap (sector×month) band on the proposed worst-regime.

## Pre-committed gates (target 90%, seeds {0,1,2})
| Gate | Statistic | Ship if |
|---|---|---|
| **G-SEAM** (primary) | proposed worst-regime coverage − static-gqaci-max worst-regime | **≥ +0.03** (learned regime axis strictly helps the regime-blind method) |
| **G-NOMINAL** | proposed worst-regime coverage | **≥ 0.90 − 0.05 = 0.85** |
| **G-WORSTGROUP** | proposed worst-group (static axes) vs static-gqaci-max worst-group | **≥ static − 0.02** (regime axis does not sacrifice the static guarantee) |
| **G-EFF** | proposed mean width vs static-gqaci-max width | **≤ 1.10×** (not blanket-conservative) |
| **G-RECOVER** (mechanism) | learned-regime balanced accuracy vs true `regime_state` (test) | **≥ 0.70** (the trigger IS identifiable [redacted: refers to a non-public confirmation run]) |
| **G-NULL** (mechanism) | worst-regime lift with **shuffled** regime labels | collapses to ≈ static (no spurious lift) |
| G-POGO (secondary, report) | POGO-style worst-regime vs tuned proposed | within ±0.03 **without a tuned γ** |

## Verdict rule (pre-committed)
- **CONFIRM (seam joins):** G-SEAM ∧ G-NOMINAL ∧ G-WORSTGROUP ∧ G-EFF ∧ G-RECOVER (G-NULL corroborating).
  Statement: on a learnable-regime substrate a learned dynamic regime axis recovers worst-regime
  coverage the regime-blind Camp-B method misses, without sacrificing the static-axis guarantee or
  blowing up width — [redacted: refers to a non-public confirmation run], demonstrated where the
  regime exists. Report the identifiability contrast (G-RECOVER here vs [redacted: refers to a non-public confirmation run]).
- **REJECT-as-finding:** if G-SEAM fails (static gqaci-max already covers every regime via global
  online adaptation) → even with a learnable regime, per-regime conditioning adds nothing; the online
  global term suffices. Honest negative; no knob retune.
- **QUALIFIED:** seam helps worst-regime but violates G-WORSTGROUP or G-EFF (a trade-off, not a free win).

## Threats pre-registered (methodology)
- **Leakage.** Regime detector uses only lagged dispersion (`disp_lag`, `known_at ≤ t`); test labels
  causal forward filter; conformal never sees `y_t`. Assert future-perturbation invariance.
- **Identifiability / spurious states.** Jump penalty λ persistence-selected; report seed stability +
  G-RECOVER vs true state. G-NULL shuffled-regime placebo must collapse the lift.
- **Not a free lunch.** G-EFF + G-WORSTGROUP guard against "fix regime by blanket widening / by
  abandoning the static axes."
- **Honesty (T2).** The online method is a *procedure*; the POGO arm is a parameter-free *member* of
  the family POGO (2606.00419) proves — we show the no-tuning payoff empirically, we do **not** claim to
  re-derive the theorem.

## 4-lens (after)
Reproduce (deterministic, seeded) · Leakage (forward-filter causality assert) · Statistics (3 seeds ±sd,
clustered band, min_n=50) · Mechanism (G-RECOVER + G-NULL: is the lift from the regime, and is the
regime real?).

## Amendment log (honesty)
- **Remedy switched to the union `aci_cqr_multi`** (from the calibrated 9-level `gqaci_max`) for
  tractability — amended *before* any coverage was seen. The union became the pre-registered primary
  gate.
- **The calibrated node `cqr_conditional` was added AFTER the union REJECT** → that arm is
  **exploratory / hypothesis-generating, NOT pre-registered**; it needs a fresh pre-registered confirm.
  See `reports/p12_regime_axis_results.md`.

## Run
`python src/synthetic_regime.py --seed {0,1,2}` then `python src/run_p12.py`
→ mirror into `reports/p12_regime_axis_results.md`.
