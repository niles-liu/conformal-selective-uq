# Pre-registration — P1 baselines + gate freeze (2026-06-30)

> **Gate-first.** This doc is frozen **before** the method (P2) runs. Gate *structures*
> and *decision rules* are committed here now (P0). Three margins (Δ, X, the
> G-SHIFT naive-failure margin) are **data-dependent**: they are set from a
> **method-blind, baseline-only** read at the *start* of P1 and written into the
> "frozen value" column — never adjusted after seeing a method result. Mirror into a
> paired `reports/p1_baselines_results.md` after the runs.

## Question
Establish, on the public PIT substrate, **(a)** that naive split-conformal holds
*marginal* coverage but **fails conditional / per-regime coverage** under the
documented shift — the failure P2 must fix — and **(b)** the baseline bar
(naive split-conformal, unconditional CQR, random abstention, and 1a-only oracle
abstention) that the selective method must clear.

## Data / splits
- **Layer 1a** (known-ceiling synthetic) — `src/synthetic.py` → `data/processed/synth_1a_seed{S}.parquet`.
  Built on the public PIT panel (`src/panel.py` → `public_pit_panel.parquet`): 93
  liquid US large-caps, 11 GICS sectors, 1634 sessions (2018-07-05 … 2024-12-31),
  16 lagged features. Target `y` = allocator log-weight tilt; **per-name ground truth**
  `recoverable_fraction` / `recoverable_mask` / `irreducible_sigma`.
  - First-read face validity (seed 0): ceiling R²(y ~ recoverable term) = **0.386**
    (target ≈ 0.40 ✓); 38/93 recoverable names; corr(recoverable_fraction,
    irreducible_sigma) = **−0.775** (hidden-driven [term edited Oct 2026] names carry the noise, by construction).
- **Layer 1b** (real public task) — same panel; target = **cross-sectional 21-session
  forward return** (preferred per SPEC §4; decided at P1). No recoverability mask, so
  1b carries G-PRIMARY / G-SHIFT / G-NULL only, never G-MECHANISM.
- **PIT cutoff.** Every feature lagged ≥ 1 session (`known_at ≤ decision_time`).
  Conformal calibration never sees test-fold labels.
- **Splits.** Walk-forward / blocked over time. **G-SHIFT split = train pre-drawdown
  (report_date < 20220101) → test drawdown (20220101–20221231)**, the regime where 1a
  amplifies the irreducible component (overlay ×1.8) and 1b sees the real 2022 bear.
- **Cluster unit for inference.** Sector × calendar-month block (the
  shift/regime/day cluster — never the row). Bands via clustered/blocked bootstrap.

## Baselines (computed first; numbers land in the results doc)
| Baseline | Role |
|---|---|
| Naive split-conformal (marginal) | coverage floor; **demonstrate conditional/regime failure** |
| Unconditional CQR | heteroscedastic-interval baseline (marginal only) |
| Random abstention | **the G-PRIMARY(i) comparator** — risk–coverage at matched retention |
| Oracle abstention (1a only) | knows `irreducible_sigma`; upper bound on achievable selection |

## Pre-committed gates (structures + decision rules frozen now)

| Gate | Statistic | Threshold | Ship if |
|---|---|---|---|
| **G-PRIMARY (i)** useful abstention | retained-set MAE (and mean interval width) vs **random** abstention at matched retention **X%** | **Δ = 0.119** (target-units MAE; *frozen* P1); **X = 70%** | retained MAE reduction ≥ Δ vs random at X% retention, multi-seed band excludes 0 |
| **G-PRIMARY (ii)** conditional coverage | per-group empirical coverage on retained set vs nominal; target **90%** | **τ = 5pp** | every group (sector / cap_tier / mom_decile) within ±τ of 90% |
| **G-SHIFT** | worst-group coverage on the test-drawdown split; **report naive gap first** | **τ_shift = 7pp**; naive worst-group = **0.507**, gap **0.393**, close ≥ 50% ⇒ worst-group **≥ 0.704** (*frozen* P1) | method worst-group within ±τ_shift **and** worst-group ≥ 0.704 |
| **G-MECHANISM (1a)** | (a) confident∩recoverable AUC; (b) Spearman(per-name width, irreducible_sigma) | **θ = 0.70**; **θ′ = 0.40** | AUC ≥ θ **and** ρ ≥ θ′, multi-seed bands clear the thresholds |
| **G-NULL** | label-shuffled `y`: G-PRIMARY(i) advantage | within band of random abstention | selective advantage collapses to random (no leakage) |

**Decision rules for the three data-dependent margins** (set at P1-start from
baseline-only statistics, method-blind, **now FROZEN** — values from the 2026-06-30
baseline read, multi-seed {0,1,2}; see `p1_baselines_results.md`):
- **Δ (G-PRIMARY i) = 0.119** — one-half of the oracle-vs-random retained-MAE gap at
  X=70% (random 2.029 − oracle 1.791, mean over seeds). The method must capture ≥ half
  the headroom random abstention leaves on the table.
- **X (retention) = 70%** — a priori; the headline operating point. Full risk–coverage
  curve reported regardless.
- **G-SHIFT naive-failure margin** — naive worst-group coverage on the test-drawdown
  slice = **0.507** (gap 0.393 below nominal); method must close ≥ 50% ⇒ worst-group
  **≥ 0.704** *and* within ±τ_shift of nominal.

Margins set a priori from the generator design (not data-dependent, frozen now):
target coverage **90%**, τ = 5pp, τ_shift = 7pp, θ = 0.70, θ′ = 0.40, X = 70%.

## Seeds / reproducibility
Seeds ≥ 3: **{0, 1, 2}** (extend to 5 if a band straddles a threshold). Every reported
number is the multi-seed mean ± clustered-bootstrap band. The synthetic target is
bit-for-bit deterministic in the seed (`np.random.default_rng(seed)`; verified). Run:
`python src/panel.py` (once) → `python src/synthetic.py --seed {S}` → `python src/run_p1.py`
(P1 baseline driver, to be added).

## 4-lens adversarial check (fill in the results doc, after)
- **Reproduce** — numbers bit-for-bit from the seeded run?
- **Leakage** — calibration never saw test labels? G-NULL collapses? every feature
  `known_at ≤ decision_time`? walk-forward (no random K-fold across the shift)?
- **Statistics** — bands from sector×month clustered bootstrap, not row bootstrap?
  per-group power adequate (don't over-slice mom_decile × sector)?
- **Mechanism** — does confident∩recoverable AUC mean the method abstains in the
  *right* place, or is it picking up a width/volatility confound that merely correlates
  with `irreducible_sigma`? (Check: does the AUC survive controlling for `vol_63`?)
