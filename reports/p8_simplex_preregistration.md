# Pre-registration — P8 *conformal on the simplex* (compositional target) (2026-07-01)

> **Gate-first.** Variant spec + gate STRUCTURE frozen before the method runs; the product-region
> joint-coverage failure number is frozen from the **method-blind baseline read**
> (`python src/run_p8.py --baseline`). Gate thresholds **reuse the repo-wide frozen tolerances**
> (`τ=0.05`, `τ_shift=0.07`, `θ_AUC=0.70`, `θ_ρ=0.40` — the same numbers P2's G-PRIMARY(ii)/G-SHIFT/
> G-MECHANISM use, not new picks). Paired with `reports/p8_simplex_results.md`. Highest-leverage
> extension (SPEC §11); the frozen P1–P9 variants are untouched; `src/simplex.py` is additive.

## Question
The 1a [redacted: refers to a non-public confirmation run] target is an **allocation vector** (weights ≥ 0, sum to 1 per session — a point on the
simplex), yet the pipeline certifies each weight as an **independent scalar** interval. `N` marginal-90%
intervals have joint coverage `~0.90^N ≈ 0` at `N ≈ 90` names, so the pipeline **cannot certify the
allocation as a vector at all.** Can a **joint conformal region on the simplex** certify the whole
allocation at valid coverage, hold it **under shift**, and — the headline — show that the
**compositionally-hardest names coincide with the known irreducibly hidden-driven [term edited Oct 2026] names** (G-MECHANISM ported
to the simplex)? This is the intersection arXiv 2511.18141 (marginal-only simplex conformal) leaves open:
*simplex region × selection/known-floor × shift.*

## The region (frozen spec — `src/simplex.py`)
A **scalar** compositional nonconformity score `S(w_true, p̂)` respecting simplex geometry; the region is
the metric ball `{v : S(v, p̂) ≤ Q}`, `Q` the finite-sample conformal quantile of calibration scores.
Scalar score ⇒ the region is a **ball, not a grid** — grid-refined HDR is infeasible at ~90 dims, so the
ball is the laptop-tractable realization of "one region on the simplex" (an explicit, honest design
choice vs arXiv 2511.18141's low-dim grid-HDR). Two frozen scores:
- **Aitchison (primary)** — Euclidean distance in centred-log-ratio space, with **multiplicative
  zero-replacement** (`ε=1e-6` floor + re-closure) for tiny parts (CoDA-standard); the canonical simplex
  metric, matching the log-ratio spirit of the cited simplex-conformal work.
- **Total variation (companion)** — `0.5·Σ|w−p̂|`, the half-L1 allocation error, bounded/robust; reported
  beside Aitchison as an interpretability + robustness check.

Predicted composition `p̂ = softmax_session(base_logit + ŷ)`, `ŷ` = the frozen LightGBM mean model,
`base_logit = 0.5·z_date(log_dollar_vol)` (the generator's base, reconstructed exactly — verified
max|·|=1e-17). True composition = the emitted `weight`, re-closed over the session's present names.

**Two region variants:** `static` (one `Q` from calibration) and `ACI` (online — per-session `Q_s` from
the calib score quantile at a carried `α_eff`, updated `α_eff += γ(α − miss_s)` after each session's
whole-vector coverage is revealed; `γ=0.05`, PIT-valid). The independent-scalar **product region**
(per-name CQR intervals, joint coverage = all names contained in a session) is the baseline foil.

## Data / splits
**Layer 1a only** — the compositional substrate (1b is forward *returns*, not a composition; [redacted: refers to a non-public confirmation run]). PIT panel; train `<20210101` |
calib 2021 | test `≥20220101`; shift = 2022 drawdown sessions. Coverage is **per session** (one region
per session); band = calendar-month-blocked session bootstrap.

## Baselines (computed first — `run_p8.py --baseline`)
Independent-scalar product-region JOINT coverage (the failure the joint region fixes) + the
compositional-score scale.

> **FROZEN FROM BASELINE READ** (`python src/run_p8.py --baseline`, 2026-07-01, 1a seeds {0,1,2}):
> independent-scalar **product-region JOINT coverage = 0.0000 ± 0.0000** (base) / **0.0000 ± 0.0000**
> (shift) — exactly 0: 93 marginal-90% intervals never all contain the truth at once (multiplicity).
> Compositional-score 90th-pct calib radius: **Aitchison 17.83**, **TV 0.757** (the allocation is
> genuinely hard to predict — mostly hidden-driven by design [term edited Oct 2026] — so the joint region is necessarily large; the
> point is that a *valid* joint region exists at all, which the product region cannot provide).

## Pre-committed gates (thresholds reuse frozen tolerances)
| Gate | Statistic | Ship if |
|---|---|---|
| **G-COMPO-JOINT (primary)** | simplex-region JOINT base coverage vs product-region joint coverage, multi-seed (1a) | simplex within **±τ=0.05** of 0.90 **AND** product region **< 0.90−τ** (independent scalars fail joint certification) |
| **G-COMPO-SHIFT (co-primary)** | simplex **ACI** joint coverage, shift regime (1a) | within **±τ_shift=0.07** of 0.90 **AND** > the static region's shift coverage (online adaptation helps) |
| **G-COMPO-MECH (headline mechanism, 1a)** | per-name compositional difficulty (mean CLR deviation) vs the known floor | **AUC(−difficulty → recoverable_mask) ≥ θ_AUC=0.70** AND **Spearman(difficulty, σ_irr) ≥ θ_ρ=0.40**, surviving a `vol_63` partial — the compositionally-hardest names are the irreducibly hidden-driven [term edited Oct 2026] ones |

**Verdict rule (pre-committed).** P8 **ships** iff **G-COMPO-JOINT ∧ G-COMPO-SHIFT ∧ G-COMPO-MECH** on
1a. If JOINT + MECH pass but the ACI shift coverage misses the ±τ_shift band on the deliberately-severe
synthetic (the family-wide horizon wall P6/P7 hit), the honest verdict is **QUALIFIED** — the
compositional certification + mechanism hold, shift-robustness bounded — **no post-hoc retune** of
`ε, γ, the score, or the reused tolerances`. [redacted: refers to a non-public confirmation run]

## Seeds / reproducibility
Seeds **{0,1,2}**, target 90%. Deterministic (seeded synthetic, frozen learners, deterministic scores/
ACI). Run: `python src/run_p8.py`. Every number reproduces bit-for-bit.

## 4-lens adversarial check (fill in results doc, after)
- **Reproduce** — bit-for-bit from the seeded driver; base reconstruction exact.
- **Leakage** — `p̂` uses only public features + the frozen mean model; the region uses only calib scores;
  ACI at session `s` uses sessions `<s` (`y_s` after emission → PIT). No test label touches calibration.
- **Statistics** — per-session coverage with a **month-blocked session bootstrap** band; multi-seed ±sd.
- **Mechanism** — is the difficulty→recoverability link real or a `vol_63`/size confound? Partial-Spearman
  guard (as in P2). Is the joint-coverage "win" just the trivial 90% of any scalar score? No — the content
  is (a) it is a **joint** statement where independent scalars give ≈0, and (b) the difficulty localizes
  on the known hidden-driven [term edited Oct 2026] names; the 90% marginal alone is not the claim.
