# Pre-registration — P10 *localized / kernel-conditional* comparator arm (2026-07-01)

> **Gate-first.** Frozen before `python src/run_p10.py`. P10 is a **comparator arm**, not a
> ship/reject method (SPEC §11: "no gate redefinition; slots into the existing conditional-coverage
> table"), so the gates are **descriptive**. Paired with `reports/p10_localized_results.md`. The
> frozen P1–P7 variants are untouched; `conformal.localized_cqr` is a new additive comparator.

## Question
Conditional coverage in this repo is **Mondrian (discrete groups)**. Does a **continuous**
approximate-conditional method — kernel-reweighted split CQR (Guan 2023 *Localized Conformal
Prediction*; Hore & Barber 2024 *Conformal prediction with local weights*), a modern reference beyond
Gibbs–Cherian–Candès — **tighten intervals at equal conditional coverage**, or do the discrete groups
already capture the conditioning structure on this task?

## The comparator (frozen spec — `conformal.localized_cqr`)
Per test point `x`, the conformal radius `Q(x)` is the `(1−α)` quantile of the CQR conformity scores
`E` **weighted by a Gaussian kernel** on z-scored features: `w_j(x) = exp(−‖z(x)−z(x_j)‖² / 2h²)`,
cumulative-mass threshold `(1−α)·(Σ_j w_j + 1)` with a **self-weight 1** for the test atom
(Hore–Barber local weighting). Interval `[qlo(x) − Q(x), qhi(x) + Q(x)]`. **Frozen knobs (laptop
tractability — the full 23k×70k kernel is out of core):** calibration **subsample = 6000** (seed
`20260701`), **median-heuristic bandwidth** `h` (median pairwise distance on a 1000-pt sub-sample),
block size 2000. Static (calibration-based), like Mondrian — this is the **conditioning-axis**
reference, not a shift method.

## Data / splits (identical to P1–P9)
Layers **1a** (gated synthetic) + **1b** (real 21d fwd return). PIT panel; train `<20210101` | calib
2021 | test `≥20220101`. Cluster unit: sector × calendar-month.

## Baselines (computed first, in the same table)
Re-run head-to-head in `run_p10.py`: `naive`, `cqr` (marginal), `mondrian` (discrete group-conditional),
`cqr_conditional` (static coupled-shape conditional). Their numbers set the bar `localized` is read
against (Mondrian is the direct discrete-vs-continuous foil).

## Pre-committed gates (descriptive; τ = 0.05 repo-wide tolerance)
| Gate | Statistic | Read |
|---|---|---|
| **G-LOCAL-VALID** | `localized` **base** marginal coverage, multi-seed (1a) | within **±τ=0.05** of 0.90 — the localized comparator is calibrated (a validity check, not ship/reject) |
| **G-LOCAL-TIGHTER** | `localized` mean width vs `mondrian` width **at comparable (±τ) base worst-group coverage** (1a) | localized **tighter** ⇒ continuous conditioning buys efficiency; **not tighter** ⇒ the discrete groups already capture the structure — either is a legitimate reported finding |

No ship/reject; [redacted: refers to a non-public confirmation run]. The result is a **positioning read**: whether the poster should
cite a continuous-conditional SOTA arm as tightening the intervals, or report that Mondrian suffices
on this task (both are honest, publishable statements).

## Seeds / reproducibility
Seeds **{0,1,2}** (1a), seed 0 (1b), target 90%. Kernel subsample + bandwidth seeded (`20260701`) →
`localized` is deterministic. Run: `python src/run_p10.py`. Every number reproduces bit-for-bit.

## 4-lens adversarial check (fill in results doc, after)
- **Reproduce** — deterministic (seeded subsample + bandwidth, no RNG in the interval math otherwise);
  frozen comparators reproduce their P1–P7 values within this run.
- **Leakage** — kernel weights + conformal quantile use only calibration `E` and calibration/test
  **features** (never test labels); every feature PIT-lagged. Static method → no online PIT concern.
- **Statistics** — multi-seed mean ± sd; worst-group `min_n=50`; widths compared at matched (±τ) base
  conditional coverage, not blindly.
- **Mechanism** — is any width difference a real conditioning effect or a bandwidth artifact? Check the
  median-heuristic `h` is sane (kernel neither near-uniform nor near-degenerate) and localized base
  coverage is actually held (not bought by over-wide intervals).
