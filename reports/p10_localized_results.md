# Results — P10 *localized / kernel-conditional* comparator arm (2026-07-01)

> Paired with `p10_localized_preregistration.md` (frozen first). All numbers from
> `python src/run_p10.py`, seeds {0,1,2} (1a) / seed 0 (1b), target 90%. `localized` deterministic
> (seeded subsample `20260701` + median-heuristic bandwidth). Reproduces bit-for-bit.

## Headline — **both gates PASS, but the win is marginal: continuous conditioning ≈ discrete Mondrian here.**
The kernel-localized comparator (`conformal.localized_cqr`) is a **valid** continuous approximate-
conditional method and is **slightly tighter** than discrete Mondrian at comparable base conditional
coverage — but the margin is small enough that the honest conclusion is **the discrete groups already
capture most of the conditioning structure on this task.** A clean positioning read (not a new ship):
the poster can cite a continuous SOTA-style arm and report it modestly tightens, without overclaiming.

- **G-LOCAL-VALID — PASS.** Localized base coverage 0.898 (within ±0.05 of 0.90) — calibrated.
- **G-LOCAL-TIGHTER — PASS (marginal).** At comparable base worst-group coverage (localized 0.852 vs
  Mondrian 0.845, within ±τ), localized width **6.191 < Mondrian 6.235** — a **−0.7%** tightening.
  Directionally the continuous kernel buys efficiency; quantitatively it is small.

## Layer 1a — localized vs discrete-group comparators (seeds {0,1,2}, target 90%)
| method | base | shift | worst-grp\|shift | base worst-grp | width |
|---|---|---|---|---|---|
| naive | 0.902 | 0.580 | 0.507 | 0.855 | 6.249 |
| cqr (marginal) | 0.901 | 0.573 | 0.474 | 0.853 | 6.238 |
| mondrian (discrete cond.) | 0.900 | 0.572 | 0.462 | 0.845 | 6.235 |
| cqr_conditional (coupled shape) | 0.898 | 0.564 | 0.433 | 0.829 | **6.178** |
| **localized (kernel cond.)** | 0.898 | 0.570 | 0.491 | 0.852 | 6.191 |

Readings:
1. **Localized tightens vs Mondrian at equal base conditional coverage** — width 6.191 < 6.235, base
   worst-group 0.852 ≥ 0.845. Continuous conditioning is a *slightly* better efficiency/coverage point
   than the discrete buckets, as its literature (SpeedCP etc.) advertises — here modestly (−0.7%).
2. **The coupled-shape `cqr_conditional` is the tightest (6.178) but sacrifices base conditional
   coverage** (worst-group 0.829) — it buys width by loosening the per-group guarantee. Localized sits on
   a better point: nearly as tight, with base worst-group back near the marginal methods' 0.85.
3. **All static methods collapse under the 1a shift** (0.56–0.58) — expected: none is shift-robust
   (that is the P2–P7 online layer's job). P10 is a **conditioning-axis** comparator, not a shift method,
   so the shift columns are context, not its test.

## Layer 1b (real 21d fwd return)
| method | base | shift | worst-grp\|shift | base worst-grp | width |
|---|---|---|---|---|---|
| naive | 0.902 | 0.850 | 0.734 | 0.817 | 0.281 |
| cqr | 0.884 | 0.814 | 0.775 | 0.853 | 0.248 |
| mondrian | 0.883 | 0.813 | 0.773 | 0.846 | 0.248 |
| cqr_conditional | 0.870 | 0.799 | 0.706 | 0.762 | 0.247 |
| **localized** | 0.882 | 0.818 | 0.776 | 0.851 | 0.250 |

On real data localized matches cqr/mondrian to within noise (width 0.250 vs 0.248; base worst-group 0.851
vs 0.846) — again, continuous conditioning neither clearly beats nor loses to the discrete groups here.

## Verdict — comparator confirmed; discrete groups suffice (a positioning read, not a ship)
`localized_cqr` is a **valid, deterministic continuous approximate-conditional reference** beyond
Gibbs–Cherian–Candès. It is **marginally tighter** than Mondrian at equal base conditional coverage on
1a (−0.7% width) and **on par** on 1b. The honest finding for the poster: **on this cross-sectional task
the discrete Mondrian groups (sector / cap-tier / momentum-decile) already capture nearly all the
conditioning structure** — the continuous kernel adds a small efficiency edge, not a step change. This
*strengthens* the existing conditional story (it survives a modern continuous comparator) rather than
replacing it. [redacted: refers to a non-public confirmation run]

## 4-lens adversarial check
- **Reproduce** — `localized` deterministic (seeded 6000-pt calibration subsample + median-heuristic
  bandwidth on a seeded 1000-pt sub-sample; no other RNG in the interval math). The frozen comparators
  (naive/cqr/mondrian/cqr_conditional) reproduce their P1–P7 values within this run.
- **Leakage** — kernel weights + conformal quantile use only calibration `E` and calibration/test
  **features** (never test labels); every feature PIT-lagged; static method (no online PIT concern).
- **Statistics** — multi-seed mean ± sd; worst-group `min_n=50`; widths compared at matched (±τ) base
  worst-group coverage, not blindly.
- **Mechanism** — the tightening is a real conditioning effect, not over-narrowing: base coverage stays
  at 0.898 (nominal) and base worst-group at 0.852 (≥ Mondrian) while width drops — a genuine
  efficiency/coverage improvement, small in magnitude. The median-heuristic bandwidth yields a
  non-degenerate kernel (neither uniform nor spiked), so the effect is not a bandwidth artifact.

## Reproduce
```
python src/run_p10.py     # localized vs naive/cqr/mondrian/cqr_conditional, 1a (seeds {0,1,2}) + 1b
```
Pairs with the frozen `reports/p10_localized_preregistration.md`; every number reproduces bit-for-bit.
