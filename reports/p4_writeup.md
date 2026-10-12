# Selective conformal regression under distribution shift
## Checked against a known irreducible-noise floor

**Niles Liu · 2026-06-30, revised Oct 2026 · short note (P4)**

> A pre-registered study of selective conformal regression under distribution shift, on a testbed
> where each stock's unrecoverable noise is known. The abstention score finds the noisy stocks
> (AUC 0.81), and online conformal recovers most of the coverage lost under the shift (worst-group
> 0.51 → 0.82, target 0.90). Two of the five pre-registered gates fail, and both are reported
> exactly as frozen. Design document: [`SPEC.md`](../SPEC.md); per-phase pre-registration and
> results: [`reports/`](.).
>
> *Revised Oct 2026* so that every verdict matches its pre-registration. The first version called
> group coverage "qualified" and coverage under shift a pass; see the Corrections block at the top
> of [`p2_method_results.md`](p2_method_results.md).

---

## 1. The gap

Conformal prediction can certify that an interval has nominal coverage; selective prediction can
trade coverage for accuracy by abstaining. Neither can usually show whether a model **abstains in
the right place**, because real tasks carry no ground truth for *which points were ever
recoverable*. So "the model is well calibrated and abstains on 30% of stocks" is unfalsifiable as a
claim about *whether those were the right 30%*.

This project builds a regression task that **does** carry that ground truth and uses it to test the
selective layer. The demonstration task is a semi-synthetic portfolio-allocation problem on public
US equity data. Each stock's weight mixes a part the public features can recover with a hidden
component the model never sees, so the unrecoverable share is known for every stock. The hidden
share is a parameter of the generator; it is not claimed to mirror any real task. The nearest prior
work ("When Alpha Breaks", 2026) applies conformal methods to return and ranking prediction, a
setting without this ground truth.

## 2. Data

- **Layer 1a: known-ceiling semi-synthetic (the controlled experiment).** The generator is
  `w = base(cap-diversified) × momentum_tilt(public features) + λ·hidden_overlay + ε`, where the
  overlay is a seeded function the learner is **not** given and `ε` is irreducible noise. So the
  recoverable fraction and the irreducible-noise magnitude are known for every stock, which is what
  makes the mechanism testable. The generator is tuned so the public-recoverable R² is about 0.40
  (a deliberately hard but non-trivial recoverable share), and its 2022 test period carries a
  deliberately severe shift.
- **Layer 1b: real public task (external validity).** 21-day cross-sectional forward returns from
  the same point-in-time panel, with a real train-calm, test-drawdown split. It guards against a
  self-fulfilling synthetic. It has no recoverability mask, so it cannot test the mechanism.

Some pre-registrations refer to a confirmation run on non-public data; it is not part of this
release.

## 3. Method (public features only; knobs frozen before the method ran)

Base learners (LightGBM mean and pinball quantiles) feed **CQR** heteroscedastic intervals (Romano
2019), then **Mondrian / group-conditional** calibration (per momentum decile), then
**group-conditional Adaptive Conformal Inference** (Gibbs and Candès 2021) as the shift-robust
headline method, with **weighted/covariate-shift CQR** (Tibshirani 2019) as an ablation. The
**selective** layer abstains on a **per-stock calibration-residual scale** `u_i` (the shrunk MAD of
the stock's own calibration residuals). If the irreducible component persists per stock, a stock's
own historical error scale is a point-in-time proxy for its irreducible noise, which raw interval
width misses (P1).

`u_i` is fit on calibration-fold residuals, and the conformal layer calibrates on the same fold, so
coverage on the retained set is **empirical**, not a finite-sample guarantee.

## 4. Pre-registered gates and what they returned (Layer 1a, seeds {0, 1, 2})

The decision rules were frozen in [`p1_baselines_preregistration.md`](p1_baselines_preregistration.md)
and [`p2_method_preregistration.md`](p2_method_preregistration.md) **before** the method ran; no
knob was tuned afterwards. Full numbers: [`p2_method_results.md`](p2_method_results.md).

| Check | Rule (frozen before the run) | Result (3 seeds) | Verdict |
|---|---|---|---|
| Useful abstention (G-PRIMARY i) | retained MAE beats random at 70% retention by ≥ 0.119 | 0.178 better, ~75% of the oracle's headroom | **PASS** |
| Group coverage (G-PRIMARY ii) | every group within 5pp of 90% on the retained set | cap tier and momentum decile within; sector off by 6.0pp | **FAIL** (by 1pp) |
| Coverage under shift (G-SHIFT) | worst-group within 7pp of 90% **and** ≥ 0.704 | 0.815 (static conformal: 0.507) | **FAIL** (meets the floor, misses the band) |
| Abstention in the right place (G-MECHANISM) | AUC ≥ 0.70 and ρ ≥ 0.40 against the known noise | AUC 0.81, ρ 0.86, survives a `vol_63` volatility control | **PASS** |
| Leakage guard (G-NULL) | shuffled labels erase the advantage | 0.178 → 0.002 | **PASS** |

Online conformal (group-conditional ACI) recovers most of the coverage lost under the shift
(worst-group 0.51 → 0.82, target 0.90) and lifts overall shift coverage from 0.58 to 0.87, but its
worst group stays 8.5pp below nominal, outside the pre-registered 7pp band. Static Mondrian
calibration alone does not help (worst-group 0.46): the shift is temporal, not a static group
effect, which isolates the online layer as the active ingredient. On the retained set, group
coverage holds on momentum decile (the axis the method conditions on) and on cap tier, and misses on
sector. SPEC §5 counts a failure of either G-PRIMARY part as a reject, so the headline method does
not ship as pre-registered; the misses motivated the multi-axis follow-ups P2.1 to P2.3.

Three figures (`figures/`, generated by `src/figures.py`, all from seed 0):

- **`fig1_risk_coverage.png`**: the per-stock score (blue) tracks the noise-aware **oracle** (green)
  and sits well below random abstention (grey), so abstention is useful, not cosmetic.
- **`fig2_coverage_shift.png`**: under the drawdown, naive split-conformal's worst-group coverage
  falls to 0.48 while group-conditional ACI recovers it to 0.79 (seed 0; the 3-seed means are 0.51
  and 0.82, against a band edge of 0.83).
- **`fig3_mechanism.png`**: `u_i` rises with the known injected noise σ_irr (Spearman ρ = 0.89 on
  seed 0; 0.86 over three seeds). Recoverable stocks cluster low-left and hidden-component stocks
  high-right: the abstention lands where the noise is known to be irreducible.

**Layer 1b (real returns).** Online conformal lifts worst-group coverage from 0.734 to 0.858 under
the real 2022 shift and holds overall shift coverage at 0.898. By the same G-SHIFT rule that is a
pass (0.042 from 0.90, above 0.704), while static conformal's 0.734 meets the floor but misses the
band. The selective gain is real but much weaker (retained MAE 0.0712 → 0.0636 at 70%), as expected
for returns, which are close to a martingale.

## 5. Limits

- **No distribution-free exact conditional coverage.** It is impossible in general (Foygel Barber
  et al. 2021); the coverage here is group-conditional and empirical. The sector miss is the visible
  edge of that limit.
- **Retained-set coverage is empirical.** The abstention score and the conformal layer use the same
  calibration fold (§3). A selection-aware calibration is the natural next experiment.
- **Some verdicts sit on a cutoff.** In the full log, P9's G-RISK-SHIFT is 0.0798 against 0.08 and
  P11's base-only false alarm is e = 20.02 against 20, so tiny numerical drift could flip either
  printed verdict. P2's sector gap (0.0601) sat 0.0001 past the original driver's non-pre-registered
  0.06 "marginal" cutoff; against the pre-registered 0.05 it fails by 1pp.
- **The known noise is a generator parameter.** It makes the mechanism checkable; it says nothing
  about how much of any real allocation is recoverable.
- **No trading claim.** This is an uncertainty-quantification method on a semi-synthetic task; no
  returns or P&L are claimed.

## 6. Reproduce

```
pip install -r requirements.txt
python src/panel.py       # build the public panel (downloads Yahoo Finance data)
python src/run_p1.py      # baselines: marginal coverage holds; conditional coverage fails under shift
python src/run_p2.py      # the five headline gates, three seeds, on 1a and 1b
python src/figures.py     # the three figures above
```

About six minutes on a laptop. Every number a driver prints reproduces exactly from a fresh clone
(verified Oct 2026, pandas 2.3 and 3.0); the Yahoo download is not frozen. The README lists the
tested versions.

## References

CQR: Romano, Patterson and Candès (2019). Weighted conformal under covariate shift: Tibshirani et
al. (2019). Adaptive Conformal Inference: Gibbs and Candès (2021). Limits of distribution-free
conditional inference: Foygel Barber et al. (2021). Conformal prediction with conditional
guarantees: Gibbs, Cherian and Candès (2023). Nearest prior work: "When Alpha Breaks" (2026, arXiv
2603.13252).
