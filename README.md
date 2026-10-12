# conformal-selective-uq

[![CI](https://github.com/niles-liu/conformal-selective-uq/actions/workflows/ci.yml/badge.svg)](https://github.com/niles-liu/conformal-selective-uq/actions/workflows/ci.yml)

A pre-registered study of selective conformal regression under distribution shift, on a testbed
where each stock's unrecoverable noise is known. The abstention score finds the noisy stocks (AUC
0.81), and online conformal recovers most of the coverage a regime shift destroys (worst-group 0.51
to 0.82, target 0.90). Every check is reported as frozen before the run, including the two that
missed.

Runs on a laptop: the headline experiment takes about 4 minutes from a fresh clone, and
`python reproduce.py` checks every printed number against the recorded output. The method installs
as a small, tested package.

| Abstention against the known noise | Coverage under the shift |
|:--:|:--:|
| ![abstention score against the known noise](figures/fig3_mechanism.png) | ![worst-group coverage under the shift](figures/fig2_coverage_shift.png) |
| Each dot is a stock. Its abstention score, from calibration data only, rises with the noise the model never sees. Seed 0 (ρ 0.89); over three seeds AUC 0.81, ρ 0.86. | Worst-group coverage under the shift: 0.48 for static split conformal, 0.79 for online group-conditional conformal (ACI). Seed 0; over three seeds 0.51 and 0.82, target 0.90. |

## The idea

Selective prediction lets a model abstain where it is unsure; conformal prediction puts a calibrated
interval on what it keeps. On real data neither can show the model abstains in the right place,
because nobody knows which points were ever predictable. This repo builds a task where that is known.

The task is semi-synthetic portfolio allocation on public US equity data. Each stock's weight mixes a
part the public features can recover with a hidden part the model never sees, so the unrecoverable
share is known for every stock. The model trains before 2021, calibrates on calm 2021 data, and is
tested on 2022 to 2024, with a deliberately severe shift in 2022. A second layer predicts real 21-day
forward returns through the 2022 drawdown, as an external check.

## Results

The headline experiment (P2): three seeds, target coverage 90%, each check as frozen in
[its pre-registration](reports/p2_method_preregistration.md).

| Check | Rule, frozen before the run | Result | Verdict |
|---|---|---|---|
| Useful abstention | At 70% retention, kept-stock error beats random abstention by at least 0.119 | 0.178 better, about 75% of an oracle's gain | Pass |
| Abstention in the right place | The score ranks stocks by their known noise: AUC at least 0.70, ρ at least 0.40 | AUC 0.81, ρ 0.86; survives a volatility control | Pass |
| Leakage guard | Shuffled labels erase the advantage | 0.178 drops to 0.002 | Pass |
| Coverage under the shift | Worst-group coverage within 7 points of 90%, and at least 0.704 | 0.815 online (ACI), against 0.507 static | Fail: misses the band by 1.5 points |
| Group coverage | On the kept stocks, every group within 5 points of 90% | Sector off by 6.0 points | Fail, by 1 point |

On the real-returns layer, online conformal lifts worst-group coverage through the 2022 drawdown
from 0.734 to 0.858, passing the same rule.

![retained error against retention for random, selective and oracle abstention](figures/fig1_risk_coverage.png)

Abstaining on the per-stock score recovers most of an oracle's gain over random abstention (seed 0).

Further along the same line:

- **More group structure.** A union over three grouping axes lifts worst-group coverage under the
  shift to 0.881 at 12% more width, but over-covers, so the two-sided check fails from above (P2.1).
  A calibrated conditional threshold is 32% narrower than the union, but misses on sector for the
  kept stocks and is not robust to the shift (P2.2).
- **Risk instead of coverage.** Online selective conformal risk control holds the kept stocks'
  capped miss-magnitude at the 0.08 target through the drawdown (0.0798, right on the threshold),
  where a static radius reaches 0.184 (P9).
- **Regimes that recur.** A second synthetic layer replaces the single shift with three recurring
  regimes (calm, normal, stress; spells of about 28 sessions). A learned regime axis (a Statistical
  Jump Model on yesterday's cross-sectional dispersion, labelled causally) adds nothing to the
  conservative union, which widens for stress on its own: +0.007, matched by a shuffled axis (P12, a
  pre-registered miss). On the tighter calibrated threshold it helps: a confirm on three fresh seeds
  passes all six gates, +0.055 worst-regime coverage, matching a true-regime oracle (P12b). That lift
  is noisy per seed (-0.022 to +0.150); a parameter-free online version lifts on every seed (+0.037),
  within 0.008 of the hand-tuned one.
- **The kept-set caveat, tested.** P2 fits the abstention score and the conformal radius on the same
  fold, so its kept-stock coverage has no guarantee, and it over-covers (0.928). Scoring stocks on
  cross-fitted training residuals and calibrating on the kept stocks only gives 0.896 at 12% less
  width, with a slightly better score (AUC 0.845). It passes on the three-seed mean but is noisy per
  seed (0.869 to 0.927); the online version holds 0.901 to 0.904 on every seed (P13).

## What didn't work, and what it taught

- **The severe synthetic shift has a floor no single-shape variant clears.** Adapting one fitted
  conditional shape online per group, in width space (P2.3) or quantile-level space (P6, P7), fixes
  what each targeted (P6 brings calm coverage from 0.966 back to 0.900 at 26% less width than the
  union), but none clears the 0.704 worst-group floor. On the milder real shift they do.
- **Abstention and group coverage interact.** Abstention drops more stocks from some sectors than
  others, so kept-stock group coverage is a harder target than the one conformal calibrates for.
  That is the sector miss in the table.
- **A joint region on the simplex breaks under the shift.** One region for the whole allocation gives
  0.923 joint coverage in calm data, where per-stock intervals give 0.000, but falls to 0.000 under
  the shift (P8).
- **Two extensions added little.** Kernel localization passes but is only 0.7% tighter than discrete
  groups (P10). An e-value monitor flags the coverage break within about two weeks on every seed, but
  also fires on the robust method and once on calm data, failing its detection and false-alarm checks
  (P11).
- **One-shot calibration is noisy where online calibration is not.** Twice (P12b, P13) a static
  calibrated fit is right on average but swings from seed to seed, while its online form is stable on
  every seed.

## Use the method

```
pip install -e .
```

[`examples/quickstart.py`](examples/quickstart.py) runs in a few seconds on toy data with no
network: split conformal (`cqr`), online group-conditional conformal (`aci_cqr`) and abstention on a
per-unit score (`name_uncertainty`, `selective_scores`), on 30 units with a noise jump partway
through.

```
$ python examples/quickstart.py
static CQR: coverage 0.91 before the shift, 0.69 after (worst tier 0.45; target 0.90)
online ACI: coverage 0.88 before the shift, 0.81 after (worst tier 0.79; target 0.90)
abstaining on the 30% least predictable rows: MAE 2.36 -> 1.89
```

Methods fit LightGBM quantile models by default. Pass your own as `learner=` (any object with
`fit_mean(X, y)` and `fit_quantile(X, y, level)`), or hand the conformal step bands you already have:

```python
from selective_conformal import aci_from_quantiles

# lo_*/hi_*: your model's 5% and 95% quantile predictions; g_*: group labels; t_te: time step
lo, hi = aci_from_quantiles(lo_cal, hi_cal, y_cal, g_cal, lo_te, hi_te, y_te, t_te, g_te)
```

`cqr_from_quantiles` and `mondrian_from_quantiles` do the same for split and group-wise calibration.
On the default model's bands, all three match `cqr`, `mondrian_cqr` and `aci_cqr` exactly (tested).

## Reproduce

Python 3.10 or later. CI tests 3.10, 3.12 and 3.13; the published numbers come from 3.12 with
`requirements-lock.txt`.

```
pip install -r requirements-lock.txt && pip install -e . --no-deps   # exact tested versions
# or: pip install -e ".[experiments]"                                  # latest compatible versions
python reproduce.py           # data, P1, P2 and the figures: about 4 minutes
python reproduce.py --all     # everything: about 2.5 hours, or 1 hour with --jobs 3
```

`reproduce.py` runs each driver in [`experiments/`](experiments/) and reports "matches" or "differs"
against the output recorded in `experiments/expected_output/`; logs and differences go to
`build/reproduce/`. On 12 October 2026, from a fresh clone with fresh data, all 30 runs matched.
LightGBM runs deterministically with no seed set, so its sampling seeds keep their fixed defaults,
and the outputs are the same at 4 threads and on all cores.

The data is public daily prices from Yahoo (via yfinance) over a pinned 2018 to 2024 window,
downloaded fresh because Yahoo's terms don't allow committing it. `experiments/manifest.py` checks
each built table against fingerprints of the published data, so a revised history shows up as a
named difference. One revision is known (Honeywell, about one part in ten million) and changes no
printed number. `panel.py` stops on any ticker that fails to download (`--allow-missing` skips it).

On Windows without long-path support, keep the virtual environment's path short (scikit-learn nests
about 110 characters deep, against a 260-character limit).

## Research log

Each phase froze its decision rule in a pre-registration, then recorded the outcome in a results
file. Results that needed fixing carry a dated "Corrections (Oct 2026)" block, with the original text
left as it was.

| Phase | What it tested | Verdict, as pre-registered | Files |
|---|---|---|---|
| P1 | Plain conformal under the shift; interval width as the abstention score | Baseline, no gate: 0.902 in calm data, worst group 0.507 under the shift; width no better than random | [pre-reg](reports/p1_baselines_preregistration.md), [results](reports/p1_baselines_results.md) |
| P2 | An abstention score plus online group-conditional calibration | 3 of 5 checks pass (table above) | [pre-reg](reports/p2_method_preregistration.md), [results](reports/p2_method_results.md) |
| P2.1 | A union over three grouping axes | Not met: over-covers, 12% wider | [pre-reg](reports/p21_multiaxis_preregistration.md), [results](reports/p21_multiaxis_results.md) |
| P2.2 | A calibrated multi-group threshold | Not met: misses on sector, not shift-robust | [pre-reg](reports/p22_conditional_preregistration.md), [results](reports/p22_conditional_results.md) |
| P2.3 | Per-group online correction | Not met: worst group 0.599 to 0.651, short of the floor | [pre-reg](reports/p23_adaptive_preregistration.md), [results](reports/p23_adaptive_results.md) |
| P3 | Confirmation on non-public data | Not in this release | |
| P4, P5 | Methods note; longer archival note | Writeups | [p4](reports/p4_writeup.md), [p5](reports/p5_archival_note.md) |
| P6 | Quantile-level adaptation on one conditional shape | Not met: calm coverage nominal, floor missed | [pre-reg](reports/p6_alphaspace_preregistration.md), [results](reports/p6_alphaspace_results.md) |
| P7 | Combine grouping axes by the most conservative one | Not met: +0.004 worst-group on the synthetic, +0.037 on real returns | [pre-reg](reports/p7_levelcomb_preregistration.md), [results](reports/p7_levelcomb_results.md) |
| P8 | One joint region on the simplex | Not met: 0.923 in calm data, 0.000 under the shift | [pre-reg](reports/p8_simplex_preregistration.md), [results](reports/p8_simplex_results.md) |
| P9 | Selective conformal risk control | Passes: 0.080 online against 0.184 static under the shift | [pre-reg](reports/p9_riskcontrol_preregistration.md), [results](reports/p9_riskcontrol_results.md) |
| P10 | Kernel localization against discrete groups | Passes, but only 0.7% tighter | [pre-reg](reports/p10_localized_preregistration.md), [results](reports/p10_localized_results.md) |
| P11 | An e-value monitor for the coverage break | Not met: flags the break early, but also raises false alarms | [pre-reg](reports/p11_evalue_preregistration.md), [results](reports/p11_evalue_results.md) |
| P12 | A learned regime axis when regimes recur | Not met: +0.007, matched by a shuffled axis. Post hoc, +0.068 on the calibrated threshold, equal to an oracle | [pre-reg](reports/p12_regime_axis_preregistration.md), [results](reports/p12_regime_axis_results.md) |
| P12b | Confirm of that post-hoc result on three fresh seeds | Passes, 6 of 6 gates: +0.055, noisy per seed; the online version lifts on every seed | [pre-reg](reports/p12b_calibrated_seam_preregistration.md), [results](reports/p12b_calibrated_seam_results.md) |
| P13 | Selection-aware calibration for the kept stocks | Passes on the seed mean: 0.896 against P2's 0.928, 12% narrower; noisy per seed, online nominal on every seed | [pre-reg](reports/p13_selection_aware_preregistration.md), [results](reports/p13_selection_aware_results.md) |

## Limits

- The synthetic layer shows how the method behaves when the noise floor is known, not how large that
  floor is in any real task; the real-returns layer is the external check.
- P2's kept-stock coverage is empirical. P13's is finite-sample valid under exchangeability, and
  nominal on the seed mean but not on every seed.
- Two verdicts sit on their thresholds (P9: 0.0798 against 0.08; P11: e = 20.02 against 20), so tiny
  numerical drift could flip either.
- Passages that refer to the non-public confirmation run, or to documents kept with it, are replaced
  by a marked redaction. Otherwise the frozen files changed in October 2026 only by a few marked term
  edits and the dated Corrections blocks. P13 was registered and committed before it ran.
- [`SPEC.md`](SPEC.md) is the June 2026 design document; where results differ, the README and
  `reports/` are the record.

## Layout

| Path | What |
|---|---|
| `src/selective_conformal/` | The package: conformal methods, learners, abstention score, metrics and extensions |
| `experiments/` | Data pipeline, data manifest, one driver per phase, figures, recorded outputs |
| `tests/` | pytest, no network, under a minute |
| `examples/` | The quickstart |
| `reports/` | A pre-registration and a results file per phase, plus two notes |
| `figures/` | The committed figures |
| `reproduce.py` | Rerun and check the published results |
| `data/` | Gitignored, built locally |

The frozen reports name the drivers by their v0.2.0 paths (`src/run_p2.py`); they now live in
`experiments/`.

## Citation and license

Cite via [`CITATION.cff`](CITATION.cff). MIT license ([`LICENSE`](LICENSE)).
