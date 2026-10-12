# SPEC: Selective Conformal Regression with Conditional Coverage Under Shift

> **Note (Oct 2026).** This is the June 2026 design document. It records the planned gates, methods
> and phases; it is not a results document, and the results differ from the plan in places (two of
> the five headline gates failed as pre-registered). See the [`README.md`](README.md) and
> [`reports/`](reports/) for what was found, in particular the short note
> [`reports/p4_writeup.md`](reports/p4_writeup.md) and the archival note
> [`reports/p5_archival_note.md`](reports/p5_archival_note.md).
>
> Author: Niles Liu.

---

## 0. One-paragraph thesis

Conformal prediction can certify that an interval has nominal coverage; selective prediction can
trade coverage for accuracy by abstaining. What neither can usually show is whether a model
**abstains in the right place**, because real tasks carry no ground truth for "which points were
ever recoverable." The plan is to build a regression task that **does** carry that ground truth (a
known recoverable-vs-irreducible decomposition), use it to build a selective conformal regressor,
test whether that regressor keeps **group-conditional coverage under a documented distribution
shift**, and check its abstention against the known noise. The method is portable. The
demonstration task is a semi-synthetic portfolio-allocation problem on public US equity data. Each
stock's weight mixes a part the public features can recover with a hidden component the model never
sees, so the unrecoverable share is known for every stock.

---

## 1. Motivation

On real data, "is the model abstaining on the right points?" has no answer, because nobody knows
which points were ever predictable. A task whose unrecoverable share is fixed by construction turns
it into a precise, *checkable* question: abstain where the hidden component dominates; commit where
the public features carry the weight. The hidden share is a generator parameter, chosen to make the
check possible, with no claim that it mirrors any real task. The aim is a portable
uncertainty-quantification method (conditional and shift-robust selective prediction) whose
abstention can be checked against ground truth.

---

## 2. Where the literature is, and the gap we take

The space is active. What exists:

- **Nearest prior work: "When Alpha Breaks: Two-Level Uncertainty for Safe Deployment of
  Cross-Sectional Stock Rankers" (2026, arXiv 2603.13252).** Conformal plus aleatoric/epistemic
  uncertainty on cross-sectional stock **rankers**, regime-conditional coverage, and abstention for
  safe trading deployment. The nearest neighbour, but it is **return/ranking prediction for
  trading**, with external feeds and deployment framing.
- **Selective Conformal Risk Control (2025, arXiv 2512.12844).** Select-then-calibrate with two λ's,
  selective coverage plus conditional risk, but **classification only**, on generic datasets. The
  paper gestures at a regression setting but does not build it, so the selective conformal
  **regression** lane (item 2 below) stays open (see §11 P9).
- **Field movement** (sharpens positioning and seeds §11): compositional-data conformal on the
  simplex (arXiv 2511.18141, 2026), marginal-only, with no selection, shift or known floor (→ §11
  P8); fast kernel/RKHS **continuous** conditional coverage (SpeedCP arXiv 2509.24100;
  Shape-Adaptive 2603.23374; Enhanced Localized CP, JASA 2026), a modern comparator beyond
  Gibbs–Cherian–Candès (→ §11 P10); e-value / anytime-valid coverage monitoring (arXiv 2503.13050;
  "When Your Model Stops Working" 2603.13156), an optional shift-break detector (→ §11 P11).
- **Backbone:** CQR (Romano et al. 2019), weighted conformal under covariate shift (Tibshirani et al.
  2019), Adaptive Conformal Inference (Gibbs & Candès 2021), conditional-guarantee conformal (Gibbs,
  Cherian & Candès 2023), conformal beyond exchangeability (Barber et al. 2023). Hard wall: the
  **distribution-free conditional-coverage impossibility** (Foygel Barber et al. 2021). Respect it;
  never claim exact distribution-free conditional coverage.

**The gap we take**, distinct from the above on several axes at once:

1. **A target with a known unrecoverable share**, not returns or rankings: a supervised regression
   whose irreducible-noise floor is set by the generator, unlike the efficient-market near-zero-R²
   setting.
2. **Selective *regression*** with conditional coverage. SCRC is classification; selective conformal
   *regression* with both selective coverage and group-conditional coverage is comparatively open.
3. **A documented shift**: a train-calm, test-drawdown split on the real 2022 drawdown, with
   concrete, pre-registered shift axes (on 1a the generator also amplifies the hidden component in
   that window).
4. **The known-floor mechanism check**, the central test: *does the abstention region coincide with
   the genuinely recoverable signal?* Studies on real data cannot ask this; here the task is built to
   have ground truth for recoverability.
5. **Pre-registration discipline**: gate-first ship/reject, shuffle nulls, clustered bootstrap,
   multi-seed bands and a 4-lens adversarial check.

---

## 3. The planned contribution

> **Planned (June 2026):** a pre-registered selective conformal regression framework, tested for
> group-conditional coverage under a documented distribution shift and, on a task with a known
> recoverable/irreducible decomposition, checked for whether the abstained-from region coincides
> with the hidden-component signal while retained-set intervals stay calibrated.

- **Headline (the defensible hook):** the *known-floor evaluation*, i.e. selective UQ that can be
  checked for abstaining in the *right* place, not merely for abstaining.
- **Portable:** the method is domain-agnostic; finance is only the demonstration.
- **Honest boundary:** group-conditional, approximate and validated coverage, never distribution-free
  exact conditional coverage (Foygel Barber 2021). No trading claim.

As run (see `reports/p2_method_results.md`): useful abstention, the mechanism check and the leakage
guard passed; group coverage on the retained set and worst-group coverage under the shift failed as
pre-registered.

---

## 4. Substrate: layered, public-primary

### Layer 1a: known-ceiling semi-synthetic (the controlled experiment)
A transparent, fully public generator we build ourselves:
`w_i = base(cap-diversified) × momentum_tilt_i(public features) + λ · hidden_overlay_i + ε_i`,
where `hidden_overlay_i` is a seeded function the *learner is not given* (the hidden component) and
`ε_i` is irreducible noise. **We therefore know, per stock, how recoverable the weight is** (the
recoverable fraction = variance from public-feature terms / total). This is the only way to test
G-MECHANISM. Knobs (frozen in the pre-registration): overlay strength `λ`, noise scale,
cross-section size, number of periods, the shift generator. Tuned so the public-recoverable R² lands
near 0.40 (a deliberately hard but non-trivial recoverable share); the share is a generator
parameter, not a claim about any real task.

### Layer 1b: real public task (external validity; guards against a self-fulfilling synthetic)
Predict a genuinely hard *public* target from the same point-in-time panel, with a real
**train-pre-drawdown, test-drawdown** split: 21-day cross-sectional forward returns, a classic and
honest target with a real regime shift and near-zero mean R², so the selective story is stark. 1b
has no ground-truth recoverability mask (so it cannot carry G-MECHANISM), but it tests the method on
a real, non-synthetic financial regression with a real shift (G-PRIMARY / G-SHIFT / G-NULL).

### Not in this repository
Some pre-registrations refer to a confirmation run on non-public data; it is not part of this
release.

---

## 5. Pre-registered gates (the single ship/reject)

Structures are fixed here; **numbers are frozen in `reports/p1_baselines_preregistration.md` after a
first baseline read and before the method runs.** Everything is evaluated over at least three seeds
with bands; coverage CIs come from a clustered/blocked bootstrap over the sector/regime cluster.

### G-PRIMARY: "the UQ layer adds something" (the gate that decides ship/reject)
At a pre-specified target coverage (default **90%**), the selective conformal regressor must do
**both**, jointly:
- **(i) useful abstention**: at ≥ **X%** retention, retained-set predictive error (and/or mean
  interval width) is reduced by ≥ **Δ** versus *random* abstention at the same retention; **and**
- **(ii) conditional coverage held**: empirical coverage per group (sector / cap tier / momentum
  decile) stays within **±τ** of nominal on the retained set.

Beating the risk–coverage curve *while* holding conditional coverage, not one or the other, is the
bar. Failing either ⇒ **reject**.

### G-SHIFT: shift-robust conditional coverage
On the out-of-regime split (train calm, test drawdown), the shift-robust variant holds worst-group
coverage within **±τ_shift**, in a setting where **naive split-conformal demonstrably fails** (we
report the naive worst-group coverage gap first; the method must close it by at least a
pre-committed margin).

### G-MECHANISM: abstention lands in the right place (1a only; the headline)
On the known-ceiling task: (a) confident (non-abstained) stocks vs the truly-recoverable mask →
**AUC ≥ θ**; (b) per-stock uncertainty vs injected irreducible-noise magnitude → **Spearman ρ ≥ θ′**
(planned here as interval width; the P2 pre-registration uses the calibration-residual score `u_i`,
because P1 found width uninformative). This is "validate the mechanism, not the P&L", applied to UQ.

### G-NULL: leakage guard
Re-run with **label-shuffled** targets: the selective advantage (G-PRIMARY (i)) must collapse to the
random-abstention baseline (within band). If it survives the shuffle there is leakage: stop and fix.

---

## 6. Method stack and baselines

**Base learners.** Mean: GBM (LightGBM). Quantile: LightGBM quantile (or linear quantile) at the
lower and upper levels for CQR.

**Conformal layers** (each a thin, composable module in `src/conformal.py`):
1. **Naive split-conformal** (marginal coverage): baseline 1.
2. **CQR** (Romano 2019): heteroscedastic intervals; baseline 2 (marginal) and the body of the method.
3. **Mondrian / group-conditional**: separate calibration per group (sector / cap tier / momentum
   decile) for conditional coverage.
4. **Shift-robust**: weighted conformal with covariate-shift weights (Tibshirani 2019) and/or
   Adaptive Conformal Inference (Gibbs & Candès 2021) over the regime axis.
5. **Conditional-guarantee comparator**: Gibbs–Cherian–Candès (2023) as the reference point.

**Selective layer** (`src/selective.py`): abstain by interval width or normalized uncertainty, and
sweep the abstention threshold to get a **risk–coverage curve**. Optionally cast as a two-stage
select-then-calibrate procedure (the SCRC structure) but in **regression**, the open variant (see
§11 P9).

**Baselines before models** (`src/eval.py` computes these first): naive split-conformal (marginal
only), unconditional CQR, **random abstention** (the G-PRIMARY comparator) and, on 1a only, an
**oracle abstention** that knows the noise (an upper bound on how good selection can get).

---

## 7. Phases and deliverables

| Phase | Deliverable | Gate(s) |
|---|---|---|
| **P0** setup + pre-registration | public panel built; `reports/p1_baselines_preregistration.md` with frozen numbers; 1a generator spec | n/a |
| **P1** baselines | naive + CQR marginal coverage on 1a and 1b; **demonstrate naive conditional failure** | establishes the G-SHIFT target |
| **P2** method | group-conditional + shift-robust selective CQR; **risk–coverage curve**; conditional-coverage table (bands); 1a mechanism plots | **G-PRIMARY / G-SHIFT / G-MECHANISM / G-NULL** |
| **P2.1 to P2.3** | multi-axis union, then a calibrated coupled shape, then per-group online (the conditional-coverage lineage; report files `p21`, `p22`, `p23`) | per-phase pre-registration |
| **P4** writeup | short note + README; committed figures | n/a |
| **P5** archival note | the single contribution sharpened; the five-method tradeoff surface and its diagnosed design tension | n/a |
| **P6/P7** | alpha-space group-conditional adaptation; cross-axis level combination (address the P5 tension) | per-phase pre-registration |
| **P8 to P11** | extension sprints (§11): simplex, risk control, localized comparator, e-value monitor | per-phase pre-registration |

**Minimum complete result (P2 + P4), as planned:** a clean repo and note in which the risk–coverage
curve beats the baselines *while* conditional coverage holds under a real shift, and the abstention
region is shown (on 1a) to coincide with the recoverable signal, reproducible on a laptop. As run,
the conditional-coverage half did not hold (see `reports/p2_method_results.md`).

---

## 8. Risks and mitigations
- **Overlap with the nearest prior work ("When Alpha Breaks").** Lean on the differentiators (§2),
  especially the **known-floor target and mechanism check**, which that paper does not have. Position
  the work as method plus evaluation, not trading deployment.
- **Conditional-coverage impossibility (Foygel Barber 2021).** State the coverage claim honestly
  (group-conditional, validated empirically), never distribution-free exact.
- **Self-fulfilling synthetic.** 1b (the real public task) is the external-validity guard; G-NULL is
  the leakage guard.
- **Thin cross-sections.** Blocked/clustered resampling; report power; don't slice conditional groups
  more finely than the sample supports.
- **Scope creep toward a trading strategy.** Hard non-goal: no alpha, no backtest P&L. UQ method only.

---

## 9. Open design knobs (fixed in the P1 pre-registration, not before)
- G-PRIMARY numbers: target coverage (default 90%), retention X%, error/width margin Δ, tolerance τ.
- G-SHIFT: which drawdown window defines the shift; τ_shift; the naive-failure margin.
- G-MECHANISM: AUC θ, uncertainty-noise ρ θ′.
- 1a generator: overlay strength λ, noise scale, cross-section size, number of periods and shift
  generator, tuned so the public-recoverable R² is about 0.40.

---

## 10. References (anchors)
- Vovk, Gammerman, Shafer: *Algorithmic Learning in a Random World* (conformal foundations).
- Romano, Patterson, Candès (2019): **Conformalized Quantile Regression**.
- Tibshirani, Foygel Barber, Candès, Ramdas (2019): **Conformal under covariate shift** (weighted).
- Gibbs & Candès (2021): **Adaptive Conformal Inference under distribution shift**.
- Foygel Barber, Candès, Ramdas, Tibshirani (2021): **Limits of distribution-free conditional inference** (impossibility).
- Gibbs, Cherian, Candès (2023): **Conformal Prediction with Conditional Guarantees** (arXiv 2305.12616).
- Barber, Candès, Ramdas, Tibshirani (2023): **Conformal Prediction Beyond Exchangeability**.
- Angelopoulos & Bates: *A Gentle Introduction to Conformal Prediction* / conformal risk control.
- El-Yaniv & Wiener: selective prediction / risk–coverage.
- "When Alpha Breaks" (2026, arXiv 2603.13252): nearest prior work (cross-sectional stock rankers).
- Selective Conformal Risk Control (2025, arXiv 2512.12844): select-then-calibrate (classification).
- Conformal Prediction for Compositional Data (arXiv 2511.18141, 2026): simplex-geometry conformal regions (→ P8).
- SpeedCP (arXiv 2509.24100, 2025); Shape-Adaptive (arXiv 2603.23374, 2026); Enhanced Localized CP (JASA 2026): localized-conditional comparators (→ P10).
- E-Values Expand the Scope of Conformal Prediction (arXiv 2503.13050, 2025); "When Your Model Stops Working" (arXiv 2603.13156, 2026): anytime-valid monitor (→ P11).

---

## 11. Extension sprints (P8 to P11)

Ranked by leverage (P8 highest). Each keeps every binding constraint (public data only, gate-first
pre-registration, no alpha or P&L claim, honest group-conditional claims). A sprint's
ship/reject numbers are frozen in `reports/p<N>_*_preregistration.md` *after* a method-blind
baseline read and *before* the method runs, never at planning time; the descriptions below fix
*structure* only. All four were run; outcomes are in `reports/p{8,9,10,11}_*` (results files open
with an Oct 2026 Corrections block where one was needed) and in the archival note §13.

### P8: conformal on the simplex (the target is compositional; highest leverage)
The allocation is a **vector** (non-negative and summing to one, i.e. compositional), but a
scalar-per-weight pipeline treats each coordinate independently. Simplex-geometry conformal exists
(arXiv 2511.18141) but is **marginal-only and non-selective, with no shift and no known floor**. The
plan: test the intersection *simplex region × selective abstention × group-conditional-under-shift ×
known-floor mechanism*, certifying the allocation as one point on the simplex rather than as N
unrelated scalars. New score and region constructor (`src/simplex.py`), coverage redefined as *true
weight vector ∈ region*, and **G-MECHANISM redefined on the simplex** (does compositional difficulty
track the known irreducible σ?).

### P9: selective conformal *risk* control in regression (the "open variant" §6 names)
Upgrade the selective layer from coverage only to a **monotone bounded risk** (a capped
miss-magnitude loss) with two-λ (select-then-calibrate) control, turning "we abstain and coverage
holds" into "we **control a risk** on the retained set". Mostly `src/riskcontrol.py` (a second
calibration pass plus an online variant), reusing the risk–coverage machinery in `src/eval.py`. New
G-RISK; G-NULL must still collapse the advantage. (As run, the selection score uses calibration
labels, so the retained-set control is empirical; see the P9 results.)

### P10: localized / kernel-conditional comparator arm
Conditional coverage is Mondrian (discrete groups). Add a **continuous** approximate-conditional
comparator (`conformal.localized_cqr`; SpeedCP / Shape-Adaptive / Enhanced Localized CP) as a modern
reference beyond Gibbs–Cherian–Candès, and check whether it tightens intervals at equal conditional
coverage. It slots into the existing conditional-coverage table; no gate redefinition.

### P11: e-value anytime-valid coverage-break monitor over the regime axis
An **anytime-valid detector for *when* conditional coverage breaks** in the drawdown window (e-value
conformal; testing by betting). A monitoring utility (`src/evalue_monitor.py`) over the walk-forward
stream: reporting only, a diagnostic companion to the G-SHIFT story, not a change to the interval
methods.

**Do not open:** training-conditional coverage bounds (arXiv 2405.16594 / 2602.16537), which are
theory-heavy and not a laptop experiment; and any LLM or foundation-model abstention tie-in, which is
off scope.
