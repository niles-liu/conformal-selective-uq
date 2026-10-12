# Conditional coverage under shift for *selective* regression
## A calibrated-shape / online-correction tradeoff surface, checked against a known noise floor

**Niles Liu · 2026-06-30, extended 2026-07-01, revised Oct 2026 · archival methods note (P5)**

> **How to read this note.** It was written incrementally: §1 to §11 on 2026-06-30, then §12
> (P6/P7) and §13 (P8 to P11) as those phases ran. **§12 and §13 supersede §5 and §8** wherever they
> disagree. The Oct 2026 revision makes every verdict match its pre-registration (see the
> Corrections blocks at the top of the results files) and removes overclaims; the rest is unchanged.
>
> **What this note is.** The short note ([`p4_writeup.md`](p4_writeup.md)) gives the gap, the
> headline gates and the mechanism figure. This is the archival version. Its contribution is not any
> one estimator but the **lineage**: a family of conditional-coverage methods, run head to head, that
> maps the calibrated-vs-online, union-vs-coupled, global-vs-group **and width-space-vs-alpha-space**
> tradeoff surface for *selective* CQR under distribution shift, on a testbed whose irreducible noise
> is **known**. The surface first ended in one named design tension (§5.3); a pre-registered pair of
> follow-ups (**P6/P7**, §12) removed the base over-coverage behind it, at a cost in shift coverage.
> Four later sprints (**P8 to P11**, §13) added a joint region on the simplex, online risk control, a
> localized comparator and a break monitor; their verdicts are mixed and stated there. Every number a
> driver prints reproduces exactly from a fresh clone (verified Oct 2026, pandas 2.3 and 3.0); the
> Yahoo download is not frozen.

---

## 1. Contribution, in one paragraph

We study **group-conditional coverage under distribution shift** for a *selective* conformalized
quantile regressor, a setting where two literatures meet but are rarely tested together: conditional
conformal (coverage within groups) and adaptive conformal (coverage through time). There are two
contributions. **(C1) A checkable abstention test.** On a task where the recoverable-vs-irreducible
split is injected rather than assumed, the selective layer abstains where the noise is known to be
irreducible: the stocks it drops are mostly the hidden-component ones (AUC 0.81, ρ 0.86 over three
seeds, surviving a volatility control; `fig3`). **(C2) A tradeoff surface, not a point claim.** We
place five conditional estimators on one axis (*static-calibrated*, *single-axis online*,
*multi-axis union*, *global-online on the calibrated shape* and *group-online on the calibrated
shape*) and show that moving along the "add structure" direction raises worst-group coverage under
shift (0.433 → 0.599 → 0.651 on the adversarial synthetic; 0.706 → 0.740 → 0.766 on real returns)
at a width that stays **under the union**. The endpoint is a **named design tension**: a
*width-space* online correction on a *single fitted* conditional shape over-covers the base when
severe-shift and base regimes interleave, where a frozen *alpha-space* per-group rule does not.

**Headline gates (P2), as pre-registered** (rules and numbers in [`p4_writeup.md`](p4_writeup.md) §4
and the Corrections block of [`p2_method_results.md`](p2_method_results.md)): useful abstention
**PASS** (retained MAE 0.178 below random at 70%, ~75% of the oracle's headroom); group coverage
**FAIL** (sector 6.0pp off against 5pp); coverage under shift **FAIL** (online conformal's worst-group
0.815 meets the 0.704 floor but misses the ±7pp band; static conformal 0.507); abstention in the
right place **PASS** (AUC 0.81, ρ 0.86); leakage guard **PASS** (0.178 → 0.002).

The honest framing: **C1 is a clean PASS and is the headline; C2 is a mapped surface whose one named
tension is diagnosed and then partly resolved (P6/P7, §12).** Its value is the whole arc: why each
rung helps, where the coupled-shape family stops short, and how an alpha-space reconciliation removes
the width-space over-coverage at a cost in shift coverage. On the most adversarial synthetic no
coupled-shape variant clears the 0.704 worst-group floor, while single-axis ACI (0.815) and the union
(0.881) do; the floor miss belongs to the coupled-shape family under the severe shift, not to every
variant.

## 2. The gap (related work)

Conformal prediction certifies nominal **marginal** coverage; selective prediction trades coverage
for accuracy by **abstaining**. Neither, alone, can show whether a model abstains in the *right
place*: real tasks carry no ground truth for which points were ever recoverable, so "calibrated and
abstains on 30%" is unfalsifiable as a claim about *which* 30%.

- **Conditional coverage** is unattainable distribution-free (Foygel Barber et al. 2021); the
  achievable targets are **group-conditional / Mondrian** coverage (Vovk) and the **linear-class**
  relaxation of Gibbs, Cherian and Candès (2023): one coupled threshold from a quantile regression of
  conformity scores on group features.
- **Shift-robust coverage** through time is the domain of **Adaptive Conformal Inference** (Gibbs and
  Candès 2021) and **weighted/covariate-shift conformal** (Tibshirani et al. 2019).
- **Conformal methods in finance** (nearest prior work: "When Alpha Breaks", 2026) work on return and
  ranking prediction, with no known-noise check.

The intersection this note works in: **both coverage axes (group and time) on a *selective*
regressor, on a semi-synthetic target whose unrecoverable share is known by construction.** The last
clause is what lets us check the mechanism (C1); the first is what the method lineage explores (C2).

## 3. Substrate

- **Layer 1a: known-ceiling semi-synthetic (the controlled experiment).** The demonstration task is a
  semi-synthetic portfolio-allocation problem on public US equity data. Each stock's weight mixes a
  part the public features can recover with a hidden component the model never sees, so the
  unrecoverable share is known for every stock. Concretely,
  `w = base(cap-diversified) × momentum_tilt(public features) + λ·hidden_overlay + ε`, where the
  overlay is a seeded function the learner is **not** given and `ε` is irreducible noise. The hidden
  share is a generator parameter (tuned to a public-recoverable R² of about 0.40, a deliberately hard
  but non-trivial recoverable share), not a claim about any real task. The shift is a deliberately
  **severe** train-calm, test-drawdown regime, with base and shift sessions interleaved in the test
  stream.
- **Layer 1b: real public task (external validity).** 21-day cross-sectional forward returns from the
  same point-in-time panel, with a real calm-to-drawdown split. It guards against a self-fulfilling
  synthetic, and its realised shift is **milder** than 1a's deliberately severe one (static-conditional
  shift coverage 0.799 vs 0.564).

## 4. Method family (public features only; every knob frozen before each variant ran)

Common stack: LightGBM mean and pinball quantiles, then **CQR** heteroscedastic intervals (Romano
2019) with conformity score `E = max(qlo − y, y − qhi)`, then a conditional or online coverage layer
(the variable below), then a **selective** layer that abstains on a per-stock calibration-residual
scale `u_i` (the shrunk MAD of the stock's own calibration residuals, a point-in-time proxy for its
irreducible noise). `u_i` does not depend on the conformal interval, so it is identical across all
five variants, and the mechanism result (C1) and the leakage null are shared. Because `u_i` is fit
on calibration residuals and the conformal layer calibrates on the same fold, coverage on the
retained set is **empirical**, not a finite-sample guarantee.

The five conditional coverage layers, i.e. the tradeoff surface (C2):

| # | Variant (`conformal.*`) | What it adds | Phase |
|---|---|---|---|
| 1 | `aci_cqr`: **single-axis online** | ACI on one axis (momentum decile); per-group α-level update via empirical quantiles (**alpha-space**) | P2 |
| 2 | `aci_cqr_multi`: **multi-axis union** | independent ACI per axis, then the **union** of the three intervals | P2.1 |
| 3 | `cqr_conditional`: **static calibrated shape** | one coupled threshold `t(x)=φ(x)ᵀβ` from a **pinball regression of `E` on multi-axis group indicators** (Gibbs–Cherian–Candès), additive in sector + cap + momentum | P2.2 |
| 4 | `cqr_conditional_aci`: **+ global online** | shape (3) carried online with a single **global** width-space offset `c` | P2.2 |
| 5 | `cqr_conditional_gaci`: **+ group online** | shape (3) carried online with `c` **plus** per-(axis, level) deviation offsets `d[k,ℓ]`; `Δ(x)=c+Σ_k d[k,ℓ_k(x)]` | P2.3 |

**The unification (the spine of the surface).** Variant 5 is the general object: `Δ≡0` recovers the
static calibrated shape (3); `d≡0` recovers the global hybrid (4); the per-group `d` is the shift fix
that (1) and (2) achieve via per-group empirical quantiles or a union, here applied additively to a
*single fitted* shape. `c` owns the average drift and `d[k,ℓ]` owns each group's *residual* (each is
nudged toward the realised session marginal `err_t`), so the marginal is not triple-counted across
the three axes. `α=0.10`, a shared `γ=0.05` and `scale = median|t|` are frozen across all five.

## 5. Results: the surface (Layer 1a, seeds {0,1,2}, target 90%)

Pre-registered gates were frozen **before each variant ran** (`reports/*_preregistration.md`); no
knob was tuned afterwards, and `aci_cqr` / `aci_cqr_multi` / `cqr_conditional` /
`cqr_conditional_aci` were never re-fit when later variants were added.

| variant | base | shift | **worst-group \| shift** | mean width |
|---|---|---|---|---|
| single-axis ACI (P2) | 0.904 | 0.867 | 0.815 | 8.09 |
| multi-axis union (P2.1) | 0.934 | 0.915 | **0.881** | 9.08 |
| static calibrated shape (P2.2) | 0.898 | 0.564 | 0.433 | **6.18** |
| + global online (P2.2) | 0.965 | 0.702 | 0.599 | 8.38 |
| **+ group online (P2.3)** | 0.966 | 0.704 | **0.651** | 8.35 |

Three readings, each a finding:

1. **The width win.** The calibrated static shape is the **tightest** method on the board (6.18, −32%
   vs the union's 9.08) while marginally calibrated (0.898): coupling one threshold across axes beats
   unioning three separately calibrated intervals. It collapses under the shift (worst-group 0.433).
   *(P2.2 G-EFFICIENCY: PASS.)*
2. **The monotone shift signal.** Worst-group under shift climbs **static 0.433 → global 0.599 →
   group-online 0.651**; each rung adds exactly the structure pre-registered for it (static to online,
   then global to group-conditional). The group term lifts worst-group **+0.052 over the global offset
   at equal width** (8.35 vs 8.38), so in width-space per-group adaptation is the better online
   correction (§12 finds the reverse in alpha-space). The width win survives going online: gACI stays
   **under the union** (8.35 < 9.08). *(P2.3 G-EFFICIENCY: PASS.)*
3. **The wall (the named design tension).** On this *deliberately severe* synthetic the online family
   does **not** reach the union's worst-group (0.651 vs 0.881) and **over-covers the base** (0.966).
   The over-coverage is **not** a group-conditionality effect: the global hybrid already sits at
   0.965, and the group term adds 0.001. It is a **width-space ACI × interleaved-regime** artifact:
   under severe shift sessions the width-space offset `c` inflates and relaxes only at rate γ·scale
   per session, so the more numerous base sessions inherit the inflated width. The frozen
   **alpha-space** `aci_cqr` does **not** show this (base 0.904): its per-group α-level update with
   empirical quantiles corrects itself in a way an additive width-space offset on a *single fitted
   shape* does not. (P2.3's shift-marginal, 0.704, is also outside the ±0.07 band; see the
   Corrections block of `p23_adaptive_results.md`.)

## 6. Results: external validity (Layer 1b, real 21-day forward return)

On the real substrate the unification behaves as designed, because the realised shift is milder and
the width-space offset never over-inflates:

| variant | base | shift | worst-group \| shift | width |
|---|---|---|---|---|
| single-axis ACI | 0.898 | 0.898 | 0.858 | 0.275 |
| multi-axis union | 0.933 | 0.934 | 0.896 | 0.306 |
| static calibrated shape | 0.870 | 0.799 | 0.706 | 0.247 |
| + global online | 0.906 | 0.829 | 0.740 | 0.269 |
| **+ group online (P2.3)** | **0.910** | 0.832 | **0.766** | **0.268** |

As of P2.3, group-online was the **best conditional method on the real board** (§12 has better ones):
base near nominal (0.910, within ±τ), worst-group **0.766** (clears the 0.704 floor, beats the global
hybrid's 0.740), width 0.268 < union 0.306. The 1a/1b contrast is the lesson: the base over-coverage
and the floor miss on 1a come from the synthetic's deliberately severe shift acting on the
width-space offset. On a milder real shift the family delivers a calibrated, tighter-than-union,
group-robust combination that no single earlier variant reached.

## 7. Scope of this release

Everything in this note runs on public data with the code in this repository. Some
pre-registrations refer to a confirmation run on non-public data; it is not part of this release.

## 8. The open question: posed here, then answered (P6/P7, §12)

The surface ended on **one** well-identified tension, not a knob: a *width-space* additive offset on a
*single fitted* conditional shape over-covers the base under interleaved severe-shift and base
regimes. A frozen *alpha-space* per-group rule does not, but it seemed to need a per-group
empirical-quantile object, exactly the Mondrian per-group sparsity the coupled shape was built to
avoid. The open move we logged was **alpha-space group-conditional adaptation reconciled with the
single-regression shape**: adapt the *quantile level* per group instead of adding width, without
reintroducing per-group sparsity, as a **fresh pre-registered experiment**, not a retune.

**That experiment was run (P6), with its own follow-up (P7); both are in §12.** In short: an
alpha-space update read off a *fitted level family* `t_τ(x)` (one additive pinball regression per
level, all on the full calibration set, so no sparsity) **removes the base over-coverage at the
lineage's lowest width**, and a most-conservative cross-axis combination fixes an
additive-cancellation artifact it surfaced. Two things it does not do. It does not clear the
worst-group floor on the deliberately severe synthetic, a floor that bounds the coupled-shape
variants specifically (single-axis ACI, 0.815, and the always-over-covering union, 0.881, clear it).
And the base fix costs shift coverage (shift-marginal 0.704 → 0.679). Details, tables and verdicts:
§12.

## 9. Limits

- **No distribution-free exact conditional coverage.** It is impossible in general (Foygel Barber et
  al. 2021); the coverage here is group-conditional and empirical. The un-conditioned-axis miss
  (sector, on the retained set) is the visible edge of that limit. On the retained subset it is a
  **selective × conditional interaction** (abstention keeps the easier low-`u` stocks within a
  sector, so the retained subset over-covers), a property of the selective layer, logged and not
  tuned away.
- **Retained-set coverage and risk are empirical.** The abstention score `u_i` and the conformal
  (or risk-control) layer use the same calibration fold (§4). A selection-aware calibration is the
  natural next experiment.
- **The width-space over-coverage (§5.3)** is a real limitation of the coupled-shape online family
  under severe interleaved shift, reported in full; it is why P2.3 is QUALIFIED, not SHIP, on 1a.
- **Some verdicts sit on a cutoff.** P9's G-RISK-SHIFT is 0.0798 against 0.08 and P11's base-only
  false alarm is e = 20.02 against 20, so tiny numerical drift could flip either printed verdict.
  P2's sector gap (0.0601) sat 0.0001 past the original driver's non-pre-registered 0.06 "marginal"
  cutoff; against the pre-registered 0.05 it fails by 1pp.
- **The known noise is a generator parameter.** It makes the mechanism checkable; it says nothing
  about how much of any real allocation is recoverable.
- **No trading claim.** This is an uncertainty-quantification method on a semi-synthetic task; no
  returns or P&L are claimed.

## 10. Reproduce

```
pip install -r requirements.txt
python src/panel.py               # build the public panel (downloads Yahoo Finance data)
python src/run_p1.py              # baselines: marginal coverage holds; conditional fails under shift
python src/run_p2.py              # the headline gates, 1a + 1b
python src/run_p21.py             # P2.1: multi-axis union
python src/run_p22.py             # P2.2: calibrated coupled shape (static + global hybrid)
python src/run_p23.py             # P2.3: group-online on the coupled shape
python src/run_p6.py              # P6: width-space vs alpha-space 2x2
python src/run_p7.py              # P7: cross-axis level combination
python src/run_p8.py --baseline   # P8: method-blind baseline read
python src/run_p8.py              # P8: joint region on the simplex
python src/run_p9.py --baseline   # P9: method-blind baseline read
python src/run_p9.py              # P9: selective conformal risk control
python src/run_p10.py             # P10: localized comparator
python src/run_p11.py             # P11: e-value break monitor
python src/figures.py             # the three headline figures (seed 0)
```

The panel, P1, P2 and the figures take about 6 minutes; the full set takes about 97 minutes (P6 and
P7 are the long runs). Each driver pairs with a frozen `reports/<phase>_preregistration.md` and a
`reports/<phase>_results.md` (files `p21`, `p22` and `p23` are phases P2.1, P2.2 and P2.3). Every
number a driver prints reproduces exactly from a fresh clone (verified Oct 2026, pandas 2.3 and
3.0); the Yahoo download is not frozen.

## 11. References (anchors)

Vovk, Gammerman and Shafer, *Algorithmic Learning in a Random World* (conformal foundations; Mondrian).
CQR: Romano, Patterson and Candès (2019). Weighted conformal under covariate shift: Tibshirani, Foygel
Barber, Candès and Ramdas (2019). Adaptive Conformal Inference: Gibbs and Candès (2021). Limits of
distribution-free conditional inference: Foygel Barber, Candès, Ramdas and Tibshirani (2021).
Conditional (linear-class) conformal: Gibbs, Cherian and Candès (2023). Nearest prior work: "When
Alpha Breaks" (2026, arXiv 2603.13252).

---

## 12. Update: the base over-coverage removed (P6/P7, 2026-06-30)

The §5.3 endpoint named **one** tension: the P2.3 *width-space* online correction on the single
fitted conditional shape over-covers the base (1a base 0.966) under interleaved severe-shift and base
regimes, where a frozen *alpha-space* rule does not. §8 logged the reconciliation as a fresh
pre-registered experiment. It was run (P6) with a follow-up (P7). Both are **QUALIFIED**; together
they remove the base over-coverage, at a cost in shift coverage. Pre-registrations and results:
`reports/p6_alphaspace_*`, `reports/p7_levelcomb_*`.

**The new object.** A **fitted level family** `t_τ(x)`: one additive pinball regression of the CQR
conformity scores on the multi-axis group indicators per quantile level `τ∈{0.50,…,0.99}`, each on
the **full** calibration set (monotone-rearranged, linearly interpolated in level). It is the
conditional analogue of `aci_cqr`'s per-group empirical quantile *function*, **with no per-group
sparsity**, which is exactly what makes the alpha-space update runnable against the *single* coupled
shape. Online, the ACI update (the frozen `aci_cqr` form and rate, γ=0.05) adapts the quantile
**level** per group instead of adding width. This extends the surface to a full **(width-space vs
alpha-space) × (global vs group)** 2×2 on one shape, plus a cross-axis-combination axis (P7).

**The seven-variant surface (1a, seeds {0,1,2}, target 90%):**
| variant | correction | base | shift | worst-grp\|shift | width |
|---|---|---|---|---|---|
| ACI multi-axis union (P2.1) | n/a | 0.934 | 0.915 | **0.881** | 9.076 |
| static calibrated shape (P2.2) | n/a | 0.898 | 0.564 | 0.433 | 6.178 |
| + global online (P2.2) | width | 0.965 | 0.702 | 0.599 | 8.376 |
| + group online (P2.3) | width | 0.966 | 0.704 | 0.651 | 8.348 |
| + global online (P6) | **alpha** | 0.904 | 0.797 | 0.683 | 7.605 |
| + group online (P6) | **alpha** | **0.900** | 0.679 | 0.655 | 6.715 |
| + group, cross-axis max (P7) | **alpha** | 0.905 | 0.770 | **0.687** | 7.401 |

**What P6/P7 establish:**
1. **The base over-coverage is a width-space artifact, and alpha-space removes it (P6), at a cost.**
   Swapping only the correction *space* on the same fitted shape moves base **0.966 → 0.900** (on
   nominal) at the lineage's **lowest width** (6.715, −26% vs the union). Both alpha variants sit at
   nominal; both width variants over-cover (|base − nominal| alpha 0.000 vs width 0.066).
   **G-BASE-CALIB and G-EFFICIENCY pass.** The cost: the shift-marginal falls from 0.704 to 0.679 (both
   outside the ±0.07 band), with worst-group about level (0.651 → 0.655). An α_eff probe suggested
   the adaptation stays interior to its clamp, but no released driver regenerates it (see the
   Corrections block of `p6_alphaspace_results.md`).
2. **A cross-axis cancellation, then fixed (P7).** P6's *additive* cross-axis level combination let a
   stock that is easy on two axes and hard on one lose its widening to cancellation (its worst group
   drifted to a spurious momentum decile). Combining axes by the **most conservative** one (`min`, a
   union in level space on the single shape) restores group > global on worst-group and moves the
   worst group back to sector = tech. **G-GROUP-HELPS passes** (small on 1a, +0.004; **material on
   real 1b, +0.037**).
3. **The residual is a floor the coupled-shape family cannot reach on 1a.** No coupled-shape variant
   in either space clears the 0.704 worst-group floor on the *deliberately severe* synthetic, while
   single-axis ACI (0.815) and the union (0.881) do, so P6 and P7 are **QUALIFIED**. On the milder
   **real** shift (1b) the reconciled family is the **best coupled method in the lineage**:
   `cond+gqACI-max` reaches base 0.899 and shift 0.896 (both near nominal), worst-group **0.860**
   (approaching the union's 0.896), at width 0.281 < union 0.306.

**Takeaway.** The surface's named design tension is diagnosed, and its base over-coverage is removed
by a principled alpha-space reconciliation that costs shift coverage on 1a. The remaining gap, the
worst-group floor on the severe synthetic, bounds the coupled-shape family specifically. C1 remains
the headline. No frozen knob was tuned after the fact anywhere in the lineage.

---

## 13. Update: extension sprints (P8 to P11, 2026-07-01)

After §12, four **optional** sprints ran end to end, each gate-first (method-blind baseline read,
frozen pre-registration, method, 4-lens check, three seeds) and with **no frozen P1 to P7 knob
touched**. Pre-registrations and results: `reports/p{8,9,10,11}_*`.

**P8: conformal on the simplex (QUALIFIED).** The allocation is a *composition*, so P8 asks whether
the **whole weight vector** can be certified as one point on the simplex instead of as N separate
scalar intervals. In the base regime it can: a joint Aitchison-metric region covers the true vector
**0.923** of the time over three seeds (seed-0 band [0.892, 0.964]; a bounded-TV companion gives
0.904, an online ACI variant 0.902), where the independent-scalar **product region covers it 0.0000**
of the time (93 marginal 90% intervals never all contain the truth at once). **Under the shift the
joint region fails**: coverage **0.000** for the Aitchison region, static and online (TV 0.230),
because the whole-vector shift pushes every session's compositional distance past the calibration
support, and a quantile-reading online rule is capped by that support. **G-COMPO-SHIFT misses.**
G-COMPO-MECH passes as computed (AUC 0.861, Spearman 0.967, surviving the `vol_63` partial), but its
"compositional difficulty" is each stock's realized test error against the true weights, so it shows
that noisier stocks are harder to predict, not that an abstention mechanism ports to the simplex (see
the Corrections block of `p8_simplex_results.md`). A grid-refined HDR region over the simplex is
infeasible at about 90 stocks; the scalar-score metric ball is the stated laptop realization. 1b is
forward returns, not a composition, so P8 runs on 1a only.

**P9: selective conformal *risk* control in regression (SHIP on 1a by the pre-registered rule).**
Every variant above controls *coverage* (a 0/1 miss rate); P9 controls a **bounded risk**, capped
miss *magnitude*, via select-then-calibrate conformal risk control (the Angelopoulos–Bates bound),
published for classification and here built for regression. The rule (G-RISK-BASE ∧ G-RISK-SHIFT ∧
G-NULL) passes, on **one** substantive leg: the **online** `scrc_aci` holds retained-pool risk under
the drawdown at **0.080 vs the static 0.184** (retained shift coverage 0.471 → 0.738), with the
radius adapting (λ path 0 → 3.5 under the shift → 0 after). That leg sits on its threshold (0.0798
against 0.08). **G-RISK-BASE passes trivially** (at α_risk = 0.05 the batch CRC picks λ = 0, because
the raw CQR quantiles already meet the calm target) and **G-EFFICIENCY is a null** (no radius to
allocate, so "select-then-calibrate is tighter" is not shown on this task); neither was retuned.
G-NULL is clean (0.0108 → 0.0002). Because the selection score uses calibration labels, the
retained-set risk control is empirical, not a finite-sample guarantee. The lesson mirrors the whole
project's: selection helps a little, and the shift-robust interval machinery carries the result.

**P10: localized / kernel-conditional comparator (both descriptive gates pass, marginally).** A
continuous approximate-conditional arm (kernel-reweighted split CQR; Guan 2023, Hore and Barber 2024)
is **valid** (base 0.898), but at comparable base worst-group coverage it is only **0.7% tighter than
Mondrian** (6.191 vs 6.235; on par on 1b). Reading: the discrete groups already capture nearly all
the conditioning structure on this task, so the coupled-shape methods were not leaving a
continuous-conditioning win on the table. A comparator that confirms the design choice, not a new
method.

**P11: e-value coverage-break monitor (both pre-registered criteria fail; kept as a diagnostic).** A
testing-by-betting (Ville) monitor catches naive split-conformal's coverage break **early and inside
the shift on all three seeds** (median session 17). But G-DETECT also required group-ACI never to
fire, and its mild residual under-coverage trips the monitor late on 2 of 3 seeds (median session
144), so G-DETECT fails as written; the separation is by timing, which the pre-registration did not
specify. G-NOFALSE fails too: one base-only false alarm (seed 0, e = 20.02 against a threshold of
20; the frozen bet cap λ_cap = 0.5 is aggressive, and it was not retuned). Not promoted to a
headline.

**Takeaway.** C1, abstention checked against the known noise, stays the headline. P8 shows that a
valid joint region exists in the base regime where independent scalars certify nothing, but the
region collapses under the shift and its mechanism statistic is realized error, so it does not extend
C1. P9 adds online risk control under shift, an empirical result that sits on its threshold. P10 and
P11 are supporting arms: a continuous-conditional comparator that supports the Mondrian choice, and a
break monitor whose pre-registered criteria were not met. No frozen knob was tuned after the fact
anywhere in the lineage.
