# Results — P2.1 multi-axis conditional coverage (2026-06-30)

> Paired with `p21_multiaxis_preregistration.md` (frozen first). All numbers from
> `python src/run_p21.py`, seeds {0,1,2}, target 90%. The frozen single-axis `aci_cqr`
> (dim=`mom_decile`) is untouched; `aci_cqr_multi` is the additive variant. Reproduces
> bit-for-bit.

## Headline — **QUALIFIED (one-sided win, two-sided over-correction).**
The multi-axis union does exactly what it was built to do — **eliminates worst-group
*under*-coverage across all three axes** — but it does so by **over-covering**, so on the
two-sided ±τ retained-coverage metric it does not cleanly ship. It is a *conservative* fix:
the right tool when the disease is under-coverage, too blunt when the task is already calibrated.

## Layer 1a — coverage + width (mean ± sd over seeds; target 90%)
| method | base | shift | worst-group \| shift | mean width |
|---|---|---|---|---|
| ACI single-axis (frozen) | 0.904 ± 0.001 | 0.867 ± 0.015 | 0.815 ± 0.029 (sector=utilities) | 8.09 |
| **ACI multi-axis (P2.1)** | 0.934 ± 0.001 | **0.915 ± 0.010** | **0.881 ± 0.021** (sector=utilities) | 9.08 |

- **Worst-group\|shift lifts 0.815 → 0.881** and shift coverage 0.867 → 0.915 — the union ⊇ each
  axis's interval, so no axis under-covers. Seed-0 clustered band on multi-axis shift coverage
  **0.908 [0.897, 0.918]**.
- **Cost:** marginal coverage over-shoots (base 0.904 → 0.934, shift 0.867 → 0.915 past nominal)
  and mean width inflates **+12%** (8.09 → 9.08). This is the efficiency price of a distribution-
  free multi-group guarantee via the conservative union.

## G-COND-MULTI — retained-70% conditional coverage, max |gap| (mean over seeds)
| axis | single-axis | **multi-axis** | vs τ=5pp |
|---|---|---|---|
| sector (the P2 failing axis) | 0.060 | 0.077 | ✗ |
| cap_tier | 0.034 | 0.057 | ✗ |
| mom_decile | 0.032 | 0.059 | ✗ |

**The two-sided metric gets *worse*, not better.** Because the union over-covers, retained-set
group coverage now sits **above** 90% by more than τ on every axis — a |gap| miss in the other
direction. So multi-axis conditioning does **not** deliver *calibrated* (±τ) conditional coverage;
it delivers *conservative* (≥ nominal, no under-coverage) conditional coverage. An honest negative
on the gate as written, and an honest positive on the one-sided property.

## G-SHIFT re-confirm — **PASS** (and dominates single-axis)
Multi-axis worst-group\|shift **0.881 ≥ 0.815** (single) **≥ 0.704** (floor); shift coverage 0.915
within ±7pp. The variant strictly improves the worst-group under-coverage G-SHIFT measures.

## G-PRIMARY (i) / G-NULL — unchanged
The selective score `u_i` is the per-name calibration-residual scale, independent of the conformal
interval, so the abstention results are identical to P2 (G-PRIMARY i PASS, G-NULL clean). The
multi-axis change touches only interval *width*, not the abstention *ranking*.

## Layer 1b (real 21d fwd return)
| method | base | shift | worst-group \| shift | width |
|---|---|---|---|---|
| ACI single-axis | 0.898 | 0.898 | 0.858 (materials) | 0.275 |
| **ACI multi-axis** | 0.933 | 0.934 | **0.896** (materials) | 0.306 |
Same pattern on real returns: worst-group lifts 0.858 → 0.896, marginal over-shoots, width +11%.

## Verdict (1a/1b)
**QUALIFIED — do NOT declare a clean fix (per the pre-committed rule).** Multi-axis ACI provably
removes worst-group *under*-coverage across all axes (G-SHIFT improves and dominates single-axis),
but at the cost of conservative over-coverage and ~12% wider intervals, so calibrated (±τ
two-sided) conditional coverage is **not** achieved on the already-calibrated synthetic.

[redacted: refers to a non-public confirmation run] The *calibrated* multi-group variant (finest-cell with hierarchical fallback,
or a Gibbs–Cherian–Candès linear-class conditional conformal that does not systematically
over-cover) is logged as a future pre-registered experiment — **not** retuned here.
