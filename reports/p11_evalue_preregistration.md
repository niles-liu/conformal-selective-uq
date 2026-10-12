# Pre-registration — P11 *e-value coverage-break monitor* (2026-07-01)

> **Gate-first.** Frozen before `python src/run_p11.py`. P11 is a **diagnostic** (SPEC §11, lowest
> priority — "keep it a diagnostic or it drifts into a second paper"), so the gates are **descriptive**;
> [redacted: refers to a non-public confirmation run]. Paired with `reports/p11_evalue_results.md`.

## Question
Can an **anytime-valid** detector flag *when* conditional coverage breaks over the walk-forward stream —
a companion to G-SHIFT (flag the break, don't merely tolerate it) — that fires for a method whose
coverage collapses under the drawdown and **stays silent** for a shift-robust method and under nominal
coverage (no false alarm)?

## The monitor (frozen spec — `src/evalue_monitor.py`)
Testing-by-betting. Under `H0` (per-session miscoverage rate = `α=0.10`) the wealth process
`W_t = ∏_{s≤t} (1 + λ_s (miss_rate_s − α))` is a non-negative martingale with `E[W_t] ≤ 1`, so by
Ville's inequality `P(sup_t W_t ≥ 1/δ) ≤ δ`. Crossing `1/δ` is an **anytime-valid** alarm that coverage
dropped below nominal (over-coverage never triggers it — the payoff is signed toward under-coverage).
The bet `λ_s` is a clamped aGRAPA strategy computed **prequentially** from past sessions only (preserves
the martingale property) — a betting strategy, not a validity-affecting knob. **Frozen:** `δ=0.05` (alarm
threshold `1/δ=20`), bet cap `λ_cap=0.5`, `α=0.10`.

## Data / streams
Layer **1a**, seeds {0,1,2}. Test stream in date order: **2022 drawdown (shift) first, then 2023-24
(base)**. Per-session miscoverage rate from the frozen P2 interval methods (`naive` split-conformal;
`aci` group-conditional ACI). Three streams:
- **naive / full** — expected to fire inside the 2022 shift window (coverage collapses, P1: shift 0.58);
- **group-ACI / full** — expected NOT to fire (shift-robust, holds coverage);
- **naive / base-only** (2023-24) — the false-alarm control (naive is nominal in base, P1: 0.90).

## Pre-committed gates (descriptive)
| Gate | Statistic | Read |
|---|---|---|
| **G-DETECT** | naive full-stream e-value crossing vs group-ACI | naive **fires within the shift window** on all seeds **AND** group-ACI does **not** fire — the monitor discriminates break from hold |
| **G-NOFALSE** | naive base-only e-value | does **not** cross on any seed — no false alarm under nominal coverage (Ville validity in practice) |

No ship/reject. The result is a diagnostic add-on to the G-SHIFT narrative.

## Seeds / reproducibility
Seeds {0,1,2}; deterministic (seeded synthetic, frozen learners, deterministic monitor). Run:
`python src/run_p11.py`. Reproduces bit-for-bit.

## 4-lens adversarial check (fill in results doc, after)
- **Reproduce** — deterministic; interval methods are the frozen P2 objects.
- **Leakage** — the monitor consumes only realised per-session miscoverage (post-hoc labels are legitimate
  for a *monitor*); the bet `λ_s` uses only sessions `<s` (prequential) → the martingale/Ville guarantee
  holds; no interval method sees test labels in calibration.
- **Statistics** — anytime-valid by construction (no multiple-testing correction needed); report the
  e-value path + fire session, not a p-value.
- **Mechanism** — does the alarm land in the *shift* window (a real break) rather than firing on base
  noise? G-NOFALSE (base-only control) is the guard; check the fire session's date/regime.
