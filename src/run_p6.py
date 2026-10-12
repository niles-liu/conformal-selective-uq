"""P6 driver — *alpha-space* group-conditional adaptation on the single-regression shape.

The 2x2 the experiment is built on, on ONE fitted conditional shape `t_tau(x)`:
                         global online            group online
  width-space  :  cond+ACI  (P2.2)          cond+gACI  (P2.3)   <- over-covers base (0.966)
  alpha-space  :  cond+qACI (P6)            cond+gqACI (P6)      <- HEADLINE: does alpha-space
                                                                    fix the base over-coverage?

The width row (and the single-axis ACI / union / static shape) are the FROZEN lineage, re-run
untouched for the head-to-head. The alpha row reads the conditional threshold off a level FAMILY
`t_tau(x)` (one additive pinball regression per level in `_QACI_GRID`, all on the full calibration
set -> no per-group sparsity), adapting the quantile LEVEL per group instead of adding width.

Gate-first spec: reports/p6_alphaspace_preregistration.md. Ship the headline (cond+gqACI) iff
G-BASE-CALIB (base within +/-tau, the P5-tension target gACI missed) AND G-SHIFT-COND (worst-group
floor) AND G-EFFICIENCY (tighter than union) all pass on 1a. G-ALPHA-FIX (report) checks the
mechanism: alpha-space base closer to nominal than width-space. A report-only alpha_eff probe
on seed 0 (where the realised level sits against its clamp) is printed after it.

Run: python src/run_p6.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import run_p2 as P
from _common import PROC
from conformal import (
    _QACI_GRID,
    aci_cqr,
    aci_cqr_multi,
    cqr_conditional,
    cqr_conditional_aci,
    cqr_conditional_gaci,
    cqr_conditional_gqaci,
    cqr_conditional_qaci,
)
from eval import clustered_bootstrap, coverage, mean_width
from panel import FEATURE_COLS, GROUP_COLS
from run_p2 import (
    ALPHA,
    GAMMA,
    MIN_N,
    SEEDS,
    SHIFT_HI,
    SHIFT_LO,
    TARGET,
    TAU,
    TAU_SHIFT,
    WG_FLOOR,
    _selective,
    _split,
    _xy,
)
from run_p22 import _axis_labels, _cov3, _ev, _retained_signed_gaps
from run_p23 import _full_base_signed_gaps
from synthetic import generate

METHODS = ["ACI single-axis", "ACI multi-axis", "cqr_conditional", "cond+ACI",
           "cond+gACI", "cond+qACI", "cond+gqACI"]
SHIP = "cond+gqACI"        # the P6 headline (group, alpha-space)
WIDTH_GROUP = "cond+gACI"  # the P2.3 width-space counterpart (for G-ALPHA-FIX)


def _alpha_eff_probe(trace):
    """Report-only mechanism probe on one seed: where the realised per-point alpha_eff of
    the headline cond+gqACI sits against its clamp, and how the global offset `a` behaves
    around the shift. Regenerates the alpha_eff probe in reports/p6_alphaspace_results.md."""
    lmin, lmax = 1 - _QACI_GRID[-1], 1 - _QACI_GRID[0]      # alpha_eff clamp [0.01, 0.50]
    omin = lmin - ALPHA                                     # lower anti-windup bound on a
    ae = np.concatenate(trace["alpha_eff"])
    sess, a = np.asarray(trace["session"]), np.asarray(trace["a"])
    at_lo, at_hi = ae <= lmin + 1e-12, ae >= lmax - 1e-12
    pinned = np.isclose(a, omin)
    shift, post = (sess >= SHIFT_LO) & (sess <= SHIFT_HI), sess > SHIFT_HI
    stay = int(np.argmin(pinned[post])) if (~pinned[post]).any() else int(post.sum())
    print(f"\n[alpha_eff probe, seed 0, report-only] realised per-point alpha_eff of {SHIP} "
          f"({len(ae):,} test points)")
    print(f"  median {np.median(ae):.3f} (nominal {ALPHA}) | interior to the "
          f"[{lmin:.2f}, {lmax:.2f}] clamp for {100 * np.mean(~at_lo & ~at_hi):.1f}% of points")
    print(f"  upper clamp (narrowest) binds for {100 * at_hi.mean():.1f}% | lower clamp "
          f"(widest, the {_QACI_GRID[-1]:.2f} node) binds for {100 * at_lo.mean():.1f}%")
    print(f"  global offset a at its lower anti-windup bound ({omin:+.2f}): {100 * pinned.mean():.1f}% "
          f"of all {len(sess)} test sessions, {100 * pinned[shift].mean():.1f}% of the "
          f"{int(shift.sum())} shift sessions")
    print(f"  after the shift ends: a stays at the bound for {stay} session(s), then leaves it; "
          f"at the bound in {int(pinned[post].sum())} of {int(post.sum())} post-shift sessions")


def _build_intervals(tr, ca, te, trace=None):
    Xtr, ytr = _xy(tr, "y"); Xca, yca = _xy(ca, "y"); Xte, yte = _xy(te, "y")
    g1c, g1t = P._grp(ca), P._grp(te)                      # single axis (mom_decile)
    gac, gat = _axis_labels(ca), _axis_labels(te)          # all 3 axes
    order = te.report_date.to_numpy()
    return {
        "ACI single-axis": aci_cqr(Xtr, ytr, Xca, yca, g1c, Xte, yte, order, g1t,
                                   ALPHA, GAMMA, MIN_N),
        "ACI multi-axis": aci_cqr_multi(Xtr, ytr, Xca, yca, gac, Xte, yte, order, gat,
                                        ALPHA, GAMMA, MIN_N),
        "cqr_conditional": cqr_conditional(Xtr, ytr, Xca, yca, gac, Xte, gat, ALPHA),
        "cond+ACI": cqr_conditional_aci(Xtr, ytr, Xca, yca, gac, Xte, yte, order, gat,
                                        ALPHA, GAMMA),
        "cond+gACI": cqr_conditional_gaci(Xtr, ytr, Xca, yca, gac, Xte, yte, order, gat,
                                          ALPHA, GAMMA),
        "cond+qACI": cqr_conditional_qaci(Xtr, ytr, Xca, yca, gac, Xte, yte, order, gat,
                                          ALPHA, GAMMA),
        "cond+gqACI": cqr_conditional_gqaci(Xtr, ytr, Xca, yca, gac, Xte, yte, order, gat,
                                            ALPHA, GAMMA, trace=trace),
    }


def run_1a():
    print("=" * 96)
    print("P6 — width-space (P2.2/P2.3) vs ALPHA-SPACE (P6) online on one fitted shape | Layer 1a")
    print("=" * 96)
    evs = {m: [] for m in METHODS}
    sel_per = []
    probe_trace: dict = {}                 # alpha_eff probe, recorded on the first seed only
    for s in SEEDS:
        path = PROC / f"synth_1a_seed{s}.parquet"
        if not path.exists():
            generate(s)
        df = pd.read_parquet(path).dropna(subset=FEATURE_COLS + ["y"])
        tr, ca, te = _split(df)
        ivs = _build_intervals(tr, ca, te, trace=probe_trace if s == SEEDS[0] else None)
        for m in METHODS:
            evs[m].append(_ev(te, ivs[m]))
        sel_per.append(_selective(tr, ca, te, "y", score_seed=s)["u"])

    # ---- coverage + width ----
    print(f"\nCoverage (mean +/- sd over seeds {SEEDS}; target {TARGET:.0%})")
    print(f"{'method':<18} {'base':>14} {'shift':>14} {'worst-grp|shift':>18} {'width':>8}")
    widths, wg_by, sh_by, base_by = {}, {}, {}, {}
    for m in METHODS:
        c = np.array([_cov3(e)[:3] for e in evs[m]], float)
        b, sh, wg = c.mean(0); bs, shs, wgs = c.std(0)
        w = float(np.mean([mean_width(e.lo.values, e.hi.values) for e in evs[m]]))
        widths[m], wg_by[m], sh_by[m], base_by[m] = w, wg, sh, b
        _, _, _, wd, wl = _cov3(evs[m][0])
        print(f"{m:<18} {b:>7.3f}+/-{bs:<5.3f} {sh:>7.3f}+/-{shs:<5.3f} "
              f"{wg:>8.3f}+/-{wgs:<5.3f}  {w:>7.3f}  ({wd}={wl})")

    # ---- G-BASE-CALIB (co-primary, the point of P6): alpha-space keeps base calibrated ----
    base_ok = abs(base_by[SHIP] - TARGET) <= TAU
    print(f"\n[G-BASE-CALIB] {SHIP} base {base_by[SHIP]:.3f} within +/-{TAU} of {TARGET}? "
          f"(width-space {WIDTH_GROUP} was {base_by[WIDTH_GROUP]:.3f}) -> "
          f"{'PASS' if base_ok else 'MISS'}")

    # ---- G-SHIFT-COND (co-primary): clears the worst-group floor under shift ----
    wg_g, sh_g = wg_by[SHIP], sh_by[SHIP]
    shift_cond = (wg_g >= WG_FLOOR) and (abs(sh_g - TARGET) <= TAU_SHIFT)
    print(f"[G-SHIFT-COND] {SHIP} worst-grp|shift {wg_g:.3f} (floor {WG_FLOOR}; "
          f"width-group {wg_by[WIDTH_GROUP]:.3f}) & shift {sh_g:.3f} (+/-{TAU_SHIFT}) -> "
          f"{'PASS' if shift_cond else 'MISS'}")

    # ---- G-EFFICIENCY (co-primary): tighter than the union ----
    eff_pass = widths[SHIP] < widths["ACI multi-axis"]
    print(f"[G-EFFICIENCY] width {SHIP} {widths[SHIP]:.3f} < union "
          f"{widths['ACI multi-axis']:.3f}? -> {'PASS' if eff_pass else 'MISS'}")

    # ---- G-ALPHA-FIX (report): the mechanism — alpha-space base closer to nominal ----
    d_alpha = abs(base_by[SHIP] - TARGET)
    d_width = abs(base_by[WIDTH_GROUP] - TARGET)
    print(f"\n[G-ALPHA-FIX report] |base-nominal|: alpha-space {SHIP} {d_alpha:.3f} vs "
          f"width-space {WIDTH_GROUP} {d_width:.3f} -> "
          f"{'alpha-space closer (mechanism confirmed)' if d_alpha < d_width else 'NOT closer'}")
    # global counterparts too (isolates global vs group within each space)
    print(f"  global pair base: width cond+ACI {base_by['cond+ACI']:.3f} | "
          f"alpha cond+qACI {base_by['cond+qACI']:.3f}")
    _alpha_eff_probe(probe_trace)

    # ---- G-CALIB-MULTI (report): full-base (headline) + retained (selective interaction) ----
    print(f"\n[G-CALIB-MULTI report] per-axis SIGNED worst gap (mean over seeds; +/-tau={TAU})")
    print(f"{'axis':<12} {'FULL base (gqACI)':>20} {'retained-70% (gqACI)':>24}")
    for dim in GROUP_COLS:
        gf = np.mean([_full_base_signed_gaps(e).get(dim, np.nan) for e in evs[SHIP]])
        gr = np.mean([_retained_signed_gaps(e, u).get(dim, np.nan)
                      for e, u in zip(evs[SHIP], sel_per)])
        fb = 'OK' if abs(gf) <= TAU else 'over' if gf > 0 else 'under'
        rb = 'OK' if abs(gr) <= TAU else 'over' if gr > 0 else 'under'
        print(f"{dim:<12} {gf:>+16.3f} {fb:>4} {gr:>+18.3f} {rb:>4}")

    # clustered band on gqACI base coverage, seed 0
    ev0 = evs[SHIP][0]; ba0 = ev0[ev0.regime == "base"].reset_index(drop=True)
    p, lo, hi = clustered_bootstrap(ba0, lambda d: coverage(d.y.values, d.lo.values, d.hi.values),
                                    n_boot=300)
    print(f"\n[seed0 clustered band] {SHIP} BASE coverage {p:.3f} [{lo:.3f},{hi:.3f}]")

    verdict = "SHIP" if (base_ok and shift_cond and eff_pass) else "QUALIFIED/REJECT"
    print(f"\n==> P6 verdict (1a): {verdict}  (G-BASE-CALIB {'pass' if base_ok else 'miss'}, "
          f"G-SHIFT-COND {'pass' if shift_cond else 'miss'}, "
          f"G-EFFICIENCY {'pass' if eff_pass else 'miss'})")
    return verdict


def run_1b():
    print("\n" + "=" * 96)
    print("P6 — alpha-space group-conditional | Layer 1b (real 21d fwd return)")
    print("=" * 96)
    df = P._build_1b()
    tr, ca, te = _split(df)
    ivs = _build_intervals(tr, ca, te)
    print(f"{'method':<18} {'base':>8} {'shift':>8} {'worst-grp|shift':>22} {'width':>8}")
    for m in METHODS:
        ev = _ev(te, ivs[m]); b, sh, wg, wd, wl = _cov3(ev)
        print(f"  {m:<16} {b:>8.3f} {sh:>8.3f} {wg:>10.3f} ({wd}={wl})  "
              f"{mean_width(ev.lo.values, ev.hi.values):>8.3f}")


if __name__ == "__main__":
    v = run_1a()
    run_1b()
