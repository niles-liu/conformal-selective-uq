"""P2.3 driver — *per-group adaptive* conditional conformal (SPEC §"P2.3").

Head-to-head on 1a (+ 1b external read), the full progression in one table:
  * ACI single-axis (frozen)        -- shift-robust, but only the conditioned axis
  * ACI multi-axis union (P2.1)     -- all axes, but OVER-covers (~+12% width)
  * cqr_conditional (P2.2)          -- calibrated shape, tightest, but STATIC -> shift-fragile
  * cqr_conditional+ACI (P2.2)      -- + a single GLOBAL offset: margin only, not group-specific
  * cqr_conditional+gACI (P2.3)     -- + per-GROUP online correction on the calibrated shape:
                                       calibrated AND tighter than the union AND group-conditionally
                                       shift-robust (the unification the others each missed one of)

Gate-first spec: reports/p23_adaptive_preregistration.md. Primary gate G-SHIFT-COND
(worst-group|shift >= 0.704, which the static method 0.433 and the global hybrid 0.599 both
missed), co-primary G-EFFICIENCY (tighter than the union) + G-BASE-CALIB (base stays calibrated).
The frozen aci_cqr / aci_cqr_multi / cqr_conditional / cqr_conditional_aci are untouched.

Run: python experiments/run_p23.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import run_p2 as P
from _common import PROC
from panel import FEATURE_COLS, GROUP_COLS
from run_p2 import ALPHA, GAMMA, MIN_N, SEEDS, TARGET, TAU, TAU_SHIFT, WG_FLOOR, _selective, _split, _xy
from run_p22 import _axis_labels, _cov3, _ev, _full_base_signed_gaps, _retained_signed_gaps
from selective_conformal.conformal import (
    aci_cqr,
    aci_cqr_multi,
    cqr_conditional,
    cqr_conditional_aci,
    cqr_conditional_gaci,
)
from selective_conformal.metrics import clustered_bootstrap, coverage, mean_width
from synthetic import generate

METHODS = ["ACI single-axis", "ACI multi-axis", "cqr_conditional", "cond+ACI", "cond+gACI"]
SHIP = "cond+gACI"   # the P2.3 method under test


def _build_intervals(tr, ca, te):
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
    }


def run_1a():
    print("=" * 90)
    print("P2.3 — single / union / conditional / +global / +GROUP-ADAPTIVE | Layer 1a (known floor)")
    print("=" * 90)
    evs = {m: [] for m in METHODS}
    sel_per = []
    for s in SEEDS:
        path = PROC / f"synth_1a_seed{s}.parquet"
        if not path.exists():
            generate(s)
        df = pd.read_parquet(path).dropna(subset=FEATURE_COLS + ["y"])
        tr, ca, te = _split(df)
        ivs = _build_intervals(tr, ca, te)
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

    # ---- G-SHIFT-COND (primary): group-adaptive clears the worst-group floor ----
    wg_g, sh_g = wg_by[SHIP], sh_by[SHIP]
    shift_cond = (wg_g >= WG_FLOOR) and (abs(sh_g - TARGET) <= TAU_SHIFT)
    print(f"\n[G-SHIFT-COND] {SHIP} worst-grp|shift {wg_g:.3f} (floor {WG_FLOOR}; "
          f"static {wg_by['cqr_conditional']:.3f}, global-hybrid {wg_by['cond+ACI']:.3f}) "
          f"& shift {sh_g:.3f} (+/-{TAU_SHIFT}) -> {'PASS' if shift_cond else 'MISS'}")

    # ---- G-EFFICIENCY (co-primary): tighter than the union ----
    eff_pass = widths[SHIP] < widths["ACI multi-axis"]
    print(f"[G-EFFICIENCY] width {SHIP} {widths[SHIP]:.3f} < union "
          f"{widths['ACI multi-axis']:.3f}? -> {'PASS' if eff_pass else 'MISS'}")

    # ---- G-BASE-CALIB (co-primary): online adaptation must not break base calibration ----
    base_ok = abs(base_by[SHIP] - TARGET) <= TAU
    print(f"[G-BASE-CALIB] base {base_by[SHIP]:.3f} within +/-{TAU} of {TARGET}? -> "
          f"{'PASS' if base_ok else 'MISS'}")

    # ---- G-CALIB-MULTI (report): full-base (headline) + retained (selective interaction) ----
    print(f"\n[G-CALIB-MULTI report] per-axis SIGNED worst gap (mean over seeds; +/-tau={TAU})")
    print(f"{'axis':<12} {'FULL base (cond+gACI)':>22} {'retained-70% (cond+gACI)':>26}")
    for dim in GROUP_COLS:
        gf = np.mean([_full_base_signed_gaps(e).get(dim, np.nan) for e in evs[SHIP]])
        gr = np.mean([_retained_signed_gaps(e, u).get(dim, np.nan)
                      for e, u in zip(evs[SHIP], sel_per)])
        fb = 'OK' if abs(gf) <= TAU else 'over' if gf > 0 else 'under'
        rb = 'OK' if abs(gr) <= TAU else 'over' if gr > 0 else 'under'
        print(f"{dim:<12} {gf:>+18.3f} {fb:>4} {gr:>+20.3f} {rb:>4}")
    print("  (retained-set sector residual = the P2.2 selective x conditional interaction; "
          "pre-committed as reported, NOT a P2.3 ship-blocker)")

    # clustered band on gACI base coverage, seed 0
    ev0 = evs[SHIP][0]; ba0 = ev0[ev0.regime == "base"].reset_index(drop=True)
    p, lo, hi = clustered_bootstrap(ba0, lambda d: coverage(d.y.values, d.lo.values, d.hi.values),
                                    n_boot=300)
    print(f"\n[seed0 clustered band] {SHIP} BASE coverage {p:.3f} [{lo:.3f},{hi:.3f}]")

    verdict = "SHIP" if (shift_cond and eff_pass and base_ok) else "QUALIFIED/REJECT"
    print(f"\n==> P2.3 verdict (1a): {verdict}  (G-SHIFT-COND {'pass' if shift_cond else 'miss'}, "
          f"G-EFFICIENCY {'pass' if eff_pass else 'miss'}, "
          f"G-BASE-CALIB {'pass' if base_ok else 'miss'})")
    return verdict


def run_1b():
    print("\n" + "=" * 90)
    print("P2.3 — group-adaptive conditional | Layer 1b (real 21d fwd return)")
    print("=" * 90)
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
