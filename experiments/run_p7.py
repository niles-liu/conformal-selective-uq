"""P7 driver — *cross-axis level combination* for alpha-space group adaptation.

Follows the P6 reversal (the additive group term lost to the global one on 1a, and the worst
group changed axis -> cross-axis additive cancellation). Tests two fixes on the SAME fitted
level family, leaving the per-group update untouched:
  * cond+gqACI-max  (headline) -- combine axes by the MOST-CONSERVATIVE axis (min offset),
                                  not the sum: removes cancellation (union-in-level-space on
                                  the single shape).
  * cond+gqACI-asym (report)   -- additive, but slower to tighten than to widen (rho=0.5).

Head-to-head vs the P6 alpha pair (global cond+qACI, additive cond+gqACI) + the union (for
efficiency) + the static shape. Gate-first spec: reports/p7_levelcomb_preregistration.md.
Primary gate G-GROUP-HELPS: gqACI-max worst-group|shift beats BOTH cond+qACI (~0.683) and
cond+gqACI (~0.655); co-primaries G-BASE-CALIB (+/-tau) + G-EFFICIENCY (< union).

Run: python experiments/run_p7.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import run_p2 as P
from _common import PROC
from panel import FEATURE_COLS, GROUP_COLS
from run_p2 import ALPHA, GAMMA, MIN_N, SEEDS, TARGET, TAU, TAU_SHIFT, WG_FLOOR, _selective, _split, _xy
from run_p22 import _axis_labels, _cov3, _ev, _retained_signed_gaps
from run_p23 import _full_base_signed_gaps
from selective_conformal.conformal import (
    _alpha_precompute,
    aci_cqr_multi,
    cqr_conditional,
    cqr_conditional_gqaci,
    cqr_conditional_gqaci_asym,
    cqr_conditional_gqaci_max,
    cqr_conditional_qaci,
)
from selective_conformal.metrics import clustered_bootstrap, coverage, mean_width
from synthetic import generate

METHODS = ["ACI multi-axis", "cqr_conditional", "cond+qACI", "cond+gqACI",
           "cond+gqACI-max", "cond+gqACI-asym"]
SHIP = "cond+gqACI-max"      # P7 headline (cross-axis max)
GLOBAL_ALPHA = "cond+qACI"   # P6 global alpha baseline
ADD_GROUP = "cond+gqACI"     # P6 additive group baseline


def _build_intervals(tr, ca, te):
    Xtr, ytr = _xy(tr, "y"); Xca, yca = _xy(ca, "y"); Xte, yte = _xy(te, "y")
    gac, gat = _axis_labels(ca), _axis_labels(te)          # all 3 axes
    order = te.report_date.to_numpy()
    # The four alpha-space variants share the CQR pieces + level family; fit ONCE and
    # reuse (bit-identical, ~4x cheaper than fitting inside each). Non-alpha methods
    # (union, static shape) keep their own self-contained fits.
    pc = _alpha_precompute(Xtr, ytr, Xca, yca, gac, Xte, gat, ALPHA)
    return {
        "ACI multi-axis": aci_cqr_multi(Xtr, ytr, Xca, yca, gac, Xte, yte, order, gat,
                                        ALPHA, GAMMA, MIN_N),
        "cqr_conditional": cqr_conditional(Xtr, ytr, Xca, yca, gac, Xte, gat, ALPHA),
        "cond+qACI": cqr_conditional_qaci(Xtr, ytr, Xca, yca, gac, Xte, yte, order, gat,
                                          ALPHA, GAMMA, precomp=pc),
        "cond+gqACI": cqr_conditional_gqaci(Xtr, ytr, Xca, yca, gac, Xte, yte, order, gat,
                                            ALPHA, GAMMA, precomp=pc),
        "cond+gqACI-max": cqr_conditional_gqaci_max(Xtr, ytr, Xca, yca, gac, Xte, yte,
                                                    order, gat, ALPHA, GAMMA, precomp=pc),
        "cond+gqACI-asym": cqr_conditional_gqaci_asym(Xtr, ytr, Xca, yca, gac, Xte, yte,
                                                      order, gat, ALPHA, GAMMA, precomp=pc),
    }


def run_1a():
    print("=" * 96)
    print("P7 — cross-axis MAX vs additive vs global, alpha-space group adaptation | Layer 1a")
    print("=" * 96)
    import time
    evs = {m: [] for m in METHODS}
    sel_per = []
    t0 = time.time()
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
        print(f"  [progress] seed {s} done @ {time.time()-t0:.0f}s "
              f"({SEEDS.index(s)+1}/{len(SEEDS)} seeds)", flush=True)

    print(f"\nCoverage (mean +/- sd over seeds {SEEDS}; target {TARGET:.0%})")
    print(f"{'method':<20} {'base':>14} {'shift':>14} {'worst-grp|shift':>18} {'width':>8}")
    widths, wg_by, sh_by, base_by = {}, {}, {}, {}
    for m in METHODS:
        c = np.array([_cov3(e)[:3] for e in evs[m]], float)
        b, sh, wg = c.mean(0); bs, shs, wgs = c.std(0)
        w = float(np.mean([mean_width(e.lo.values, e.hi.values) for e in evs[m]]))
        widths[m], wg_by[m], sh_by[m], base_by[m] = w, wg, sh, b
        _, _, _, wd, wl = _cov3(evs[m][0])
        print(f"{m:<20} {b:>7.3f}+/-{bs:<5.3f} {sh:>7.3f}+/-{shs:<5.3f} "
              f"{wg:>8.3f}+/-{wgs:<5.3f}  {w:>7.3f}  ({wd}={wl})")

    # ---- G-GROUP-HELPS (primary): max beats BOTH global and additive on worst-group ----
    helps = (wg_by[SHIP] > wg_by[GLOBAL_ALPHA]) and (wg_by[SHIP] > wg_by[ADD_GROUP])
    print(f"\n[G-GROUP-HELPS] {SHIP} worst-grp|shift {wg_by[SHIP]:.3f} > global "
          f"{GLOBAL_ALPHA} {wg_by[GLOBAL_ALPHA]:.3f} AND > additive {ADD_GROUP} "
          f"{wg_by[ADD_GROUP]:.3f}? -> {'PASS' if helps else 'MISS'}")
    # per-seed paired (same data/fold)
    wmax = np.array([_cov3(e)[2] for e in evs[SHIP]])
    wglo = np.array([_cov3(e)[2] for e in evs[GLOBAL_ALPHA]])
    print(f"  per-seed worst-grp: max {np.round(wmax,3)} vs global {np.round(wglo,3)} "
          f"(max-global {np.round(wmax-wglo,3)})")

    # ---- G-BASE-CALIB (co-primary) ----
    base_ok = abs(base_by[SHIP] - TARGET) <= TAU
    print(f"[G-BASE-CALIB] {SHIP} base {base_by[SHIP]:.3f} within +/-{TAU} of {TARGET}? -> "
          f"{'PASS' if base_ok else 'MISS'}")

    # ---- G-EFFICIENCY (co-primary): tighter than union ----
    eff_pass = widths[SHIP] < widths["ACI multi-axis"]
    print(f"[G-EFFICIENCY] width {SHIP} {widths[SHIP]:.3f} < union "
          f"{widths['ACI multi-axis']:.3f}? -> {'PASS' if eff_pass else 'MISS'}")

    # ---- G-SHIFT-COND (report): worst-group floor + shift band ----
    shift_cond = (wg_by[SHIP] >= WG_FLOOR) and (abs(sh_by[SHIP] - TARGET) <= TAU_SHIFT)
    print(f"[G-SHIFT-COND report] {SHIP} worst-grp|shift {wg_by[SHIP]:.3f} (floor {WG_FLOOR}) "
          f"& shift {sh_by[SHIP]:.3f} (+/-{TAU_SHIFT}) -> {'PASS' if shift_cond else 'MISS'}")

    # ---- G-CALIB-MULTI (report): full-base + retained signed gaps ----
    print(f"\n[G-CALIB-MULTI report] {SHIP} per-axis SIGNED worst gap (mean over seeds; +/-tau={TAU})")
    print(f"{'axis':<12} {'FULL base':>14} {'retained-70%':>16}")
    for dim in GROUP_COLS:
        gf = np.mean([_full_base_signed_gaps(e).get(dim, np.nan) for e in evs[SHIP]])
        gr = np.mean([_retained_signed_gaps(e, u).get(dim, np.nan)
                      for e, u in zip(evs[SHIP], sel_per)])
        fb = 'OK' if abs(gf) <= TAU else 'over' if gf > 0 else 'under'
        rb = 'OK' if abs(gr) <= TAU else 'over' if gr > 0 else 'under'
        print(f"{dim:<12} {gf:>+10.3f} {fb:>4} {gr:>+12.3f} {rb:>4}")

    ev0 = evs[SHIP][0]; ba0 = ev0[ev0.regime == "base"].reset_index(drop=True)
    p, lo, hi = clustered_bootstrap(ba0, lambda d: coverage(d.y.values, d.lo.values, d.hi.values),
                                    n_boot=300)
    print(f"\n[seed0 clustered band] {SHIP} BASE coverage {p:.3f} [{lo:.3f},{hi:.3f}]")

    direction = "SHIPS (direction)" if (helps and base_ok and eff_pass) else "no"
    full = "FULL SHIP" if (helps and base_ok and eff_pass and shift_cond) else "QUALIFIED/REJECT"
    print(f"\n==> P7 verdict (1a): {full}  (G-GROUP-HELPS {'pass' if helps else 'miss'}, "
          f"G-BASE-CALIB {'pass' if base_ok else 'miss'}, "
          f"G-EFFICIENCY {'pass' if eff_pass else 'miss'}, "
          f"G-SHIFT-COND {'pass' if shift_cond else 'miss'}; direction {direction})")
    return full


def run_1b():
    print("\n" + "=" * 96)
    print("P7 — cross-axis level combination | Layer 1b (real 21d fwd return)")
    print("=" * 96)
    df = P._build_1b()
    tr, ca, te = _split(df)
    ivs = _build_intervals(tr, ca, te)
    print(f"{'method':<20} {'base':>8} {'shift':>8} {'worst-grp|shift':>22} {'width':>8}")
    for m in METHODS:
        ev = _ev(te, ivs[m]); b, sh, wg, wd, wl = _cov3(ev)
        print(f"  {m:<18} {b:>8.3f} {sh:>8.3f} {wg:>10.3f} ({wd}={wl})  "
              f"{mean_width(ev.lo.values, ev.hi.values):>8.3f}")


if __name__ == "__main__":
    v = run_1a()
    run_1b()
