"""P2.2 driver — *calibrated* multi-group conditional conformal (SPEC §"P2.2").

Three-way head-to-head on 1a (+ 1b external read):
  * ACI single-axis (frozen)        -- calibrated, but only the conditioned axis
  * ACI multi-axis union (P2.1)     -- fixes worst-group but OVER-covers (~+12% width)
  * cqr_conditional (P2.2)          -- one coupled threshold over the joint group class;
                                       targets calibrated (two-sided +/-tau) coverage on ALL
                                       axes at lower width than the union
  * cqr_conditional+ACI (report)    -- the calibrated shape made shift-robust (global offset)

Question (gate-first spec: reports/p22_conditional_preregistration.md): does the coupled
conditional threshold deliver the *calibrated* multi-group coverage the union sacrificed,
strictly tighter than the union? The frozen aci_cqr / aci_cqr_multi are untouched.
Also prints the report-only full-base diagnostic (table (b) of the results file).

Run: python src/run_p22.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import run_p2 as P
from _common import PROC
from conformal import aci_cqr, aci_cqr_multi, cqr_conditional, cqr_conditional_aci
from eval import clustered_bootstrap, coverage, grouped_coverage, mean_width, worst_group_coverage
from panel import FEATURE_COLS, GROUP_COLS
from run_p2 import ALPHA, GAMMA, MIN_N, SEEDS, TARGET, TAU, TAU_SHIFT, WG_FLOOR, X_RET, _selective, _split, _xy
from synthetic import generate


def _axis_labels(df):
    """List of per-axis group-label arrays (strings), one per GROUP_COLS axis."""
    out = []
    for c in GROUP_COLS:
        out.append(df[c].fillna(-1).astype(str).to_numpy() if df[c].dtype != object
                   else df[c].fillna("na").astype(str).to_numpy())
    return out


def _ev(te, iv):
    keep = list(dict.fromkeys(["report_date", "regime", "ticker", *GROUP_COLS]))
    out = te[keep].copy()
    out["y"] = te["y"].to_numpy()
    out["lo"], out["hi"], out["pred"] = iv.lo, iv.hi, iv.pred
    return out.reset_index(drop=True)


def _cov3(ev):
    sh, ba = ev[ev.regime == "shift"], ev[ev.regime == "base"]
    wg, dim, lab = worst_group_coverage(sh, GROUP_COLS, TARGET, MIN_N)
    return (coverage(ba.y.values, ba.lo.values, ba.hi.values),
            coverage(sh.y.values, sh.lo.values, sh.hi.values), wg, dim, lab)


def _retained_signed_gaps(ev, u_by_name):
    """Per-axis SIGNED worst |gap| on the lowest-u X_RET% retained set (base regime).
    Returns {axis: gap with the largest magnitude} — sign matters here (over- vs
    under-cover), since the whole P2.2 point is two-sided calibration."""
    e = ev[ev.regime == "base"].copy()
    e["u"] = e["ticker"].map(u_by_name).fillna(u_by_name.median())
    ret = e[e["u"] <= np.quantile(e["u"], X_RET)]
    gaps = {}
    for dim in GROUP_COLS:
        gc = grouped_coverage(ret, dim, TARGET)
        gc = gc[gc["n"] >= MIN_N]
        if not gc.empty:
            gaps[dim] = gc.loc[gc["gap"].abs().idxmax(), "gap"]
    return gaps


def _full_base_signed_gaps(ev):
    """Per-axis SIGNED worst |gap| on the FULL base set (no abstention): the distribution
    the conditional shape calibrates, distinct from the retained subset above."""
    e = ev[ev.regime == "base"]
    gaps = {}
    for dim in GROUP_COLS:
        gc = grouped_coverage(e, dim, TARGET)
        gc = gc[gc["n"] >= MIN_N]
        if not gc.empty:
            gaps[dim] = gc.loc[gc["gap"].abs().idxmax(), "gap"]
    return gaps


METHODS = ["ACI single-axis", "ACI multi-axis", "cqr_conditional", "cond+ACI"]


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
    }


def run_1a():
    print("=" * 84)
    print("P2.2 — single / union / CONDITIONAL ACI | Layer 1a (known-floor synthetic)")
    print("=" * 84)
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
    widths = {}
    for m in METHODS:
        c = np.array([_cov3(e)[:3] for e in evs[m]], float)
        b, sh, wg = c.mean(0); bs, shs, wgs = c.std(0)
        w = float(np.mean([mean_width(e.lo.values, e.hi.values) for e in evs[m]]))
        widths[m] = w
        _, _, _, wd, wl = _cov3(evs[m][0])
        print(f"{m:<18} {b:>7.3f}+/-{bs:<5.3f} {sh:>7.3f}+/-{shs:<5.3f} "
              f"{wg:>8.3f}+/-{wgs:<5.3f}  {w:>7.3f}  ({wd}={wl})")

    # ---- G-CALIB-MULTI: two-sided retained conditional coverage on BASE ----
    print(f"\n[G-CALIB-MULTI] retained-{X_RET:.0%} BASE conditional coverage, SIGNED worst gap "
          f"(mean over seeds; +/-tau={TAU})")
    print(f"{'axis':<12} {'union (P2.1)':>14} {'conditional':>14}  verdict(cond, two-sided)")
    cond_ok = {}
    for dim in GROUP_COLS:
        gu = np.mean([_retained_signed_gaps(e, u).get(dim, np.nan)
                      for e, u in zip(evs["ACI multi-axis"], sel_per)])
        gc = np.mean([_retained_signed_gaps(e, u).get(dim, np.nan)
                      for e, u in zip(evs["cqr_conditional"], sel_per)])
        cond_ok[dim] = abs(gc) <= TAU
        print(f"{dim:<12} {gu:>+14.3f} {gc:>+14.3f}  {'OK' if cond_ok[dim] else 'MISS'}")
    # marginal base must also be within +/-tau (not over-covering)
    base_cond = np.mean([_cov3(e)[0] for e in evs["cqr_conditional"]])
    marg_ok = abs(base_cond - TARGET) <= TAU
    calib_pass = all(cond_ok.values()) and marg_ok
    print(f"marginal base cov (conditional) {base_cond:.3f} within +/-{TAU}? "
          f"{'OK' if marg_ok else 'MISS'}")
    print(f"-> G-CALIB-MULTI {'PASS' if calib_pass else 'MISS'}  "
          f"(all axes two-sided +/-tau AND marginal not over-covering)")

    # ---- diagnostic (report-only, not a gate): the FULL base set, no abstention ----
    # table (b) of reports/p22_conditional_results.md, from the same intervals as above
    print(f"\n[diagnostic, not a gate] FULL base set (no abstention), SIGNED worst gap "
          f"(mean over seeds; +/-tau={TAU})")
    print(f"{'axis':<12} {'union (P2.1)':>14} {'conditional':>14} {'single-axis':>14}  "
          f"cond within +/-tau?")
    for dim in GROUP_COLS:
        fu, fc, fs = (np.mean([_full_base_signed_gaps(e).get(dim, np.nan) for e in evs[m]])
                      for m in ("ACI multi-axis", "cqr_conditional", "ACI single-axis"))
        print(f"{dim:<12} {fu:>+14.3f} {fc:>+14.3f} {fs:>+14.3f}  {'OK' if abs(fc) <= TAU else 'MISS'}")

    # ---- G-EFFICIENCY: conditional strictly tighter than union ----
    eff_pass = widths["cqr_conditional"] < widths["ACI multi-axis"]
    print(f"\n[G-EFFICIENCY] width conditional {widths['cqr_conditional']:.3f} < union "
          f"{widths['ACI multi-axis']:.3f}? -> {'PASS' if eff_pass else 'MISS'}  "
          f"(vs single-axis {widths['ACI single-axis']:.3f})")

    # ---- G-SHIFT (report): worst-group|shift for each method ----
    print(f"\n[G-SHIFT report] worst-grp|shift (floor {WG_FLOOR}); shift cov within +/-{TAU_SHIFT}")
    for m in METHODS:
        wg = np.mean([_cov3(e)[2] for e in evs[m]])
        sh = np.mean([_cov3(e)[1] for e in evs[m]])
        tag = "ok-floor" if wg >= WG_FLOOR else "under-floor"
        print(f"  {m:<18} worst-grp {wg:.3f} ({tag}) shift {sh:.3f}")

    # clustered band on conditional base coverage, seed 0
    ev0 = evs["cqr_conditional"][0]; ba0 = ev0[ev0.regime == "base"].reset_index(drop=True)
    p, lo, hi = clustered_bootstrap(ba0, lambda d: coverage(d.y.values, d.lo.values, d.hi.values),
                                    n_boot=300)
    print(f"\n[seed0 clustered band] conditional BASE coverage {p:.3f} [{lo:.3f},{hi:.3f}]")

    verdict = "SHIP" if (calib_pass and eff_pass) else "QUALIFIED/REJECT"
    print(f"\n==> P2.2 verdict (1a): {verdict}  "
          f"(G-CALIB-MULTI {'pass' if calib_pass else 'miss'}, "
          f"G-EFFICIENCY {'pass' if eff_pass else 'miss'})")
    return verdict


def run_1b():
    print("\n" + "=" * 84)
    print("P2.2 — conditional conformal | Layer 1b (real 21d fwd return)")
    print("=" * 84)
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
