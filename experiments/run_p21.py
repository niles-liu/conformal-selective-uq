"""P2.1 driver — single-axis vs MULTI-axis group-conditional ACI (SPEC §"P2.1").

Head-to-head on 1a (+ 1b external read): does conditioning ACI on ALL THREE axes
(sector / cap_tier / mom_decile) via the union interval close the un-conditioned-axis
conditional-coverage gap P2 left open (sector ~1pp over tau), without over-correcting?
Gate-first spec: reports/p21_multiaxis_preregistration.md.
The frozen single-axis aci_cqr is untouched; this only adds aci_cqr_multi.

Run: python experiments/run_p21.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import run_p2 as P
from _common import PROC
from panel import FEATURE_COLS, GROUP_COLS
from run_p2 import ALPHA, GAMMA, MIN_N, SEEDS, TARGET, TAU, TAU_SHIFT, WG_FLOOR, X_RET, _selective, _split, _xy
from selective_conformal.conformal import aci_cqr, aci_cqr_multi
from selective_conformal.metrics import (
    clustered_bootstrap,
    coverage,
    grouped_coverage,
    mean_width,
    worst_group_coverage,
)
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


def _retained_gaps(ev, u_by_name):
    """Per-axis max|gap| on the lowest-u X_RET% retained set."""
    e = ev.copy()
    e["u"] = e["ticker"].map(u_by_name).fillna(u_by_name.median())
    ret = e[e["u"] <= np.quantile(e["u"], X_RET)]
    gaps = {}
    for dim in GROUP_COLS:
        gc = grouped_coverage(ret, dim, TARGET)
        gc = gc[gc["n"] >= MIN_N]
        if not gc.empty:
            gaps[dim] = gc["gap"].abs().max()
    return gaps


def run_1a():
    print("=" * 78)
    print("P2.1 — single-axis vs MULTI-axis ACI | Layer 1a (known-floor synthetic)")
    print("=" * 78)
    single, multi, sel_per = [], [], []
    for s in SEEDS:
        path = PROC / f"synth_1a_seed{s}.parquet"
        if not path.exists():
            generate(s)
        df = pd.read_parquet(path).dropna(subset=FEATURE_COLS + ["y"])
        tr, ca, te = _split(df)
        Xtr, ytr = _xy(tr, "y"); Xca, yca = _xy(ca, "y"); Xte, yte = _xy(te, "y")
        g1c, g1t = P._grp(ca), P._grp(te)                      # single axis (mom_decile)
        gac, gat = _axis_labels(ca), _axis_labels(te)          # all 3 axes
        order = te.report_date.to_numpy()
        iv_s = aci_cqr(Xtr, ytr, Xca, yca, g1c, Xte, yte, order, g1t, ALPHA, GAMMA, MIN_N)
        iv_m = aci_cqr_multi(Xtr, ytr, Xca, yca, gac, Xte, yte, order, gat, ALPHA, GAMMA, MIN_N)
        single.append(_ev(te, iv_s)); multi.append(_ev(te, iv_m))
        sel_per.append(_selective(tr, ca, te, "y", score_seed=s)["u"])

    # coverage + width
    print(f"\nCoverage (mean +/- sd over seeds {SEEDS}; target {TARGET:.0%})")
    print(f"{'method':<16} {'base':>14} {'shift':>14} {'worst-grp|shift':>18} {'width':>8}")
    for name, evs in [("ACI single-axis", single), ("ACI multi-axis", multi)]:
        c = np.array([_cov3(e)[:3] for e in evs], float)
        b, sh, wg = c.mean(0); bs, shs, wgs = c.std(0)
        w = np.mean([mean_width(e.lo.values, e.hi.values) for e in evs])
        _, _, _, wd, wl = _cov3(evs[0])
        print(f"{name:<16} {b:>7.3f}+/-{bs:<5.3f} {sh:>7.3f}+/-{shs:<5.3f} "
              f"{wg:>8.3f}+/-{wgs:<5.3f}  {w:>7.3f}  ({wd}={wl})")

    # G-COND-MULTI: retained-set per-axis conditional coverage
    print(f"\n[G-COND-MULTI] retained-{X_RET:.0%} conditional coverage, max|gap| (mean over seeds)")
    print(f"{'axis':<12} {'single-axis':>14} {'multi-axis':>14}  verdict(multi vs tau=%.2f)" % TAU)
    res = {}
    for dim in GROUP_COLS:
        gs = np.mean([_retained_gaps(e, u)[dim] for e, u in zip(single, sel_per) if dim in _retained_gaps(e, u)])
        gm = np.mean([_retained_gaps(e, u)[dim] for e, u in zip(multi, sel_per) if dim in _retained_gaps(e, u)])
        res[dim] = gm
        print(f"{dim:<12} {gs:>14.3f} {gm:>14.3f}  {'OK' if gm <= TAU else 'MISS'}")
    cond_pass = all(v <= TAU for v in res.values())

    # G-SHIFT re-confirm (multi)
    wgm = np.mean([_cov3(e)[2] for e in multi]); shm = np.mean([_cov3(e)[1] for e in multi])
    wgs_ = np.mean([_cov3(e)[2] for e in single])
    gshift = (wgm >= WG_FLOOR and abs(shm - TARGET) <= TAU_SHIFT and wgm >= wgs_)
    print(f"\n[G-SHIFT re-confirm] multi worst-grp|shift {wgm:.3f} (>= single {wgs_:.3f} and >= {WG_FLOOR}); "
          f"shift cov {shm:.3f} (+/-{TAU_SHIFT}) -> {'PASS' if gshift else 'CHECK'}")

    ev0 = multi[0]; sh0 = ev0[ev0.regime == "shift"].reset_index(drop=True)
    p, lo, hi = clustered_bootstrap(sh0, lambda d: coverage(d.y.values, d.lo.values, d.hi.values), n_boot=300)
    print(f"[seed0 clustered band] multi-axis ACI shift coverage {p:.3f} [{lo:.3f},{hi:.3f}]")

    verdict = "SHIP" if (cond_pass and gshift) else "QUALIFIED/REJECT"
    print(f"\n==> P2.1 verdict (1a): {verdict}  (G-COND-MULTI {'pass' if cond_pass else 'miss'}, "
          f"G-SHIFT {'pass' if gshift else 'check'})")
    return verdict


def run_1b():
    print("\n" + "=" * 78)
    print("P2.1 — multi-axis ACI | Layer 1b (real 21d fwd return)")
    print("=" * 78)
    df = P._build_1b()
    tr, ca, te = _split(df)
    Xtr, ytr = _xy(tr, "y"); Xca, yca = _xy(ca, "y"); Xte, yte = _xy(te, "y")
    g1c, g1t = P._grp(ca), P._grp(te)
    gac, gat = _axis_labels(ca), _axis_labels(te)
    order = te.report_date.to_numpy()
    iv_s = aci_cqr(Xtr, ytr, Xca, yca, g1c, Xte, yte, order, g1t, ALPHA, GAMMA, MIN_N)
    iv_m = aci_cqr_multi(Xtr, ytr, Xca, yca, gac, Xte, yte, order, gat, ALPHA, GAMMA, MIN_N)
    for name, iv in [("single-axis", iv_s), ("multi-axis", iv_m)]:
        ev = _ev(te, iv); b, sh, wg, wd, wl = _cov3(ev)
        print(f"  ACI {name:<11} base {b:.3f} shift {sh:.3f} worst-grp|shift {wg:.3f} "
              f"({wd}={wl}) width {mean_width(ev.lo.values, ev.hi.values):.3f}")


if __name__ == "__main__":
    v = run_1a()
    run_1b()
