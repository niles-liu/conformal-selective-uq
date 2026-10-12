"""P12 driver: learned regime axis (regime × conformal) on the multi-regime substrate.

Tests whether adding a LEARNED regime axis to a group-conditional remedy recovers worst-regime
coverage that the regime-blind (static-axes) method misses. Eval is on the TRUE latent regime
(ground truth); the method triggers on the LEARNED regime (anti-circularity).

Remedy = the P2.1 multi-axis UNION ACI (`aci_cqr_multi` / `_aci_one_axis`): per-axis empirical
group quantiles + online ACI, unioned across axes. Chosen over the calibrated `gqaci_max` for
tractability (gqaci_max's 9-solve QuantileRegressor level-family is ~100x slower per axis-set) — a
pre-reg amendment made before any coverage result was seen; the union is the cleaner, faster testbed
for the seam (its worst-group coverage dominates each axis, so adding the regime axis directly tests
the lift). The GBM CQR pieces are fit ONCE per seed and shared across all axis-sets. Calibrated
`gqaci_max` + the POGO parameter-free arm are the deferred secondary (see pre-reg).

Gates + apparatus: reports/p12_regime_axis_preregistration.md.
Run: python experiments/synthetic_regime.py --seed {0,1,2}  then  python experiments/run_p12.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import PROC
from panel import FEATURE_COLS, GROUP_COLS
from run_p2 import ALPHA, GAMMA, MIN_N, SEEDS, TARGET, _split, _xy
from selective_conformal import conformal as C
from selective_conformal import regime as R
from selective_conformal.metrics import (
    clustered_bootstrap,
    coverage,
    grouped_coverage,
    mean_width,
    worst_group_coverage,
)

LAM_GRID = (10, 25, 50, 100, 200)
CALIB_END = 20220101
DET = ["disp_lag", "disp_ma"]
G_SEAM, G_NOMINAL, G_WG_DROP, G_EFF, G_RECOVER = 0.03, 0.85, 0.02, 1.10, 0.70


def _pf(b):
    return "PASS" if b else "MISS"


def _sess_detector(df):
    return (df.groupby("report_date")[DET].first().reset_index()
            .sort_values("report_date").reset_index(drop=True))


def _learn_regime(df, seed=0):
    """Jump model on train+calib sessions (all regimes present -> identified), causal forward
    filter on test. Returns per-session label map (stress-ordered 0..K-1) + diagnostics."""
    sess = _sess_detector(df)
    dates = sess.report_date.to_numpy()
    is_te = dates >= CALIB_END
    fitm = ~is_te
    Xraw = sess[DET].to_numpy(float)
    X = (Xraw - Xraw[fitm].mean(0)) / (Xraw[fitm].std(0) + 1e-12)
    lam = R.select_lambda(X[fitm], LAM_GRID, k=3, seed=seed)
    mu, path_fit = R.fit_jump(X[fitm], k=3, lam=lam, seed=seed)
    ranks = R.order_states_by(mu, col=0)                 # ascending disp -> 0=calm..2=stress
    remap = {st: r for r, st in enumerate(ranks)}
    labels = np.empty(len(dates), dtype=int)
    labels[fitm] = np.array([remap[s] for s in path_fit])
    filt = R.forward_filter(X[is_te], mu, lam, path_fit[-1])
    labels[is_te] = np.array([remap[s] for s in filt])
    diag = dict(lam=lam, n_switch=R.n_switches(labels), stress_share=float((labels == 2).mean()))
    return dict(zip(dates, labels)), diag


def _thr_map(df):
    sess = _sess_detector(df)
    dates = sess.report_date.to_numpy()
    lab = R.threshold_labels(sess["disp_lag"].to_numpy(), dates < CALIB_END)
    return dict(zip(dates, lab))


def _axis_labels(df):
    out = []
    for c in GROUP_COLS:
        s = df[c]
        out.append(s.fillna(-1).astype(str).to_numpy() if s.dtype != object
                   else s.fillna("na").astype(str).to_numpy())
    return out


def _ev(te, pred, lo, hi):
    out = te[["report_date", "regime", "regime_state", "ticker", *GROUP_COLS]].copy()
    out["y"] = te["y"].to_numpy()
    out["lo"], out["hi"], out["pred"] = lo, hi, pred
    return out.reset_index(drop=True)


def _worst_regime(ev):
    gc = grouped_coverage(ev, "regime_state", TARGET)
    gc = gc[gc["n"] >= MIN_N]
    return float(gc["coverage"].min()), int(gc.loc[gc["coverage"].idxmin(), "regime_state"])


def _summary(ev):
    wr, wlab = _worst_regime(ev)
    wg, _, _ = worst_group_coverage(ev, GROUP_COLS, TARGET, MIN_N)
    per = {int(k): coverage(g.y.values, g.lo.values, g.hi.values)
           for k, g in ev.groupby("regime_state")}
    return dict(marg=coverage(ev.y.values, ev.lo.values, ev.hi.values), wr=wr, wr_lab=wlab,
                wg=wg, width=mean_width(ev.lo.values, ev.hi.values), per=per)


def _bal_acc(true, pred, k=3):
    return float(np.mean([float((pred[true == c] == c).mean()) for c in range(k) if (true == c).any()]))


def main():
    variants = ["naive", "static", "regime_marg", "proposed", "oracle", "threshold",
                "cal_static", "cal_proposed", "cal_oracle"]
    per_seed = {v: [] for v in variants}
    recover, nulls, lams = [], [], []

    for s in SEEDS:
        print(f"[seed {s}] load + learn regime ...", flush=True)
        df = pd.read_parquet(PROC / f"synth_1b_seed{s}.parquet").dropna(subset=FEATURE_COLS + ["y"])
        reg_map, diag = _learn_regime(df, seed=s); lams.append(diag["lam"])
        thr_map = _thr_map(df)
        df = df.copy()
        df["reg_learn"] = df.report_date.map(reg_map).astype(int)
        df["reg_thr"] = df.report_date.map(thr_map).astype(int)

        tr, ca, te = _split(df)
        Xtr, ytr = _xy(tr, "y"); Xca, yca = _xy(ca, "y"); Xte, yte = _xy(te, "y")
        order = te.report_date.to_numpy()
        gac, gat = _axis_labels(ca), _axis_labels(te)

        def A(col):     # (calib, test) label arrays for a regime column
            return ca[col].astype(str).to_numpy(), te[col].astype(str).to_numpy()  # noqa: B023 — closure is defined and consumed within the same loop iteration
        rlc, rlt = A("reg_learn"); roc, rot = A("regime_state"); rtc, rtt = A("reg_thr")

        # GBM CQR pieces fit ONCE per seed, shared across every axis-set
        pred, lo_q, hi_q, E = C._cqr_pieces(Xtr, ytr, Xca, yca, Xte, ALPHA)
        print(f"[seed {s}] lam={diag['lam']} stress_share={diag['stress_share']:.2f}; unioning axes ...", flush=True)

        def union(cal_axes, te_axes):
            lo = np.full(len(Xte), np.inf); hi = np.full(len(Xte), -np.inf)  # noqa: B023 — closure is defined and consumed within the same loop iteration
            for gc, gt in zip(cal_axes, te_axes):
                lk, hk = C._aci_one_axis(lo_q, hi_q, E, gc, gt, yte, order, ALPHA, GAMMA, MIN_N)  # noqa: B023 — closure is defined and consumed within the same loop iteration
                lo = np.minimum(lo, lk); hi = np.maximum(hi, hk)
            return lo, hi

        def calibrated(cal_axes, te_axes):
            """TIGHT calibrated conditional node (Gibbs-Cherian-Candes; T2 theorem node): one
            pinball regression of E on the group-indicator design at 1-alpha. Fast: 1 LP/axis-set."""
            from sklearn.linear_model import QuantileRegressor
            Phi_c, Phi_t = C._multiaxis_design(cal_axes, te_axes)
            qr = QuantileRegressor(quantile=1 - ALPHA, alpha=0.0, solver="highs", fit_intercept=False)
            qr.fit(Phi_c, E)  # noqa: B023 — closure is defined and consumed within the same loop iteration
            t = qr.predict(Phi_t)
            lo, hi = lo_q - t, hi_q + t  # noqa: B023 — closure is defined and consumed within the same loop iteration
            bad = lo > hi
            if bad.any():
                mid = 0.5 * (lo[bad] + hi[bad]); lo[bad], hi[bad] = mid, mid
            return lo, hi

        # naive marginal floor: global CQR conformity quantile, no groups (shared pieces)
        q = C._conf_quantile(E, ALPHA)
        per_seed["naive"].append(_ev(te, pred, lo_q - q, hi_q + q))
        for name, ca_ax, te_ax in [
            ("static", gac, gat),
            ("regime_marg", [rlc], [rlt]),
            ("proposed", gac + [rlc], gat + [rlt]),
            ("oracle", gac + [roc], gat + [rot]),
            ("threshold", gac + [rtc], gat + [rtt])]:
            lo, hi = union(ca_ax, te_ax)
            per_seed[name].append(_ev(te, pred, lo, hi))
        for name, ca_ax, te_ax in [
            ("cal_static", gac, gat),
            ("cal_proposed", gac + [rlc], gat + [rlt]),
            ("cal_oracle", gac + [roc], gat + [rot])]:
            lo, hi = calibrated(ca_ax, te_ax)
            per_seed[name].append(_ev(te, pred, lo, hi))

        recover.append(_bal_acc(te.regime_state.to_numpy(), te.reg_learn.to_numpy()))

        # G-NULL: shuffle learned regime labels across TEST sessions
        rng = np.random.default_rng(1200 + s)
        td = np.sort(te.report_date.unique())
        shuf = dict(zip(td, rng.permutation([reg_map[d] for d in td])))
        rnt = te.report_date.map(shuf).astype(str).to_numpy()
        lo, hi = union(gac + [rlc], gat + [rnt])
        nulls.append(_worst_regime(_ev(te, pred, lo, hi))[0])
        print(f"[seed {s}] done (bal-acc {recover[-1]:.3f})", flush=True)

    # ---- aggregate ----
    print("=" * 90)
    print("P12 — LEARNED REGIME AXIS (regime × conformal), UNION remedy, multi-regime substrate")
    print("=" * 90)
    print(f"seeds {SEEDS} | test>= {CALIB_END} | target {TARGET:.0%} | lambda {lams} | "
          f"eval on TRUE regime, trigger on LEARNED")
    lbl = {"naive": "naive (marginal floor)", "static": "static union (Camp B, regime-blind)",
           "regime_marg": "regime-only ACI (Camp A)", "proposed": "PROPOSED union +learned regime",
           "oracle": "oracle union +true regime", "threshold": "threshold union (ablation)",
           "cal_static": "CALIB static (tight, Camp B)", "cal_proposed": "CALIB +learned regime",
           "cal_oracle": "CALIB +true regime"}
    agg = {}
    print(f"\n{'variant':<34}{'marg':>7}{'worst-regime':>14}{'worst-grp':>11}{'width':>8}  per-regime[calm/norm/stress]")
    for v in variants:
        S = [_summary(e) for e in per_seed[v]]
        marg = np.mean([x["marg"] for x in S]); wr = np.mean([x["wr"] for x in S])
        wrs = np.std([x["wr"] for x in S]); wg = np.mean([x["wg"] for x in S])
        w = np.mean([x["width"] for x in S])
        per = {k: np.mean([x["per"].get(k, np.nan) for x in S]) for k in (0, 1, 2)}
        agg[v] = dict(marg=marg, wr=wr, wg=wg, width=w)
        print(f"{lbl[v]:<34}{marg:>7.3f}{wr:>9.3f}+/-{wrs:<4.3f}{wg:>11.3f}{w:>8.3f}  "
              f"[{per[0]:.3f}/{per[1]:.3f}/{per[2]:.3f}]")

    ev0 = per_seed["proposed"][0]; wl = _summary(ev0)["wr_lab"]
    sub = ev0[ev0.regime_state == wl].reset_index(drop=True)
    p, lo, hi = clustered_bootstrap(sub, lambda d: coverage(d.y.values, d.lo.values, d.hi.values), n_boot=300)
    print(f"\n[seed0 clustered band] proposed worst-regime (state {wl}) {p:.3f} [{lo:.3f},{hi:.3f}]")
    print(f"regime recovery (balanced acc vs true, test): {np.mean(recover):.3f} per-seed {np.round(recover,3)}")
    print(f"G-NULL shuffled-regime worst-regime: {np.mean(nulls):.3f} (vs static {agg['static']['wr']:.3f})")

    prop, stat, orc = agg["proposed"], agg["static"], agg["oracle"]
    g_seam = (prop["wr"] - stat["wr"]) >= G_SEAM
    g_nom = prop["wr"] >= G_NOMINAL
    g_wg = prop["wg"] >= stat["wg"] - G_WG_DROP
    g_eff = prop["width"] <= stat["width"] * G_EFF
    g_rec = np.mean(recover) >= G_RECOVER
    g_null = np.mean(nulls) <= stat["wr"] + 0.02

    print("\n" + "-" * 90)
    print("PRE-COMMITTED GATES")
    print(f"[G-SEAM]       proposed wr {prop['wr']:.3f} - static wr {stat['wr']:.3f} = "
          f"{prop['wr']-stat['wr']:+.3f} >= +{G_SEAM} -> {_pf(g_seam)}")
    print(f"[G-NOMINAL]    proposed wr {prop['wr']:.3f} >= {G_NOMINAL}                 -> {_pf(g_nom)}")
    print(f"[G-WORSTGROUP] proposed wg {prop['wg']:.3f} >= static wg {stat['wg']:.3f} - {G_WG_DROP} -> {_pf(g_wg)}")
    print(f"[G-EFF]        proposed width {prop['width']:.3f} <= {G_EFF}x static {stat['width']:.3f} -> {_pf(g_eff)}")
    print(f"[G-RECOVER]    regime bal-acc {np.mean(recover):.3f} >= {G_RECOVER}          -> {_pf(g_rec)}")
    print(f"[G-NULL]       shuffled wr {np.mean(nulls):.3f} <= static+.02          -> {_pf(g_null)} (collapses)")
    print(f"  (context: oracle-regime wr {orc['wr']:.3f} = identifiability upper bound)")
    # calibrated (tight) family — where the seam most plausibly lives (union over-covers)
    cs, cp, co = agg["cal_static"], agg["cal_proposed"], agg["cal_oracle"]
    print("\n[CALIBRATED family — tight remedy] seam test on the calibrated node")
    print(f"  cal_static wr {cs['wr']:.3f} (marg {cs['marg']:.3f} w {cs['width']:.2f}) | "
          f"cal+learned wr {cp['wr']:.3f} ({cp['wr']-cs['wr']:+.3f}) | "
          f"cal+true(oracle) wr {co['wr']:.3f} ({co['wr']-cs['wr']:+.3f})")
    print(f"  [G-SEAM-CALIB] cal+learned - cal_static = {cp['wr']-cs['wr']:+.3f} "
          f"(>= +{G_SEAM}? {_pf((cp['wr']-cs['wr'])>=G_SEAM)}); "
          f"oracle headroom {co['wr']-cs['wr']:+.3f}")

    if g_seam and g_nom and g_wg and g_eff and g_rec:
        v = ("CONFIRM: a learned dynamic regime axis recovers worst-regime coverage the regime-blind "
             "Camp-B union misses, without sacrificing the static guarantee or width — the seam joins "
             "where the regime is learnable.")
    elif not g_seam:
        v = ("REJECT-as-finding: even with a learnable regime, the static union already covers every "
             "regime (its cross-axis conservatism absorbs the regime effect) — the regime axis adds nothing.")
    else:
        v = "QUALIFIED: seam helps worst-regime but trades off worst-group or width (see gates)."
    print(f"\nVERDICT -> {v}")


if __name__ == "__main__":
    main()
