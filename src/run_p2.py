"""P2 method driver — resolve G-PRIMARY / G-SHIFT / G-MECHANISM / G-NULL (SPEC §7, P2).

The method (all public-feature-only, knobs frozen in reports/p2_method_preregistration.md):
  * Mondrian (group-conditional) CQR        -> static conditional coverage   [G-PRIMARY ii]
  * group-conditional ACI on CQR scores     -> shift-robust worst-group cov   [G-SHIFT]  (headline)
  * selective layer: per-name calib residual scale u_i -> risk-coverage      [G-PRIMARY i]
  * G-MECHANISM (1a): does low-u coincide with recoverable names, surviving vol_63 control
  * G-NULL: shuffle y within date -> the selective advantage must collapse

Splits identical to P1: train rd<20210101 | calib 2021 | test rd>=20220101.
Run: python src/run_p2.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import PANEL, PROC, RETURNS
from conformal import aci_cqr, cqr, mondrian_cqr, naive_split, weighted_cqr
from eval import clustered_bootstrap, coverage, grouped_coverage, mean_width, risk_coverage, worst_group_coverage
from learners import fit_mean
from panel import FEATURE_COLS, GROUP_COLS
from selective import mechanism_auc, mechanism_corr, name_uncertainty, selective_scores
from synthetic import generate

ALPHA = 0.10
TARGET = 1 - ALPHA
TRAIN_END, CALIB_END = 20210101, 20220101
SHIFT_LO, SHIFT_HI = 20220101, 20221231
SEEDS = [0, 1, 2]
RETENTIONS = np.round(np.arange(0.3, 1.01, 0.1), 2)
X_RET = 0.70

# frozen knobs (pre-reg)
GAMMA = 0.05
MIN_N = 50
MOND_DIM = "mom_decile"

# frozen gate thresholds (from P1)
DELTA = 0.119
WG_FLOOR = 0.704
TAU = 0.05
TAU_SHIFT = 0.07
THETA_AUC = 0.70
THETA_RHO = 0.40


def _grp(df: pd.DataFrame) -> np.ndarray:
    """Mondrian/ACI group label from MOND_DIM (NaN -> 'na'), as strings."""
    return df[MOND_DIM].fillna(-1).astype(int).astype(str).to_numpy()


def _split(df):
    return (df[df.report_date < TRAIN_END],
            df[(df.report_date >= TRAIN_END) & (df.report_date < CALIB_END)],
            df[df.report_date >= CALIB_END])


def _xy(df, target):
    return df[FEATURE_COLS].to_numpy(), df[target].to_numpy()


def _ev(te, iv, target):
    keep = list(dict.fromkeys(["report_date", "regime", "ticker", *GROUP_COLS]))
    out = te[keep].copy()
    out["y"] = te[target].to_numpy()
    out["lo"], out["hi"], out["pred"], out["width"] = iv.lo, iv.hi, iv.pred, iv.width
    return out.reset_index(drop=True)


def _cov3(ev):
    """(base, shift, worst-group|shift across all 3 dims)."""
    sh, ba = ev[ev.regime == "shift"], ev[ev.regime == "base"]
    wg, dim, lab = worst_group_coverage(sh, GROUP_COLS, TARGET, MIN_N)
    return (coverage(ba.y.values, ba.lo.values, ba.hi.values),
            coverage(sh.y.values, sh.lo.values, sh.hi.values),
            wg, dim, lab)


def _gshift_verdict(name: str, wg: float) -> str:
    """The pre-registered G-SHIFT rule, applied to worst-group coverage under the shift:
    worst-group >= WG_FLOOR AND within +/-TAU_SHIFT of nominal. Both must hold."""
    floor_ok = wg >= WG_FLOOR
    band_ok = abs(wg - TARGET) <= TAU_SHIFT
    return (f"{name} worst-group|shift = {wg:.3f}: >= floor {WG_FLOOR}? "
            f"{'yes' if floor_ok else 'no'}; within +/-{TAU_SHIFT} of {TARGET}? "
            f"{'yes' if band_ok else 'no'} ({abs(wg - TARGET):.3f} off) -> "
            f"{'PASS' if (floor_ok and band_ok) else 'FAIL'}")


def _intervals(tr, ca, te, target):
    """All methods on one (train, calib, test) split. Returns dict name -> ev frame."""
    Xtr, ytr = _xy(tr, target)
    Xca, yca = _xy(ca, target)
    Xte, yte = _xy(te, target)
    gca, gte = _grp(ca), _grp(te)
    out = {}
    out["naive"] = _ev(te, naive_split(Xtr, ytr, Xca, yca, Xte, ALPHA), target)
    out["cqr"] = _ev(te, cqr(Xtr, ytr, Xca, yca, Xte, ALPHA), target)
    out["mondrian"] = _ev(te, mondrian_cqr(Xtr, ytr, Xca, yca, gca, Xte, gte,
                                           ALPHA, MIN_N), target)
    out["weighted"] = _ev(te, weighted_cqr(Xtr, ytr, Xca, yca, Xte, ALPHA), target)
    out["aci"] = _ev(te, aci_cqr(Xtr, ytr, Xca, yca, gca, Xte, yte,
                                 te.report_date.to_numpy(), gte,
                                 ALPHA, GAMMA, MIN_N), target)
    return out


def _selective(tr, ca, te, target, score_seed=0):
    """Risk-coverage: name-uncertainty vs random vs (1a) oracle. Returns curves + u."""
    Xtr, ytr = _xy(tr, target)
    Xca, yca = _xy(ca, target)
    Xte, yte = _xy(te, target)
    mean = fit_mean(Xtr, ytr)
    cadf = ca[["ticker"]].copy()
    cadf["resid"] = np.abs(yca - mean.predict(Xca))
    u = name_uncertainty(cadf)
    pred_te = mean.predict(Xte)
    score = selective_scores(te, u)
    rng = np.random.default_rng(score_seed)
    rand = [risk_coverage(yte, pred_te, rng.standard_normal(len(yte)), RETENTIONS)
            for _ in range(20)]
    rc_rand = {r: float(np.mean([c[r] for c in rand])) for r in RETENTIONS}
    rc_meth = risk_coverage(yte, pred_te, score, RETENTIONS)
    return dict(rand=rc_rand, method=rc_meth, u=u, pred=pred_te, y=yte, score=score)


# --------------------------------------------------------------------------- #
def run_1a():
    print("=" * 74)
    print("LAYER 1a — known-ceiling synthetic | METHOD (P2)")
    print("=" * 74)
    per = []
    for s in SEEDS:
        path = PROC / f"synth_1a_seed{s}.parquet"
        if not path.exists():
            generate(s)
        df = pd.read_parquet(path).dropna(subset=FEATURE_COLS + ["y"])
        tr, ca, te = _split(df)
        ivs = _intervals(tr, ca, te, "y")

        sel = _selective(tr, ca, te, "y", score_seed=s)
        # oracle uses true per-name irreducible noise
        isig = te["irreducible_sigma"].to_numpy()
        rc_oracle = risk_coverage(sel["y"], sel["pred"], isig, RETENTIONS)

        # G-MECHANISM ground truth per name (+ per-name mean vol_63 for the confound)
        gt = (df.groupby("ticker")
                .agg(recoverable_mask=("recoverable_mask", "first"),
                     irreducible_sigma=("irreducible_sigma", "first"),
                     vol_63=("vol_63", "mean")).reset_index())
        auc = mechanism_auc(sel["u"], gt)
        mc = mechanism_corr(sel["u"], gt)

        # G-NULL: shuffle y within date, redo selective advantage
        rngn = np.random.default_rng(1000 + s)
        dfn = df.copy()
        dfn["y"] = dfn.groupby("report_date")["y"].transform(
            lambda v: v.to_numpy()[rngn.permutation(len(v))])  # noqa: B023 (closure is defined and consumed within the same loop iteration)
        trn, can, ten = _split(dfn)
        seln = _selective(trn, can, ten, "y", score_seed=s)

        per.append(dict(ivs=ivs, sel=sel, oracle=rc_oracle, auc=auc, mc=mc,
                        nulladv=seln["rand"][X_RET] - seln["method"][X_RET]))

    # ---- coverage table (mean +/- sd over seeds) ----
    print(f"\nCoverage (mean +/- sd over seeds {SEEDS}; target {TARGET:.0%})")
    print(f"{'method':<10} {'base':>14} {'shift':>14} {'worst-grp|shift':>18}")
    for name in ["naive", "cqr", "mondrian", "weighted", "aci"]:
        c = np.array([_cov3(p["ivs"][name]) [:3] for p in per], float)
        b, sh, wg = c.mean(0)
        bs, shs, wgs = c.std(0)
        ex = per[0]
        _, _, _, wdim, wlab = _cov3(ex["ivs"][name])
        print(f"{name:<10} {b:>7.3f}+/-{bs:<5.3f} {sh:>7.3f}+/-{shs:<5.3f} "
              f"{wg:>8.3f}+/-{wgs:<5.3f}  ({wdim}={wlab})")

    # clustered band on headline (aci) shift worst-dim coverage, seed 0
    ev0 = per[0]["ivs"]["aci"]
    sh0 = ev0[ev0.regime == "shift"].reset_index(drop=True)
    p, lo, hi = clustered_bootstrap(
        sh0, lambda d: coverage(d.y.values, d.lo.values, d.hi.values), n_boot=300)
    print(f"\n[seed0 clustered band] ACI shift coverage {p:.3f} [{lo:.3f},{hi:.3f}]")

    # ---- G-PRIMARY (i): risk-coverage @ X_RET ----
    rr = np.mean([p["sel"]["rand"][X_RET] for p in per])
    rm = np.mean([p["sel"]["method"][X_RET] for p in per])
    ro = np.mean([p["oracle"][X_RET] for p in per])
    full = np.mean([p["sel"]["rand"][1.0] for p in per])
    red = rr - rm
    print(f"\n[G-PRIMARY i] retained MAE @ {X_RET:.0%}  (mean over seeds)")
    print(f"  full {full:.4f} | random {rr:.4f} | METHOD(u) {rm:.4f} | oracle {ro:.4f}")
    print(f"  reduction vs random = {red:.4f}  (gate Delta={DELTA})  -> "
          f"{'PASS' if red >= DELTA else 'FAIL'}")
    # band on the reduction across seeds
    reds = np.array([p["sel"]["rand"][X_RET] - p["sel"]["method"][X_RET] for p in per])
    print(f"  per-seed reductions {np.round(reds,4)}  (min {reds.min():.4f})")
    print("  risk-coverage curve (mean over seeds):")
    for r in RETENTIONS:
        print(f"    ret {r:.0%}: random {np.mean([p['sel']['rand'][r] for p in per]):.4f} "
              f"| method {np.mean([p['sel']['method'][r] for p in per]):.4f} "
              f"| oracle {np.mean([p['oracle'][r] for p in per]):.4f}")

    # ---- G-PRIMARY (ii): per-group coverage on retained set (headline method) ----
    print(f"\n[G-PRIMARY ii] per-group coverage on retained {X_RET:.0%} (ACI, per seed)")
    seed_gaps = []
    for s_i, p in enumerate(per):  # noqa: B007 (index kept for readability of the seed loop)
        ev = p["ivs"]["aci"].copy()
        ev["u"] = selective_scores(ev, p["sel"]["u"])
        ret = ev[ev["u"] <= np.quantile(ev["u"], X_RET)]  # keep lowest-u 70%
        dim_gap = {}
        for dim in GROUP_COLS:
            gc = grouped_coverage(ret, dim, TARGET)
            gc = gc[gc["n"] >= MIN_N]
            if not gc.empty:
                dim_gap[dim] = gc["gap"].abs().max()
        seed_gaps.append(dim_gap)
    for dim in GROUP_COLS:
        v = np.array([g[dim] for g in seed_gaps if dim in g])
        print(f"  {dim}: max |gap| over seeds {np.round(v,3)}  mean {v.mean():.3f}")
    allg = np.array([max(g.values()) for g in seed_gaps])
    # pre-registered rule (reports/p2_method_preregistration.md): plain pass/fail at tau
    print(f"  worst-dim |gap| per seed {np.round(allg,3)}  mean {allg.mean():.3f}  "
          f"(gate tau={TAU}) -> {'PASS' if allg.mean() <= TAU else 'FAIL'}")
    print(f"  non-pre-registered reading: a MARGINAL band up to tau+0.01 = {TAU + 0.01:.2f}; "
          f"mean {allg.mean():.4f} -> "
          f"{'PASS' if allg.mean() <= TAU else 'MARGINAL' if allg.mean() <= TAU + 0.01 else 'FAIL'}")

    # ---- G-SHIFT verdict (headline = aci) ----
    # pre-registered rule: WORST-GROUP coverage under the shift must clear the floor
    # AND sit within +/-tau_shift of nominal (reports/p1_baselines_preregistration.md)
    wg_aci = np.mean([_cov3(p["ivs"]["aci"])[2] for p in per])
    sh_aci = np.mean([_cov3(p["ivs"]["aci"])[1] for p in per])
    print(f"\n[G-SHIFT] {_gshift_verdict('ACI', wg_aci)}")
    print(f"  non-pre-registered reading: band tested on overall shift coverage {sh_aci:.3f} "
          f"instead of worst-group (within +/-{TAU_SHIFT} of {TARGET}?) -> "
          f"{'PASS' if (wg_aci >= WG_FLOOR and abs(sh_aci-TARGET) <= TAU_SHIFT) else 'PARTIAL/FAIL'}")

    # ---- G-MECHANISM ----
    auc = np.array([p["auc"] for p in per])
    rho = np.array([p["mc"]["rho"] for p in per])
    rho_c = np.array([p["mc"]["rho_ctrl_vol"] for p in per])
    print(f"\n[G-MECHANISM] AUC(-u -> recoverable) {auc.mean():.3f}+/-{auc.std():.3f} "
          f"(theta {THETA_AUC}) | Spearman(u,sigma_irr) {rho.mean():.3f}+/-{rho.std():.3f} "
          f"(theta' {THETA_RHO})")
    print(f"  confound: Spearman(u,sigma_irr | vol_63) {rho_c.mean():.3f}+/-{rho_c.std():.3f}"
          f"  -> survives vol control? {'YES' if rho_c.mean() >= THETA_RHO else 'WEAKENED'}")
    print(f"  -> {'PASS' if (auc.mean() >= THETA_AUC and rho.mean() >= THETA_RHO and rho_c.mean() >= THETA_RHO) else 'CHECK'}")

    # ---- G-NULL ----
    nul = np.array([p["nulladv"] for p in per])
    print(f"\n[G-NULL] shuffled-y selective advantage @ {X_RET:.0%}: "
          f"{nul.mean():.4f}+/-{nul.std():.4f}  (real {red:.4f}) -> "
          f"{'PASS (collapses)' if abs(nul.mean()) < DELTA * 0.5 else 'CHECK (leakage?)'}")
    return per


def _build_1b():
    panel = pd.read_parquet(PANEL)
    r = pd.read_parquet(RETURNS).pivot(index="date", columns="ticker", values="ret")
    L = np.log1p(r.clip(lower=-0.95))
    fwd = L.rolling(21).sum().shift(-21).stack().rename("y").reset_index()
    fwd.columns = ["date", "ticker", "y"]
    df = panel.merge(fwd, on=["date", "ticker"], how="inner").dropna(
        subset=FEATURE_COLS + ["y"])
    df["regime"] = np.where(df.report_date.between(SHIFT_LO, SHIFT_HI), "shift", "base")
    return df


def run_1b():
    print("\n" + "=" * 74)
    print("LAYER 1b — real public task (21d fwd return) | METHOD (P2)")
    print("=" * 74)
    df = _build_1b()
    tr, ca, te = _split(df)
    ivs = _intervals(tr, ca, te, "y")
    print(f"{'method':<10} {'base':>8} {'shift':>8} {'worst-grp|shift':>22} {'width':>8}")
    for name in ["naive", "cqr", "mondrian", "weighted", "aci"]:
        b, sh, wg, dim, lab = _cov3(ivs[name])
        w = mean_width(ivs[name].lo.values, ivs[name].hi.values)
        print(f"{name:<10} {b:>8.3f} {sh:>8.3f} {wg:>10.3f} ({dim}={lab})  {w:>8.3f}")

    # G-SHIFT on real returns, same pre-registered rule as 1a (worst-group, floor AND band)
    print(f"\n[G-SHIFT] {_gshift_verdict('ACI', _cov3(ivs['aci'])[2])}")
    print(f"  context, same rule on the static baseline: {_gshift_verdict('naive', _cov3(ivs['naive'])[2])}")

    sel = _selective(tr, ca, te, "y", score_seed=0)
    rr, rm = sel["rand"][X_RET], sel["method"][X_RET]
    # G-NULL on 1b
    rngn = np.random.default_rng(2000)
    dfn = df.copy()
    dfn["y"] = dfn.groupby("report_date")["y"].transform(
        lambda v: v.to_numpy()[rngn.permutation(len(v))])
    trn, can, ten = _split(dfn)
    seln = _selective(trn, can, ten, "y", score_seed=0)
    print(f"\n[G-PRIMARY i] retained MAE @ {X_RET:.0%}: random {rr:.4f} | method {rm:.4f} "
          f"| reduction {rr-rm:.4f}")
    print(f"[G-NULL] shuffled advantage {seln['rand'][X_RET]-seln['method'][X_RET]:.4f}")


if __name__ == "__main__":
    run_1a()
    run_1b()
