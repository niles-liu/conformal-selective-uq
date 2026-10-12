"""P13 driver: selection-aware calibration for the kept set.

P2 builds the abstention score u_i from calibration-fold residuals and calibrates the conformal
layer on the same fold, so coverage on the kept 70% has no finite-sample guarantee. P13 compares
that arrangement with a selection-aware one: u_i from cross-fitted TRAINING residuals (no
calibration or test label touched), then split CQR calibrated on the kept calibration rows only.

Gates (kept base-regime test rows, mean over seeds {0,1,2}):
  G-VALID   V2 coverage within +/-0.02 of 0.90                                (primary)
  G-TIGHT   V2 width / V2all width <= 0.95                                    (primary)
  G-MECH    u_train AUC >= 0.70 and retained-MAE reduction vs random >= 0.119 (primary)

Pre-reg + apparatus: reports/p13_selection_aware_preregistration.md.
Run: python experiments/run_p13.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import PROC
from panel import FEATURE_COLS, GROUP_COLS
from run_p2 import ALPHA, DELTA, GAMMA, MIN_N, RETENTIONS, SEEDS, TARGET, THETA_AUC, X_RET, _grp, _split, _xy
from selective_conformal import (
    aci_from_quantiles,
    clustered_bootstrap,
    coverage,
    cqr_from_quantiles,
    grouped_coverage,
    mean_width,
    name_uncertainty,
    risk_coverage,
    selective_scores,
)
from selective_conformal.learners import fit_mean, fit_quantile
from selective_conformal.selective import mechanism_auc
from synthetic import generate

N_BLOCKS = 5               # contiguous time blocks for the cross-fitted training residuals
HALF = 20210701            # u_half: calibration residuals before this date only
G_VALID_TOL, G_TIGHT = 0.02, 0.95
VARIANTS = ["V0", "V1", "V2all", "V2", "V3", "V2-ACI"]
LABEL = {"V0": "V0 u_cal, calib all (P2)", "V1": "V1 u_cal, calib kept",
         "V2all": "V2all u_train, calib all", "V2": "V2 u_train, calib kept (PROPOSED)",
         "V3": "V3 u_half, calib kept H2", "V2-ACI": "V2-ACI u_train, online kept"}


def _pf(b):
    return "PASS" if b else "FAIL"


def _oof_residuals(tr: pd.DataFrame) -> np.ndarray:
    """|y - mean(x)| on the training fold, each block predicted by a model fit on the others."""
    X, y = _xy(tr, "y")
    rd = tr.report_date.to_numpy()
    res = np.empty(len(tr))
    for block in np.array_split(np.sort(np.unique(rd)), N_BLOCKS):
        hold = np.isin(rd, block)
        m = fit_mean(X[~hold], y[~hold])
        res[hold] = np.abs(y[hold] - m.predict(X[hold]))
    return res


def _u(rows: pd.DataFrame, resid: np.ndarray) -> pd.Series:
    return name_uncertainty(pd.DataFrame({"ticker": rows["ticker"].to_numpy(), "resid": resid}))


def run_seed(s: int) -> dict:
    path = PROC / f"synth_1a_seed{s}.parquet"
    if not path.exists():
        generate(s)
    df = pd.read_parquet(path).dropna(subset=FEATURE_COLS + ["y"])
    tr, ca, te = _split(df)
    Xtr, ytr = _xy(tr, "y"); Xca, yca = _xy(ca, "y"); Xte, yte = _xy(te, "y")

    mean = fit_mean(Xtr, ytr)
    qlo, qhi = fit_quantile(Xtr, ytr, ALPHA / 2), fit_quantile(Xtr, ytr, 1 - ALPHA / 2)
    lo_c, hi_c, lo_t, hi_t = qlo.predict(Xca), qhi.predict(Xca), qlo.predict(Xte), qhi.predict(Xte)

    res_ca = np.abs(yca - mean.predict(Xca))
    h1 = ca.report_date.to_numpy() < HALF
    scores = {"u_cal": _u(ca, res_ca), "u_train": _u(tr, _oof_residuals(tr)),
              "u_half": _u(ca[h1], res_ca[h1])}

    def kept(score):   # P2's label-free threshold, applied to both folds
        tau = np.quantile(selective_scores(te, scores[score]), X_RET)
        return selective_scores(ca, scores[score]) <= tau, selective_scores(te, scores[score]) <= tau

    k_cal, k_train, k_half = kept("u_cal"), kept("u_train"), kept("u_half")
    plan = {"V0": (k_cal, np.ones(len(ca), bool)), "V1": (k_cal, k_cal[0]),
            "V2all": (k_train, np.ones(len(ca), bool)), "V2": (k_train, k_train[0]),
            "V3": (k_half, k_half[0] & ~h1)}
    base = te.regime.to_numpy() == "base"
    out = {}
    for name, ((_, keep_te), cal_rows) in plan.items():
        lo, hi = cqr_from_quantiles(lo_c[cal_rows], hi_c[cal_rows], yca[cal_rows], lo_t, hi_t, ALPHA)
        out[name] = dict(keep=keep_te, lo=lo, hi=hi)
    kc, kt = k_train
    alo, ahi = np.full(len(te), np.nan), np.full(len(te), np.nan)
    alo[kt], ahi[kt] = aci_from_quantiles(lo_c[kc], hi_c[kc], yca[kc], _grp(ca)[kc], lo_t[kt], hi_t[kt],
                                          yte[kt], te.report_date.to_numpy()[kt], _grp(te)[kt],
                                          ALPHA, GAMMA, MIN_N)
    out["V2-ACI"] = dict(keep=kt, lo=alo, hi=ahi)

    row = {}
    for name, v in out.items():
        kb, ks = v["keep"] & base, v["keep"] & ~base
        row[name] = dict(cov_base=coverage(yte[kb], v["lo"][kb], v["hi"][kb]),
                         width_base=mean_width(v["lo"][kb], v["hi"][kb]),
                         cov_shift=coverage(yte[ks], v["lo"][ks], v["hi"][ks]),
                         n_base=int(kb.sum()), n_cal=int(plan[name][1].sum()) if name in plan else int(kc.sum()))

    def kept_base_frame(name):
        v = out[name]
        m = v["keep"] & base
        f = te.loc[m, ["report_date", "sector", "cap_tier", "mom_decile"]].reset_index(drop=True)
        f["y"], f["lo"], f["hi"] = yte[m], v["lo"][m], v["hi"][m]
        return f

    gaps = {}
    for name in ("V0", "V2"):
        f = kept_base_frame(name)
        worst = 0.0
        for dim in GROUP_COLS:
            gc = grouped_coverage(f, dim, TARGET)
            gc = gc[gc["n"] >= MIN_N]
            if not gc.empty:
                worst = max(worst, float(gc["gap"].abs().max()))
        gaps[name] = worst

    gt = df.groupby("ticker").agg(recoverable_mask=("recoverable_mask", "first")).reset_index()
    pred_te = mean.predict(Xte)
    rng = np.random.default_rng(s)
    rand = [risk_coverage(yte, pred_te, rng.standard_normal(len(yte)), RETENTIONS) for _ in range(20)]
    rand70 = float(np.mean([c[X_RET] for c in rand]))
    mech = {}
    for sc in ("u_cal", "u_train", "u_half"):
        rc = risk_coverage(yte, pred_te, selective_scores(te, scores[sc]), RETENTIONS)
        mech[sc] = dict(auc=mechanism_auc(scores[sc], gt), red=rand70 - rc[X_RET])
    agree = float(np.mean(k_cal[1] == k_train[1]))
    return dict(row=row, gaps=gaps, mech=mech, agree=agree, band_frame=kept_base_frame("V2") if s == SEEDS[0] else None)


def main():
    print("=" * 92)
    print("P13: SELECTION-AWARE CALIBRATION FOR THE KEPT SET (Layer 1a, seeds %s)" % SEEDS)
    print("=" * 92)
    per = []
    for s in SEEDS:
        per.append(run_seed(s))
        r = per[-1]["row"]
        print(f"[seed {s}] kept-base coverage: " + "  ".join(f"{v} {r[v]['cov_base']:.3f}" for v in VARIANTS),
              flush=True)

    def mean_sd(v, key):
        a = np.array([p["row"][v][key] for p in per])
        return a.mean(), a.std()

    print(f"\nKept set = lowest-score {X_RET:.0%} of test rows (P2's rule). Target {TARGET:.0%}. Mean +/- sd over seeds.")
    print(f"{'variant':<34}{'cov base':>15}{'width base':>12}{'cov shift':>15}{'n base':>8}{'n calib':>9}")
    for v in VARIANTS:
        cb, cbs = mean_sd(v, "cov_base"); w, _ = mean_sd(v, "width_base"); cs, css = mean_sd(v, "cov_shift")
        nb = int(np.mean([p["row"][v]["n_base"] for p in per])); nc = int(np.mean([p["row"][v]["n_cal"] for p in per]))
        print(f"{LABEL[v]:<34}{cb:>8.3f}+/-{cbs:<5.3f}{w:>11.3f}{cs:>9.3f}+/-{css:<5.3f}{nb:>7d}{nc:>9d}")

    p, lo, hi = clustered_bootstrap(per[0]["band_frame"],
                                    lambda d: coverage(d.y.values, d.lo.values, d.hi.values), n_boot=300)
    print(f"\n[seed0 clustered band] V2 kept-base coverage {p:.3f} [{lo:.3f},{hi:.3f}]")
    for v in ("V0", "V2"):
        g = np.array([q["gaps"][v] for q in per])
        print(f"worst-group |gap| on kept base rows, {v}: per seed {np.round(g, 3)}  mean {g.mean():.3f}")
    for sc in ("u_cal", "u_train", "u_half"):
        auc = np.array([q["mech"][sc]["auc"] for q in per]); red = np.array([q["mech"][sc]["red"] for q in per])
        print(f"score {sc:<8} AUC {auc.mean():.3f}+/-{auc.std():.3f} | retained-MAE reduction vs random "
              f"@{X_RET:.0%} {red.mean():.4f} (per seed {np.round(red, 4)})")
    print(f"kept-set agreement u_cal vs u_train (share of test rows with the same keep/drop call): "
          f"{np.mean([q['agree'] for q in per]):.3f}")

    v2, v2all = mean_sd("V2", "cov_base")[0], mean_sd("V2all", "cov_base")[0]
    wr = mean_sd("V2", "width_base")[0] / mean_sd("V2all", "width_base")[0]
    auc_t = np.mean([q["mech"]["u_train"]["auc"] for q in per])
    red_t = np.mean([q["mech"]["u_train"]["red"] for q in per])
    g_valid = abs(v2 - TARGET) <= G_VALID_TOL
    g_tight = wr <= G_TIGHT
    g_mech = auc_t >= THETA_AUC and red_t >= DELTA
    v0, v1 = mean_sd("V0", "cov_base")[0], mean_sd("V1", "cov_base")[0]
    v3, w3 = mean_sd("V3", "cov_base")[0], mean_sd("V3", "width_base")[0]

    print("\n" + "-" * 92)
    print("PRE-COMMITTED GATES (reports/p13_selection_aware_preregistration.md)")
    print(f"[G-VALID]  V2 kept-base coverage {v2:.3f}, within +/-{G_VALID_TOL} of {TARGET}? "
          f"({abs(v2 - TARGET):.3f} off) -> {_pf(g_valid)}")
    print(f"[G-TIGHT]  V2 width / V2all width = {wr:.3f} <= {G_TIGHT} -> {_pf(g_tight)} "
          f"(V2all coverage {v2all:.3f})")
    print(f"[G-MECH]   u_train AUC {auc_t:.3f} >= {THETA_AUC} and reduction {red_t:.4f} >= {DELTA} -> {_pf(g_mech)}")
    print(f"[G-CAVEAT] (report) V0 kept-base coverage {v0:.3f}: {v0 - TARGET:+.3f} from nominal")
    print(f"[G-REUSE]  (report) V1 - V2 kept-base coverage = {v1 - v2:+.3f} (V1 {v1:.3f})")
    print(f"[G-SPLIT]  (report) V3 coverage {v3:.3f}, width {w3:.3f} vs V2 width {mean_sd('V2', 'width_base')[0]:.3f}")
    print(f"[G-SHIFT]  (report) kept shift coverage: V2 {mean_sd('V2', 'cov_shift')[0]:.3f}, "
          f"V2-ACI {mean_sd('V2-ACI', 'cov_shift')[0]:.3f}")
    if g_valid and g_tight and g_mech:
        verdict = ("CONFIRM: a selection-aware score plus calibration on the kept stocks gives nominal "
                   "kept-set coverage, narrower intervals than calibrating on every stock, and still finds "
                   "the unrecoverable stocks.")
    elif g_valid:
        verdict = "QUALIFIED: kept-set coverage is nominal, but " + (
            "the width gain is below the gate" if not g_tight else "the selection-aware score is weaker") + "."
    else:
        verdict = (f"REJECT-as-finding: the selection-aware kept-set coverage is {v2:.3f}, "
                   f"{'above' if v2 > TARGET else 'below'} the +/-{G_VALID_TOL} band.")
    print(f"\nVERDICT -> {verdict}")


if __name__ == "__main__":
    main()
