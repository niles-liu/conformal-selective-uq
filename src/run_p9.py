"""P9 driver — selective conformal RISK control in regression (SPEC §11 P9).

Phase A (this commit): the **method-blind baseline read** that freezes the G-RISK
target/margin BEFORE the method runs (gate-first). It reports, on the would-be-retained
pool (lowest-u 70%), the risk (capped miss-magnitude) and coverage under a single GLOBAL
conformal radius calibrated on the FULL calibration fold — base vs shift — plus the
random-selection comparator and the no-selection reference. Those numbers set the bar the
select-then-calibrate method must clear.

Phase B (after freeze): the SCRC method (`riskcontrol.crc_radius` on the SELECTED calib
fold + an online shift-robust variant) and the G-RISK / G-NULL gate evaluation.

Run: python src/run_p9.py --baseline     (Phase A, method-blind)
     python src/run_p9.py                 (full, after the pre-reg is frozen)
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import run_p2 as P
from _common import PROC
from conformal import _conf_quantile
from eval import clustered_bootstrap, coverage
from learners import fit_mean, fit_quantile
from panel import FEATURE_COLS
from riskcontrol import crc_radius, loss_miss_magnitude, scrc, scrc_aci
from run_p2 import ALPHA, SEEDS, TARGET, _split, _xy
from selective import name_uncertainty, selective_scores
from synthetic import generate

R_RET = 0.70                       # retention (a priori, matches P2 X_RET)
NBOOT_RAND = 20                    # random-selection draws to average

# --- FROZEN gate constants (reports/p9_riskcontrol_preregistration.md) ------ #
ALPHA_RISK = 0.05                  # capped miss-magnitude target
TAU_RISK = 0.03                    # shift band -> ceiling 0.08
MARGIN_SHIFT = 0.03                # online must beat static-radius shift risk by this
GAMMA = 0.05                       # eta = GAMMA * W0 (the cqr_conditional_aci step)
N_GRID = 121                       # CRC radius grid resolution


def _base_pieces(tr, ca, te, target):
    """Fit CQR base once; return everything both baseline and method need:
    pred_te, qlo/qhi on calib & test, calib conformity E, per-name u -> test score."""
    Xtr, ytr = _xy(tr, target); Xca, yca = _xy(ca, target); Xte, yte = _xy(te, target)
    mean = fit_mean(Xtr, ytr)
    qlo = fit_quantile(Xtr, ytr, ALPHA / 2)
    qhi = fit_quantile(Xtr, ytr, 1 - ALPHA / 2)
    qlo_ca, qhi_ca = qlo.predict(Xca), qhi.predict(Xca)
    qlo_te, qhi_te = qlo.predict(Xte), qhi.predict(Xte)
    E = np.maximum(qlo_ca - yca, yca - qhi_ca)
    # per-name uncertainty from calib mean residuals (the frozen selective score)
    cadf = ca[["ticker"]].copy()
    cadf["resid"] = np.abs(yca - mean.predict(Xca))
    u = name_uncertainty(cadf)
    score_te = selective_scores(te, u)
    return dict(pred=mean.predict(Xte), qlo_ca=qlo_ca, qhi_ca=qhi_ca,
                qlo_te=qlo_te, qhi_te=qhi_te, E=E, yca=yca, yte=yte,
                u=u, score_te=score_te)


def _risk_cov(y, lo, hi, w0):
    """(risk, coverage) for a set of points: risk = mean capped miss-magnitude."""
    return (float(np.mean(loss_miss_magnitude(y, lo, hi, w0))),
            float(coverage(y, lo, hi)))


def baseline_read(layer="1a"):
    print("=" * 92)
    print(f"P9 baseline read (method-blind) | Layer {layer} | retention r={R_RET:.0%}, "
          f"target coverage {TARGET:.0%}")
    print("=" * 92)
    seeds = SEEDS if layer == "1a" else [0]
    rows = {k: [] for k in ("W0", "cov_full_base", "cov_full_shift",
                            "risk_full_base", "risk_full_shift",
                            "cov_ret_base", "cov_ret_shift",
                            "risk_ret_base", "risk_ret_shift", "width_ret",
                            "cov_rnd_base", "cov_rnd_shift",
                            "risk_rnd_base", "risk_rnd_shift", "width_rnd")}
    for s in seeds:
        if layer == "1a":
            path = PROC / f"synth_1a_seed{s}.parquet"
            if not path.exists():
                generate(s)
            df = pd.read_parquet(path).dropna(subset=FEATURE_COLS + ["y"])
        else:
            df = P._build_1b()
        tr, ca, te = _split(df)
        bp = _base_pieces(tr, ca, te, "y")
        Qg = _conf_quantile(bp["E"], ALPHA)                 # GLOBAL radius (full calib)
        yte = bp["yte"]
        lo = bp["qlo_te"] - Qg; hi = bp["qhi_te"] + Qg      # global-cal intervals
        regime = te["regime"].to_numpy()
        base_m, shift_m = regime == "base", regime == "shift"
        # frozen scale W0 = median base-regime interval width (seed-mean frozen in pre-reg)
        w0 = float(np.median((hi - lo)[base_m]))
        rows["W0"].append(w0)

        # no-selection reference (full test)
        rb, cb = _risk_cov(yte[base_m], lo[base_m], hi[base_m], w0)
        rs, cs = _risk_cov(yte[shift_m], lo[shift_m], hi[shift_m], w0)
        rows["risk_full_base"].append(rb); rows["cov_full_base"].append(cb)
        rows["risk_full_shift"].append(rs); rows["cov_full_shift"].append(cs)

        # SELECTED pool (lowest-u r fraction) under the same GLOBAL radius
        keep = bp["score_te"] <= np.quantile(bp["score_te"], R_RET)
        kb, ks = keep & base_m, keep & shift_m
        rb, cb = _risk_cov(yte[kb], lo[kb], hi[kb], w0)
        rs, cs = _risk_cov(yte[ks], lo[ks], hi[ks], w0)
        rows["risk_ret_base"].append(rb); rows["cov_ret_base"].append(cb)
        rows["risk_ret_shift"].append(rs); rows["cov_ret_shift"].append(cs)
        rows["width_ret"].append(float(np.mean((hi - lo)[keep])))

        # RANDOM selection comparator (avg over draws)
        rng = np.random.default_rng(s)
        rrb = rrs = crb = crs = wr = 0.0
        for _ in range(NBOOT_RAND):
            rk = rng.random(len(yte)) <= R_RET
            rkb, rks = rk & base_m, rk & shift_m
            a, c = _risk_cov(yte[rkb], lo[rkb], hi[rkb], w0); rrb += a; crb += c
            a, c = _risk_cov(yte[rks], lo[rks], hi[rks], w0); rrs += a; crs += c
            wr += float(np.mean((hi - lo)[rk]))
        rows["risk_rnd_base"].append(rrb / NBOOT_RAND); rows["cov_rnd_base"].append(crb / NBOOT_RAND)
        rows["risk_rnd_shift"].append(rrs / NBOOT_RAND); rows["cov_rnd_shift"].append(crs / NBOOT_RAND)
        rows["width_rnd"].append(wr / NBOOT_RAND)

    def ms(k):
        v = np.array(rows[k], float)
        return v.mean(), v.std()

    print(f"\nFrozen scale  W0 (median base width) = {ms('W0')[0]:.3f} +/- {ms('W0')[1]:.3f}")
    print(f"\n{'pool':<22}{'cov|base':>12}{'cov|shift':>12}"
          f"{'risk|base':>12}{'risk|shift':>12}{'width':>10}")
    print(f"{'full (no select)':<22}{ms('cov_full_base')[0]:>12.3f}{ms('cov_full_shift')[0]:>12.3f}"
          f"{ms('risk_full_base')[0]:>12.3f}{ms('risk_full_shift')[0]:>12.3f}{'-':>10}")
    print(f"{'retained (u, global)':<22}{ms('cov_ret_base')[0]:>12.3f}{ms('cov_ret_shift')[0]:>12.3f}"
          f"{ms('risk_ret_base')[0]:>12.3f}{ms('risk_ret_shift')[0]:>12.3f}{ms('width_ret')[0]:>10.3f}")
    print(f"{'retained (random)':<22}{ms('cov_rnd_base')[0]:>12.3f}{ms('cov_rnd_shift')[0]:>12.3f}"
          f"{ms('risk_rnd_base')[0]:>12.3f}{ms('risk_rnd_shift')[0]:>12.3f}{ms('width_rnd')[0]:>10.3f}")
    print("\n(risk = mean capped miss-magnitude in [0,1]; target to be frozen from these numbers)")
    return rows


def _method_metrics(df):
    """One (train/calib/test) split: build scrc + scrc_aci + comparators, return the
    per-regime risk/coverage/width metrics and the label-shuffle advantage for G-NULL."""
    tr, ca, te = _split(df)
    bp = _base_pieces(tr, ca, te, "y")
    Qg = _conf_quantile(bp["E"], ALPHA)
    yte = bp["yte"]; regime = te["regime"].to_numpy()
    base_m, shift_m = regime == "base", regime == "shift"
    lo_g, hi_g = bp["qlo_te"] - Qg, bp["qhi_te"] + Qg     # global-radius intervals
    w0 = float(np.median((hi_g - lo_g)[base_m]))
    lossf = lambda y, lo, hi: loss_miss_magnitude(y, lo, hi, w0)
    grid = np.linspace(0.0, 3.0 * Qg, N_GRID)
    score_ca = selective_scores(ca, bp["u"])
    score_te = bp["score_te"]
    order = te["report_date"].to_numpy()

    S = scrc(bp["qlo_ca"], bp["qhi_ca"], bp["yca"], bp["qlo_te"], bp["qhi_te"],
             score_ca, score_te, R_RET, lossf, ALPHA_RISK, grid)
    A = scrc_aci(bp["qlo_ca"], bp["qhi_ca"], bp["yca"], bp["qlo_te"], bp["qhi_te"],
                 yte, order, score_ca, score_te, R_RET, lossf, ALPHA_RISK, grid,
                 eta=GAMMA * w0)
    lam_full = crc_radius(bp["qlo_ca"], bp["qhi_ca"], bp["yca"], lossf, ALPHA_RISK, grid)

    def rc(lo, hi, keep, m):
        sel = keep & m
        return (float(np.mean(lossf(yte[sel], lo[sel], hi[sel]))),
                float(coverage(yte[sel], lo[sel], hi[sel])), int(sel.sum()))

    keep = S["keep"]
    out = dict(w0=w0, Qg=Qg, lam_sel=S["lam"], lam_full=lam_full, lam_path=A["lam_path"])
    out["scrc"] = dict(base=rc(S["lo"], S["hi"], keep, base_m),
                       shift=rc(S["lo"], S["hi"], keep, shift_m),
                       width=float(np.mean((S["hi"] - S["lo"])[keep])))
    out["scrc_aci"] = dict(base=rc(A["lo"], A["hi"], keep, base_m),
                          shift=rc(A["lo"], A["hi"], keep, shift_m),
                          width=float(np.mean((A["hi"] - A["lo"])[keep])))
    # efficiency comparator: full-fold CRC radius on the SAME retained test pool
    lo_f, hi_f = bp["qlo_te"] - lam_full, bp["qhi_te"] + lam_full
    out["fullcrc"] = dict(base=rc(lo_f, hi_f, keep, base_m),
                         width=float(np.mean((hi_f - lo_f)[keep])))
    # seed-0 clustered-band frame for scrc base retained risk
    out["_ev"] = dict(report_date=te["report_date"].to_numpy(),
                      sector=te["sector"].to_numpy(), y=yte,
                      lo=S["lo"], hi=S["hi"], keep=keep, base=base_m, w0=w0)

    # --- G-SELECT + G-NULL: u-selection vs random advantage (mixed, global radius) ---
    out["adv_real"] = _sel_advantage(yte, score_te, lo_g, hi_g, lossf)
    return out


def _sel_advantage(yte, score_te, lo, hi, lossf):
    """Retained-risk advantage of u-selection over random selection (mixed test, at the
    given radius): random_risk - u_risk. Positive => u-selection is informative."""
    k = score_te <= np.quantile(score_te, R_RET)
    u_risk = float(np.mean(lossf(yte[k], lo[k], hi[k])))
    rng = np.random.default_rng(0)
    rr = []
    for _ in range(NBOOT_RAND):
        m = rng.random(len(yte)) <= R_RET
        rr.append(float(np.mean(lossf(yte[m], lo[m], hi[m]))))
    return float(np.mean(rr)) - u_risk


def _shuffle_adv(df, seed):
    """G-NULL: shuffle y within report_date, rebuild the base pieces, recompute the
    u-vs-random selective risk advantage. Should collapse toward 0 (no leakage)."""
    rng = np.random.default_rng(3000 + seed)
    dfn = df.copy()
    dfn["y"] = dfn.groupby("report_date")["y"].transform(
        lambda v: v.to_numpy()[rng.permutation(len(v))])
    tr, ca, te = _split(dfn)
    bp = _base_pieces(tr, ca, te, "y")
    Qg = _conf_quantile(bp["E"], ALPHA)
    yte = bp["yte"]
    lo_g, hi_g = bp["qlo_te"] - Qg, bp["qhi_te"] + Qg
    w0 = float(np.median((hi_g - lo_g)[te["regime"].to_numpy() == "base"]))
    lossf = lambda y, lo, hi: loss_miss_magnitude(y, lo, hi, w0)
    return _sel_advantage(yte, bp["score_te"], lo_g, hi_g, lossf)


def run_1a():
    print("=" * 92)
    print("P9 — SELECTIVE CONFORMAL RISK CONTROL (regression) | Layer 1a (gated)")
    print(f"     retention r={R_RET:.0%}, alpha_risk={ALPHA_RISK}, tau_risk={TAU_RISK}, "
          f"eta={GAMMA}*W0")
    print("=" * 92)
    import time
    t0 = time.time()
    per = []
    for s in SEEDS:
        path = PROC / f"synth_1a_seed{s}.parquet"
        if not path.exists():
            generate(s)
        df = pd.read_parquet(path).dropna(subset=FEATURE_COLS + ["y"])
        m = _method_metrics(df)
        m["adv_null"] = _shuffle_adv(df, s)
        per.append(m)
        print(f"  [progress] seed {s} done @ {time.time()-t0:.0f}s", flush=True)

    def col(meth, reg, i):     # i: 0 risk, 1 cov
        return np.array([p[meth][reg][i] for p in per], float)

    print(f"\nRetained-pool RISK (capped miss-magnitude) & coverage; mean+/-sd over {SEEDS}")
    print(f"{'method':<14}{'risk|base':>16}{'risk|shift':>16}{'cov|base':>12}{'cov|shift':>12}{'width':>9}")
    for meth in ("scrc", "scrc_aci"):
        rb, rs = col(meth, "base", 0), col(meth, "shift", 0)
        cb, cs = col(meth, "base", 1), col(meth, "shift", 1)
        w = np.array([p[meth]["width"] for p in per])
        print(f"{meth:<14}{rb.mean():>8.3f}+/-{rb.std():<5.3f}{rs.mean():>8.3f}+/-{rs.std():<5.3f}"
              f"{cb.mean():>12.3f}{cs.mean():>12.3f}{w.mean():>9.3f}")
    fb = col("fullcrc", "base", 0); fw = np.array([p["fullcrc"]["width"] for p in per])
    print(f"{'full-fold CRC':<14}{fb.mean():>8.3f}+/-{fb.std():<5.3f}{'-':>16}{'-':>12}{'-':>12}{fw.mean():>9.3f}")

    lam_sel = np.array([p["lam_sel"] for p in per]); lam_full = np.array([p["lam_full"] for p in per])
    print(f"\nradius: scrc lambda_sel {lam_sel.mean():.3f} (selected calib) vs full-fold "
          f"{lam_full.mean():.3f}; Qg {np.mean([p['Qg'] for p in per]):.3f}")

    # ---- gates ----
    grb = col("scrc", "base", 0).mean()
    g_base = grb <= ALPHA_RISK
    print(f"\n[G-RISK-BASE] scrc retained base risk {grb:.3f} <= {ALPHA_RISK}? -> "
          f"{'PASS' if g_base else 'MISS'}")

    aci_shift = col("scrc_aci", "shift", 0).mean()
    scrc_shift = col("scrc", "shift", 0).mean()
    g_shift = (aci_shift <= ALPHA_RISK + TAU_RISK) and (aci_shift <= scrc_shift - MARGIN_SHIFT)
    print(f"[G-RISK-SHIFT] scrc_aci shift risk {aci_shift:.3f} <= {ALPHA_RISK+TAU_RISK:.2f} "
          f"AND <= static scrc {scrc_shift:.3f} - {MARGIN_SHIFT} ({scrc_shift-MARGIN_SHIFT:.3f})? -> "
          f"{'PASS' if g_shift else 'MISS'}")

    adv_r = np.mean([p["adv_real"] for p in per]); adv_n = np.mean([p["adv_null"] for p in per])
    g_null = abs(adv_n) < 0.5 * abs(adv_r) if adv_r != 0 else abs(adv_n) < 1e-6
    print(f"[G-NULL] selective risk advantage real {adv_r:.4f} vs shuffled {adv_n:.4f} "
          f"(|null| < 0.5*|real|={0.5*abs(adv_r):.4f})? -> {'PASS (collapses)' if g_null else 'CHECK'}")

    g_eff = lam_sel.mean() < lam_full.mean()
    scrc_w = np.mean([p["scrc"]["width"] for p in per])
    print(f"[G-EFFICIENCY report] scrc width {scrc_w:.3f} (lambda_sel {lam_sel.mean():.3f}) < "
          f"full-fold CRC {fw.mean():.3f} (lambda_full {lam_full.mean():.3f})? -> "
          f"{'PASS' if g_eff else 'MISS'}")
    print(f"[G-SELECT report] u-selection risk advantage over random {adv_r:.4f} > 0? -> "
          f"{'PASS' if adv_r > 0 else 'MISS'}")

    # seed-0 clustered band on scrc base retained risk
    ev = per[0]["_ev"]; w0 = ev["w0"]
    mask = ev["keep"] & ev["base"]
    band = pd.DataFrame(dict(report_date=ev["report_date"][mask], sector=ev["sector"][mask],
                             y=ev["y"][mask], lo=ev["lo"][mask], hi=ev["hi"][mask]))
    p, lo, hi = clustered_bootstrap(
        band, lambda d: float(np.mean(loss_miss_magnitude(d.y.values, d.lo.values, d.hi.values, w0))),
        n_boot=300)
    print(f"\n[seed0 clustered band] scrc base retained risk {p:.3f} [{lo:.3f},{hi:.3f}]")

    # lambda path adaptation check (mechanism): the ONLINE lambda is not grid-bounded
    # (only the batch CRC uses the grid); a healthy path RISES under shift then RELAXES.
    lp = np.array([v for _, v in per[0]["lam_path"]])
    adapts = (lp.max() > lp[0] + 1e-9) and (lp[-1] < lp.max() - 1e-9)
    print(f"[mechanism] scrc_aci lambda path seed0: start {lp[0]:.3f}, max {lp.max():.3f}, "
          f"end {lp[-1]:.3f} — {'adapts (rises under shift, relaxes after)' if adapts else 'STATIC'}")

    ships = g_base and g_shift and g_null
    verdict = "FULL SHIP" if ships else ("QUALIFIED" if (g_base and g_null and aci_shift < scrc_shift)
                                         else "REJECT")
    print(f"\n==> P9 verdict (1a): {verdict}  (G-RISK-BASE {'pass' if g_base else 'miss'}, "
          f"G-RISK-SHIFT {'pass' if g_shift else 'miss'}, G-NULL {'pass' if g_null else 'miss'})")
    return verdict


def run_1b():
    print("\n" + "=" * 92)
    print("P9 — selective conformal risk control | Layer 1b (real 21d fwd return, external read)")
    print("=" * 92)
    df = P._build_1b()
    m = _method_metrics(df)
    print(f"{'method':<14}{'risk|base':>12}{'risk|shift':>12}{'cov|base':>12}{'cov|shift':>12}{'width':>9}")
    for meth in ("scrc", "scrc_aci"):
        print(f"{meth:<14}{m[meth]['base'][0]:>12.3f}{m[meth]['shift'][0]:>12.3f}"
              f"{m[meth]['base'][1]:>12.3f}{m[meth]['shift'][1]:>12.3f}{m[meth]['width']:>9.3f}")
    print(f"{'full-fold CRC':<14}{m['fullcrc']['base'][0]:>12.3f}{'-':>12}{'-':>12}{'-':>12}"
          f"{m['fullcrc']['width']:>9.3f}")
    print(f"radius: lambda_sel {m['lam_sel']:.3f} vs full-fold {m['lam_full']:.3f}; Qg {m['Qg']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true", help="Phase A method-blind read")
    args = ap.parse_args()
    if args.baseline:
        baseline_read("1a")
        baseline_read("1b")
    else:
        run_1a()
        run_1b()
