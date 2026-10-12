"""P8 driver — conformal on the simplex (compositional target) — SPEC §11 P8, highest leverage.

Phase A (--baseline, method-blind): the failure the joint region must fix — the INDEPENDENT-
scalar product region's JOINT coverage (all ~90 names' marginal intervals containing the truth
at once), which multiplicity drives ~0. Plus the compositional-score distribution.

Phase B (full): the JOINT simplex region (static + online-ACI on the scalar compositional score),
its regime coverage vs the product region, and G-COMPO-MECH — do the compositionally-hardest names
coincide with the low-recoverable / high-irreducible-noise names (the known-floor mechanism, ported
to the simplex).

Layers: 1a only (the compositional substrate). 1b is forward *returns*, not a composition, so
P8 has no real-data read in this repository. Run:
    python experiments/run_p8.py --baseline
    python experiments/run_p8.py
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from _common import PROC
from panel import FEATURE_COLS
from run_p2 import ALPHA, SEEDS, TARGET, TAU, TAU_SHIFT, THETA_AUC, THETA_RHO, _split, _xy  # reuse frozen tolerances
from selective_conformal.conformal import cqr
from selective_conformal.learners import fit_mean
from selective_conformal.simplex import aci_radius_path, aitchison, clr_dev, region_radius, softmax, tv
from synthetic import generate

GAMMA = 0.05


def _base_logit(sub: pd.DataFrame) -> np.ndarray:
    """Reconstruct the generator base: 0.5 * per-date z-score of log_dollar_vol."""
    v = sub["log_dollar_vol"].to_numpy(float)
    mu, sd = np.nanmean(v), np.nanstd(v)
    z = (v - mu) / sd if sd > 0 else np.zeros_like(v)
    return 0.5 * np.nan_to_num(z)


def _session_pieces(te: pd.DataFrame, yhat: np.ndarray):
    """Per session: true composition (closed `weight`), predicted composition
    softmax(base_logit + yhat), regime, and the per-name CLR deviation. Returns lists of
    per-session (aitchison, tv, regime) and a per-name difficulty accumulator."""
    te = te.copy()
    te["_yhat"] = yhat
    rows = []
    name_dev, name_cnt = {}, {}
    for _, sub in te.groupby("report_date"):
        sub = sub.sort_values("ticker")
        w_true = sub["weight"].to_numpy(float)
        p_hat = softmax(_base_logit(sub) + sub["_yhat"].to_numpy(float))
        rows.append((aitchison(w_true, p_hat), tv(w_true, p_hat),
                     sub["regime"].iloc[0]))
        dev = clr_dev(w_true, p_hat)
        for tk, dv in zip(sub["ticker"].to_numpy(), dev):
            name_dev[tk] = name_dev.get(tk, 0.0) + dv
            name_cnt[tk] = name_cnt.get(tk, 0) + 1
    df = pd.DataFrame(rows, columns=["ait", "tv", "regime"])
    diff = pd.Series({t: name_dev[t] / name_cnt[t] for t in name_dev}, name="difficulty")
    return df, diff


def _product_joint(te: pd.DataFrame, iv) -> tuple[float, float]:
    """Independent-scalar PRODUCT region joint coverage per session (all names' marginal
    intervals contain the truth), base & shift."""
    d = te[["report_date", "regime"]].copy()
    d["cov"] = ((te["y"].to_numpy() >= iv.lo) & (te["y"].to_numpy() <= iv.hi))
    js = d.groupby(["report_date", "regime"])["cov"].all().reset_index()
    base = js[js.regime == "base"]["cov"].mean()
    shift = js[js.regime == "shift"]["cov"].mean()
    return float(base), float(shift)


def _fit(df):
    tr, ca, te = _split(df)
    Xtr, ytr = _xy(tr, "y"); Xca, _ = _xy(ca, "y"); Xte, _ = _xy(te, "y")
    mean = fit_mean(Xtr, ytr)
    ivp = cqr(Xtr, ytr, Xca, _xy(ca, "y")[1], Xte, ALPHA)   # per-name product intervals
    ca_pieces, _ = _session_pieces(ca, mean.predict(Xca))
    te_pieces, diff = _session_pieces(te, mean.predict(Xte))
    return tr, ca, te, ivp, ca_pieces, te_pieces, diff


def baseline_read():
    print("=" * 92)
    print("P8 baseline read (method-blind) | Layer 1a | product-region joint coverage failure")
    print("=" * 92)
    pj_b, pj_s, ait_q, tv_q = [], [], [], []
    for s in SEEDS:
        path = PROC / f"synth_1a_seed{s}.parquet"
        if not path.exists():
            generate(s)
        df = pd.read_parquet(path).dropna(subset=FEATURE_COLS + ["y"])
        tr, ca, te, ivp, ca_p, te_p, _ = _fit(df)
        b, sh = _product_joint(te, ivp)
        pj_b.append(b); pj_s.append(sh)
        ait_q.append(np.quantile(ca_p["ait"], 0.9)); tv_q.append(np.quantile(ca_p["tv"], 0.9))
    print("\nINDEPENDENT-scalar product region — JOINT coverage (all names in at once):")
    print(f"  base  {np.mean(pj_b):.4f} +/- {np.std(pj_b):.4f}   "
          f"shift {np.mean(pj_s):.4f} +/- {np.std(pj_s):.4f}   (target {TARGET:.2f})")
    print(f"  -> the multiplicity failure: {int(round(93))} marginal-90% intervals => joint ~0.")
    print(f"\nCompositional score 90th-pct on calib (radius scale): "
          f"Aitchison {np.mean(ait_q):.3f}, TV {np.mean(tv_q):.4f}")
    print("\n(Gate thresholds reuse frozen tolerances: tau=%.2f, tau_shift=%.2f, "
          "theta_AUC=%.2f, theta_rho=%.2f)" % (TAU, TAU_SHIFT, THETA_AUC, THETA_RHO))


def _partial_spearman(x, y, z):
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    Z = np.c_[np.ones_like(rz), rz]
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def run_1a():
    from sklearn.metrics import roc_auc_score
    print("=" * 92)
    print("P8 — JOINT conformal region on the SIMPLEX vs independent-scalar product region | 1a")
    print("=" * 92)
    import time
    t0 = time.time()
    res = {k: [] for k in ("pj_b", "pj_s", "sj_b", "sj_s", "aci_b", "aci_s",
                           "tvj_b", "tvj_s", "auc", "rho", "rho_c")}
    seed0 = None
    for s in SEEDS:
        path = PROC / f"synth_1a_seed{s}.parquet"
        if not path.exists():
            generate(s)
        df = pd.read_parquet(path).dropna(subset=FEATURE_COLS + ["y"])
        tr, ca, te, ivp, ca_p, te_p, diff = _fit(df)

        # product-region joint coverage
        b, sh = _product_joint(te, ivp)
        res["pj_b"].append(b); res["pj_s"].append(sh)

        # simplex JOINT region (static), Aitchison
        Q = region_radius(ca_p["ait"].to_numpy(), ALPHA)
        cov = te_p["ait"].to_numpy() <= Q
        res["sj_b"].append(cov[te_p.regime == "base"].mean())
        res["sj_s"].append(cov[te_p.regime == "shift"].mean())
        # TV region (companion)
        Qtv = region_radius(ca_p["tv"].to_numpy(), ALPHA)
        covtv = te_p["tv"].to_numpy() <= Qtv
        res["tvj_b"].append(covtv[te_p.regime == "base"].mean())
        res["tvj_s"].append(covtv[te_p.regime == "shift"].mean())

        # simplex ACI region (online, Aitchison) over time-ordered sessions
        # (te_p rows are already in groupby(report_date) ascending order)
        covered, _ = aci_radius_path(ca_p["ait"].to_numpy(), te_p["ait"].to_numpy(),
                                     ALPHA, GAMMA)
        res["aci_b"].append(covered[te_p.regime == "base"].mean())
        res["aci_s"].append(covered[te_p.regime == "shift"].mean())

        # ---- G-COMPO-MECH: compositional difficulty vs known floor ----
        gt = (df.groupby("ticker")
                .agg(recoverable_mask=("recoverable_mask", "first"),
                     irreducible_sigma=("irreducible_sigma", "first"),
                     vol_63=("vol_63", "mean")).reset_index().set_index("ticker"))
        gt = gt.join(diff, how="inner").dropna()
        y = gt["recoverable_mask"].astype(int).to_numpy()
        d = gt["difficulty"].to_numpy()
        res["auc"].append(roc_auc_score(y, -d) if len(np.unique(y)) > 1 else np.nan)
        res["rho"].append(stats.spearmanr(d, gt["irreducible_sigma"]).statistic)
        res["rho_c"].append(_partial_spearman(d, gt["irreducible_sigma"].to_numpy(),
                                              gt["vol_63"].to_numpy()))
        if s == SEEDS[0]:
            seed0 = dict(te_p=te_p, Q=Q, df=df, te=te)
        print(f"  [progress] seed {s} done @ {time.time()-t0:.0f}s", flush=True)

    def ms(k):
        v = np.array(res[k], float); return v.mean(), v.std()

    print(f"\nJOINT coverage — the true allocation VECTOR in the region (mean+/-sd over {SEEDS})")
    print(f"{'region':<28}{'base':>16}{'shift':>16}")
    print(f"{'independent product (N scalars)':<28}{ms('pj_b')[0]:>10.4f}+/-{ms('pj_b')[1]:<5.4f}"
          f"{ms('pj_s')[0]:>10.4f}+/-{ms('pj_s')[1]:<5.4f}")
    print(f"{'simplex JOINT (Aitchison)':<28}{ms('sj_b')[0]:>10.3f}+/-{ms('sj_b')[1]:<5.3f}"
          f"{ms('sj_s')[0]:>10.3f}+/-{ms('sj_s')[1]:<5.3f}")
    print(f"{'simplex JOINT (TV companion)':<28}{ms('tvj_b')[0]:>10.3f}+/-{ms('tvj_b')[1]:<5.3f}"
          f"{ms('tvj_s')[0]:>10.3f}+/-{ms('tvj_s')[1]:<5.3f}")
    print(f"{'simplex ACI (Aitchison,online)':<28}{ms('aci_b')[0]:>10.3f}+/-{ms('aci_b')[1]:<5.3f}"
          f"{ms('aci_s')[0]:>10.3f}+/-{ms('aci_s')[1]:<5.3f}")

    # ---- gates ----
    sj_b = ms("sj_b")[0]
    g_joint = (abs(sj_b - TARGET) <= TAU) and (ms("pj_b")[0] < TARGET - TAU)
    print(f"\n[G-COMPO-JOINT] simplex base {sj_b:.3f} within +/-{TAU} of {TARGET} "
          f"AND product region {ms('pj_b')[0]:.4f} << target? -> {'PASS' if g_joint else 'MISS'}")

    aci_s, sj_s = ms("aci_s")[0], ms("sj_s")[0]
    g_shift = (abs(aci_s - TARGET) <= TAU_SHIFT) and (aci_s > sj_s)
    print(f"[G-COMPO-SHIFT] simplex ACI shift {aci_s:.3f} within +/-{TAU_SHIFT} of {TARGET} "
          f"AND > static {sj_s:.3f}? -> {'PASS' if g_shift else 'MISS'}")

    auc, rho, rho_c = ms("auc")[0], ms("rho")[0], ms("rho_c")[0]
    g_mech = (auc >= THETA_AUC) and (rho >= THETA_RHO) and (rho_c >= THETA_RHO)
    print(f"[G-COMPO-MECH] AUC(-difficulty->recoverable) {auc:.3f} (theta {THETA_AUC}); "
          f"Spearman(difficulty,sigma_irr) {rho:.3f} (theta' {THETA_RHO}); "
          f"partial|vol {rho_c:.3f} -> {'PASS' if g_mech else 'CHECK'}")

    # seed-0 band on simplex base joint coverage: MONTH-BLOCKED session bootstrap (coverage
    # is per whole-vector session, so the cluster unit is the calendar month, not sector×month).
    tp = seed0["te_p"][seed0["te_p"].regime == "base"].reset_index(drop=True)
    cov_vec = (tp["ait"].to_numpy() <= seed0["Q"]).astype(float)
    blocks = (seed0["te"][seed0["te"].regime == "base"]
              .drop_duplicates("report_date")["report_date"].to_numpy() // 100)
    rng = np.random.default_rng(0)
    uniq = np.unique(blocks)
    boots = []
    for _ in range(300):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([np.where(blocks == b)[0] for b in pick])
        boots.append(cov_vec[idx].mean())
    print(f"\n[seed0 month-blocked band] simplex base joint coverage {cov_vec.mean():.3f} "
          f"[{np.quantile(boots,0.05):.3f},{np.quantile(boots,0.95):.3f}]")

    ships = g_joint and g_shift and g_mech
    verdict = ("FULL SHIP" if ships else
               "QUALIFIED" if (g_joint and g_mech) else "REJECT")
    print(f"\n==> P8 verdict (1a): {verdict}  (G-COMPO-JOINT {'pass' if g_joint else 'miss'}, "
          f"G-COMPO-SHIFT {'pass' if g_shift else 'miss'}, "
          f"G-COMPO-MECH {'pass' if g_mech else 'miss'})")
    print("    (1b is forward returns, not compositional -> P8 has no real-data read here)")
    return verdict


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    args = ap.parse_args()
    if args.baseline:
        baseline_read()
    else:
        run_1a()
