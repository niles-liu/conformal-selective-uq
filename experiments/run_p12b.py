"""P12b driver: CONFIRMATORY calibrated-node regime seam + POGO parameter-free arm.

P12 (union remedy) rejected the seam and a POST-HOC probe on the tight calibrated node
(`cqr_conditional`, Gibbs-Cherian-Candes) found a +0.068 worst-regime lift. That probe was
exploratory (added after the union REJECT). P12b turns it into a pre-registered confirm decided on
FRESH held-out seeds {3,4,5} (independent regime realizations that did not inform the +0.068
hypothesis, which came from {0,1,2}). Exploratory {0,1,2} are reported for continuity only.

Gates (held-out {3,4,5}):
  G-SEAM-CALIB     cal_proposed wr - cal_static wr >= +0.03      (primary)
  G-NULL-CALIB     cal_shuffled wr - cal_static wr <  +0.03      (primary; lift is regime-specific)
  G-ORACLE-CAPTURE (cal_proposed lift)/(cal_oracle lift) >= 0.60 (mechanism)
  G-EFF-CALIB      cal_proposed width / cal_static width <= 1.20 (reallocation, not blanket widening)
  G-POGO-SEAM      pogo_proposed wr - pogo_static wr >= +0.03    (secondary)
  G-POGO-MATCH     |pogo_proposed - gqaci_max_proposed| <= 0.03  (secondary; parameter-free ~ tuned)

Pre-reg + apparatus: reports/p12b_calibrated_seam_preregistration.md. Synthetic data only.
Run: python experiments/synthetic_regime.py --seed {0,...,5}
     python experiments/run_p12b.py --seed N      (each N in 0..5; seeds can run in parallel)
     python experiments/run_p12b.py               (aggregate the six per-seed results + gates)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import PROC
from panel import FEATURE_COLS
from run_p2 import ALPHA, GAMMA, TARGET, _split, _xy
from run_p12 import _axis_labels, _bal_acc, _ev, _learn_regime, _summary
from selective_conformal import conformal as C
from selective_conformal import pogo as P
from selective_conformal.metrics import clustered_bootstrap, coverage

CONFIRM_SEEDS = [3, 4, 5]          # fresh held-out — decide the gates here
EXPLORE_SEEDS = [0, 1, 2]          # P12 seeds — report-only continuity
G_SEAM, G_NULL, G_CAPTURE, G_EFF = 0.03, 0.03, 0.60, 1.20
G_POGO_SEAM, G_POGO_MATCH = 0.03, 0.03


def _pf(b):
    return "PASS" if b else "MISS"


def _calibrated(lo_q, hi_q, E, cal_axes, te_axes):
    """Tight calibrated node (Gibbs-Cherian-Candes): 1 pinball solve of E on the group design at
    1-alpha, shared CQR pieces. Identical to conformal.cqr_conditional with pieces passed in."""
    from sklearn.linear_model import QuantileRegressor
    Phi_c, Phi_t = C._multiaxis_design(cal_axes, te_axes)
    qr = QuantileRegressor(quantile=1 - ALPHA, alpha=0.0, solver="highs", fit_intercept=False)
    qr.fit(Phi_c, E)
    t = qr.predict(Phi_t)
    lo, hi = lo_q - t, hi_q + t
    bad = lo > hi
    if bad.any():
        mid = 0.5 * (lo[bad] + hi[bad]); lo[bad], hi[bad] = mid, mid
    return lo, hi


def run_seed(s, do_pogo=True):
    df = pd.read_parquet(PROC / f"synth_1b_seed{s}.parquet").dropna(subset=FEATURE_COLS + ["y"])
    reg_map, diag = _learn_regime(df, seed=s)
    df = df.copy()
    df["reg_learn"] = df.report_date.map(reg_map).astype(int)

    tr, ca, te = _split(df)
    Xtr, ytr = _xy(tr, "y"); Xca, yca = _xy(ca, "y"); Xte, yte = _xy(te, "y")
    order = te.report_date.to_numpy()
    gac, gat = _axis_labels(ca), _axis_labels(te)

    def A(col):
        return ca[col].astype(str).to_numpy(), te[col].astype(str).to_numpy()
    rlc, rlt = A("reg_learn"); roc, rot = A("regime_state")

    # shuffled learned regime across TEST sessions (deterministic; same construction as P12 G-NULL)
    rng = np.random.default_rng(1200 + s)
    td = np.sort(te.report_date.unique())
    shuf = dict(zip(td, rng.permutation([reg_map[d] for d in td])))
    rst = te.report_date.map(shuf).astype(str).to_numpy()
    rsc = rlc                                            # calib axis unchanged; only test labels shuffled

    # ---- shared CQR pieces (fit ONCE) --------------------------------------
    pred, lo_q, hi_q, E = C._cqr_pieces(Xtr, ytr, Xca, yca, Xte, ALPHA)

    out = {}
    # ---- PRIMARY: calibrated node (1 LP / axis-set) ------------------------
    for name, cax, tax in [("cal_static", gac, gat),
                           ("cal_proposed", gac + [rlc], gat + [rlt]),
                           ("cal_oracle", gac + [roc], gat + [rot]),
                           ("cal_shuffled", gac + [rsc], gat + [rst])]:
        lo, hi = _calibrated(lo_q, hi_q, E, cax, tax)
        out[name] = _ev(te, pred, lo, hi)
    print(f"[seed {s}] calibrated done (lam={diag['lam']}, "
          f"bal-acc={_bal_acc(te.regime_state.to_numpy(), te.reg_learn.to_numpy()):.3f})", flush=True)

    # ---- SECONDARY: POGO parameter-free + tuned gqaci_max (level family) ----
    if do_pogo:
        def precomp_for(cax, tax):
            Phi_c, Phi_t = C._multiaxis_design(cax, tax)
            T = C._fit_level_family(Phi_c, E, Phi_t)      # 9 HiGHS solves, reuses shared E
            return (pred, lo_q, hi_q, T)
        pc_static = precomp_for(gac, gat)
        pc_prop = precomp_for(gac + [rlc], gat + [rlt])
        iv = P.cqr_conditional_gqaci_max_pogo(Xtr, ytr, Xca, yca, gac, Xte, yte, order, gat,
                                              alpha=ALPHA, precomp=pc_static)
        out["pogo_static"] = _ev(te, pred, iv.lo, iv.hi)
        iv = P.cqr_conditional_gqaci_max_pogo(Xtr, ytr, Xca, yca, gac + [rlc], Xte, yte, order,
                                              gat + [rlt], alpha=ALPHA, precomp=pc_prop)
        out["pogo_proposed"] = _ev(te, pred, iv.lo, iv.hi)
        iv = C.cqr_conditional_gqaci_max(Xtr, ytr, Xca, yca, gac + [rlc], Xte, yte, order,
                                         gat + [rlt], alpha=ALPHA, gamma=GAMMA, precomp=pc_prop)
        out["gqaci_max_proposed"] = _ev(te, pred, iv.lo, iv.hi)
        print(f"[seed {s}] POGO + tuned gqaci_max done", flush=True)

    recover = _bal_acc(te.regime_state.to_numpy(), te.reg_learn.to_numpy())
    return out, recover


def _summ_json(ev):
    s = _summary(ev)
    return dict(marg=float(s["marg"]), wr=float(s["wr"]), wr_lab=int(s["wr_lab"]),
                wg=float(s["wg"]), width=float(s["width"]),
                per={str(k): float(v) for k, v in s["per"].items()})


def run_one(seed, do_pogo=True):
    """Run one seed, return a JSON-able dict {variant: summary} + recover + cal_proposed band."""
    import json
    out, recover = run_seed(seed, do_pogo=do_pogo)
    summ = {v: _summ_json(ev) for v, ev in out.items()}
    ev0 = out["cal_proposed"]; wl = _summary(ev0)["wr_lab"]
    sub = ev0[ev0.regime_state == wl].reset_index(drop=True)
    p, lo, hi = clustered_bootstrap(sub, lambda d: coverage(d.y.values, d.lo.values, d.hi.values),
                                    n_boot=300)
    rec = dict(seed=int(seed), recover=float(recover), band=[float(p), float(lo), float(hi)],
               band_state=int(wl), summ=summ)
    path = PROC / f"p12b_seed{seed}.json"
    path.write_text(json.dumps(rec))
    print(f"[seed {seed}] wrote {path.name} (band state {wl}: {p:.3f} [{lo:.3f},{hi:.3f}])", flush=True)
    return rec


def _agg(recs, variants):
    """mean over seeds of each variant's stored summary. recs = list of run_one dicts."""
    a = {}
    for v in variants:
        Ss = [r["summ"][v] for r in recs if v in r["summ"]]
        if not Ss:
            continue
        a[v] = dict(marg=np.mean([x["marg"] for x in Ss]),
                    wr=np.mean([x["wr"] for x in Ss]), wr_sd=np.std([x["wr"] for x in Ss]),
                    wg=np.mean([x["wg"] for x in Ss]), width=np.mean([x["width"] for x in Ss]),
                    per={k: np.mean([x["per"].get(str(k), np.nan) for x in Ss]) for k in (0, 1, 2)})
    return a


def _table(a, variants, lbl):
    print(f"\n{'variant':<30}{'marg':>7}{'worst-regime':>14}{'worst-grp':>11}{'width':>8}  "
          f"per-regime[calm/norm/stress]")
    for v in variants:
        x = a[v]; p = x["per"]
        print(f"{lbl[v]:<30}{x['marg']:>7.3f}{x['wr']:>9.3f}+/-{x['wr_sd']:<4.3f}"
              f"{x['wg']:>11.3f}{x['width']:>8.3f}  [{p[0]:.3f}/{p[1]:.3f}/{p[2]:.3f}]")


def main():
    variants = ["cal_static", "cal_proposed", "cal_oracle", "cal_shuffled",
                "pogo_static", "pogo_proposed", "gqaci_max_proposed"]
    lbl = {"cal_static": "cal_static (tight, regime-blind)", "cal_proposed": "cal +learned regime",
           "cal_oracle": "cal +true regime (oracle)", "cal_shuffled": "cal +SHUFFLED (G-NULL)",
           "pogo_static": "POGO static (param-free)", "pogo_proposed": "POGO +learned (param-free)",
           "gqaci_max_proposed": "gqaci_max +learned (tuned g)"}

    import json
    print("=" * 92)
    print("P12b — CONFIRMATORY calibrated-node regime seam (held-out {3,4,5}) + POGO parameter-free arm")
    print("=" * 92)

    def load(seeds):
        recs = []
        for s in seeds:
            p = PROC / f"p12b_seed{s}.json"
            if not p.exists():
                raise SystemExit(f"missing {p.name}: run `python experiments/run_p12b.py --seed {s}` first")
            recs.append(json.loads(p.read_text()))
        return recs

    conf_recs = load(CONFIRM_SEEDS); expl_recs = load(EXPLORE_SEEDS)
    conf = _agg(conf_recs, variants); expl = _agg(expl_recs, variants)
    conf_rec = [r["recover"] for r in conf_recs]

    print("\n" + "#" * 92)
    print(f"# CONFIRMATION SET — held-out seeds {CONFIRM_SEEDS} (these decide the gates) "
          f"| target {TARGET:.0%} | eval TRUE regime, trigger LEARNED")
    print("#" * 92)
    _table(conf, [v for v in variants if v in conf], lbl)
    print(f"regime recovery (balanced acc vs true): {np.mean(conf_rec):.3f} "
          f"per-seed {np.round(conf_rec,3)}")
    for r in conf_recs:
        b = r["band"]
        print(f"[seed{r['seed']} clustered band] cal_proposed worst-regime (state {r['band_state']}) "
              f"{b[0]:.3f} [{b[1]:.3f},{b[2]:.3f}]")

    print("\n" + "-" * 92)
    print(f"# EXPLORATORY SET — P12 seeds {EXPLORE_SEEDS} (report-only, do NOT decide gates)")
    _table(expl, [v for v in variants if v in expl], lbl)

    # ---- gates decided on the CONFIRMATION set -----------------------------
    cs, cp, co, csh = conf["cal_static"], conf["cal_proposed"], conf["cal_oracle"], conf["cal_shuffled"]
    seam = cp["wr"] - cs["wr"]; nul = csh["wr"] - cs["wr"]; orc = co["wr"] - cs["wr"]
    capture = (seam / orc) if orc > 1e-9 else float("nan")
    effr = cp["width"] / cs["width"]
    g_seam = seam >= G_SEAM
    g_null = nul < G_NULL
    g_capture = (not np.isnan(capture)) and capture >= G_CAPTURE
    g_eff = effr <= G_EFF

    ps, pp, gm = conf["pogo_static"], conf["pogo_proposed"], conf["gqaci_max_proposed"]
    pogo_seam = pp["wr"] - ps["wr"]; pogo_match = abs(pp["wr"] - gm["wr"])
    g_pogo_seam = pogo_seam >= G_POGO_SEAM
    g_pogo_match = pogo_match <= G_POGO_MATCH

    print("\n" + "=" * 92)
    print("PRE-COMMITTED GATES (decided on held-out {3,4,5})")
    print("=" * 92)
    print(f"[G-SEAM-CALIB]     cal_proposed wr {cp['wr']:.3f} - cal_static wr {cs['wr']:.3f} = "
          f"{seam:+.3f} >= +{G_SEAM}  -> {_pf(g_seam)}")
    print(f"[G-NULL-CALIB]     cal_shuffled wr {csh['wr']:.3f} - cal_static = {nul:+.3f} < +{G_NULL} "
          f"(regime-specific?) -> {_pf(g_null)}")
    print(f"[G-ORACLE-CAPTURE] seam {seam:+.3f} / oracle {orc:+.3f} = {capture:.2f} >= {G_CAPTURE} "
          f"-> {_pf(g_capture)}")
    print(f"[G-EFF-CALIB]      width ratio {effr:.3f} <= {G_EFF}x  -> {_pf(g_eff)}")
    print(f"[G-POGO-SEAM]      pogo_proposed wr {pp['wr']:.3f} - pogo_static {ps['wr']:.3f} = "
          f"{pogo_seam:+.3f} >= +{G_POGO_SEAM}  -> {_pf(g_pogo_seam)}")
    print(f"[G-POGO-MATCH]     |pogo {pp['wr']:.3f} - tuned gqaci_max {gm['wr']:.3f}| = "
          f"{pogo_match:.3f} <= {G_POGO_MATCH}  -> {_pf(g_pogo_match)}")

    primary = g_seam and g_null and g_capture and g_eff
    if primary:
        v = ("CONFIRM: on the TIGHT calibrated (Gibbs-Cherian-Candes) node, a learned dynamic regime "
             "axis recovers worst-regime coverage the regime-blind node misses; the lift is "
             "regime-specific (shuffled placebo does not reproduce it), captures the true-regime oracle "
             "headroom, and is a reallocation (not blanket widening) — confirmed out-of-sample on fresh "
             "seeds. The seam lives on the tight node, not the self-absorbing union (the P12 boundary).")
    elif not g_seam:
        v = ("REJECT-as-finding: the exploratory +0.068 did NOT replicate on held-out seeds; the "
             "calibrated seam is not robust.")
    else:
        v = ("QUALIFIED: seam lifts worst-regime but a mechanism/efficiency gate failed "
             "(G-NULL-CALIB / G-ORACLE-CAPTURE / G-EFF-CALIB) — see gates above.")
    print(f"\nVERDICT (primary) -> {v}")
    print(f"POGO (secondary): seam {_pf(g_pogo_seam)}, parameter-free-matches-tuned {_pf(g_pogo_match)} "
          f"-> {'corroborates' if (g_pogo_seam and g_pogo_match) else 'see gates'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None, help="run ONE seed -> p12b_seed{N}.json")
    ap.add_argument("--nopogo", action="store_true", help="calibrated only (for report-only seeds)")
    a = ap.parse_args()
    if a.seed is not None:
        run_one(a.seed, do_pogo=not a.nopogo)
    else:
        main()
