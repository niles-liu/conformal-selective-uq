"""Layer 1b: the MULTI-REGIME semi-synthetic target (the regime x conformal experiment, P12).

Layer 1a (`synthetic.py`) injects a SINGLE documented shift window. P12 asks: **when a stable,
RECURRING latent regime genuinely exists, does adding a LEARNED regime axis to the
group-conditional remedy recover per-regime / worst-regime coverage that a regime-blind
(static-axes-only) method misses?**

Generative model — identical recoverable/overlay/base structure as Layer 1a, EXCEPT the single
shift window is replaced by a K-state latent regime **Markov chain over sessions** with persistent
dwell. Each state scales the irreducible (overlay) term by `REGIME_AMP[state]`:

    raw_tilt_{i,t} = beta_pub*pub_signal_{i,t}                       # recoverable
                   + REGIME_AMP[state_t] * lambda_i * overlay_{i,t}  # regime-amplified irreducible
                   + sigma_eps * eps_{i,t}
    w, y as in Layer 1a.

The regime is LEARNABLE from a PIT observable: the cross-sectional dispersion of yesterday's
realised target (`disp_lag`, known at decision time — analogous to a realised-vol / market-state
index). In stress states the amplified overlay widens the target cross-section, so `disp_lag`
jumps: a clean, causal regime signal. True `regime_state` is persisted so per-regime coverage and regime-recovery
accuracy are checkable ground truth.

FROZEN generator knobs (pre-registration apparatus, P12; do not tune post-hoc). Determinism:
everything flows from `seed`; each seed is a fresh regime realisation (the multi-seed bands).
See reports/p12_regime_axis_preregistration.md.

Run: python experiments/synthetic_regime.py --seed 0   (writes data/processed/synth_1b_seed0.parquet)
As in Layer 1a, the cache is validated and then written through a temp file and an atomic rename.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _common import PROC, write_parquet_atomic
from panel import PANEL
from synthetic import BETA_PUB, LAMBDA_HI, LAMBDA_LO, OVERLAY_RHO, PUB_FEATURES, REC_CUT, SIGMA_EPS, _zscore_by_date

# ---- FROZEN P12 regime knobs (pre-registration; do not tune post-hoc) ------ #
K_STATES = 3
REGIME_AMP = np.array([0.6, 1.0, 2.4])          # calm / normal / stress overlay amplifier
REGIME_LABELS = np.array(["calm", "normal", "stress"])
P_STAY = 0.96                                    # self-transition -> ~25-session mean dwell
DISP_MA = 5                                      # rolling mean window for the detector feature


def _markov_regime(n_sessions: int, rng: np.random.Generator) -> np.ndarray:
    """Persistent K-state chain over sessions: stay w.p. P_STAY, else jump uniformly."""
    off = (1.0 - P_STAY) / (K_STATES - 1)
    Tm = np.full((K_STATES, K_STATES), off)
    np.fill_diagonal(Tm, P_STAY)
    s = np.empty(n_sessions, dtype=int)
    s[0] = rng.integers(K_STATES)
    for t in range(1, n_sessions):
        s[t] = rng.choice(K_STATES, p=Tm[s[t - 1]])
    return s


def generate(seed: int = 0) -> pd.DataFrame:
    panel = pd.read_parquet(PANEL).sort_values(["date", "ticker"]).reset_index(drop=True)
    rng = np.random.default_rng(seed)

    # --- recoverable public signal (identical to Layer 1a) ------------------
    pub = pd.Series(0.0, index=panel.index)
    for f, load in PUB_FEATURES.items():
        if f not in panel.columns:
            raise SystemExit(f"panel missing feature {f} — rebuild with experiments/panel.py")
        pub = pub + load * _zscore_by_date(panel, f).fillna(0.0)
    panel["pub_signal"] = pub

    names = sorted(panel["ticker"].unique())
    name_ix = {t: k for k, t in enumerate(names)}
    panel["_ni"] = panel["ticker"].map(name_ix).to_numpy()
    lam = rng.uniform(LAMBDA_LO, LAMBDA_HI, size=len(names))
    panel["lambda_i"] = lam[panel["_ni"].to_numpy()]

    # --- hidden per-name AR(1) overlay (identical to Layer 1a) --------------
    overlay = np.zeros(len(panel), dtype=float)
    for t in names:
        idx = panel.index[panel["ticker"] == t]
        n = len(idx)
        innov = rng.standard_normal(n)
        z = np.empty(n)
        z[0] = innov[0]
        for k in range(1, n):
            z[k] = OVERLAY_RHO * z[k - 1] + np.sqrt(1 - OVERLAY_RHO**2) * innov[k]
        overlay[panel.index.get_indexer(idx)] = z
    panel["overlay"] = overlay
    panel["overlay"] = _zscore_by_date(panel, "overlay").fillna(0.0)

    # --- MULTI-REGIME latent chain over sessions ----------------------------
    sessions = np.sort(panel["report_date"].unique())
    state = _markov_regime(len(sessions), rng)
    s_by_date = dict(zip(sessions, state))
    panel["regime_state"] = panel["report_date"].map(s_by_date).to_numpy()
    panel["regime"] = REGIME_LABELS[panel["regime_state"].to_numpy()]
    amp = REGIME_AMP[panel["regime_state"].to_numpy()]

    # --- assemble target (regime-amplified irreducible term) ----------------
    eps = rng.standard_normal(len(panel))
    recoverable_term = BETA_PUB * panel["pub_signal"].to_numpy()
    irreducible_term = (amp * panel["lambda_i"].to_numpy() * panel["overlay"].to_numpy()
                        + SIGMA_EPS * eps)
    panel["raw_tilt"] = recoverable_term + irreducible_term

    size = _zscore_by_date(panel, "log_dollar_vol").fillna(0.0).to_numpy()
    panel["_base_logit"] = 0.5 * size

    def _per_date(g: pd.DataFrame) -> pd.DataFrame:
        logw = g["_base_logit"] + g["raw_tilt"]
        logZ = np.log(np.exp(logw).sum())
        g = g.copy()
        g["weight"] = np.exp(logw - logZ)
        g["y"] = logw - logZ - (g["_base_logit"] - np.log(np.exp(g["_base_logit"]).sum()))
        return g

    # select columns explicitly so the "date" key is passed through on pandas>=3
    # (groupby.apply no longer hands grouping columns to the function there)
    panel = panel.groupby("date", group_keys=False)[panel.columns.tolist()].apply(_per_date)

    # --- PIT regime detector: dispersion of YESTERDAY's target (known at t) --
    disp = (panel.groupby("report_date")["y"].std()
            .reindex(sessions).rename("disp"))
    disp_lag = disp.shift(1)                                    # known_at <= decision_time
    disp_ma = disp_lag.rolling(DISP_MA, min_periods=1).mean()
    det = pd.DataFrame({"report_date": sessions,
                        "disp_lag": disp_lag.to_numpy(),
                        "disp_ma": disp_ma.to_numpy()})
    det["disp_lag"] = det["disp_lag"].bfill()
    det["disp_ma"] = det["disp_ma"].bfill()
    panel = panel.merge(det, on="report_date", how="left")

    # --- per-name known ceiling (kept for parity with Layer 1a eval) --------
    panel["_irr"] = panel["raw_tilt"] - BETA_PUB * panel["pub_signal"]
    gt = panel.groupby("ticker").agg(
        var_recoverable=("pub_signal", lambda s: (BETA_PUB * s).var()),
        var_total=("raw_tilt", "var")).reset_index()
    irr = panel.groupby("ticker")["_irr"].std().rename("irreducible_sigma")
    gt = gt.merge(irr, on="ticker")
    gt["recoverable_fraction"] = (gt["var_recoverable"] /
                                  gt["var_total"].replace(0, np.nan)).clip(0, 1)
    gt["recoverable_mask"] = gt["recoverable_fraction"] >= REC_CUT
    panel = panel.merge(gt[["ticker", "recoverable_fraction", "recoverable_mask",
                            "irreducible_sigma"]], on="ticker", how="left")

    panel = panel.drop(columns=["_ni", "_base_logit", "_irr"])
    need = {"date", "report_date", "ticker", "regime", "regime_state", "y", "weight",
            "disp_lag", "disp_ma", "recoverable_fraction", "recoverable_mask", "irreducible_sigma"}
    lost = sorted(need - set(panel.columns))
    if lost:
        raise RuntimeError(f"generator output is malformed (missing {lost}); nothing was cached")
    out = PROC / f"synth_1b_seed{seed}.parquet"
    write_parquet_atomic(panel, out)

    # face-validity reads
    yv = panel["y"].to_numpy()
    rec = recoverable_term - recoverable_term.mean()
    ss_tot = ((yv - yv.mean()) ** 2).sum()
    b = np.dot(rec, yv - yv.mean()) / np.dot(rec, rec)
    r2 = 1 - (((yv - yv.mean()) - b * rec) ** 2).sum() / ss_tot
    dwell = np.mean([len(list(g)) for _, g in __import__("itertools").groupby(state)])
    print("=== Layer 1b MULTI-REGIME synthetic allocator (P12) ===")
    print(f"  seed={seed} | {len(panel):,} rows | {len(names)} names | {len(sessions)} sessions")
    print("  regime sessions: " + ", ".join(
        f"{REGIME_LABELS[k]} {int((state==k).sum())}" for k in range(K_STATES))
        + f" | mean dwell {dwell:.1f} | switches {int((np.diff(state)!=0).sum())}")
    print(f"  overlay amp per state: {dict(zip(REGIME_LABELS.tolist(), REGIME_AMP.tolist()))}")
    print("  disp_lag by true regime (detector separation): " + ", ".join(
        f"{REGIME_LABELS[k]} {panel.loc[panel.regime_state==k,'disp_lag'].mean():.3f}"
        for k in range(K_STATES)))
    print(f"  ceiling R^2 (y ~ recoverable_term): {r2:.3f}  (target ~0.40)")
    print(f"  wrote {out.name}")
    return panel


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    generate(args.seed)
