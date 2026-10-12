"""Layer 1a: the known-ceiling semi-synthetic target (the controlled experiment).

The demonstration task is a semi-synthetic portfolio-allocation problem on public US equity data.
Each stock's weight mixes a part the public features can recover with a hidden component the model never sees, so the unrecoverable share is known for every stock.

That known per-stock split is what makes G-MECHANISM ("does the method abstain in
the *right* place?") checkable against ground truth.

Generative model (per stock i, session t), as in SPEC §4:

    raw_tilt_{i,t} = beta_pub * pub_signal_{i,t}        # recoverable from the panel
                   + lambda_i  * overlay_{i,t}          # hidden component (never given)
                   + sigma_eps * eps_{i,t}              # irreducible iid noise
    w_{i,t}        = base_i * exp(raw_tilt_{i,t}),  renormalised to sum 1 per session
    y_{i,t}        = log w_{i,t} - log base_i  =  raw_tilt_{i,t} - logZ_t   # the target

`pub_signal` is a fixed linear combination of per-date-standardised public features
(so a learner *can* recover it). `overlay` is a seeded per-stock AR(1) latent that is
independent of the panel and never given to the learner.

The size of the hidden share is a generator parameter, not an estimate of anything:
each stock's hidden-component weight lambda_i is drawn from U[LAMBDA_LO, LAMBDA_HI],
on top of iid noise of scale SIGMA_EPS. Because lambda_i varies across stocks, some
are mostly recoverable and some are mostly hidden (and should be abstained on). That
per-stock heterogeneity is the ground truth G-MECHANISM scores against.

Known-ceiling ground truth (per stock, persisted):
  * recoverable_fraction_i = Var_t(beta_pub*pub_signal) / Var_t(raw_tilt)   in [0,1]
  * recoverable_mask_i     = recoverable_fraction_i >= REC_CUT
  * irreducible_sigma_i    = std_t(lambda_i*overlay + sigma_eps*eps)        (G-MECH b)

Documented shift (G-SHIFT): in the SHIFT_REGIME window the hidden component is
amplified (SHIFT_AMP on the overlay weight), so *marginal* coverage can still hold
while *conditional / per-regime* coverage breaks: the failure naive split-conformal
must exhibit and the method must fix.

The knobs below are the frozen pre-registration knobs (SPEC §9), set so the
recoverable R^2 lands near 0.40 (a deliberately hard but non-trivial recoverable
fraction). Determinism: given the panel file, everything flows from `seed`, so the
same seed gives an identical target.

Run: python experiments/synthetic.py --seed 0     (writes data/processed/synth_1a_seed0.parquet)
The cache is written only after the target is fully built, via a temp file and an
atomic rename, so a failed run never leaves a broken cache behind.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _common import PROC, write_parquet_atomic
from panel import PANEL  # ensures panel module import path; PANEL path reused below

# ---- FROZEN generator knobs (pre-registration; do not tune post-hoc) ------- #
PUB_FEATURES = {  # public feature -> signed loading in the recoverable signal
    "resid_mom_63": 1.0,
    "mom_252": 0.6,
    "vol_63": -0.5,       # low-vol tilt
    "dist_252hi": 0.4,    # mild reversal/quality
}
BETA_PUB = 1.0            # recoverable-signal gain
LAMBDA_LO, LAMBDA_HI = 0.3, 2.5   # per-stock hidden-component weight range (heterogeneity)
SIGMA_EPS = 0.4           # irreducible iid noise scale
OVERLAY_RHO = 0.95        # per-name AR(1) persistence of the hidden overlay
REC_CUT = 0.50            # recoverable_fraction >= REC_CUT  =>  recoverable name
SHIFT_REGIME = (20220101, 20221231)  # the documented drawdown window (report_date int)
SHIFT_AMP = 1.8           # overlay-weight amplifier inside the shift regime


def _zscore_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.groupby("date")[col]
    return (df[col] - g.transform("mean")) / g.transform("std").replace(0, np.nan)


def generate(seed: int = 0, panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the Layer 1a target for `seed` from the cached panel and cache it.

    `panel` (for tests): build from this frame instead of the panel file; nothing is
    written. The draws are the same either way, so a frame equal to the panel file
    gives the identical target."""
    from_file = panel is None
    panel = pd.read_parquet(PANEL) if from_file else panel.copy()
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    n_rows = len(panel)
    rng = np.random.default_rng(seed)

    # --- recoverable public signal (per-date standardised feature combo) ----
    pub = pd.Series(0.0, index=panel.index)
    for f, load in PUB_FEATURES.items():
        if f not in panel.columns:
            raise SystemExit(f"panel missing feature {f} — rebuild with experiments/panel.py")
        pub = pub + load * _zscore_by_date(panel, f).fillna(0.0)
    panel["pub_signal"] = pub

    names = sorted(panel["ticker"].unique())
    name_ix = {t: k for k, t in enumerate(names)}
    panel["_ni"] = panel["ticker"].map(name_ix).to_numpy()

    # --- per-stock hidden-component weight lambda_i (the heterogeneity) -----
    lam = rng.uniform(LAMBDA_LO, LAMBDA_HI, size=len(names))
    panel["lambda_i"] = lam[panel["_ni"].to_numpy()]

    # --- hidden overlay: per-stock AR(1) latent, independent of the panel ----
    # built per name over its own (date-sorted) sessions, then z-scored per date.
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

    # --- regime-dependent overlay amplification (the documented shift) -------
    in_shift = panel["report_date"].between(*SHIFT_REGIME)
    panel["regime"] = np.where(in_shift, "shift", "base")
    amp = np.where(in_shift, SHIFT_AMP, 1.0)

    # --- assemble the target -------------------------------------------------
    eps = rng.standard_normal(len(panel))
    recoverable_term = BETA_PUB * panel["pub_signal"].to_numpy()
    irreducible_term = (amp * panel["lambda_i"].to_numpy() * panel["overlay"].to_numpy()
                        + SIGMA_EPS * eps)
    raw_tilt = recoverable_term + irreducible_term

    # base weight: softmax of the per-date standardised log dollar volume, kept
    # mild so the tilt (the thing we predict) carries the cross-sectional signal.
    size = _zscore_by_date(panel, "log_dollar_vol").fillna(0.0).to_numpy()
    base_logit = 0.5 * size
    panel["raw_tilt"] = raw_tilt
    panel["_base_logit"] = base_logit

    # renormalise per date: w = base*exp(tilt) / sum; target y = log w - log base
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

    # --- per-name known-ceiling ground truth --------------------------------
    gt = (
        panel.groupby("ticker")
        .agg(
            var_recoverable=("pub_signal", lambda s: (BETA_PUB * s).var()),
            var_total=("raw_tilt", "var"),
            irreducible_sigma=("raw_tilt", lambda s: np.nan),  # filled below
            lambda_i=("lambda_i", "first"),
        )
        .reset_index()
    )
    # irreducible_sigma per name = std of (raw_tilt - recoverable_term)
    panel["_irr"] = panel["raw_tilt"] - BETA_PUB * panel["pub_signal"]
    irr = panel.groupby("ticker")["_irr"].std().rename("irreducible_sigma")
    gt = gt.drop(columns="irreducible_sigma").merge(irr, on="ticker")
    gt["recoverable_fraction"] = (gt["var_recoverable"] /
                                  gt["var_total"].replace(0, np.nan)).clip(0, 1)
    gt["recoverable_mask"] = gt["recoverable_fraction"] >= REC_CUT

    panel = panel.merge(
        gt[["ticker", "recoverable_fraction", "recoverable_mask", "irreducible_sigma"]],
        on="ticker", how="left",
    )

    # tidy: drop scratch cols
    panel = panel.drop(columns=["_ni", "_base_logit", "_irr"])

    # sanity check before anything is cached: drivers reuse this file by name
    need = {"date", "report_date", "ticker", "regime", "y", "weight",
            "recoverable_fraction", "recoverable_mask", "irreducible_sigma"}
    lost = sorted(need - set(panel.columns))
    if lost or len(panel) != n_rows:
        raise RuntimeError(f"generator output is malformed (missing {lost}, "
                           f"{len(panel)} rows vs {n_rows} in); nothing was cached")
    out = PROC / f"synth_1a_seed{seed}.parquet"

    # face-validity read: cross-sectional R^2 of y on the recoverable term alone
    yv = panel["y"].to_numpy()
    rec = recoverable_term - recoverable_term.mean()
    ss_tot = ((yv - yv.mean()) ** 2).sum()
    # regress y on recoverable_term (1 predictor) for the ceiling read
    b = np.dot(rec, yv - yv.mean()) / np.dot(rec, rec)
    resid = (yv - yv.mean()) - b * rec
    r2_ceiling = 1 - (resid**2).sum() / ss_tot

    print("=== Layer 1a synthetic allocator ===")
    print(f"  seed={seed} | {len(panel):,} rows | {len(names)} names "
          f"| {panel.date.nunique()} sessions")
    print(f"  recoverable names (frac>={REC_CUT}): "
          f"{int(gt['recoverable_mask'].sum())}/{len(gt)}  "
          f"(frac: med {gt['recoverable_fraction'].median():.2f}, "
          f"IQR [{gt['recoverable_fraction'].quantile(.25):.2f},"
          f"{gt['recoverable_fraction'].quantile(.75):.2f}])")
    print(f"  ceiling R^2 (y ~ recoverable_term): {r2_ceiling:.3f}  "
          f"(design target ~0.40)")
    print(f"  shift regime {SHIFT_REGIME}: overlay amp x{SHIFT_AMP}")
    if from_file:
        write_parquet_atomic(panel, out)  # only now, after every step above succeeded
        print(f"  wrote {out.name}")
    return panel


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    generate(args.seed)
