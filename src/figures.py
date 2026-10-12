"""P4: the three headline figures (Layer 1a semi-synthetic target, seed 0).

All three use one seed, so their numbers are seed-0 numbers; the 3-seed means are
in the results tables (reports/p2_method_results.md). Each title says so.
  fig1_risk_coverage.png      G-PRIMARY(i): retained MAE vs retention (random / method / oracle)
  fig2_coverage_shift.png     G-SHIFT: base vs drawdown coverage, naive vs group-conditional ACI
  fig3_mechanism.png          G-MECHANISM: per-name uncertainty u_i vs the KNOWN irreducible floor

Reuses the frozen P2 pipeline (run_p2). Run: python src/figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_p2 as R
from _common import PROC
from eval import risk_coverage
from panel import FEATURE_COLS
from selective import _spearman
from synthetic import generate

FIGDIR = Path(__file__).resolve().parents[1] / "figures"
SEED = 0
BLUE, GREY, GREEN, RED = "#1f4e79", "#999999", "#2e7d32", "#c0392b"


def _load(seed=SEED):
    path = PROC / f"synth_1a_seed{seed}.parquet"
    if not path.exists():
        generate(seed)
    df = pd.read_parquet(path).dropna(subset=FEATURE_COLS + ["y"])
    return df, R._split(df)


def fig_risk_coverage(df, splits):
    tr, ca, te = splits
    sel = R._selective(tr, ca, te, "y", score_seed=SEED)
    isig = te["irreducible_sigma"].to_numpy()
    oracle = risk_coverage(sel["y"], sel["pred"], isig, R.RETENTIONS)
    xs = [r * 100 for r in R.RETENTIONS]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(xs, [sel["rand"][r] for r in R.RETENTIONS], "o--", color=GREY, label="random abstention")
    ax.plot(xs, [sel["method"][r] for r in R.RETENTIONS], "o-", color=BLUE, lw=2.2,
            label="selective (per-name $u_i$)")
    ax.plot(xs, [oracle[r] for r in R.RETENTIONS], "o-", color=GREEN, label="oracle (knows the floor)")
    ax.axvline(70, color=RED, ls=":", lw=1, alpha=0.7)
    ax.set_xlabel("retention (%)"); ax.set_ylabel("retained-set MAE")
    ax.set_title("Risk-coverage: abstaining on per-name uncertainty\n"
                 "recovers most of the oracle's gain over random (synthetic, seed 0)")
    ax.legend(frameon=False); ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(FIGDIR / "fig1_risk_coverage.png", dpi=150); plt.close(fig)


def fig_coverage_shift(df, splits):
    tr, ca, te = splits
    ivs = R._intervals(tr, ca, te, "y")
    methods = [("naive", "naive split-conformal", GREY), ("aci", "group-conditional ACI", BLUE)]
    labels, base_c, shift_c, wg_c = [], [], [], []
    for key, lab, _ in methods:
        b, sh, wg, _, _ = R._cov3(ivs[key])
        labels.append(lab); base_c.append(b); shift_c.append(sh); wg_c.append(wg)
    x = np.arange(len(methods)); w = 0.26
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(x - w, base_c, w, label="base (calm)", color="#a9cce3")
    ax.bar(x, shift_c, w, label="shift (drawdown)", color=BLUE)
    ax.bar(x + w, wg_c, w, label="worst-group | shift", color="#0b2f4a")
    ax.axhline(0.90, color=RED, ls="--", lw=1.2, label="nominal 90%")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("empirical coverage"); ax.set_ylim(0.3, 1.0)
    ax.set_title("Coverage under distribution shift (synthetic, seed 0):\n"
                 "ACI recovers most of the coverage lost under the shift")
    ax.legend(frameon=False, fontsize=8, ncol=2); ax.grid(axis="y", alpha=0.25); fig.tight_layout()
    fig.savefig(FIGDIR / "fig2_coverage_shift.png", dpi=150); plt.close(fig)


def fig_mechanism(df, splits):
    tr, ca, te = splits
    sel = R._selective(tr, ca, te, "y", score_seed=SEED)
    gt = (df.groupby("ticker")
            .agg(irreducible_sigma=("irreducible_sigma", "first"),
                 recoverable_mask=("recoverable_mask", "first")).reset_index())
    u = sel["u"]
    m = gt.set_index("ticker").reindex(u.index)
    isig = m["irreducible_sigma"].to_numpy()
    rec = m["recoverable_mask"].astype(bool).to_numpy()
    uu = u.to_numpy()
    ok = np.isfinite(uu) & np.isfinite(isig)
    rho = _spearman(uu[ok], isig[ok])
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.scatter(isig[ok & rec], uu[ok & rec], s=18, color=GREEN, alpha=0.7, label="recoverable name")
    ax.scatter(isig[ok & ~rec], uu[ok & ~rec], s=18, color=RED, alpha=0.7, label="mostly-hidden name (not recoverable)")
    ax.set_xlabel("injected irreducible noise $\\sigma_{irr}$ (ground truth)")
    ax.set_ylabel("per-name uncertainty $u_i$ (from calibration)")
    ax.set_title("Mechanism check: abstention score vs the known noise floor\n"
                 f"Spearman $\\rho$ = {rho:.2f} on seed 0 (synthetic)")
    ax.legend(frameon=False); ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(FIGDIR / "fig3_mechanism.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    FIGDIR.mkdir(exist_ok=True)
    df, splits = _load()
    fig_risk_coverage(df, splits)
    fig_coverage_shift(df, splits)
    fig_mechanism(df, splits)
    print(f"wrote 3 figures to {FIGDIR}")
