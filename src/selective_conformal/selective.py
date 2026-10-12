"""Selective layer — abstention score + G-MECHANISM diagnostics (SPEC §6, P2).

The abstention score is a **per-name calibration residual scale** `u_i`: the shrunk
MAD of name i's residuals on the calm 2021 calibration fold. Why this and not raw
interval width (which P1 showed is uninformative): the injected overlay is **per-name
persistent** (lambda_i fixed, overlay AR(1) rho=0.95), so a name's OWN historical error
scale is a public, PIT-legal proxy for its irreducible noise. Abstain on highest u_i.

G-MECHANISM checks the abstention lands in the *right* place: low-u names should be the
recoverable ones, and u_i should track the injected irreducible_sigma — and that link
must survive partialling out per-name vol_63 (else it's just a volatility confound).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

SHRINK_K = 20  # FROZEN (pre-reg): per-name uncertainty shrinkage toward the global scale


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(np.median(np.abs(x - np.median(x))) * 1.4826)


def name_uncertainty(cal_df: pd.DataFrame, resid_col: str = "resid",
                     name_col: str = "ticker", k: int = SHRINK_K) -> pd.Series:
    """Per-name shrunk residual scale u_i (indexed by name). Names with few calib
    points shrink toward the global scale: u_i = (n_i*MAD_i + k*MAD_glob)/(n_i+k)."""
    glob = _mad(cal_df[resid_col].to_numpy())
    g = cal_df.groupby(name_col)[resid_col]
    mad = g.apply(lambda s: _mad(s.to_numpy()))
    n = g.size()
    return ((n * mad + k * glob) / (n + k)).rename("u")


def selective_scores(test_df: pd.DataFrame, u_by_name: pd.Series,
                     name_col: str = "ticker") -> np.ndarray:
    """Map per-name uncertainty onto test rows; unseen names get the global median."""
    return test_df[name_col].map(u_by_name).fillna(u_by_name.median()).to_numpy()


# --------------------------------------------------------------------------- #
# G-MECHANISM diagnostics (1a only)                                           #
# --------------------------------------------------------------------------- #
def mechanism_auc(u_by_name: pd.Series, gt: pd.DataFrame,
                  name_col: str = "ticker") -> float:
    """AUC of the confidence score (-u_i) predicting recoverable_mask_i. >= theta?"""
    from sklearn.metrics import roc_auc_score

    m = gt.set_index(name_col).reindex(u_by_name.index)
    y = m["recoverable_mask"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, -u_by_name.to_numpy()))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(stats.spearmanr(a, b).statistic)


def _partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Spearman(x, y) controlling for z: correlate the rank-residuals of x|z and y|z."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    Z = np.c_[np.ones_like(rz), rz]
    bx = np.linalg.lstsq(Z, rx, rcond=None)[0]
    by = np.linalg.lstsq(Z, ry, rcond=None)[0]
    ex, ey = rx - Z @ bx, ry - Z @ by
    return float(np.corrcoef(ex, ey)[0, 1])


def mechanism_corr(u_by_name: pd.Series, gt: pd.DataFrame,
                   name_col: str = "ticker") -> dict:
    """Spearman(u_i, irreducible_sigma_i) raw and partialling out per-name vol_63."""
    m = gt.set_index(name_col).reindex(u_by_name.index)
    u = u_by_name.to_numpy()
    isig = m["irreducible_sigma"].to_numpy()
    vol = m["vol_63"].to_numpy()
    ok = np.isfinite(u) & np.isfinite(isig) & np.isfinite(vol)
    return dict(
        rho=_spearman(u[ok], isig[ok]),
        rho_ctrl_vol=_partial_spearman(u[ok], isig[ok], vol[ok]),
        auc_ctrl_vol_note="see mechanism_auc; partial-rho is the confound guard",
    )
