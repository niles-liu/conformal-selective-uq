"""Evaluation harness — coverage, risk-coverage, clustered bands (SPEC §4 guards).

Every reported number is multi-seed mean +/- a band from **clustered/blocked**
bootstrap over the sector x calendar-month cluster (never the row — that would
overstate precision under cross-sectional + serial correlation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_width(lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean(hi - lo))


def grouped_coverage(df: pd.DataFrame, group: str, target: float = 0.90) -> pd.DataFrame:
    """Per-group empirical coverage and signed gap vs target. df has y, lo, hi, <group>."""
    rows = []
    for g, sub in df.groupby(group, observed=True):
        cov = coverage(sub["y"].to_numpy(), sub["lo"].to_numpy(), sub["hi"].to_numpy())
        rows.append({group: g, "n": len(sub), "coverage": cov, "gap": cov - target})
    return pd.DataFrame(rows).sort_values("coverage").reset_index(drop=True)


def worst_group_coverage(df: pd.DataFrame, groups: list[str], target: float = 0.90,
                         min_n: int = 50) -> tuple[float, str, str]:
    """Lowest per-group coverage across all group dimensions (groups with >= min_n)."""
    worst_cov, worst_dim, worst_lab = 1.0, "", ""
    for dim in groups:
        gc = grouped_coverage(df, dim, target)
        gc = gc[gc["n"] >= min_n]
        if gc.empty:
            continue
        row = gc.iloc[0]
        if row["coverage"] < worst_cov:
            worst_cov, worst_dim, worst_lab = row["coverage"], dim, str(row[dim])
    return worst_cov, worst_dim, worst_lab


def risk_coverage(y: np.ndarray, pred: np.ndarray, score: np.ndarray,
                  retentions: np.ndarray) -> dict[float, float]:
    """Retained-set MAE as a function of retention, abstaining on the HIGHEST score
    first (score = uncertainty proxy: random / oracle-noise / interval width)."""
    abs_err = np.abs(y - pred)
    order = np.argsort(score)  # ascending: keep lowest-score (most confident) first
    out = {}
    n = len(y)
    for r in retentions:
        k = max(1, int(round(r * n)))
        keep = order[:k]
        out[float(r)] = float(np.mean(abs_err[keep]))
    return out


def cluster_key(df: pd.DataFrame) -> pd.Series:
    """sector x calendar-month block id for clustered resampling."""
    month = (df["report_date"] // 100).astype(int)  # YYYYMM
    return df["sector"].astype(str) + "|" + month.astype(str)


def clustered_bootstrap(df: pd.DataFrame, stat_fn, n_boot: int = 500,
                        seed: int = 0, ci: float = 0.90) -> tuple[float, float, float]:
    """(point, lo, hi) for stat_fn(df) via resampling whole clusters with replacement."""
    point = stat_fn(df)
    rng = np.random.default_rng(seed)
    key = cluster_key(df)
    clusters = key.unique()
    groups = {c: df.index[key == c].to_numpy() for c in clusters}
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(clusters, size=len(clusters), replace=True)
        idx = np.concatenate([groups[c] for c in pick])
        boots[b] = stat_fn(df.loc[idx])
    a = (1 - ci) / 2
    return point, float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))
