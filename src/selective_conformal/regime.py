"""Learned regime state: the dynamic axis for regime x conformal (P12). Pure method, no data.

A K-state Statistical Jump Model (Nystrup/Kolm/Shu, arXiv:2402.05272): the jump penalty lambda
enforces regime persistence, killing the spurious flipping a vanilla HMM produces. Fit
UNSUPERVISED on a per-session market-state feature series (dispersion of yesterday's realised
target); it never sees the conformal target at decision time.

PIT / no-lookahead. Because the training window spans MANY regime cycles (the multi-regime
substrate, `experiments/synthetic_regime.py`), centroids are identified offline on train+calib
and the test labels are assigned by a CAUSAL FORWARD FILTER with frozen centroids (each label uses
only data <= t). The design needs every regime to appear before the test window: with a single
regime in the calibration window the states are not identifiable.

See reports/p12_regime_axis_preregistration.md.
"""
from __future__ import annotations

import numpy as np

MAX_ITER = 100


def _kmeanspp_init(X, k, rng):
    n = len(X)
    c = [int(rng.integers(n))]
    for _ in range(1, k):
        d2 = np.min(((X[:, None, :] - X[np.array(c)][None, :, :]) ** 2).sum(-1), axis=1)
        p = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
        c.append(int(rng.choice(n, p=p)))
    return X[np.array(c)].copy()


def _viterbi_path(X, mu, lam):
    """Min-cost state path: sum_t ||x_t-mu_{s_t}||^2 + lam*sum_t 1[s_t != s_{t-1}]."""
    T, k = len(X), len(mu)
    emit = ((X[:, None, :] - mu[None, :, :]) ** 2).sum(-1)
    C = np.empty((T, k)); B = np.zeros((T, k), dtype=int)
    C[0] = emit[0]
    for t in range(1, T):
        for j in range(k):
            trans = C[t - 1] + lam * (np.arange(k) != j)
            B[t, j] = int(np.argmin(trans))
            C[t, j] = emit[t, j] + trans[B[t, j]]
    s = np.empty(T, dtype=int)
    s[-1] = int(np.argmin(C[-1]))
    for t in range(T - 1, 0, -1):
        s[t - 1] = B[t, s[t]]
    return s


def fit_jump(X, k=3, lam=50.0, seed=0):
    """Coordinate descent: (Viterbi assign) <-> (recompute centroids). Returns (mu, path)."""
    rng = np.random.default_rng(seed)
    mu = _kmeanspp_init(X, k, rng)
    path = _viterbi_path(X, mu, lam)
    for _ in range(MAX_ITER):
        new_mu = mu.copy()
        for j in range(k):
            m = path == j
            if m.any():
                new_mu[j] = X[m].mean(0)
        new_path = _viterbi_path(X, new_mu, lam)
        if np.array_equal(new_path, path):
            mu, path = new_mu, new_path
            break
        mu, path = new_mu, new_path
    return mu, path


def n_switches(path):
    return int((np.diff(path) != 0).sum())


def order_states_by(mu, col):
    """Return states ordered by centroid value on `col` (ascending) -> stress = last."""
    return list(np.argsort(mu[:, col]))


def select_lambda(X, grid, k=3, seed=0, target_switch_rate=0.05):
    """Persistence-only lambda selection (NEVER on coverage): smallest lambda whose smoothed
    path switch-rate <= target_switch_rate * T. Pre-registered rule."""
    Tn = len(X)
    for lam in sorted(grid):
        _, path = fit_jump(X, k, float(lam), seed)
        if n_switches(path) <= target_switch_rate * Tn:
            return float(lam)
    return float(max(grid))


def forward_filter(Xte, mu, lam, prev_state):
    """Causal deploy: s_t = argmin_k[||x_t-mu_k||^2 + lam*1[k != s_{t-1}]], frozen centroids.
    Uses only data <= t (prev_state seeded from the last train+calib state). Returns int labels."""
    k = len(mu)
    out = np.empty(len(Xte), dtype=int)
    s_prev = int(prev_state)
    for t in range(len(Xte)):
        cost = ((Xte[t] - mu) ** 2).sum(1) + lam * (np.arange(k) != s_prev)
        s_prev = int(np.argmin(cost))
        out[t] = s_prev
    return out


def threshold_labels(disp, calib_mask, qs=(0.5, 0.9)):
    """Ordinal cheap-trigger baseline (causal): bucket session dispersion by CALIB quantiles
    into {0,1,2}. Calib quantiles are known before test."""
    lo, hi = np.quantile(disp[calib_mask], qs[0]), np.quantile(disp[calib_mask], qs[1])
    return (disp > lo).astype(int) + (disp > hi).astype(int)
