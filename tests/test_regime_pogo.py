"""The learned regime detector (Statistical Jump Model + causal filter) and the
parameter-free POGO-style online update."""

from __future__ import annotations

import numpy as np

from selective_conformal import coverage, regime
from selective_conformal.pogo import cqr_conditional_gqaci_max_pogo


def _planted(seed=0, T=1200, means=(0.0, 2.0, 5.0), stay=0.97):
    rng = np.random.default_rng(seed)
    s = np.empty(T, dtype=int)
    s[0] = 0
    for t in range(1, T):
        s[t] = s[t - 1] if rng.random() < stay else rng.integers(3)
    x = np.asarray(means)[s] + 0.4 * rng.standard_normal(T)
    return x[:, None], s


def _accuracy_up_to_relabel(path, truth, k=3):
    best = {j: np.bincount(truth[path == j], minlength=k).argmax() for j in np.unique(path)}
    return float(np.mean(np.array([best[j] for j in path]) == truth))


def test_jump_model_recovers_a_planted_regime_path():
    X, s = _planted()
    mu, path = regime.fit_jump(X, k=3, lam=10.0, seed=0)
    assert _accuracy_up_to_relabel(path, s) > 0.97
    assert regime.n_switches(path) <= 2 * regime.n_switches(s)        # persistence, no flicker


def test_lambda_selection_meets_the_persistence_rule():
    X, _ = _planted(seed=1)
    grid = (1, 10, 50, 200)
    lam = regime.select_lambda(X, grid, k=3, seed=0, target_switch_rate=0.05)
    _, path = regime.fit_jump(X, 3, lam, 0)
    assert lam in grid and regime.n_switches(path) <= 0.05 * len(X)


def test_forward_filter_is_causal():
    X, _ = _planted(seed=2)
    mu, path = regime.fit_jump(X[:800], k=3, lam=10.0, seed=0)
    a = regime.forward_filter(X[800:], mu, 10.0, path[-1])
    Xb = X[800:].copy()
    Xb[300:] += 50.0                                               # rewrite the future
    b = regime.forward_filter(Xb, mu, 10.0, path[-1])
    np.testing.assert_array_equal(a[:300], b[:300])


def test_threshold_labels_use_calibration_quantiles():
    disp = np.r_[np.linspace(0, 1, 100), np.full(10, 5.0)]
    calib = np.r_[np.ones(100, bool), np.zeros(10, bool)]
    lab = regime.threshold_labels(disp, calib, qs=(0.5, 0.9))
    assert set(np.unique(lab)) <= {0, 1, 2} and (lab[100:] == 2).all()
    assert regime.order_states_by(np.array([[3.0], [1.0], [2.0]]), col=0) == [1, 2, 0]


def test_pogo_is_online_valid_and_covers(toy_regression):
    d = toy_regression
    X, y, g, t, tr, ca, te = (d[k] for k in ("X", "y", "g", "t", "tr", "ca", "te"))
    gc, gt = [g[ca].astype(str)], [g[te].astype(str)]
    iv = cqr_conditional_gqaci_max_pogo(X[tr], y[tr], X[ca], y[ca], gc, X[te], y[te], t[te], gt)
    assert np.all(iv.hi >= iv.lo) and 0.85 <= coverage(y[te], iv.lo, iv.hi) <= 0.95

    s = np.unique(t[te])[10]
    y2 = y[te].copy()
    y2[t[te] >= s] += 100.0                                        # rewrite labels from step s on
    iv2 = cqr_conditional_gqaci_max_pogo(X[tr], y[tr], X[ca], y[ca], gc, X[te], y2, t[te], gt)
    upto = t[te] <= s
    np.testing.assert_array_equal(iv.lo[upto], iv2.lo[upto])
