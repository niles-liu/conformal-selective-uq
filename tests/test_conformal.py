"""Conformal methods: quantile edge cases, finite-sample coverage, group coverage, the
online method under a shift, no lookahead, and the bring-your-own-model paths."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import grouped_stream

from selective_conformal import (
    LightGBMLearner,
    aci_cqr,
    aci_from_quantiles,
    conformity_scores,
    coverage,
    cqr,
    cqr_from_quantiles,
    mondrian_cqr,
    mondrian_from_quantiles,
    naive_split,
)
from selective_conformal.conformal import _adaptive_q, _conf_quantile


# --------------------------------------------------------------------------- #
# quantile edge cases                                                         #
# --------------------------------------------------------------------------- #
def test_conf_quantile_uses_the_finite_sample_level():
    scores = np.arange(1.0, 20.0)                 # n = 19
    # k = ceil((n + 1)(1 - alpha)) = ceil(20 * 0.9) = 18 -> the 18th smallest score
    assert _conf_quantile(scores, 0.10) == 18.0
    assert _conf_quantile(scores[::-1], 0.10) == 18.0   # input order does not matter


def test_conf_quantile_clamps_to_the_largest_score_when_n_is_small():
    scores = np.array([3.0, 1.0, 2.0])            # ceil(4 * 0.9) = 4 > n: capped at the max
    assert _conf_quantile(scores, 0.10) == 3.0
    assert _conf_quantile(np.array([5.0]), 0.10) == 5.0


def test_adaptive_quantile_extremes():
    s = np.sort(np.array([0.2, 0.5, 0.9, 1.4]))
    assert _adaptive_q(s, 0.0) == 1.4             # alpha <= 0: widest available score
    assert _adaptive_q(s, -0.3) == 1.4
    assert _adaptive_q(s, 1.0) == 0.0             # alpha >= 1: zero radius
    assert _adaptive_q(s, 0.5) == _conf_quantile(s, 0.5)


def test_conformity_scores_are_signed_distance_outside_the_band():
    E = conformity_scores(np.zeros(3), np.ones(3), np.array([-0.5, 0.5, 3.0]))
    np.testing.assert_allclose(E, [0.5, -0.5, 2.0])


# --------------------------------------------------------------------------- #
# finite-sample marginal coverage                                             #
# --------------------------------------------------------------------------- #
def test_split_cqr_finite_sample_coverage_holds_for_any_band():
    """Under exchangeability split conformal covers with probability in
    [1 - alpha, 1 - alpha + 1/(n + 1)], however bad the base band is. Averaged over
    300 calibration draws the mean coverage must land in that range (Monte Carlo
    standard error about 0.002)."""
    rng = np.random.default_rng(0)
    n_cal, n_te, alpha = 99, 1000, 0.10
    covs = []
    for _ in range(300):
        y = rng.standard_t(df=3, size=n_cal + n_te)          # heavy tails
        lo, hi = np.full_like(y, 0.3), np.full_like(y, 0.4)  # a deliberately wrong band
        tlo, thi = cqr_from_quantiles(lo[:n_cal], hi[:n_cal], y[:n_cal], lo[n_cal:], hi[n_cal:], alpha)
        covs.append(coverage(y[n_cal:], tlo, thi))
    mean = float(np.mean(covs))
    assert 1 - alpha - 0.01 <= mean <= 1 - alpha + 1 / (n_cal + 1) + 0.01


def test_cqr_and_naive_split_cover_iid_data(toy_regression):
    d = toy_regression
    X, y, tr, ca, te = d["X"], d["y"], d["tr"], d["ca"], d["te"]
    for method in (cqr, naive_split):
        iv = method(X[tr], y[tr], X[ca], y[ca], X[te])
        assert 0.86 <= coverage(y[te], iv.lo, iv.hi) <= 0.95
        assert np.all(iv.hi >= iv.lo)


def test_mondrian_fixes_group_coverage_that_one_radius_misses():
    rng = np.random.default_rng(3)
    _, gc, yc = grouped_stream(rng, 40, 30)
    _, gt, yt = grouped_stream(rng, 100, 30)

    def band(y):
        return -np.ones_like(y), np.ones_like(y)

    def worst(lo, hi):
        return min(coverage(yt[gt == k], lo[gt == k], hi[gt == k]) for k in range(3))

    slo, shi = cqr_from_quantiles(*band(yc), yc, *band(yt))
    mlo, mhi = mondrian_from_quantiles(*band(yc), yc, gc, *band(yt), gt)
    assert worst(slo, shi) < 0.80                  # one radius undercovers the noisy group
    assert worst(mlo, mhi) >= 0.84                 # a radius per group repairs it


# --------------------------------------------------------------------------- #
# online conformal (ACI) under a shift                                        #
# --------------------------------------------------------------------------- #
def _shift_case(seed=11):
    rng = np.random.default_rng(seed)
    _, gc, yc = grouped_stream(rng, 40, 30)                     # calm calibration
    tt, gt, yt = grouped_stream(rng, 400, 30, shift_at=200)     # noise x1.5 halfway
    return dict(lo_c=-np.ones_like(yc), hi_c=np.ones_like(yc), yc=yc, gc=gc,
                lo_t=-np.ones_like(yt), hi_t=np.ones_like(yt), yt=yt, tt=tt, gt=gt)


def test_aci_recovers_coverage_that_static_conformal_loses():
    c = _shift_case()
    slo, shi = cqr_from_quantiles(c["lo_c"], c["hi_c"], c["yc"], c["lo_t"], c["hi_t"])
    alo, ahi = aci_from_quantiles(c["lo_c"], c["hi_c"], c["yc"], c["gc"], c["lo_t"], c["hi_t"],
                                  c["yt"], c["tt"], c["gt"])
    post, yt, gt = c["tt"] >= 200, c["yt"], c["gt"]

    def worst(lo, hi):
        return min(coverage(yt[post & (gt == k)], lo[post & (gt == k)], hi[post & (gt == k)])
                   for k in range(3))

    assert coverage(yt[post], slo[post], shi[post]) < 0.85
    assert coverage(yt[post], alo[post], ahi[post]) >= 0.87
    assert worst(slo, shi) < 0.65
    assert worst(alo, ahi) >= 0.86


def test_aci_has_no_lookahead():
    """The interval at step s may depend only on labels from steps before s."""
    c = _shift_case()
    args = (c["lo_c"], c["hi_c"], c["yc"], c["gc"], c["lo_t"], c["hi_t"])
    lo1, hi1 = aci_from_quantiles(*args, c["yt"], c["tt"], c["gt"])
    s = 250
    y2 = c["yt"].copy()
    y2[c["tt"] >= s] += 100.0                       # rewrite every label from step s on
    lo2, hi2 = aci_from_quantiles(*args, y2, c["tt"], c["gt"])
    upto = c["tt"] <= s                             # steps before s, and step s itself
    np.testing.assert_array_equal(lo1[upto], lo2[upto])
    np.testing.assert_array_equal(hi1[upto], hi2[upto])
    assert not np.array_equal(lo1[~upto], lo2[~upto])


# --------------------------------------------------------------------------- #
# bring your own model                                                        #
# --------------------------------------------------------------------------- #
def test_from_quantiles_reproduce_the_fitted_methods_exactly(toy_regression):
    d = toy_regression
    X, y, g, t, tr, ca, te = (d[k] for k in ("X", "y", "g", "t", "tr", "ca", "te"))
    lrn, alpha = LightGBMLearner(), 0.10
    qlo = lrn.fit_quantile(X[tr], y[tr], alpha / 2)
    qhi = lrn.fit_quantile(X[tr], y[tr], 1 - alpha / 2)
    band_c = (qlo.predict(X[ca]), qhi.predict(X[ca]))
    band_t = (qlo.predict(X[te]), qhi.predict(X[te]))

    iv = cqr(X[tr], y[tr], X[ca], y[ca], X[te])
    np.testing.assert_array_equal((iv.lo, iv.hi), cqr_from_quantiles(*band_c, y[ca], *band_t))

    iv = mondrian_cqr(X[tr], y[tr], X[ca], y[ca], g[ca], X[te], g[te])
    np.testing.assert_array_equal((iv.lo, iv.hi),
                                  mondrian_from_quantiles(*band_c, y[ca], g[ca], *band_t, g[te]))

    iv = aci_cqr(X[tr], y[tr], X[ca], y[ca], g[ca], X[te], y[te], t[te], g[te])
    np.testing.assert_array_equal((iv.lo, iv.hi),
                                  aci_from_quantiles(*band_c, y[ca], g[ca], *band_t, y[te], t[te], g[te]))


class _Linear:
    """Least squares, shifted by an empirical residual quantile (0 = the mean fit)."""

    def __init__(self, X, y, level):
        A = np.c_[np.ones(len(X)), X]
        self.coef = np.linalg.lstsq(A, y, rcond=None)[0]
        self.offset = float(np.quantile(y - A @ self.coef, level)) if level else 0.0

    def predict(self, X):
        return np.c_[np.ones(len(X)), X] @ self.coef + self.offset


class _LinearQuantileLearner:
    """A user's own learner. Records every call so the test can see it was used."""

    def __init__(self):
        self.calls = []

    def fit_mean(self, X, y):
        self.calls.append("mean")
        return _Linear(X, y, 0.0)

    def fit_quantile(self, X, y, level):
        self.calls.append(level)
        return _Linear(X, y, level)


@pytest.mark.parametrize("method", ["cqr", "aci_cqr"])
def test_methods_accept_a_custom_learner(toy_regression, method):
    d = toy_regression
    X, y, g, t, tr, ca, te = (d[k] for k in ("X", "y", "g", "t", "tr", "ca", "te"))
    lrn = _LinearQuantileLearner()
    if method == "cqr":
        iv = cqr(X[tr], y[tr], X[ca], y[ca], X[te], learner=lrn)
    else:
        iv = aci_cqr(X[tr], y[tr], X[ca], y[ca], g[ca], X[te], y[te], t[te], g[te], learner=lrn)
    assert lrn.calls == ["mean", 0.05, 0.95]
    np.testing.assert_allclose(iv.pred, _Linear(X[tr], y[tr], 0.0).predict(X[te]))
    assert 0.80 <= coverage(y[te], iv.lo, iv.hi) <= 0.97
