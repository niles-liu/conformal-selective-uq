"""Metrics and the selective (abstention) layer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from selective_conformal import (
    clustered_bootstrap,
    coverage,
    grouped_coverage,
    mean_width,
    name_uncertainty,
    risk_coverage,
    selective_scores,
    worst_group_coverage,
)
from selective_conformal.selective import SHRINK_K, _mad, mechanism_auc, mechanism_corr


def test_coverage_and_width_by_hand():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    lo, hi = np.array([-1.0, 1.5, 1.0, 3.0]), np.array([1.0, 2.0, 2.5, 3.0])
    assert coverage(y, lo, hi) == 0.75            # 1.0 is outside [1.5, 2.0]; endpoints count
    assert mean_width(lo, hi) == (2.0 + 0.5 + 1.5 + 0.0) / 4


def test_worst_group_skips_small_groups_and_reports_where_it_is():
    df = pd.DataFrame({
        "y": np.r_[np.zeros(60), np.zeros(60), np.zeros(5)],
        "lo": np.r_[-np.ones(60), np.r_[np.ones(15), -np.ones(45)], np.ones(5)],
        "hi": np.ones(125),
        "sector": ["a"] * 60 + ["b"] * 60 + ["c"] * 5,     # "c" misses everything but n=5
    })
    gc = grouped_coverage(df, "sector")
    assert list(gc["sector"]) == ["c", "b", "a"]           # sorted, worst first
    cov, dim, lab = worst_group_coverage(df, ["sector"], min_n=50)
    assert (cov, dim, lab) == (0.75, "sector", "b")


def test_risk_coverage_with_an_oracle_score_improves_as_we_abstain():
    rng = np.random.default_rng(0)
    err = rng.exponential(size=2000)
    y, pred = err, np.zeros(2000)
    curve = risk_coverage(y, pred, score=err, retentions=np.array([1.0, 0.7, 0.4]))
    assert curve[1.0] > curve[0.7] > curve[0.4]
    assert np.isclose(curve[1.0], err.mean())


def test_clustered_bootstrap_is_seeded_and_brackets_the_point():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "x": rng.normal(size=600),
        "sector": rng.choice(list("abcdef"), 600),
        "report_date": rng.choice([20220103, 20220201, 20220301, 20220401], 600),
    })

    def stat(d):
        return float(d["x"].mean())

    a = clustered_bootstrap(df, stat, n_boot=200, seed=3)
    b = clustered_bootstrap(df, stat, n_boot=200, seed=3)
    assert a == b
    point, lo, hi = a
    assert lo <= point <= hi


def test_name_uncertainty_matches_the_shrinkage_formula():
    rng = np.random.default_rng(2)
    cal = pd.DataFrame({"ticker": np.repeat(["A", "B", "C"], [100, 30, 5]),
                        "resid": np.r_[rng.normal(0, 0.5, 100), rng.normal(0, 2.0, 30), rng.normal(0, 1, 5)]})
    u = name_uncertainty(cal)
    glob = _mad(cal["resid"].to_numpy())
    for name, sub in cal.groupby("ticker"):
        n = len(sub)
        expect = (n * _mad(sub["resid"].to_numpy()) + SHRINK_K * glob) / (n + SHRINK_K)
        assert np.isclose(u[name], expect)
    assert u["B"] > u["A"]                          # the noisy name scores as less predictable


def test_selective_scores_fall_back_to_the_median_for_unseen_names():
    u = pd.Series({"A": 0.1, "B": 0.5, "C": 0.9})
    s = selective_scores(pd.DataFrame({"ticker": ["A", "C", "Z"]}), u)
    np.testing.assert_allclose(s, [0.1, 0.9, 0.5])


def test_mechanism_diagnostics_on_a_known_ranking():
    names = [f"n{i}" for i in range(40)]
    sigma = np.linspace(0.1, 2.0, 40)
    gt = pd.DataFrame({"ticker": names, "recoverable_mask": sigma < 1.0,
                       "irreducible_sigma": sigma, "vol_63": np.tile([0.1, 0.3], 20)})
    u = pd.Series(sigma, index=names)               # a perfect abstention score
    assert mechanism_auc(u, gt) == 1.0
    m = mechanism_corr(u, gt)
    assert np.isclose(m["rho"], 1.0)
    assert m["rho_ctrl_vol"] > 0.99
