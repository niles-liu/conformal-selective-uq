"""Extensions: conformal risk control, conformal on the simplex, the e-value monitor."""

from __future__ import annotations

import numpy as np

from selective_conformal.evalue_monitor import detect, evalue_path
from selective_conformal.riskcontrol import (
    crc_radius,
    dist_outside,
    loss_miscoverage,
    loss_miss_magnitude,
    scrc_aci,
)
from selective_conformal.simplex import aci_radius_path, aitchison, close, clr, region_radius, tv


# --------------------------------------------------------------------------- #
# conformal risk control                                                      #
# --------------------------------------------------------------------------- #
def test_losses_are_bounded_and_shrink_as_the_interval_widens():
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    lo, hi = np.full(500, -0.2), np.full(500, 0.2)
    assert np.all(dist_outside(y, lo, hi) >= 0)
    prev_m, prev_g = np.inf, np.inf
    for lam in np.linspace(0, 3, 13):
        m = loss_miscoverage(y, lo - lam, hi + lam).mean()
        g = loss_miss_magnitude(y, lo - lam, hi + lam, w0=0.5).mean()
        assert 0 <= g <= 1 and m <= prev_m and g <= prev_g
        prev_m, prev_g = m, g


def test_crc_radius_is_the_smallest_grid_value_meeting_the_bound():
    rng = np.random.default_rng(1)
    y = rng.normal(size=400)
    q = np.zeros(400)
    grid = np.linspace(0, 4, 401)
    lam = crc_radius(q, q, y, loss_miscoverage, alpha=0.10, grid=grid)
    n = len(y)

    def bound(r):
        return n / (n + 1) * loss_miscoverage(y, q - r, q + r).mean() + 1 / (n + 1)

    assert bound(lam) <= 0.10
    assert bound(grid[np.searchsorted(grid, lam) - 1]) > 0.10    # one step tighter fails


def test_crc_controls_risk_on_fresh_data_on_average():
    rng = np.random.default_rng(2)
    grid = np.linspace(0, 6, 601)
    risks = []
    for _ in range(200):
        y = rng.standard_t(df=4, size=1200)
        q = np.zeros_like(y)
        lam = crc_radius(q[:200], q[:200], y[:200], loss_miscoverage, alpha=0.10, grid=grid)
        risks.append(loss_miscoverage(y[200:], q[200:] - lam, q[200:] + lam).mean())
    assert np.mean(risks) <= 0.10 + 0.005


def test_online_risk_control_widens_after_a_shift():
    rng = np.random.default_rng(3)
    n_cal, steps, per = 400, 200, 20
    yc = rng.normal(size=n_cal)
    te_order = np.repeat(np.arange(steps), per)
    yt = rng.normal(size=steps * per) * np.where(te_order >= 100, 2.0, 1.0)
    z = np.zeros
    out = scrc_aci(z(n_cal), z(n_cal), yc, z(len(yt)), z(len(yt)), yt, te_order,
                   score_ca=z(n_cal), score_te=z(len(yt)), r=1.0, loss_fn=loss_miscoverage,
                   alpha_risk=0.10, grid=np.linspace(0, 6, 601), eta=0.5)
    path = dict(out["lam_path"])
    assert path[199] > 1.5 * path[99]               # the radius grew after the shift
    post = te_order >= 150
    assert loss_miscoverage(yt[post], out["lo"][post], out["hi"][post]).mean() < 0.15


# --------------------------------------------------------------------------- #
# conformal on the simplex                                                    #
# --------------------------------------------------------------------------- #
def test_composition_helpers():
    w = np.array([2.0, 1.0, 1.0])
    np.testing.assert_allclose(close(w), [0.5, 0.25, 0.25])
    assert abs(clr(w).sum()) < 1e-12               # clr coordinates sum to zero
    a, b = close(np.array([1.0, 2.0, 3.0])), close(np.array([3.0, 2.0, 1.0]))
    assert aitchison(a, a) == 0.0 and np.isclose(aitchison(a, b), aitchison(b, a))
    assert np.isclose(aitchison(a, b), aitchison(close(10 * a), b))   # scale invariant
    assert 0.0 <= tv(a, b) <= 1.0 and np.isclose(tv(a, b), 1 / 3)


def test_simplex_region_static_and_online():
    rng = np.random.default_rng(4)
    cal = rng.exponential(size=500)
    assert region_radius(cal, 0.10) == np.sort(cal)[int(np.ceil(501 * 0.9)) - 1]
    test = rng.exponential(size=3000) * np.r_[np.ones(1500), 1.5 * np.ones(1500)]
    covered, radii = aci_radius_path(cal, test, alpha=0.10, gamma=0.05)
    static = test <= region_radius(cal, 0.10)
    assert covered[1500:].mean() > static[1500:].mean() + 0.05
    assert abs(covered.mean() - 0.90) < 0.02


# --------------------------------------------------------------------------- #
# e-value monitor                                                             #
# --------------------------------------------------------------------------- #
def test_evalue_monitor_stays_quiet_under_the_null_and_fires_on_a_break():
    rng = np.random.default_rng(5)
    fires = 0
    for _ in range(200):
        miss = rng.binomial(30, 0.10, size=300) / 30          # nominal coverage throughout
        fires += detect(miss, alpha=0.10, delta=0.05)["fire_idx"] >= 0
    assert fires / 200 <= 0.05 + 0.03                         # Ville: P(alarm) <= delta

    miss = np.r_[rng.binomial(30, 0.10, 150), rng.binomial(30, 0.35, 150)] / 30
    out = detect(miss, alpha=0.10, delta=0.05)
    assert out["fire_idx"] >= 150 - 10                        # not (much) before the break
    assert out["fire_idx"] != -1
    assert np.all(evalue_path(miss) >= 0)
