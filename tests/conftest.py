"""Shared toy data for the tests: no network, no market data, seconds to run."""

from __future__ import annotations

import numpy as np
import pytest

SCALES = np.array([0.5, 1.0, 2.0])   # per-group noise scale: groups 0, 1, 2


def grouped_stream(rng, n_steps: int, per_step: int, shift_at: int | None = None, amp: float = 1.5):
    """Time-ordered rows in 3 groups with group-dependent noise; from step `shift_at`
    on the noise grows by `amp` (the shift). Returns (t, g, y)."""
    t = np.repeat(np.arange(n_steps), per_step)
    g = rng.integers(0, 3, len(t))
    scale = SCALES[g] * (np.where(t >= shift_at, amp, 1.0) if shift_at is not None else 1.0)
    return t, g, scale * rng.standard_normal(len(t))


@pytest.fixture
def toy_regression():
    """Train / calibration / test folds of a small heteroscedastic regression with groups
    and a time order."""
    rng = np.random.default_rng(7)
    n = 3000
    X = rng.normal(size=(n, 4))
    g = rng.integers(0, 3, n)
    t = np.arange(n) // 25
    y = X[:, 0] - 0.5 * X[:, 1] + SCALES[g] * rng.standard_normal(n)
    tr, ca, te = slice(0, 1200), slice(1200, 2000), slice(2000, n)
    return dict(X=X, y=y, g=g, t=t, tr=tr, ca=ca, te=te)
