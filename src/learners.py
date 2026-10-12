"""Base learners: LightGBM mean + quantile (SPEC §6).

Mean learner: a LightGBM regressor (squared loss). Quantile learners: LightGBM with
the pinball objective at the lower/upper levels, the body of CQR. Thin wrappers so
`conformal.py` stays about conformal math, not model config. Frozen hyper-params
(no per-phase tuning, by gate-first discipline).
"""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMRegressor

PARAMS = dict(
    n_estimators=400,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_jobs=-1,
    verbosity=-1,
)


def fit_mean(X: np.ndarray, y: np.ndarray) -> LGBMRegressor:
    m = LGBMRegressor(objective="regression_l2", **PARAMS)
    m.fit(X, y)
    return m


def fit_quantile(X: np.ndarray, y: np.ndarray, alpha: float) -> LGBMRegressor:
    """Quantile regressor at level `alpha` (pinball loss)."""
    m = LGBMRegressor(objective="quantile", alpha=alpha, **PARAMS)
    m.fit(X, y)
    return m
