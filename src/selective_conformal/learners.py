"""Base learners: LightGBM mean + quantile (SPEC §6).

Mean learner: a LightGBM regressor (squared loss). Quantile learners: LightGBM with
the pinball objective at the lower/upper levels, the body of CQR. Thin wrappers so
`conformal.py` stays about conformal math, not model config. Frozen hyper-params
(no per-phase tuning, by gate-first discipline).

Every method in `conformal.py` takes an optional `learner=`: any object with
`fit_mean(X, y)` and `fit_quantile(X, y, level)` that return fitted models with a
`.predict(X)` method. Leaving it out uses `LightGBMLearner`, the frozen default that
produced every published number.
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
    # Repeatability. No seed is set, so LightGBM's sampling seeds keep their fixed defaults
    # (bagging 3, feature fraction 2); passing random_state would change every published number.
    # deterministic + force_row_wise stop LightGBM from picking its histogram method by timing
    # both at startup. Every driver prints the same output with or without them, at 4 threads
    # and at all cores; they cost about 20% in run time.
    deterministic=True,
    force_row_wise=True,
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


class LightGBMLearner:
    """The frozen default learner: `fit_mean` and `fit_quantile` with `PARAMS`."""

    def fit_mean(self, X: np.ndarray, y: np.ndarray) -> LGBMRegressor:
        return fit_mean(X, y)

    def fit_quantile(self, X: np.ndarray, y: np.ndarray, level: float) -> LGBMRegressor:
        return fit_quantile(X, y, level)
