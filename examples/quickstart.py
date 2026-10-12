"""Quickstart on toy data: no network, about 5 s. Run from the repo root:
    python examples/quickstart.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from conformal import aci_cqr, cqr
from eval import coverage, risk_coverage
from learners import fit_mean
from selective import name_uncertainty, selective_scores

# 30 units, each with its own noise level; the noise doubles at time step 100 (the shift)
rng = np.random.default_rng(0)
n, t = 6000, np.arange(6000) // 50
X, unit = rng.normal(size=(n, 5)), rng.integers(0, 30, n)
y = X[:, 0] + np.linspace(0.3, 3.0, 30)[unit] * rng.normal(size=n) * np.where(t >= 100, 2.0, 1.0)
tier = unit // 10                                    # 3 noise tiers, the groups ACI conditions on
tr, ca, te = slice(0, 3000), slice(3000, 4500), slice(4500, n)
static = cqr(X[tr], y[tr], X[ca], y[ca], X[te])      # split conformal (CQR), calibrated once
online = aci_cqr(X[tr], y[tr], X[ca], y[ca], tier[ca], X[te], y[te], t[te], tier[te])
yt, shifted = y[te], t[te] >= 100
for name, iv in (("static CQR", static), ("online ACI", online)):
    cov = [coverage(yt[m], iv.lo[m], iv.hi[m]) for m in (~shifted, shifted)]
    worst = min(coverage(yt[m], iv.lo[m], iv.hi[m]) for m in (shifted & (tier[te] == k) for k in range(3)))
    print(f"{name}: coverage {cov[0]:.2f} before the shift, {cov[1]:.2f} after "
          f"(worst tier {worst:.2f}; target 0.90)")

# selective abstention: score each unit by its calibration error, drop the 30% least predictable
mean = fit_mean(X[tr], y[tr])
u = name_uncertainty(pd.DataFrame({"ticker": unit[ca], "resid": np.abs(y[ca] - mean.predict(X[ca]))}))
score = selective_scores(pd.DataFrame({"ticker": unit[te]}), u)
mae = risk_coverage(yt, mean.predict(X[te]), score, np.array([1.0, 0.7]))
print(f"abstaining on the 30% least predictable rows: MAE {mae[1.0]:.2f} -> {mae[0.7]:.2f}")
