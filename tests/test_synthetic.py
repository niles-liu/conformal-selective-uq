"""The Layer 1a generator (experiments/synthetic.py) on a small fake panel: same seed,
same target; the known per-stock ground truth is internally consistent; and building
from an in-memory panel writes no cache."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import synthetic


@pytest.fixture(scope="module")
def fake_panel():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2021-06-01", "2022-12-30")[::3]      # spans the 2022 shift window
    tickers = [f"T{i:02d}" for i in range(20)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    p = pd.DataFrame(index=idx).reset_index()
    for col in ("resid_mom_63", "mom_252", "vol_63", "dist_252hi", "log_dollar_vol"):
        p[col] = rng.normal(size=len(p))
    p["report_date"] = p["date"].dt.strftime("%Y%m%d").astype(int)
    return p


def test_same_seed_same_target_and_no_cache_written(fake_panel):
    seed = 987_654
    cache = synthetic.PROC / f"synth_1a_seed{seed}.parquet"
    a = synthetic.generate(seed, panel=fake_panel)
    b = synthetic.generate(seed, panel=fake_panel.sample(frac=1.0, random_state=1))   # row order shuffled
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a["y"], synthetic.generate(seed + 1, panel=fake_panel)["y"])
    assert not cache.exists()


def test_target_and_ground_truth_are_consistent(fake_panel):
    out = synthetic.generate(0, panel=fake_panel)
    assert len(out) == len(fake_panel) and np.isfinite(out["y"]).all()
    np.testing.assert_allclose(out.groupby("date")["weight"].sum(), 1.0)
    in_window = out["report_date"].between(*synthetic.SHIFT_REGIME)
    assert (out.loc[in_window, "regime"] == "shift").all() and (out.loc[~in_window, "regime"] == "base").all()

    per_name = out.groupby("ticker")[["recoverable_fraction", "recoverable_mask", "irreducible_sigma",
                                      "lambda_i"]].nunique()
    assert (per_name == 1).all().all()              # ground truth is one value per stock
    gt = out.groupby("ticker").first()
    assert gt["recoverable_fraction"].between(0, 1).all()
    assert (gt["recoverable_mask"] == (gt["recoverable_fraction"] >= synthetic.REC_CUT)).all()
    # a heavier hidden-component weight means more unrecoverable noise
    assert stats.spearmanr(gt["lambda_i"], gt["irreducible_sigma"]).statistic > 0.8
