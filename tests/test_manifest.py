"""The data-manifest fingerprint (experiments/manifest.py): blind to storage details,
sensitive to any change in the values, and specific about what moved."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import manifest


def _frame():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "date": pd.to_datetime(["2022-01-03", "2022-01-04", "2022-01-05"]),
        "ticker": ["AAPL", "MSFT", None],
        "ret": [0.01, np.nan, -0.02],
        "vol": rng.random(3),
        "n": [1, 2, 3],
    })


def test_fingerprint_ignores_column_order_and_storage_dtypes():
    a = _frame()
    b = a[["vol", "n", "ret", "ticker", "date"]].copy()
    b["ticker"] = b["ticker"].astype("string")             # text stored as pandas strings
    b["date"] = b["date"].astype("datetime64[ms]")          # timestamps at another unit
    b["n"] = b["n"].astype("float64")                       # integers stored as floats
    assert manifest.fingerprint(a)["sha256"] == manifest.fingerprint(b)["sha256"]


def test_fingerprint_catches_a_one_ulp_change_and_says_where():
    a = _frame()
    b = a.copy()
    b.loc[2, "ret"] = np.nextafter(b.loc[2, "ret"], 0)     # the smallest possible change
    fa, fb = manifest.fingerprint(a), manifest.fingerprint(b)
    assert fa["sha256"] != fb["sha256"]
    msg = manifest._diagnose(fa, fb)
    assert "ret" in msg and "floating-point" in msg

    c = a.copy()
    c.loc[0, "vol"] += 0.5
    assert "source data differs" in manifest._diagnose(fa, manifest.fingerprint(c))
    assert "rows" in manifest._diagnose(fa, manifest.fingerprint(a.iloc[:2]))


def test_the_committed_manifest_covers_every_artifact():
    data = json.loads(manifest.MANIFEST.read_text())
    assert set(data["artifacts"]) == set(manifest.ARTIFACTS)
    for fp in data["artifacts"].values():
        assert fp["rows"] > 100_000 and len(fp["sha256"]) == 64
