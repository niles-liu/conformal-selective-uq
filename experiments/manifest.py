"""Data manifest: checksums of the built data, so "reproduces exactly" can be checked.

The Yahoo data is not committed (its terms of use), so a fresh clone downloads it
again. Yahoo can revise history (a late split or dividend correction, a dropped bar),
and a different pandas or numpy can change the last bits of a rolling statistic.
Either would move the published numbers, so this module fingerprints each built table
and compares it with the fingerprints of the data behind the published results
(experiments/data_manifest.json).

A fingerprint does not depend on the parquet writer: rows in stored order, columns
sorted by name, numbers as float64 bytes, timestamps as UTC nanoseconds, text as
UTF-8. Per-column sums and absolute sums ride along, so a mismatch can say which
columns moved and by how much.

An example of why this matters: between the June and the October 2026 downloads Yahoo
revised Honeywell's (HON) price history by about 1e-7 relative; every other name was
identical. That is below anything the printed results can show (all of them still
matched), and the check reports it as tiny. A larger change would say so.

    python experiments/manifest.py            # check the cached data against the manifest
    python experiments/manifest.py --write    # maintainers: record the current data

panel.py runs the check after every build and prints the verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import PANEL, PRICES, PROC, PULL_END, RETURNS, WIN_LO

MANIFEST = Path(__file__).resolve().with_name("data_manifest.json")
ARTIFACTS: dict[str, Path] = {
    "prices": PRICES,
    "returns": RETURNS,
    "panel": PANEL,
    **{f"synth_1a_seed{s}": PROC / f"synth_1a_seed{s}.parquet" for s in (0, 1, 2)},
    **{f"synth_1b_seed{s}": PROC / f"synth_1b_seed{s}.parquet" for s in range(6)},
}
TINY = 1e-6   # relative change (of a column's absolute sum) below which a difference is float noise


def fingerprint(df: pd.DataFrame) -> dict:
    """Rows, sorted column names, a SHA-256 over a canonical rendering, numeric sums and
    absolute sums (the scale a change is measured against)."""
    h = hashlib.sha256()
    sums: dict[str, float] = {}
    scales: dict[str, float] = {}
    cols = sorted(df.columns)
    for c in cols:
        s = df[c]
        h.update(c.encode() + b"\x00")
        if pd.api.types.is_datetime64_any_dtype(s):
            ts = pd.to_datetime(s, utc=True).dt.as_unit("ns")
            h.update(ts.astype("int64").to_numpy().astype("<i8").tobytes())
        elif pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            a = s.to_numpy(dtype="float64", na_value=np.nan)
            a = np.where(np.isnan(a), np.nan, a)          # one NaN bit pattern
            h.update(a.astype("<f8").tobytes())
            sums[c] = float(np.nansum(a))
            scales[c] = float(np.nansum(np.abs(a)))
        else:
            text = "\x1f".join("\x00" if pd.isna(v) else str(v) for v in s)
            h.update(text.encode("utf-8"))
    return {"rows": int(len(df)), "columns": cols, "sha256": h.hexdigest(), "sums": sums, "scales": scales}


def _diagnose(want: dict, got: dict) -> str:
    if want["rows"] != got["rows"]:
        return f"{got['rows']:,} rows, the published data has {want['rows']:,}"
    if want["columns"] != got["columns"]:
        return f"columns differ: {sorted(set(want['columns']) ^ set(got['columns']))}"
    moved = []
    for c, w in want["sums"].items():
        g = got["sums"].get(c, np.nan)
        if g != w:
            moved.append((abs(g - w) / max(want["scales"][c], 1e-12), c))
    if not moved:
        return "same rows, columns and sums, but the values differ (row order or text)"
    worst = ", ".join(f"{c} ({r:.0e})" for r, c in sorted(moved, reverse=True)[:3])
    kind = (f"tiny (below {TINY:.0e}): floating-point noise from Yahoo or library versions"
            if max(r for r, _ in moved) < TINY else "larger than float noise: the source data differs")
    return f"values differ in {len(moved)} column(s), largest relative change {worst}; {kind}"


def check(names: list[str] | None = None) -> bool:
    """Compare each built artifact with the manifest. Prints one line per artifact and
    returns True when every built one matches (artifacts not built yet are skipped)."""
    manifest = json.loads(MANIFEST.read_text())["artifacts"]
    ok = True
    for name in names or list(ARTIFACTS):
        path = ARTIFACTS[name]
        if not path.exists():
            print(f"  {name:15s} not built yet")
            continue
        want, got = manifest[name], fingerprint(pd.read_parquet(path))
        if got["sha256"] == want["sha256"]:
            print(f"  {name:15s} matches the published data ({got['rows']:,} rows)")
        else:
            ok = False
            print(f"  {name:15s} DIFFERS: {_diagnose(want, got)}")
    if not ok:
        print("  WARNING: the data differs from the data behind the published results, so the\n"
              "  printed numbers will not match the results files exactly (see README, Reproduce).")
    return ok


def write() -> None:
    import numpy
    import pyarrow
    import yfinance

    missing = [n for n, p in ARTIFACTS.items() if not p.exists()]
    if missing:
        raise SystemExit(f"build these first (panel.py, synthetic.py, synthetic_regime.py): {missing}")
    out = {
        "note": "Fingerprints of the data behind the published results; see experiments/manifest.py.",
        "window": {"first_day": WIN_LO, "end_exclusive": PULL_END},
        "versions": {"pandas": pd.__version__, "numpy": numpy.__version__,
                     "pyarrow": pyarrow.__version__, "yfinance": yfinance.__version__},
        "artifacts": {n: fingerprint(pd.read_parquet(p)) for n, p in ARTIFACTS.items()},
    }
    MANIFEST.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {MANIFEST.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Check the cached data against the published fingerprints.")
    ap.add_argument("--write", action="store_true", help="record the current data as the reference")
    args = ap.parse_args()
    if args.write:
        write()
    else:
        print("=== data manifest ===")
        raise SystemExit(0 if check() else 1)
