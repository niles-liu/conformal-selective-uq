"""P11 driver — anytime-valid e-value coverage-break monitor (SPEC §11 P11, lowest priority).

A DIAGNOSTIC companion to the G-SHIFT story: an anytime-valid detector (`evalue_monitor`,
testing-by-betting / Ville) for *when* conditional coverage breaks over the walk-forward
session stream. Demonstration (reporting-only, no interval method changed):
  * naive split-conformal over the full test stream (2022 drawdown first, then 2023-24 base)
    -> the e-value should CROSS 1/delta inside the drawdown (detects the break);
  * a shift-robust method (group-conditional ACI) over the same stream -> should NOT cross
    (coverage holds, no alarm);
  * naive over the BASE-ONLY stream (2023-24) -> should NOT cross (no false alarm under
    nominal coverage).

Run: python experiments/run_p11.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import PROC
from panel import FEATURE_COLS
from run_p2 import ALPHA, SEEDS, SHIFT_HI, SHIFT_LO, _intervals, _split
from selective_conformal.evalue_monitor import detect
from synthetic import generate

DELTA = 0.05                 # anytime-valid level -> alarm threshold 1/delta = 20 (frozen)


def _session_missrate(ev: pd.DataFrame) -> pd.DataFrame:
    """Per-session miscoverage rate in date order, with the session's regime + date."""
    ev = ev.copy()
    ev["miss"] = ~((ev["y"].to_numpy() >= ev["lo"].to_numpy())
                   & (ev["y"].to_numpy() <= ev["hi"].to_numpy()))
    g = (ev.groupby("report_date")
           .agg(miss_rate=("miss", "mean"),
                regime=("regime", "first")).reset_index().sort_values("report_date"))
    return g


def _fire_report(g: pd.DataFrame, label: str):
    """Returns (fired: bool, in_shift: bool, idx: int)."""
    res = detect(g["miss_rate"].to_numpy(), ALPHA, DELTA)
    idx = res["fire_idx"]
    if idx < 0:
        print(f"  {label:<34} no alarm (max e-value {res['W'].max():.1f} < {res['thresh']:.0f})")
        return False, False, -1
    date = int(g["report_date"].iloc[idx]); reg = g["regime"].iloc[idx]
    in_shift = SHIFT_LO <= date <= SHIFT_HI
    print(f"  {label:<34} FIRES at session {idx} (date {date}, regime={reg}) "
          f"e-value {res['W'][idx]:.1f} — {'in shift window' if in_shift else 'OUTSIDE shift'}")
    return True, in_shift, idx


def run_1a():
    print("=" * 92)
    print("P11 — e-value coverage-break monitor | Layer 1a "
          f"(alpha={ALPHA}, delta={DELTA}, threshold {1/DELTA:.0f})")
    print("=" * 92)
    naive_shift, naive_idx, aci_fired, aci_idx, false_alarm = [], [], [], [], []
    for s in SEEDS:
        path = PROC / f"synth_1a_seed{s}.parquet"
        if not path.exists():
            generate(s)
        df = pd.read_parquet(path).dropna(subset=FEATURE_COLS + ["y"])
        tr, ca, te = _split(df)
        ivs = _intervals(tr, ca, te, "y")
        print(f"\nseed {s}:")
        gn = _session_missrate(ivs["naive"])
        ga = _session_missrate(ivs["aci"])
        nf, nsh, nidx = _fire_report(gn, "naive / full stream")
        af, _, aidx = _fire_report(ga, "group-ACI / full stream")
        # base-only stream (no shift) -> false-alarm control
        gnb = gn[gn.regime == "base"].reset_index(drop=True)
        fa, _, _ = _fire_report(gnb, "naive / BASE-ONLY (control)")
        naive_shift.append(nf and nsh); naive_idx.append(nidx if nf else -1)
        aci_fired.append(af); aci_idx.append(aidx if af else -1)
        false_alarm.append(fa)

    print("\n" + "-" * 92)
    nmed = np.median([i for i in naive_idx if i >= 0]) if any(i >= 0 for i in naive_idx) else -1
    amed = np.median([i for i in aci_idx if i >= 0]) if any(i >= 0 for i in aci_idx) else -1
    detect_ok = all(naive_shift) and (nmed < amed or amed < 0)
    print(f"[G-DETECT] naive fires in-shift {sum(naive_shift)}/{len(SEEDS)} "
          f"(median session {nmed:.0f}); group-ACI fires {sum(aci_fired)}/{len(SEEDS)} "
          f"(median session {amed:.0f}) -> {'PASS (naive detected earlier)' if detect_ok else 'CHECK'}")
    print(f"[G-NOFALSE] naive fires on base-only control {sum(false_alarm)}/{len(SEEDS)} -> "
          f"{'PASS (no false alarm)' if not any(false_alarm) else 'CHECK (false alarm; lam_cap frozen)'}")


if __name__ == "__main__":
    run_1a()
