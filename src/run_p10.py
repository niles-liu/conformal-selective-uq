"""P10 driver — localized / kernel-conditional comparator arm (SPEC §11 P10).

A CONTINUOUS approximate-conditional reference (kernel-reweighted split CQR,
`conformal.localized_cqr`; Guan 2023 / Hore-Barber 2024) slotted into the existing
conditional-coverage table beside the DISCRETE Mondrian groups. No gate redefinition:
the question is whether continuous conditioning *tightens intervals at equal conditional
coverage* vs the Mondrian/static-conditional comparators the repo already has.

Report gates (descriptive — P10 is a comparator, not a ship/reject method):
  * G-LOCAL-VALID   — localized base coverage within +/-tau of nominal (calibrated)
  * G-LOCAL-TIGHTER — localized width < Mondrian width at comparable base worst-group
                      coverage (the continuous-conditioning efficiency question)

Run: python src/run_p10.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import run_p2 as P
from _common import PROC
from conformal import cqr, cqr_conditional, localized_cqr, mondrian_cqr, naive_split
from eval import mean_width, worst_group_coverage
from panel import FEATURE_COLS, GROUP_COLS
from run_p2 import ALPHA, MIN_N, SEEDS, TARGET, TAU, _cov3, _grp, _split, _xy
from run_p22 import _axis_labels
from synthetic import generate

METHODS = ["naive", "cqr", "mondrian", "cqr_conditional", "localized"]


def _build(tr, ca, te):
    Xtr, ytr = _xy(tr, "y"); Xca, yca = _xy(ca, "y"); Xte, yte = _xy(te, "y")
    gca, gte = _grp(ca), _grp(te)
    gac, gat = _axis_labels(ca), _axis_labels(te)
    out = {}
    out["naive"] = naive_split(Xtr, ytr, Xca, yca, Xte, ALPHA)
    out["cqr"] = cqr(Xtr, ytr, Xca, yca, Xte, ALPHA)
    out["mondrian"] = mondrian_cqr(Xtr, ytr, Xca, yca, gca, Xte, gte, ALPHA, MIN_N)
    out["cqr_conditional"] = cqr_conditional(Xtr, ytr, Xca, yca, gac, Xte, gat, ALPHA)
    out["localized"] = localized_cqr(Xtr, ytr, Xca, yca, Xte, ALPHA)
    return out


def _ev(te, iv):
    keep = list(dict.fromkeys(["report_date", "regime", "ticker", *GROUP_COLS]))
    out = te[keep].copy()
    out["y"] = te["y"].to_numpy()
    out["lo"], out["hi"] = iv.lo, iv.hi
    return out.reset_index(drop=True)


def _base_worst(ev):
    """Worst-group coverage in the BASE regime (the conditioning check, calm regime)."""
    ba = ev[ev.regime == "base"]
    return worst_group_coverage(ba, GROUP_COLS, TARGET, MIN_N)[0]


def run_1a():
    print("=" * 92)
    print("P10 — localized (kernel-conditional) vs discrete-group comparators | Layer 1a")
    print("=" * 92)
    import time
    t0 = time.time()
    evs = {m: [] for m in METHODS}
    for s in SEEDS:
        path = PROC / f"synth_1a_seed{s}.parquet"
        if not path.exists():
            generate(s)
        df = pd.read_parquet(path).dropna(subset=FEATURE_COLS + ["y"])
        tr, ca, te = _split(df)
        ivs = _build(tr, ca, te)
        for m in METHODS:
            evs[m].append(_ev(te, ivs[m]))
        print(f"  [progress] seed {s} done @ {time.time()-t0:.0f}s", flush=True)

    print(f"\nCoverage (mean+/-sd over {SEEDS}; target {TARGET:.0%})")
    print(f"{'method':<18}{'base':>14}{'shift':>14}{'worst-grp|shift':>18}"
          f"{'base worst-grp':>16}{'width':>9}")
    W, BW = {}, {}
    for m in METHODS:
        c = np.array([_cov3(e)[:3] for e in evs[m]], float)
        b, sh, wg = c.mean(0); bs, shs, wgs = c.std(0)
        bw = np.mean([_base_worst(e) for e in evs[m]])
        w = float(np.mean([mean_width(e.lo.values, e.hi.values) for e in evs[m]]))
        W[m], BW[m] = w, bw
        print(f"{m:<18}{b:>7.3f}+/-{bs:<5.3f}{sh:>7.3f}+/-{shs:<5.3f}"
              f"{wg:>10.3f}+/-{wgs:<5.3f}{bw:>16.3f}{w:>9.3f}")

    base_cov = np.mean([_cov3(e)[0] for e in evs["localized"]])
    valid = abs(base_cov - TARGET) <= TAU
    print(f"\n[G-LOCAL-VALID] localized base coverage {base_cov:.3f} within +/-{TAU} of "
          f"{TARGET}? -> {'PASS' if valid else 'MISS'}")
    comparable = abs(BW["localized"] - BW["mondrian"]) <= TAU
    tighter = W["localized"] < W["mondrian"]
    print(f"[G-LOCAL-TIGHTER] localized base worst-grp {BW['localized']:.3f} vs mondrian "
          f"{BW['mondrian']:.3f} (comparable +/-{TAU}: {comparable}); width {W['localized']:.3f} "
          f"< mondrian {W['mondrian']:.3f}? -> {'PASS' if (tighter and comparable) else 'MISS'}")
    return evs


def run_1b():
    print("\n" + "=" * 92)
    print("P10 — localized comparator | Layer 1b (real 21d fwd return)")
    print("=" * 92)
    df = P._build_1b()
    tr, ca, te = _split(df)
    ivs = _build(tr, ca, te)
    print(f"{'method':<18}{'base':>8}{'shift':>8}{'worst-grp|shift':>20}{'base worst-grp':>16}{'width':>9}")
    for m in METHODS:
        ev = _ev(te, ivs[m]); b, sh, wg, dim, lab = _cov3(ev)
        print(f"{m:<18}{b:>8.3f}{sh:>8.3f}{wg:>12.3f} ({dim[:8]}) {_base_worst(ev):>10.3f}"
              f"{mean_width(ev.lo.values, ev.hi.values):>9.3f}")


if __name__ == "__main__":
    run_1a()
    run_1b()
