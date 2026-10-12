"""P1 baseline driver — the method-blind baseline read (SPEC §7, P1).

Establishes, on the public substrate, (a) that naive split-conformal holds MARGINAL
coverage but FAILS conditional/per-regime coverage under the documented shift, and
(b) the baseline bar (naive split, unconditional CQR, random + oracle abstention).
From baseline-only statistics it computes the three data-dependent gate margins
(Delta, X, the G-SHIFT naive-failure gap) frozen into the pre-registration.

No method here — selective/group-conditional/shift-robust layers are P2.

Splits (walk-forward, blocked over time; report_date = YYYYMMDD int):
  TRAIN  rd <  20210101   (fit base + quantile learners)
  CALIB  20210101..20211231  (calm, PRE-shift conformal calibration)
  TEST   rd >= 20220101   (2022 = amplified-overlay shift regime; 2023-24 = base)

Run: python experiments/run_p1.py            (1a seeds {0,1,2} + 1b)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import PROC, RETURNS
from panel import FEATURE_COLS, GROUP_COLS, PANEL
from selective_conformal.conformal import cqr, naive_split
from selective_conformal.metrics import clustered_bootstrap, coverage, mean_width, risk_coverage, worst_group_coverage
from synthetic import generate

ALPHA = 0.10
TARGET = 1 - ALPHA
TRAIN_END, CALIB_END = 20210101, 20220101
SHIFT_LO, SHIFT_HI = 20220101, 20221231
SEEDS = [0, 1, 2]
RETENTIONS = np.round(np.arange(0.3, 1.01, 0.1), 2)
X_RET = 0.70  # frozen headline retention


def _split(df: pd.DataFrame):
    tr = df[df.report_date < TRAIN_END]
    ca = df[(df.report_date >= TRAIN_END) & (df.report_date < CALIB_END)]
    te = df[df.report_date >= CALIB_END]
    return tr, ca, te


def _xy(df: pd.DataFrame, target: str):
    return df[FEATURE_COLS].to_numpy(), df[target].to_numpy()


def _eval_intervals(te: pd.DataFrame, iv, target: str) -> pd.DataFrame:
    keep = list(dict.fromkeys(["report_date", "regime", *GROUP_COLS]))
    out = te[keep].copy()
    out["y"] = te[target].to_numpy()
    out["lo"], out["hi"], out["pred"] = iv.lo, iv.hi, iv.pred
    out["width"] = iv.width
    return out.reset_index(drop=True)


def _one_seed_1a(seed: int) -> dict:
    path = PROC / f"synth_1a_seed{seed}.parquet"
    if not path.exists():
        generate(seed)
    df = pd.read_parquet(path).dropna(subset=FEATURE_COLS + ["y"])
    tr, ca, te = _split(df)
    Xtr, ytr = _xy(tr, "y")
    Xca, yca = _xy(ca, "y")
    Xte, _ = _xy(te, "y")

    res = {}
    for name, fn in [("naive", naive_split), ("cqr", cqr)]:
        iv = fn(Xtr, ytr, Xca, yca, Xte, ALPHA)
        ev = _eval_intervals(te, iv, "y")
        shift = ev[ev.regime == "shift"]
        base = ev[ev.regime == "base"]
        wg_shift, wdim, wlab = worst_group_coverage(shift, GROUP_COLS, TARGET)
        res[name] = dict(
            marg=coverage(ev.y.values, ev.lo.values, ev.hi.values),
            cov_base=coverage(base.y.values, base.lo.values, base.hi.values),
            cov_shift=coverage(shift.y.values, shift.lo.values, shift.hi.values),
            width=mean_width(ev.lo.values, ev.hi.values),
            wg_shift=wg_shift, wg_dim=wdim, wg_lab=wlab,
            ev=ev,
        )

    # risk-coverage on the mean model (shared); oracle uses true irreducible_sigma
    pred = res["naive"]["ev"]["pred"].to_numpy()
    y = res["naive"]["ev"]["y"].to_numpy()
    isig = te["irreducible_sigma"].to_numpy()
    width = res["cqr"]["ev"]["width"].to_numpy()
    rng = np.random.default_rng(seed)
    rand_curves = [risk_coverage(y, pred, rng.standard_normal(len(y)), RETENTIONS)
                   for _ in range(20)]
    rc_random = {r: float(np.mean([c[r] for c in rand_curves])) for r in RETENTIONS}
    rc_oracle = risk_coverage(y, pred, isig, RETENTIONS)
    rc_width = risk_coverage(y, pred, width, RETENTIONS)
    res["rc"] = dict(random=rc_random, oracle=rc_oracle, width=rc_width)
    return res


def _band(ev: pd.DataFrame, stat_fn) -> str:
    p, lo, hi = clustered_bootstrap(ev.reset_index(drop=True), stat_fn, n_boot=300)
    return f"{p:.3f} [{lo:.3f},{hi:.3f}]"


def run_1a():
    print("=" * 74)
    print("LAYER 1a — known-ceiling synthetic | baseline read")
    print("=" * 74)
    per = [_one_seed_1a(s) for s in SEEDS]

    for name in ["naive", "cqr"]:
        marg = np.array([p[name]["marg"] for p in per])
        cb = np.array([p[name]["cov_base"] for p in per])
        cs = np.array([p[name]["cov_shift"] for p in per])
        wg = np.array([p[name]["wg_shift"] for p in per])
        w = np.array([p[name]["width"] for p in per])
        print(f"\n[{name}]  (mean +/- sd over seeds {SEEDS}; target {TARGET:.0%})")
        print(f"  marginal coverage    : {marg.mean():.3f} +/- {marg.std():.3f}")
        print(f"  coverage | base reg  : {cb.mean():.3f} +/- {cb.std():.3f}")
        print(f"  coverage | SHIFT reg : {cs.mean():.3f} +/- {cs.std():.3f}   <-- "
              f"conditional failure" if cs.mean() < TARGET - 0.02 else "")
        print(f"  WORST-GROUP | shift  : {wg.mean():.3f} +/- {wg.std():.3f}  "
              f"(e.g. {per[0][name]['wg_dim']}={per[0][name]['wg_lab']})")
        print(f"  mean interval width  : {w.mean():.3f} +/- {w.std():.3f}")
        # clustered band on seed-0 marginal + shift for the headline
        ev0 = per[0][name]["ev"]
        sh0 = ev0[ev0.regime == "shift"].reset_index(drop=True)
        print(f"  [seed0 clustered band] marginal "
              f"{_band(ev0, lambda d: coverage(d.y.values, d.lo.values, d.hi.values))} | "
              f"shift {_band(sh0, lambda d: coverage(d.y.values, d.lo.values, d.hi.values))}")

    # risk-coverage @ X_RET, averaged over seeds
    print(f"\n[risk-coverage] retained MAE @ retention {X_RET:.0%} "
          f"(mean over seeds):")
    rr = np.mean([p["rc"]["random"][X_RET] for p in per])
    ro = np.mean([p["rc"]["oracle"][X_RET] for p in per])
    rw = np.mean([p["rc"]["width"][X_RET] for p in per])
    full = np.mean([p["rc"]["random"][1.0] for p in per])
    print(f"  full (no abstention) : {full:.4f}")
    print(f"  random abstention    : {rr:.4f}")
    print(f"  CQR-width abstention : {rw:.4f}  (proto-method, informative)")
    print(f"  ORACLE abstention    : {ro:.4f}  (knows irreducible_sigma)")
    print("  full risk-coverage curve (oracle):")
    for r in RETENTIONS:
        print(f"     ret {r:.0%}: random {np.mean([p['rc']['random'][r] for p in per]):.4f} | "
              f"oracle {np.mean([p['rc']['oracle'][r] for p in per]):.4f} | "
              f"width {np.mean([p['rc']['width'][r] for p in per]):.4f}")

    # ---- the three data-dependent margins (frozen from baseline-only stats) ----
    delta = 0.5 * (rr - ro)
    naive_wg = np.mean([p["naive"]["wg_shift"] for p in per])
    shift_gap = TARGET - naive_wg
    print("\n" + "-" * 74)
    print("FROZEN MARGINS (from method-blind baseline-only read):")
    print(f"  X (retention)            : {X_RET:.0%}  (a priori)")
    print(f"  Delta (G-PRIMARY i)      : {delta:.4f}  "
          f"= 0.5*(random {rr:.4f} - oracle {ro:.4f}) @ {X_RET:.0%}")
    print(f"  G-SHIFT naive worst-group: {naive_wg:.3f}  -> failure gap "
          f"{shift_gap:.3f}; method must close >=50% (to a gap <= {shift_gap*0.5:.3f}, "
          f"i.e. worst-group >= {naive_wg + shift_gap*0.5:.3f})")
    print("-" * 74)
    return dict(delta=delta, naive_wg=naive_wg, shift_gap=shift_gap, full=full,
                rr=rr, ro=ro, rw=rw)


def _build_1b() -> pd.DataFrame:
    """Real public task: 21-session forward LOG return as the target, on the same
    PIT panel. No recoverability mask (G-MECHANISM N/A)."""
    panel = pd.read_parquet(PANEL)
    r = pd.read_parquet(RETURNS).pivot(index="date", columns="ticker", values="ret")
    L = np.log1p(r.clip(lower=-0.95))
    fwd = L.rolling(21).sum().shift(-21)  # sum of next 21 daily log returns
    fwd = fwd.stack().rename("y").reset_index()
    fwd.columns = ["date", "ticker", "y"]
    df = panel.merge(fwd, on=["date", "ticker"], how="inner").dropna(
        subset=FEATURE_COLS + ["y"])
    # regime label from report_date (mirror 1a)
    df["regime"] = np.where(df.report_date.between(SHIFT_LO, SHIFT_HI), "shift", "base")
    return df


def run_1b():
    print("\n" + "=" * 74)
    print("LAYER 1b — real public task (21d fwd return) | baseline read")
    print("=" * 74)
    df = _build_1b()
    tr, ca, te = _split(df)
    Xtr, ytr = _xy(tr, "y")
    Xca, yca = _xy(ca, "y")
    Xte, _ = _xy(te, "y")
    for name, fn in [("naive", naive_split), ("cqr", cqr)]:
        iv = fn(Xtr, ytr, Xca, yca, Xte, ALPHA)
        ev = _eval_intervals(te, iv, "y")
        shift = ev[ev.regime == "shift"]
        base = ev[ev.regime == "base"]
        wg, wdim, wlab = worst_group_coverage(shift, GROUP_COLS, TARGET)
        print(f"\n[{name}]  marginal {coverage(ev.y.values, ev.lo.values, ev.hi.values):.3f} "
              f"| base {coverage(base.y.values, base.lo.values, base.hi.values):.3f} "
              f"| SHIFT {coverage(shift.y.values, shift.lo.values, shift.hi.values):.3f} "
              f"| worst-grp(shift) {wg:.3f} ({wdim}={wlab}) "
              f"| width {mean_width(ev.lo.values, ev.hi.values):.3f}")


if __name__ == "__main__":
    margins = run_1a()
    run_1b()
