"""P11 — anytime-valid e-value coverage-break monitor (SPEC §11 P11).

A DIAGNOSTIC (reporting-only, no change to any interval method): an anytime-valid detector
for *when* conditional coverage breaks over the walk-forward stream (e-value conformal, Vovk;
arXiv 2503.13050; "When Your Model Stops Working", 2603.13156). Testing-by-betting: under the
null H0 that per-session miscoverage rate = alpha, the wealth process

    W_t = prod_{s<=t} (1 + lambda_s * (miss_rate_s - alpha))

is a non-negative martingale with E[W_t] <= 1, so by Ville's inequality
P(sup_t W_t >= 1/delta) <= delta. Crossing 1/delta is an anytime-valid alarm that coverage
has dropped below nominal (over-widening never triggers it — the payoff is signed toward
UNDER-coverage). lambda_s is a clamped fraction of the aGRAPA optimal bet (bounded for
numerical stability); it is a betting strategy, not a tuned knob affecting validity.

The alarm should fire inside the drawdown window for a method that breaks (naive split) and
stay silent for a shift-robust method that holds coverage — that discrimination is the P11
demonstration. NOT run directly — imported by run_p11.py.
"""

from __future__ import annotations

import numpy as np

LAMBDA_CAP = 0.5           # cap |bet| well inside [0, 1/(1-alpha)) for stability (frozen)


def evalue_path(miss_rate: np.ndarray, alpha: float = 0.10,
                lam_cap: float = LAMBDA_CAP) -> np.ndarray:
    """Wealth (e-value) path over sessions from per-session miscoverage RATES.

    Payoff each session g_s = miss_rate_s - alpha (>0 under under-coverage). The bet
    lambda_s is set BEFORE seeing session s from the running estimate of the payoff mean
    and second moment (aGRAPA-style, prequential -> the martingale property is preserved),
    clamped to [0, lam_cap/(1-alpha)] so wealth stays non-negative and finite. Returns the
    cumulative-product wealth W_t (W_0 = 1)."""
    g = np.asarray(miss_rate, float) - alpha
    W = np.empty(len(g))
    w = 1.0
    s_sum, s_sq, n = 0.0, 0.0, 0            # running payoff stats (from PAST sessions only)
    hi = lam_cap / max(1.0 - alpha, 1e-6)   # upper bet bound
    for t in range(len(g)):
        mean = s_sum / n if n else 0.0
        var = (s_sq / n - mean * mean) if n else 1.0
        lam = mean / (var + 1e-6) if var > 0 else 0.0    # aGRAPA optimal bet
        lam = float(min(max(lam, 0.0), hi))              # bet only on under-coverage
        w = w * (1.0 + lam * g[t])
        w = max(w, 0.0)
        W[t] = w
        s_sum += g[t]; s_sq += g[t] * g[t]; n += 1       # update AFTER betting (prequential)
    return W


def detect(miss_rate: np.ndarray, alpha: float = 0.10, delta: float = 0.05,
           lam_cap: float = LAMBDA_CAP) -> dict:
    """Run the monitor; return the wealth path, the anytime-valid threshold 1/delta, and
    the first session index where wealth crosses it (-1 = never fired)."""
    W = evalue_path(miss_rate, alpha, lam_cap)
    thresh = 1.0 / delta
    crossed = np.where(W >= thresh)[0]
    return dict(W=W, thresh=thresh, fire_idx=int(crossed[0]) if len(crossed) else -1)
