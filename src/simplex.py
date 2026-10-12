"""P8 — conformal on the simplex (the target is compositional) — SPEC §11 P8, highest leverage.

The 1a/real target is an ALLOCATION VECTOR (weights >= 0, sum to 1 per session — a point on
the simplex), but the P1-P9 pipeline certifies each weight as an INDEPENDENT scalar interval.
N marginal-90% intervals have joint coverage ~0.90^N ~ 0 for N~90 names — so the pipeline
cannot certify the allocation *as a vector* at all. P8 builds a JOINT conformal region on the
simplex whose coverage = "the true allocation vector lies in the region", closing exactly the
gap arXiv 2511.18141 (simplex conformal) leaves open: selection x shift x known-floor mechanism.

Region construction. A scalar compositional nonconformity score S(w_true, p_hat) that respects
the simplex geometry; the region is the ball {v : S(v, p_hat) <= Q}, Q the conformal quantile
of the calibration scores. Two frozen scores (both scalar -> the region is a metric ball, NOT a
grid; grid-refined HDR is infeasible at ~90 dims, so the ball is the laptop-tractable realization
of "one region on the simplex"):
  * Aitchison (primary, canonical simplex geometry) — Euclidean distance in centred-log-ratio
    space, with standard MULTIPLICATIVE ZERO-REPLACEMENT (eps floor + re-closure) for the tiny
    parts (CoDA textbook fix).
  * total variation (companion, bounded/interpretable) — 0.5 * sum |w - p_hat|, the half-L1
    allocation error, robust to tiny parts.

Predicted composition p_hat = softmax over the session of (base_logit + yhat), base_logit
reconstructed exactly from the public size feature (0.5 * z_date(log_dollar_vol), the generator's
base). NOT run directly — imported by run_p8.py.
"""

from __future__ import annotations

import numpy as np

from conformal import _conf_quantile

COMPO_EPS = 1e-6          # multiplicative zero-replacement floor (frozen, CoDA standard)
BASE_COEF = 0.5           # base_logit = BASE_COEF * z_date(log_dollar_vol) (synthetic.py)


# --------------------------------------------------------------------------- #
# compositions                                                                #
# --------------------------------------------------------------------------- #
def close(w: np.ndarray) -> np.ndarray:
    """Closure: renormalise a non-negative vector to sum 1."""
    w = np.asarray(w, float)
    s = w.sum()
    return w / s if s > 0 else np.full_like(w, 1.0 / len(w))


def replace_close(w: np.ndarray, eps: float = COMPO_EPS) -> np.ndarray:
    """Multiplicative zero-replacement then closure (handles tiny/zero parts for CLR)."""
    w = np.maximum(np.asarray(w, float), eps)
    return close(w)


def clr(w: np.ndarray, eps: float = COMPO_EPS) -> np.ndarray:
    """Centred log-ratio of one composition (after zero-replacement)."""
    w = replace_close(w, eps)
    lw = np.log(w)
    return lw - lw.mean()


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(np.asarray(x, float) - np.max(x))
    return e / e.sum()


def aitchison(a: np.ndarray, b: np.ndarray, eps: float = COMPO_EPS) -> float:
    """Aitchison distance between two compositions = Euclidean distance of their CLRs."""
    return float(np.linalg.norm(clr(a, eps) - clr(b, eps)))


def tv(a: np.ndarray, b: np.ndarray) -> float:
    """Total-variation (half-L1) distance between two compositions, in [0, 1]."""
    return float(0.5 * np.sum(np.abs(close(a) - close(b))))


def clr_dev(a: np.ndarray, b: np.ndarray, eps: float = COMPO_EPS) -> np.ndarray:
    """Per-coordinate |CLR(a) - CLR(b)| — a name's contribution to the compositional
    nonconformity (the per-name compositional difficulty for the mechanism check)."""
    return np.abs(clr(a, eps) - clr(b, eps))


# --------------------------------------------------------------------------- #
# joint conformal region on the simplex (scalar score -> metric ball)         #
# --------------------------------------------------------------------------- #
def region_radius(cal_scores: np.ndarray, alpha: float = 0.10) -> float:
    """Static conformal radius Q = finite-sample (1-alpha) quantile of calib scores."""
    return _conf_quantile(np.asarray(cal_scores, float), alpha)


def aci_radius_path(cal_scores: np.ndarray, test_scores_in_order: np.ndarray,
                    alpha: float = 0.10, gamma: float = 0.05):
    """Online ACI on the SCALAR compositional score over time-ordered sessions (one region
    per session). Radius Q_s = (1-alpha_eff) empirical quantile of the fixed calib scores;
    after session s reveals whether the true composition fell in the region, update
    alpha_eff += gamma*(alpha - miss_s). Returns (covered[bool per session], radii). The
    session's coverage is revealed only after its region is emitted -> anytime/PIT-valid."""
    cal = np.sort(np.asarray(cal_scores, float))
    ts = np.asarray(test_scores_in_order, float)
    covered = np.empty(len(ts), bool)
    radii = np.empty(len(ts))
    a_eff = alpha
    for i, s in enumerate(ts):
        q = _emp_q(cal, a_eff)
        radii[i] = q
        miss = 1.0 if s > q else 0.0
        covered[i] = not miss
        a_eff = min(max(a_eff + gamma * (alpha - miss), 1e-3), 0.999)
    return covered, radii


def _emp_q(sorted_scores: np.ndarray, alpha_eff: float) -> float:
    """(1-alpha_eff) empirical quantile of pre-sorted scores, clamped."""
    n = len(sorted_scores)
    if alpha_eff <= 0:
        return float(sorted_scores[-1])
    if alpha_eff >= 1:
        return 0.0
    k = int(np.ceil((n + 1) * (1 - alpha_eff)))
    k = min(max(k, 1), n)
    return float(sorted_scores[k - 1])
