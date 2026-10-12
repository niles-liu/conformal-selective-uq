"""Selective conformal RISK control in regression (SPEC §11 P9 — the "open variant" §6 named).

SCRC (Selective Conformal Risk Control, arXiv 2512.12844) is *select-then-calibrate*
with two lambdas and a monotone risk — but published for **classification**. §6 flagged
select-then-calibrate in **regression** as the open variant; this module builds it.

Two knobs (both frozen in reports/p9_riskcontrol_preregistration.md):
  * lambda_sel  — the SELECTION threshold: retain the fraction r of names with the
    lowest per-name uncertainty u_i (the existing `selective.name_uncertainty` score).
    u_i is built from calibration-fold residuals, so selection uses calibration labels and
    retained-pool risk control is empirical, not a finite-sample guarantee.
  * lambda_risk — the CRC radius applied to the CQR base interval on the RETAINED pool,
    chosen by the Conformal Risk Control bound (Angelopoulos-Bates-Fisch-Lei-Schuster
    2023) *on the retained calibration fold*, aiming for retained-test risk <= target.

The risk is a **monotone, bounded** loss (this file defines it; the pre-reg freezes the
target). Two losses, both non-increasing in the radius lambda and bounded by B=1:
  * miscoverage        L = 1{y not in C_lambda(x)}       (recovers selective coverage)
  * capped miss-magnitude  L = clip(dist_outside(y, C_lambda)/W0, 0, 1)   (a proper risk,
    the P9 upgrade beyond 0/1 coverage — penalises HOW FAR outside, not just whether)

Why calibrate on the *selected* fold (the SCRC point): selection by u_i keeps the
low-noise names, whose true residuals are smaller than the pooled calibration set's.
A single global conformal radius (calibrated on the FULL fold) therefore over-covers
the retained pool and wastes width; the select-then-calibrate radius aims to be tighter at
the same target risk. Under the documented shift the static radius is not enough — the
online `scrc_aci` variant carries a risk-controlled radius over the session stream.

NOT run directly — imported by run_p9.py.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Monotone, bounded losses (risk = mean loss). Interval C_lambda(x) is the CQR   #
# base [qlo - lambda, qhi + lambda]; larger lambda => wider => smaller loss.      #
# --------------------------------------------------------------------------- #
def dist_outside(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Signed-zero distance of y OUTSIDE [lo, hi] (0 if inside), >= 0."""
    return np.maximum.reduce([lo - y, y - hi, np.zeros_like(y)])


def loss_miscoverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """0/1 miscoverage loss (B=1). Monotone non-increasing in a widening radius."""
    return ((y < lo) | (y > hi)).astype(float)


def loss_miss_magnitude(y: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                        w0: float) -> np.ndarray:
    """Capped fractional miss-magnitude loss (B=1): clip(dist_outside / w0, 0, 1).
    w0 is a FROZEN scale (median base-regime CQR width) so the loss is unit-free and
    bounded. A proper risk (how far outside), the P9 upgrade beyond 0/1 coverage."""
    return np.clip(dist_outside(y, lo, hi) / max(w0, 1e-9), 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Conformal Risk Control radius (Angelopoulos-Bates-Fisch-Lei-Schuster 2023)   #
# --------------------------------------------------------------------------- #
def crc_radius(qlo_cal: np.ndarray, qhi_cal: np.ndarray, ycal: np.ndarray,
               loss_fn, alpha: float, grid: np.ndarray, B: float = 1.0) -> float:
    """Smallest radius lambda in `grid` whose CRC upper bound on the calibration risk
    is <= alpha:  (n/(n+1)) * Rhat(lambda) + B/(n+1) <= alpha,  with Rhat the mean
    calibration loss at radius lambda. The loss must be non-increasing in lambda and
    bounded by B, so Rhat is non-increasing and the smallest passing lambda is the
    tightest interval with the guarantee. Returns grid[-1] (widest) if none pass.

    loss_fn(y, lo, hi) -> per-point loss; intervals are [qlo - lambda, qhi + lambda]."""
    n = len(ycal)
    grid = np.sort(np.asarray(grid, float))
    for lam in grid:
        rhat = float(np.mean(loss_fn(ycal, qlo_cal - lam, qhi_cal + lam)))
        if (n / (n + 1)) * rhat + B / (n + 1) <= alpha:
            return float(lam)
    return float(grid[-1])


# --------------------------------------------------------------------------- #
# Two-lambda select-then-calibrate (SCRC in regression, SPEC §11 P9)          #
# --------------------------------------------------------------------------- #
def _retain_masks(score_ca, score_te, r):
    """Selection by a shared threshold tau = the r-quantile of the TEST score
    (retention r on the test stream). Names with score <= tau are retained in BOTH
    folds -> exchangeability within the retained pool (covariate-side rule)."""
    tau = float(np.quantile(np.asarray(score_te, float), r))
    return (np.asarray(score_ca, float) <= tau,
            np.asarray(score_te, float) <= tau, tau)


def scrc(qlo_ca, qhi_ca, yca, qlo_te, qhi_te,
         score_ca, score_te, r, loss_fn, alpha_risk, grid, B=1.0):
    """P9 batch — select-then-calibrate. Select the lowest-score r fraction, then run
    Conformal Risk Control on the RETAINED calibration fold for the radius lambda_risk;
    emit [qlo - lambda, qhi + lambda] for the retained test pool. Returns dict with the
    full-test lo/hi (radius applied to all; evaluate risk on `keep` only), keep mask,
    the chosen lambda, and the selection threshold tau."""
    ret_ca, ret_te, tau = _retain_masks(score_ca, score_te, r)
    lam = crc_radius(np.asarray(qlo_ca)[ret_ca], np.asarray(qhi_ca)[ret_ca],
                     np.asarray(yca)[ret_ca], loss_fn, alpha_risk, grid, B)
    return dict(lo=np.asarray(qlo_te) - lam, hi=np.asarray(qhi_te) + lam,
                keep=ret_te, lam=lam, tau=tau)


def scrc_aci(qlo_ca, qhi_ca, yca, qlo_te, qhi_te, yte, te_order,
             score_ca, score_te, r, loss_fn, alpha_risk, grid, eta, B=1.0):
    """P9 online — the shift-robust SCRC. Warm-starts at the batch CRC radius lambda0
    (from the retained calib fold), then carries a risk-controlled radius over the
    time-ordered sessions: after each session reveals its retained points' loss, widen
    when the realised session risk exceeds target,
        lambda <- max( lambda + eta * (Rhat_sess - alpha_risk), 0 ).
    Only sessions < s inform session s's radius; y_s revealed after emission -> PIT.
    Returns the same dict shape as `scrc` (plus the per-session lambda path for the
    mechanism check). Risk is evaluated on `keep` (retained) rows only."""
    ret_ca, ret_te, tau = _retain_masks(score_ca, score_te, r)
    lam0 = crc_radius(np.asarray(qlo_ca)[ret_ca], np.asarray(qhi_ca)[ret_ca],
                      np.asarray(yca)[ret_ca], loss_fn, alpha_risk, grid, B)
    qlo_te, qhi_te = np.asarray(qlo_te, float), np.asarray(qhi_te, float)
    yte, te_order = np.asarray(yte, float), np.asarray(te_order)
    lo, hi = np.empty(len(qlo_te)), np.empty(len(qhi_te))
    lam = lam0
    path = []
    for s in np.unique(te_order):                 # ascending sessions: the online stream
        rows = np.where(te_order == s)[0]
        lo[rows] = qlo_te[rows] - lam
        hi[rows] = qhi_te[rows] + lam
        path.append((int(s), float(lam)))
        sel = rows[ret_te[rows]]                   # retained rows in this session
        if len(sel):                               # update from the retained loss only
            rhat = float(np.mean(loss_fn(yte[sel], lo[sel], hi[sel])))
            lam = max(lam + eta * (rhat - alpha_risk), 0.0)
    return dict(lo=lo, hi=hi, keep=ret_te, lam=lam0, tau=tau, lam_path=path)
