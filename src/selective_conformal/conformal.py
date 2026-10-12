"""Conformal interval constructors — baselines for P1 (SPEC §6).

Two split-conformal layers, each returning lower/upper interval bands on a test set
calibrated on a held-out calibration fold (never the training labels, never the test
labels). The finite-sample conformal quantile level is the standard
`ceil((n+1)(1-alpha)) / n` (Vovk / Romano), so the marginal guarantee is exact under
exchangeability — the point of P1 is to show that conditional/regime coverage is NOT,
under the documented shift.

P1 ships #1 (naive split) and #2 (CQR, unconditional). P2 adds the method layers
(#3 Mondrian / group-conditional, #4 group-conditional ACI for shift, #5 weighted /
covariate-shift). All build on the same CQR conformity score E = max(qlo-y, y-qhi).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .learners import LightGBMLearner

_DEFAULT_LEARNER = LightGBMLearner()


def _lrn(learner):
    """The learner to fit with: the caller's, else the frozen LightGBM default."""
    return _DEFAULT_LEARNER if learner is None else learner


def _conf_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile of nonconformity scores at level 1-alpha."""
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)  # clamp; if k>n the band is +inf, we cap at the max
    return float(np.sort(scores)[k - 1])


@dataclass
class Intervals:
    pred: np.ndarray  # point prediction (mean model)
    lo: np.ndarray
    hi: np.ndarray

    @property
    def width(self) -> np.ndarray:
        return self.hi - self.lo


def naive_split(
    Xtr: np.ndarray, ytr: np.ndarray,
    Xcal: np.ndarray, ycal: np.ndarray,
    Xte: np.ndarray, alpha: float = 0.10, *, learner=None,
) -> Intervals:
    """Baseline #1 — naive split-conformal (marginal). Nonconformity = |y - yhat|;
    symmetric interval yhat +/- q."""
    mean = _lrn(learner).fit_mean(Xtr, ytr)
    resid = np.abs(ycal - mean.predict(Xcal))
    q = _conf_quantile(resid, alpha)
    pred = mean.predict(Xte)
    return Intervals(pred=pred, lo=pred - q, hi=pred + q)


def cqr(
    Xtr: np.ndarray, ytr: np.ndarray,
    Xcal: np.ndarray, ycal: np.ndarray,
    Xte: np.ndarray, alpha: float = 0.10, *, learner=None,
) -> Intervals:
    """Baseline #2 / method body — conformalized quantile regression (Romano 2019).
    Conformity E = max(qlo - y, y - qhi); interval [qlo - Q, qhi + Q]."""
    lrn = _lrn(learner)
    mean = lrn.fit_mean(Xtr, ytr)
    qlo = lrn.fit_quantile(Xtr, ytr, alpha / 2)
    qhi = lrn.fit_quantile(Xtr, ytr, 1 - alpha / 2)
    lo_cal, hi_cal = qlo.predict(Xcal), qhi.predict(Xcal)
    E = np.maximum(lo_cal - ycal, ycal - hi_cal)
    Q = _conf_quantile(E, alpha)
    return Intervals(pred=mean.predict(Xte),
                     lo=qlo.predict(Xte) - Q,
                     hi=qhi.predict(Xte) + Q)


# --------------------------------------------------------------------------- #
# Bring your own quantiles: the conformal step alone, on precomputed bands     #
# --------------------------------------------------------------------------- #
# For a user whose model already emits lower/upper quantile predictions (any model,
# any library), these apply the calibration step without fitting anything. Each
# returns (lo, hi) arrays for the test rows. With the default learner's quantiles they
# reproduce `cqr`, `mondrian_cqr` and `aci_cqr` exactly (tests/test_conformal.py).
def conformity_scores(lo_cal: np.ndarray, hi_cal: np.ndarray, ycal: np.ndarray) -> np.ndarray:
    """CQR conformity scores E = max(lo - y, y - hi) on the calibration rows."""
    lo_cal, hi_cal, ycal = (np.asarray(a, float) for a in (lo_cal, hi_cal, ycal))
    return np.maximum(lo_cal - ycal, ycal - hi_cal)


def cqr_from_quantiles(
    lo_cal: np.ndarray, hi_cal: np.ndarray, ycal: np.ndarray,
    lo_te: np.ndarray, hi_te: np.ndarray, alpha: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Split CQR on precomputed quantile bands: one conformal radius for every test row."""
    Q = _conf_quantile(conformity_scores(lo_cal, hi_cal, ycal), alpha)
    return np.asarray(lo_te, float) - Q, np.asarray(hi_te, float) + Q


def mondrian_from_quantiles(
    lo_cal: np.ndarray, hi_cal: np.ndarray, ycal: np.ndarray, gcal: np.ndarray,
    lo_te: np.ndarray, hi_te: np.ndarray, gte: np.ndarray,
    alpha: float = 0.10, min_n: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Group-conditional (Mondrian) CQR on precomputed bands: a radius per group with at
    least `min_n` calibration rows, the global radius otherwise."""
    E = conformity_scores(lo_cal, hi_cal, ycal)
    gcal, gte = np.asarray(gcal), np.asarray(gte)
    Q_global = _conf_quantile(E, alpha)
    Qg = {g: _conf_quantile(E[gcal == g], alpha)
          for g in np.unique(gcal) if (gcal == g).sum() >= min_n}
    Q_te = np.array([Qg.get(g, Q_global) for g in gte])
    return np.asarray(lo_te, float) - Q_te, np.asarray(hi_te, float) + Q_te


def aci_from_quantiles(
    lo_cal: np.ndarray, hi_cal: np.ndarray, ycal: np.ndarray, gcal: np.ndarray,
    lo_te: np.ndarray, hi_te: np.ndarray, yte: np.ndarray, te_order: np.ndarray,
    gte: np.ndarray, alpha: float = 0.10, gamma: float = 0.05, min_n: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Group-conditional ACI on precomputed bands (the online method of `aci_cqr`).

    Test rows are processed in ascending `te_order` (e.g. the trading session); within a
    step each group's interval uses only the level carried from earlier steps, and `yte`
    for that step is read only after its intervals are set."""
    E = conformity_scores(lo_cal, hi_cal, ycal)
    yte, te_order = np.asarray(yte, float), np.asarray(te_order)
    return _aci_one_axis(np.asarray(lo_te, float), np.asarray(hi_te, float), E,
                         gcal, gte, yte, te_order, alpha, gamma, min_n)


# --------------------------------------------------------------------------- #
# P2 method layers                                                            #
# --------------------------------------------------------------------------- #
def _cqr_pieces(Xtr, ytr, Xcal, ycal, Xte, alpha, learner=None):
    """Shared CQR body: fit mean + lo/hi quantile learners, return calib conformity
    scores E and the test-set lo/hi quantile predictions + mean prediction."""
    lrn = _lrn(learner)
    mean = lrn.fit_mean(Xtr, ytr)
    qlo = lrn.fit_quantile(Xtr, ytr, alpha / 2)
    qhi = lrn.fit_quantile(Xtr, ytr, 1 - alpha / 2)
    E = np.maximum(qlo.predict(Xcal) - ycal, ycal - qhi.predict(Xcal))
    return mean.predict(Xte), qlo.predict(Xte), qhi.predict(Xte), E


def mondrian_cqr(
    Xtr: np.ndarray, ytr: np.ndarray,
    Xcal: np.ndarray, ycal: np.ndarray, gcal: np.ndarray,
    Xte: np.ndarray, gte: np.ndarray,
    alpha: float = 0.10, min_n: int = 50, *, learner=None,
) -> Intervals:
    """#3 — group-conditional (Mondrian) CQR. A separate conformal quantile Q_g per
    group (groups with >= min_n calibration points); global Q fallback otherwise.
    Gives per-group marginal coverage by construction on the calibration distribution."""
    pred, lo_q, hi_q, E = _cqr_pieces(Xtr, ytr, Xcal, ycal, Xte, alpha, learner)
    gcal, gte = np.asarray(gcal), np.asarray(gte)
    Q_global = _conf_quantile(E, alpha)
    Qg = {g: _conf_quantile(E[gcal == g], alpha)
          for g in np.unique(gcal) if (gcal == g).sum() >= min_n}
    Q_te = np.array([Qg.get(g, Q_global) for g in gte])
    return Intervals(pred=pred, lo=lo_q - Q_te, hi=hi_q + Q_te)


def _adaptive_q(sorted_scores: np.ndarray, alpha_g: float) -> float:
    """Empirical (1-alpha_g) conformal quantile of pre-sorted scores, clamped. At
    alpha_g<=0 we return the widest available score (finite stand-in for +inf)."""
    n = len(sorted_scores)
    if alpha_g <= 0:
        return float(sorted_scores[-1])
    if alpha_g >= 1:
        return 0.0
    k = int(np.ceil((n + 1) * (1 - alpha_g)))
    k = min(max(k, 1), n)
    return float(sorted_scores[k - 1])


def aci_cqr(
    Xtr: np.ndarray, ytr: np.ndarray,
    Xcal: np.ndarray, ycal: np.ndarray, gcal: np.ndarray,
    Xte: np.ndarray, yte: np.ndarray, te_order: np.ndarray, gte: np.ndarray,
    alpha: float = 0.10, gamma: float = 0.05, min_n: int = 50, *, learner=None,
) -> Intervals:
    """#4 — group-conditional Adaptive Conformal Inference (Gibbs-Candes 2021, Mondrian
    variant) on the CQR scores. Per group g, an effective level alpha_{g,t} is carried
    over time-ordered test sessions and updated AFTER each session by
    alpha_{g,t+1} = alpha_{g,t} + gamma*(alpha - err_{g,t}). The interval at session t
    uses only alpha_{g,t} (carried from the past); y_t is revealed after the interval
    is emitted -> PIT-valid. Corrects the temporal coverage drift static conformal can't.
    """
    pred, lo_q, hi_q, E = _cqr_pieces(Xtr, ytr, Xcal, ycal, Xte, alpha, learner)
    gcal, gte = np.asarray(gcal), np.asarray(gte)
    yte, te_order = np.asarray(yte, float), np.asarray(te_order)
    Es_global = np.sort(E)
    Es_grp = {g: np.sort(E[gcal == g])
              for g in np.unique(gcal) if (gcal == g).sum() >= min_n}

    lo = np.empty(len(Xte))
    hi = np.empty(len(Xte))
    alpha_g: dict = {}
    for s in np.unique(te_order):           # ascending sessions: the online stream
        sm = te_order == s
        for g in np.unique(gte[sm]):
            rows = np.where(sm & (gte == g))[0]
            ag = alpha_g.get(g, alpha)
            Q = _adaptive_q(Es_grp.get(g, Es_global), ag)
            lo[rows] = lo_q[rows] - Q
            hi[rows] = hi_q[rows] + Q
            err = 1.0 - float(np.mean((yte[rows] >= lo[rows]) & (yte[rows] <= hi[rows])))
            alpha_g[g] = min(max(ag + gamma * (alpha - err), 1e-4), 1 - 1e-4)
    return Intervals(pred=pred, lo=lo, hi=hi)


def _aci_one_axis(lo_q, hi_q, E, gcal, gte, yte, te_order, alpha, gamma, min_n):
    """One axis of group-conditional ACI on pre-computed CQR pieces. Returns (lo, hi)
    arrays for the test set. Factored out so the multi-axis variant can union axes
    without re-fitting learners."""
    gcal, gte = np.asarray(gcal), np.asarray(gte)
    Es_global = np.sort(E)
    Es_grp = {g: np.sort(E[gcal == g])
              for g in np.unique(gcal) if (gcal == g).sum() >= min_n}
    lo = np.empty(len(lo_q)); hi = np.empty(len(hi_q))
    alpha_g: dict = {}
    for s in np.unique(te_order):
        sm = te_order == s
        for g in np.unique(gte[sm]):
            rows = np.where(sm & (gte == g))[0]
            ag = alpha_g.get(g, alpha)
            Q = _adaptive_q(Es_grp.get(g, Es_global), ag)
            lo[rows] = lo_q[rows] - Q
            hi[rows] = hi_q[rows] + Q
            err = 1.0 - float(np.mean((yte[rows] >= lo[rows]) & (yte[rows] <= hi[rows])))
            alpha_g[g] = min(max(ag + gamma * (alpha - err), 1e-4), 1 - 1e-4)
    return lo, hi


def aci_cqr_multi(
    Xtr: np.ndarray, ytr: np.ndarray,
    Xcal: np.ndarray, ycal: np.ndarray, gcal_axes: list,
    Xte: np.ndarray, yte: np.ndarray, te_order: np.ndarray, gte_axes: list,
    alpha: float = 0.10, gamma: float = 0.05, min_n: int = 50, *, learner=None,
) -> Intervals:
    """P2.1 — MULTI-AXIS group-conditional ACI. Runs an independent group-conditional
    ACI on EACH conditioning axis (sector / cap_tier / momentum-decile) over the shared
    CQR scores, then emits the per-point UNION interval `[min_k lo_k, max_k hi_k]`.

    Rationale (Gibbs-Cherian-Candes 2023 conditional-guarantee spirit): the union is at
    least as wide as each axis's interval, so its coverage dominates each axis's
    per-group coverage -> worst-group coverage is controlled across ALL axes
    simultaneously, not only the single frozen conditioning axis (which left the
    un-conditioned sector/cap_tier axes adrift in P2). Cost: wider intervals (an
    efficiency/over-coverage tradeoff, reported, not hidden). The single-axis `aci_cqr`
    is left untouched — this is an additive variant, never a retune of the frozen knob.
    """
    pred, lo_q, hi_q, E = _cqr_pieces(Xtr, ytr, Xcal, ycal, Xte, alpha, learner)
    yte, te_order = np.asarray(yte, float), np.asarray(te_order)
    lo = np.full(len(Xte), np.inf)
    hi = np.full(len(Xte), -np.inf)
    for gcal, gte in zip(gcal_axes, gte_axes):
        lo_k, hi_k = _aci_one_axis(lo_q, hi_q, E, gcal, gte, yte, te_order, alpha, gamma, min_n)
        lo = np.minimum(lo, lo_k)
        hi = np.maximum(hi, hi_k)
    return Intervals(pred=pred, lo=lo, hi=hi)


def _multiaxis_design(gcal_axes: list, gte_axes: list):
    """Drop-first dummy design matrices (intercept + per-axis indicators) for calib &
    test from a list of per-axis label arrays. Columns are FIXED by the CALIB levels;
    a test-only level maps to all-zero on that axis -> pooled into the intercept + the
    other axes (the graceful-fallback property: no hard min_n, no per-group fit)."""
    cal_cols = [np.ones(len(gcal_axes[0]))]
    te_cols = [np.ones(len(gte_axes[0]))]
    for gc, gt in zip(gcal_axes, gte_axes):
        gc, gt = np.asarray(gc).astype(str), np.asarray(gt).astype(str)
        for lev in np.unique(gc)[1:]:          # drop first level = reference category
            cal_cols.append((gc == lev).astype(float))
            te_cols.append((gt == lev).astype(float))
    return np.column_stack(cal_cols), np.column_stack(te_cols)


def cqr_conditional(
    Xtr: np.ndarray, ytr: np.ndarray,
    Xcal: np.ndarray, ycal: np.ndarray, gcal_axes: list,
    Xte: np.ndarray, gte_axes: list, alpha: float = 0.10, *, learner=None,
) -> Intervals:
    """P2.2 — CALIBRATED multi-group conditional conformal (Gibbs-Cherian-Candes 2023
    "Conformal Prediction with Conditional Guarantees", linear-class construction with
    the multi-axis group indicators as the basis).

    Instead of P2.1's union of three separately-calibrated intervals (which over-covers
    by stacking three one-sided guarantees), this fits ONE coupled threshold per point:
    a pinball (quantile) regression of the CQR conformity scores E on the group-indicator
    feature map phi(x) = [intercept, sector dummies, cap_tier dummies, mom_decile dummies]
    at level 1-alpha. The threshold t(x)=phi(x)^T beta is ADDITIVE in the axes (sector +
    cap + momentum effects), and the pinball first-order conditions target the (1-alpha)
    quantile PER basis function -> each group's coverage is centred at ~1-alpha (two-sided,
    calibrated), not >= 1-alpha (one-sided, conservative). Interval [qlo - t, qhi + t].

    Rare/unseen groups are pooled into the intercept + remaining axes (see _multiaxis_design)
    -- graceful, no hard min_n fallback, the structural advantage over Mondrian. The frozen
    aci_cqr / aci_cqr_multi are untouched: this is an additive third variant, not a retune.
    """
    from sklearn.linear_model import QuantileRegressor

    pred, lo_q, hi_q, E = _cqr_pieces(Xtr, ytr, Xcal, ycal, Xte, alpha, learner)
    Phi_cal, Phi_te = _multiaxis_design(gcal_axes, gte_axes)
    qr = QuantileRegressor(quantile=1 - alpha, alpha=0.0, solver="highs",
                           fit_intercept=False)
    qr.fit(Phi_cal, E)
    t = qr.predict(Phi_te)
    lo, hi = lo_q - t, hi_q + t
    bad = lo > hi                               # guard: degenerate (negative) threshold
    if bad.any():
        mid = 0.5 * (lo[bad] + hi[bad])
        lo[bad], hi[bad] = mid, mid
    return Intervals(pred=pred, lo=lo, hi=hi)


def cqr_conditional_aci(
    Xtr: np.ndarray, ytr: np.ndarray,
    Xcal: np.ndarray, ycal: np.ndarray, gcal_axes: list,
    Xte: np.ndarray, yte: np.ndarray, te_order: np.ndarray, gte_axes: list,
    alpha: float = 0.10, gamma: float = 0.05, *, learner=None,
) -> Intervals:
    """P2.2 (secondary, report-only) — the calibrated conditional threshold made
    shift-robust by a single GLOBAL online offset. The static t(x) (from cqr_conditional)
    sets the per-point shape; an additive correction c_t, carried over time-ordered
    sessions and updated AFTER each session by c_{t+1} = c_t + gamma*scale*(err_t - alpha),
    absorbs the marginal drift the static method can't — under-coverage (err_t > alpha)
    WIDENS (c grows), the width-space analogue of ACI's alpha-space update (scale = median
    so gamma moves c by ~5% of a typical half-width per mis-coverage). y_t is revealed only
    after session t's interval is emitted -> PIT-valid. Best-of-both: conditional shape +
    ACI's marginal shift correction, without the union's over-coverage."""
    from sklearn.linear_model import QuantileRegressor

    pred, lo_q, hi_q, E = _cqr_pieces(Xtr, ytr, Xcal, ycal, Xte, alpha, learner)
    Phi_cal, Phi_te = _multiaxis_design(gcal_axes, gte_axes)
    qr = QuantileRegressor(quantile=1 - alpha, alpha=0.0, solver="highs",
                           fit_intercept=False)
    qr.fit(Phi_cal, E)
    t = qr.predict(Phi_te)
    yte, te_order = np.asarray(yte, float), np.asarray(te_order)
    scale = max(float(np.median(np.abs(t))), 1e-6)

    lo, hi = np.empty(len(Xte)), np.empty(len(Xte))
    c = 0.0
    for s in np.unique(te_order):               # ascending sessions: the online stream
        rows = np.where(te_order == s)[0]
        lo[rows] = lo_q[rows] - t[rows] - c
        hi[rows] = hi_q[rows] + t[rows] + c
        err = 1.0 - float(np.mean((yte[rows] >= lo[rows]) & (yte[rows] <= hi[rows])))
        c = max(c + gamma * scale * (err - alpha), -scale)   # under-cover -> widen; floor: no invert
    bad = lo > hi
    if bad.any():
        mid = 0.5 * (lo[bad] + hi[bad])
        lo[bad], hi[bad] = mid, mid
    return Intervals(pred=pred, lo=lo, hi=hi)


def cqr_conditional_gaci(
    Xtr: np.ndarray, ytr: np.ndarray,
    Xcal: np.ndarray, ycal: np.ndarray, gcal_axes: list,
    Xte: np.ndarray, yte: np.ndarray, te_order: np.ndarray, gte_axes: list,
    alpha: float = 0.10, gamma: float = 0.05, *, learner=None,
) -> Intervals:
    """P2.3 — *per-group ADAPTIVE* conditional conformal. The calibrated conditional
    SHAPE of `cqr_conditional` (t(x)=phi(x)^T beta, the pinball-regression additive
    threshold; reused unchanged, NOT re-fit) carried online with a group-conditional
    correction, over time-ordered sessions:

      * a GLOBAL offset c  -- absorbs marginal drift:   c <- max(c + gamma*scale*(err_t - alpha), -scale)
      * per-(axis,level) deviation offsets d[k,l]       -- absorb each group's residual:
            d[k,l] <- max(d[k,l] + gamma*scale*(err_{k,l,t} - err_t), -scale)
        each group nudged toward the realised session marginal err_t, so c owns the
        average and d[k,l] owns only the group residual -> no triple-counting of the
        marginal across the three additive axes.

    Per-point correction Delta(x) = c + sum_k d[k, level_k(x)] (additive across axes,
    matching the linear class). Interval [qlo - t - Delta, qhi + t + Delta]. At session s
    only c,d carried from sessions < s are used; y_s is revealed after the interval is
    emitted -> PIT-valid. scale = median|t| (width-space, as in cqr_conditional_aci).

    Unification: Delta==0 recovers cqr_conditional (P2.2 static); d==0 recovers
    cqr_conditional_aci (P2.2 global hybrid); the per-group d is the aci_cqr/aci_cqr_multi
    shift fix applied additively to the calibrated shape (not per-group quantiles, not a
    union). The frozen variants are untouched: this is an additive fifth variant, not a retune.
    """
    from sklearn.linear_model import QuantileRegressor

    pred, lo_q, hi_q, E = _cqr_pieces(Xtr, ytr, Xcal, ycal, Xte, alpha, learner)
    Phi_cal, Phi_te = _multiaxis_design(gcal_axes, gte_axes)
    qr = QuantileRegressor(quantile=1 - alpha, alpha=0.0, solver="highs",
                           fit_intercept=False)
    qr.fit(Phi_cal, E)
    t = qr.predict(Phi_te)
    yte, te_order = np.asarray(yte, float), np.asarray(te_order)
    gte_axes = [np.asarray(g).astype(str) for g in gte_axes]
    scale = max(float(np.median(np.abs(t))), 1e-6)

    lo, hi = np.empty(len(Xte)), np.empty(len(Xte))
    c = 0.0                                     # global marginal offset
    d: dict = {}                                # (axis_idx, level) -> deviation offset
    for s in np.unique(te_order):               # ascending sessions: the online stream
        rows = np.where(te_order == s)[0]
        delta = np.full(len(rows), c)
        for k, g in enumerate(gte_axes):        # add each axis's per-level deviation
            delta += np.array([d.get((k, lev), 0.0) for lev in g[rows]])
        lo[rows] = lo_q[rows] - t[rows] - delta
        hi[rows] = hi_q[rows] + t[rows] + delta
        cov = (yte[rows] >= lo[rows]) & (yte[rows] <= hi[rows])
        err = 1.0 - float(np.mean(cov))
        c = max(c + gamma * scale * (err - alpha), -scale)   # global toward alpha
        for k, g in enumerate(gte_axes):                     # groups toward the marginal
            gr = g[rows]
            for lev in np.unique(gr):
                err_g = 1.0 - float(np.mean(cov[gr == lev]))
                d[(k, lev)] = max(d.get((k, lev), 0.0) + gamma * scale * (err_g - err),
                                  -scale)
    bad = lo > hi
    if bad.any():
        mid = 0.5 * (lo[bad] + hi[bad])
        lo[bad], hi[bad] = mid, mid
    return Intervals(pred=pred, lo=lo, hi=hi)


# --------------------------------------------------------------------------- #
# P6 — ALPHA-SPACE (quantile-level) adaptation on the single-regression shape  #
# --------------------------------------------------------------------------- #
# Frozen level grid (reports/p6_alphaspace_preregistration.md). The conditional
# analogue of aci_cqr's per-group empirical quantile FUNCTION: one additive pinball
# regression per level, all on the FULL calibration set -> no per-group sparsity.
_QACI_GRID = np.array([0.50, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99])


def _fit_level_family(Phi_cal, E, Phi_te, grid=_QACI_GRID):
    """Pinball regression of conformity scores E on the group-indicator design at EACH
    level in `grid`; return test-set thresholds T[n_te, n_levels], monotone-rearranged
    across levels per point (sorted ascending -> removes quantile crossing, Chernozhukov-
    Fernandez-Val-Galichon 2010). Every level's fit uses ALL calibration rows + the group
    indicators, so there is no per-group empirical-quantile object (no Mondrian sparsity)."""
    from sklearn.linear_model import QuantileRegressor

    cols = []
    for tau in grid:
        qr = QuantileRegressor(quantile=float(tau), alpha=0.0, solver="highs",
                               fit_intercept=False)
        qr.fit(Phi_cal, E)
        cols.append(qr.predict(Phi_te))
    T = np.column_stack(cols)                # columns increasing in level
    T.sort(axis=1)                           # monotone rearrangement per point
    return T


def _read_level(T_rows, cover, grid=_QACI_GRID):
    """Per-point threshold at coverage level `cover[i]` (=1-alpha_eff), linear-interpolated
    in level between grid nodes (clamped to the grid range). `T_rows` is the rearranged
    (ascending) sub-block for the rows being emitted."""
    cover = np.clip(cover, grid[0], grid[-1])
    return np.array([np.interp(cover[i], grid, T_rows[i]) for i in range(len(cover))])


def _alpha_precompute(Xtr, ytr, Xcal, ycal, gcal_axes, Xte, gte_axes, alpha=0.10, learner=None):
    """Shared, deterministic prologue for every alpha-space variant: the CQR pieces
    (pred/lo_q/hi_q) + the rearranged test-set level-family thresholds T. The four
    alpha-space variants (qaci/gqaci/gqaci_max/gqaci_asym) differ ONLY in the online
    loop, so a driver can compute this ONCE per split and pass it as `precomp=` to all
    four — bit-identical to computing it inside each (the solver is deterministic), just
    ~4x cheaper (the 9 highs-solver quantile fits + 3 GBM fits run once, not four times)."""
    pred, lo_q, hi_q, E = _cqr_pieces(Xtr, ytr, Xcal, ycal, Xte, alpha, learner)
    Phi_cal, Phi_te = _multiaxis_design(gcal_axes, gte_axes)
    T = _fit_level_family(Phi_cal, E, Phi_te)
    return pred, lo_q, hi_q, T


def cqr_conditional_qaci(
    Xtr: np.ndarray, ytr: np.ndarray,
    Xcal: np.ndarray, ycal: np.ndarray, gcal_axes: list,
    Xte: np.ndarray, yte: np.ndarray, te_order: np.ndarray, gte_axes: list,
    alpha: float = 0.10, gamma: float = 0.05, precomp=None, *, learner=None,
) -> Intervals:
    """P6 (global) — ALPHA-SPACE online adaptation on the single-regression conditional
    shape. The alpha-space analogue of `cqr_conditional_aci`: instead of adding a global
    WIDTH offset c to the static t(x), carry a single global effective miscoverage LEVEL
    alpha_eff over time-ordered sessions and read the conditional threshold off the level
    family `t_tau(x)` at 1-alpha_eff. Update is the exact form/rate of aci_cqr (gamma=0.05,
    no width scaling): alpha_eff <- clip(alpha_eff + gamma*(alpha - err_t), 0.01, 0.50).
    The threshold is bounded by the family range [t_0.50(x), t_0.99(x)] -> it cannot inflate
    unboundedly the way the width-space offset can (the hypothesized base self-correction).
    y_t revealed after session t's interval -> PIT-valid."""
    if precomp is None:
        precomp = _alpha_precompute(Xtr, ytr, Xcal, ycal, gcal_axes, Xte, gte_axes, alpha, learner)
    pred, lo_q, hi_q, T = precomp
    yte, te_order = np.asarray(yte, float), np.asarray(te_order)
    lmin, lmax = 1 - _QACI_GRID[-1], 1 - _QACI_GRID[0]      # alpha_eff in [0.01, 0.50]

    lo, hi = np.empty(len(Xte)), np.empty(len(Xte))
    a_eff = alpha                            # carried clamped level (no windup)
    for s in np.unique(te_order):            # ascending sessions: the online stream
        rows = np.where(te_order == s)[0]
        t = _read_level(T[rows], np.full(len(rows), 1 - a_eff))
        lo[rows] = lo_q[rows] - t
        hi[rows] = hi_q[rows] + t
        err = 1.0 - float(np.mean((yte[rows] >= lo[rows]) & (yte[rows] <= hi[rows])))
        a_eff = min(max(a_eff + gamma * (alpha - err), lmin), lmax)
    bad = lo > hi
    if bad.any():
        mid = 0.5 * (lo[bad] + hi[bad]); lo[bad], hi[bad] = mid, mid
    return Intervals(pred=pred, lo=lo, hi=hi)


def cqr_conditional_gqaci(
    Xtr: np.ndarray, ytr: np.ndarray,
    Xcal: np.ndarray, ycal: np.ndarray, gcal_axes: list,
    Xte: np.ndarray, yte: np.ndarray, te_order: np.ndarray, gte_axes: list,
    alpha: float = 0.10, gamma: float = 0.05, precomp=None, trace: dict | None = None,
    *, learner=None,
) -> Intervals:
    """P6 (group, HEADLINE) — *group-conditional* ALPHA-SPACE adaptation on the single-
    regression shape. The alpha-space analogue of `cqr_conditional_gaci` (P2.3): the
    per-group online correction adapts the quantile LEVEL, not the width. A level-space
    additive decomposition (mirroring gACI's c + sum_k d[k,l], but in level units) sets a
    per-point effective miscoverage level:

        alpha_eff(x) = clip(alpha + a + sum_k b[k, level_k(x)], 0.01, 0.50)

      * global a   <- a + gamma*(alpha - err_t)                  (average drift)
      * per-(axis,level) b[k,l] <- b[k,l] + gamma*(err_t - err_{k,l,t})  (group residual,
        nudged to the session marginal err_t so a owns the average -> no triple-counting)

    The threshold is read off the level family `t_tau(x)` at 1-alpha_eff(x). Carried offsets
    a, b are clamped to the level range [-0.09, 0.40] (anti-windup, faithful to aci_cqr
    carrying a clamped level), so the per-group correction relaxes as fast as it tightened.
    Reconciles alpha-space group-conditional adaptation with the coupled single regression:
    NO per-group empirical quantiles, NO Mondrian sparsity. y_t revealed after the interval
    -> PIT-valid. Unification: a==0,b==0 recovers the static family node t_0.90 = cqr_conditional.

    `trace` (optional, report-only): a dict that collects, per session, the session id,
    the clipped per-point alpha_eff and the global offset `a` carried into that session.
    It only records values; the intervals are identical with or without it."""
    if precomp is None:
        precomp = _alpha_precompute(Xtr, ytr, Xcal, ycal, gcal_axes, Xte, gte_axes, alpha, learner)
    pred, lo_q, hi_q, T = precomp
    yte, te_order = np.asarray(yte, float), np.asarray(te_order)
    gte_axes = [np.asarray(g).astype(str) for g in gte_axes]
    lmin, lmax = 1 - _QACI_GRID[-1], 1 - _QACI_GRID[0]      # alpha_eff in [0.01, 0.50]
    omin, omax = lmin - alpha, lmax - alpha                 # offset clamp [-0.09, 0.40]

    lo, hi = np.empty(len(Xte)), np.empty(len(Xte))
    a = 0.0                                   # global level offset
    b: dict = {}                              # (axis_idx, level) -> per-group level offset
    for s in np.unique(te_order):             # ascending sessions: the online stream
        rows = np.where(te_order == s)[0]
        a_eff = np.full(len(rows), alpha + a)
        for k, g in enumerate(gte_axes):
            a_eff += np.array([b.get((k, lev), 0.0) for lev in g[rows]])
        a_eff = np.clip(a_eff, lmin, lmax)
        if trace is not None:
            trace.setdefault("session", []).append(s)
            trace.setdefault("alpha_eff", []).append(a_eff.copy())
            trace.setdefault("a", []).append(a)
        t = _read_level(T[rows], 1 - a_eff)
        lo[rows] = lo_q[rows] - t
        hi[rows] = hi_q[rows] + t
        cov = (yte[rows] >= lo[rows]) & (yte[rows] <= hi[rows])
        err = 1.0 - float(np.mean(cov))
        a = min(max(a + gamma * (alpha - err), omin), omax)      # global toward alpha
        for k, g in enumerate(gte_axes):                          # groups toward the marginal
            gr = g[rows]
            for lev in np.unique(gr):
                err_g = 1.0 - float(np.mean(cov[gr == lev]))
                b[(k, lev)] = min(max(b.get((k, lev), 0.0) + gamma * (err - err_g),
                                      omin), omax)
    bad = lo > hi
    if bad.any():
        mid = 0.5 * (lo[bad] + hi[bad]); lo[bad], hi[bad] = mid, mid
    return Intervals(pred=pred, lo=lo, hi=hi)


def cqr_conditional_gqaci_max(
    Xtr: np.ndarray, ytr: np.ndarray,
    Xcal: np.ndarray, ycal: np.ndarray, gcal_axes: list,
    Xte: np.ndarray, yte: np.ndarray, te_order: np.ndarray, gte_axes: list,
    alpha: float = 0.10, gamma: float = 0.05, precomp=None, *, learner=None,
) -> Intervals:
    """P7 (HEADLINE) — *cross-axis MAX* alpha-space group adaptation. Identical to
    `cqr_conditional_gqaci` EXCEPT the per-point effective level combines the three axes'
    level offsets by the **most-conservative (widest) axis**, not by their sum:

        alpha_eff(x) = clip(alpha + a + min_k b[k, level_k(x)], 0.01, 0.50)

    (min over axes = the most-negative offset = the widest demand). Motivation (P6 §Next):
    the P6 *additive* gqaci regressed on worst-group because for a point easy on two axes
    but hard on one, the easy axes' tightening offsets CANCEL the hard axis's widening one
    (the worst group even changed axis, sector -> mom_decile). Taking the max removes that
    cross-axis cancellation — a point is as wide as its worst axis demands — the level-space
    analogue of the P2.1 union, but on the SINGLE fitted shape (no per-group sparsity). The
    update of each b[k,l] is unchanged (toward the session marginal); only the COMBINATION
    differs, isolating the cancellation hypothesis. y_t revealed after the interval -> PIT."""
    if precomp is None:
        precomp = _alpha_precompute(Xtr, ytr, Xcal, ycal, gcal_axes, Xte, gte_axes, alpha, learner)
    pred, lo_q, hi_q, T = precomp
    yte, te_order = np.asarray(yte, float), np.asarray(te_order)
    gte_axes = [np.asarray(g).astype(str) for g in gte_axes]
    lmin, lmax = 1 - _QACI_GRID[-1], 1 - _QACI_GRID[0]
    omin, omax = lmin - alpha, lmax - alpha

    lo, hi = np.empty(len(Xte)), np.empty(len(Xte))
    a = 0.0
    b: dict = {}
    for s in np.unique(te_order):
        rows = np.where(te_order == s)[0]
        dev = None                                   # most-conservative axis (min offset)
        for k, g in enumerate(gte_axes):
            col = np.array([b.get((k, lev), 0.0) for lev in g[rows]])
            dev = col if dev is None else np.minimum(dev, col)
        a_eff = np.clip(alpha + a + dev, lmin, lmax)
        t = _read_level(T[rows], 1 - a_eff)
        lo[rows] = lo_q[rows] - t
        hi[rows] = hi_q[rows] + t
        cov = (yte[rows] >= lo[rows]) & (yte[rows] <= hi[rows])
        err = 1.0 - float(np.mean(cov))
        a = min(max(a + gamma * (alpha - err), omin), omax)
        for k, g in enumerate(gte_axes):
            gr = g[rows]
            for lev in np.unique(gr):
                err_g = 1.0 - float(np.mean(cov[gr == lev]))
                b[(k, lev)] = min(max(b.get((k, lev), 0.0) + gamma * (err - err_g),
                                      omin), omax)
    bad = lo > hi
    if bad.any():
        mid = 0.5 * (lo[bad] + hi[bad]); lo[bad], hi[bad] = mid, mid
    return Intervals(pred=pred, lo=lo, hi=hi)


def cqr_conditional_gqaci_asym(
    Xtr: np.ndarray, ytr: np.ndarray,
    Xcal: np.ndarray, ycal: np.ndarray, gcal_axes: list,
    Xte: np.ndarray, yte: np.ndarray, te_order: np.ndarray, gte_axes: list,
    alpha: float = 0.10, gamma: float = 0.05, rho: float = 0.5, precomp=None, *, learner=None,
) -> Intervals:
    """P7 (secondary) — *asymmetric-rate* alpha-space group adaptation. Identical to
    `cqr_conditional_gqaci` (ADDITIVE combination unchanged) EXCEPT the per-group level
    offset updates faster to WIDEN than to TIGHTEN: a tightening step (b increasing, i.e.
    a group over-covers relative to the marginal) is scaled by `rho<1`; a widening step is
    full-rate. The adaptive-conformal rationale: under-coverage is the costly direction for
    a coverage guarantee, so react fast to it and slowly to over-coverage. `rho=0.5` frozen
    (the canonical "halve the rate when removing the safety margin"; not tuned). Tests
    whether reluctance-to-tighten alone recovers the group term's value. y_t revealed after
    the interval -> PIT-valid."""
    if precomp is None:
        precomp = _alpha_precompute(Xtr, ytr, Xcal, ycal, gcal_axes, Xte, gte_axes, alpha, learner)
    pred, lo_q, hi_q, T = precomp
    yte, te_order = np.asarray(yte, float), np.asarray(te_order)
    gte_axes = [np.asarray(g).astype(str) for g in gte_axes]
    lmin, lmax = 1 - _QACI_GRID[-1], 1 - _QACI_GRID[0]
    omin, omax = lmin - alpha, lmax - alpha

    lo, hi = np.empty(len(Xte)), np.empty(len(Xte))
    a = 0.0
    b: dict = {}
    for s in np.unique(te_order):
        rows = np.where(te_order == s)[0]
        a_eff = np.full(len(rows), alpha + a)
        for k, g in enumerate(gte_axes):
            a_eff += np.array([b.get((k, lev), 0.0) for lev in g[rows]])
        a_eff = np.clip(a_eff, lmin, lmax)
        t = _read_level(T[rows], 1 - a_eff)
        lo[rows] = lo_q[rows] - t
        hi[rows] = hi_q[rows] + t
        cov = (yte[rows] >= lo[rows]) & (yte[rows] <= hi[rows])
        err = 1.0 - float(np.mean(cov))
        a = min(max(a + gamma * (alpha - err), omin), omax)
        for k, g in enumerate(gte_axes):
            gr = g[rows]
            for lev in np.unique(gr):
                err_g = 1.0 - float(np.mean(cov[gr == lev]))
                raw = gamma * (err - err_g)          # >0 tightens, <0 widens
                step = raw if raw <= 0 else rho * raw  # slower to tighten
                b[(k, lev)] = min(max(b.get((k, lev), 0.0) + step, omin), omax)
    bad = lo > hi
    if bad.any():
        mid = 0.5 * (lo[bad] + hi[bad]); lo[bad], hi[bad] = mid, mid
    return Intervals(pred=pred, lo=lo, hi=hi)


# --------------------------------------------------------------------------- #
# P10 — localized / kernel-conditional comparator (SPEC §11 P10)              #
# --------------------------------------------------------------------------- #
# A CONTINUOUS approximate-conditional reference beyond the discrete Mondrian
# groups: kernel-reweighted split CQR (Guan 2023 "Localized Conformal Prediction";
# Hore & Barber 2024 "Conformal prediction with local weights"). For each test x the
# conformal radius Q(x) is the (1-alpha) quantile of the calibration conformity scores
# E, weighted by a Gaussian kernel on standardized features — so the interval width
# varies continuously with x, no group buckets. FROZEN knobs: subsample size (laptop
# tractability — the full 23k x 70k kernel is out of core), median-heuristic bandwidth,
# self-weight = 1 for the test atom (Hore-Barber local weighting).
_LOCAL_SUBSAMPLE = 6000
_LOCAL_SEED = 20260701
_LOCAL_BLOCK = 2000


def _median_bandwidth(Z: np.ndarray, rng, m: int = 1000) -> float:
    """Median pairwise Euclidean distance on a random m-subsample (median heuristic)."""
    if len(Z) > m:
        Z = Z[rng.choice(len(Z), m, replace=False)]
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    iu = np.triu_indices(len(Z), k=1)
    med = float(np.median(np.sqrt(d2[iu])))
    return max(med, 1e-6)


def localized_cqr(
    Xtr: np.ndarray, ytr: np.ndarray,
    Xcal: np.ndarray, ycal: np.ndarray,
    Xte: np.ndarray, alpha: float = 0.10,
    subsample: int = _LOCAL_SUBSAMPLE, *, learner=None,
) -> Intervals:
    """P10 — localized (kernel-conditional) CQR. Per-test-point conformal radius Q(x) =
    kernel-weighted (1-alpha) quantile of the CQR conformity scores E, Gaussian kernel on
    z-scored features. Continuous conditional coverage; the P10 comparator to Mondrian's
    discrete groups. Static (calibration-based), like Mondrian — the conditioning-axis
    reference, not a shift method."""
    pred, lo_q, hi_q, E = _cqr_pieces(Xtr, ytr, Xcal, ycal, Xte, alpha, learner)
    Xc, Xt = np.asarray(Xcal, float), np.asarray(Xte, float)
    mu, sd = Xc.mean(0), Xc.std(0)
    sd[sd == 0] = 1.0
    Zc, Zt = (Xc - mu) / sd, (Xt - mu) / sd

    rng = np.random.default_rng(_LOCAL_SEED)
    if len(Zc) > subsample:
        idx = rng.choice(len(Zc), subsample, replace=False)
        Zc, Es = Zc[idx], E[idx]
    else:
        Es = E
    h = _median_bandwidth(Zc, rng)

    order = np.argsort(Es)
    E_sorted = Es[order]
    Zc_sorted = Zc[order]
    csq = np.sum(Zc_sorted ** 2, axis=1)                 # |c|^2 term

    Q_te = np.empty(len(Zt))
    for s in range(0, len(Zt), _LOCAL_BLOCK):
        blk = Zt[s:s + _LOCAL_BLOCK]
        d2 = (np.sum(blk ** 2, axis=1, keepdims=True) + csq
              - 2.0 * blk @ Zc_sorted.T)                 # [b, n_sub]
        W = np.exp(-np.maximum(d2, 0.0) / (2.0 * h * h))
        cumw = np.cumsum(W, axis=1)
        tot = cumw[:, -1] + 1.0                          # + self-weight (test atom)
        thresh = (1.0 - alpha) * tot                     # target cumulative mass
        k = (cumw < thresh[:, None]).sum(axis=1)         # first index reaching it
        k = np.clip(k, 0, len(E_sorted) - 1)
        Q_te[s:s + _LOCAL_BLOCK] = E_sorted[k]
    return Intervals(pred=pred, lo=lo_q - Q_te, hi=hi_q + Q_te)


def weighted_cqr(
    Xtr: np.ndarray, ytr: np.ndarray,
    Xcal: np.ndarray, ycal: np.ndarray,
    Xte: np.ndarray, alpha: float = 0.10, *, learner=None,
) -> Intervals:
    """#5 (ablation) — covariate-shift weighted CQR (Tibshirani et al. 2019). A
    classifier separates calib (0) vs test (1) covariates; calib points are reweighted
    by the likelihood ratio w(x)=p/(1-p) and the conformal quantile is the weighted
    empirical quantile (test atom approximated at the mean test weight). One shared Q."""
    from lightgbm import LGBMClassifier

    from .learners import PARAMS  # reuse frozen GBM config

    pred, lo_q, hi_q, E = _cqr_pieces(Xtr, ytr, Xcal, ycal, Xte, alpha, learner)
    clf = LGBMClassifier(**{k: v for k, v in PARAMS.items() if k != "subsample_freq"})
    Z = np.vstack([Xcal, Xte])
    yz = np.r_[np.zeros(len(Xcal)), np.ones(len(Xte))]
    clf.fit(Z, yz)
    p_cal = np.clip(clf.predict_proba(Xcal)[:, 1], 1e-4, 1 - 1e-4)
    w = p_cal / (1 - p_cal)
    p_te = np.clip(clf.predict_proba(Xte)[:, 1], 1e-4, 1 - 1e-4)
    w_te = float(np.mean(p_te / (1 - p_te)))

    order = np.argsort(E)
    Es, ws = E[order], w[order]
    total = ws.sum() + w_te
    cw = np.cumsum(ws) / total
    idx = np.searchsorted(cw, 1 - alpha)
    Q = float(Es[min(idx, len(Es) - 1)])
    return Intervals(pred=pred, lo=lo_q - Q, hi=hi_q + Q)
