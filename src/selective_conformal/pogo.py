"""POGO-style parameter-free online group-conditional update (P12, secondary arm).

The tuned method `conformal.cqr_conditional_gqaci_max` carries a single global step size gamma
(frozen at 0.05). POGO (parameter-free online group-conditional coverage, arXiv:2606.00419) proves
per-group miscoverage -> 0 for k intersecting groups SIMULTANEOUSLY with **no step-size knob**. This
module implements a member of that family: the identical cross-axis-MAX level-space update, but with
a per-(axis,level) ADAPTIVE step `eta = ETA0 / sqrt(1 + visits)` (Robbins-Monro / vanishing-rate
schedule) in place of the fixed gamma. `ETA0` is a fixed base scale (documented, NOT tuned per
dataset), so the method has no per-dataset knob to set.

HONEST SCOPE: we do NOT re-derive POGO's theorem here. We implement a parameter-free member of its
family and show empirically that it matches the tuned-gamma method WITHOUT per-dataset tuning, the
practical payoff of the guarantee. POGO (2606.00419) is the citation that establishes the family's guarantee.

No data. See reports/p12_regime_axis_preregistration.md.
"""
from __future__ import annotations

import numpy as np

from .conformal import _QACI_GRID, Intervals, _alpha_precompute, _read_level

ETA0 = 0.10   # fixed base scale of the vanishing-rate schedule (documented, not tuned per dataset)


def cqr_conditional_gqaci_max_pogo(Xtr, ytr, Xcal, ycal, gcal_axes,
                                   Xte, yte, te_order, gte_axes,
                                   alpha=0.10, precomp=None, *, learner=None) -> Intervals:
    """cross-axis-MAX group adaptation with a PARAMETER-FREE per-group step (no gamma).
    alpha_eff(x) = clip(alpha + a + min_k b[k, level_k(x)], lmin, lmax); each offset updates by
    eta = ETA0/sqrt(1+visits[key]) toward the session marginal. y_t revealed after -> PIT."""
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
    visits: dict = {}
    v_glob = 0
    for s in np.unique(te_order):
        rows = np.where(te_order == s)[0]
        dev = None
        for k, g in enumerate(gte_axes):
            col = np.array([b.get((k, lev), 0.0) for lev in g[rows]])
            dev = col if dev is None else np.minimum(dev, col)
        a_eff = np.clip(alpha + a + dev, lmin, lmax)
        t = _read_level(T[rows], 1 - a_eff)
        lo[rows] = lo_q[rows] - t
        hi[rows] = hi_q[rows] + t
        cov = (yte[rows] >= lo[rows]) & (yte[rows] <= hi[rows])
        err = 1.0 - float(np.mean(cov))
        v_glob += 1
        a = min(max(a + (ETA0 / np.sqrt(v_glob)) * (alpha - err), omin), omax)
        for k, g in enumerate(gte_axes):
            gr = g[rows]
            for lev in np.unique(gr):
                key = (k, lev)
                visits[key] = visits.get(key, 0) + 1
                eta = ETA0 / np.sqrt(visits[key])
                err_g = 1.0 - float(np.mean(cov[gr == lev]))
                b[key] = min(max(b.get(key, 0.0) + eta * (err - err_g), omin), omax)
    bad = lo > hi
    if bad.any():
        mid = 0.5 * (lo[bad] + hi[bad]); lo[bad], hi[bad] = mid, mid
    return Intervals(pred=pred, lo=lo, hi=hi)
