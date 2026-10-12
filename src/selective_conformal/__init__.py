"""selective_conformal: selective conformal regression under distribution shift.

Split, group-conditional and online conformal intervals on top of quantile
regression, plus a per-unit abstention score and the metrics used to judge them.
Each method fits its own models (LightGBM by default, or any `learner=` you pass);
the `*_from_quantiles` functions apply only the conformal step to bands your own
model already produced.
"""

from .conformal import (
    Intervals,
    aci_cqr,
    aci_cqr_multi,
    aci_from_quantiles,
    conformity_scores,
    cqr,
    cqr_conditional,
    cqr_conditional_aci,
    cqr_conditional_gaci,
    cqr_conditional_gqaci,
    cqr_conditional_gqaci_asym,
    cqr_conditional_gqaci_max,
    cqr_conditional_qaci,
    cqr_from_quantiles,
    localized_cqr,
    mondrian_cqr,
    mondrian_from_quantiles,
    naive_split,
    weighted_cqr,
)
from .learners import LightGBMLearner
from .metrics import (
    clustered_bootstrap,
    coverage,
    grouped_coverage,
    mean_width,
    risk_coverage,
    worst_group_coverage,
)
from .selective import name_uncertainty, selective_scores

__version__ = "0.3.0"

__all__ = [
    "Intervals",
    "LightGBMLearner",
    "aci_cqr",
    "aci_cqr_multi",
    "aci_from_quantiles",
    "clustered_bootstrap",
    "conformity_scores",
    "coverage",
    "cqr",
    "cqr_conditional",
    "cqr_conditional_aci",
    "cqr_conditional_gaci",
    "cqr_conditional_gqaci",
    "cqr_conditional_gqaci_asym",
    "cqr_conditional_gqaci_max",
    "cqr_conditional_qaci",
    "cqr_from_quantiles",
    "grouped_coverage",
    "localized_cqr",
    "mean_width",
    "mondrian_cqr",
    "mondrian_from_quantiles",
    "naive_split",
    "name_uncertainty",
    "risk_coverage",
    "selective_scores",
    "weighted_cqr",
    "worst_group_coverage",
]
