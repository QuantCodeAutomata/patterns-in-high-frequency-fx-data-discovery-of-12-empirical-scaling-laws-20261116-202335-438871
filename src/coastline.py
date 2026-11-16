"""Coastline-length evaluation from scaling law (12), section 2.3 of
Glattfelder, Dupuis & Olsen (arXiv:0809.1040).

Law (12): Delta_x_cum^* = (Delta_x_dc / C_cum,*)^E_cum,* for * in {tm, dc, os}.
The "coastline" is the annualised cumulative total-move Delta_x_cum^{tm} as a
function of the measurement resolution Delta_x_dc (section 2.3): decreasing
the threshold increases the measured coastline, analogous to the coastline
paradox in fractal geometry.
"""

from __future__ import annotations

import numpy as np

from src.scaling_laws import PowerLawFit


def coastline_at(fit: PowerLawFit, thresholds_pct: np.ndarray) -> np.ndarray:
    """Evaluate the fitted cumulative-total-move law at given thresholds
    (percent): Delta_x_cum = (Delta_x_dc / C)^E."""
    thresholds_pct = np.asarray(thresholds_pct, dtype=np.float64)
    return (thresholds_pct / fit.C) ** fit.E


def coastline_ratio(fit: PowerLawFit, low_pct: float, high_pct: float) -> float:
    """Ratio of coastline at ``low_pct`` to coastline at ``high_pct``
    (section 2.3: "decreasing the threshold... decreases/increases the
    coastline by a factor of...")."""
    values = coastline_at(fit, np.array([low_pct, high_pct]))
    return float(values[0] / values[1])
