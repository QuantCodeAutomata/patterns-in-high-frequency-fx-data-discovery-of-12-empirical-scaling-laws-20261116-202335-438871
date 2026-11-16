"""Fitting of the power-law scaling relations of Glattfelder, Dupuis & Olsen
(arXiv:0809.1040), section 3.3 "Data fitting".

Every scaling law in the paper has the form Y = (X / C)^E, fitted in
log-space as ln(Y) = A + B * ln(X) with E = B and C = exp(-A / B)
(section 3.3). A quadratic fit ln(Y) = A + B*ln(X) + Q*ln(X)^2 is used only
to test for systematic curvature around the linear fit (section 3.3,
"we also fit the data to a quadratic model... to check whether there is any
remaining curvature").

# statsmodels OLS (sm.add_constant) is the library used for both the linear
# and quadratic log-log fits, as specified in the paper's fitting procedure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm


@dataclass
class PowerLawFit:
    """Result of a single log-log power-law fit Y = (X / C)^E."""

    E: float
    C: float
    E_se: float
    C_se: float
    A: float  # linear-fit intercept, ln(Y) = A + B*ln(X)
    A_se: float
    r2_linear: float
    r2_adj_linear: float
    r2_quadratic: float
    delta_r2: float  # r2_quadratic - r2_linear (curvature test, section 3.3)
    n_obs: int


def fit_power_law(x: np.ndarray, y: np.ndarray) -> PowerLawFit:
    """Fit Y = (X / C)^E via log-log OLS, as in section 3.3.

    Non-positive or non-finite (x, y) pairs are dropped before fitting
    (a logarithm is undefined there); at least 3 valid points are required.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[valid], y[valid]
    if x.shape[0] < 3:
        raise ValueError(f"need at least 3 valid (x>0, y>0) points, got {x.shape[0]}")

    log_x = np.log(x)
    log_y = np.log(y)

    design_linear = sm.add_constant(log_x)
    linear_fit = sm.OLS(log_y, design_linear).fit()
    A, B = linear_fit.params[0], linear_fit.params[1]
    A_se, B_se = linear_fit.bse[0], linear_fit.bse[1]

    E = B
    C = np.exp(-A / B)
    # Error propagation for C = f(A, B) = exp(-A/B), section 3.3:
    # Var[C] ~= (df/dA)^2 Var[A] + (df/dB)^2 Var[B], ignoring Cov[A,B] as the
    # paper's appendix does not report a covariance term.
    dC_dA = -C / B
    dC_dB = C * A / B**2
    C_se = float(np.sqrt((dC_dA * A_se) ** 2 + (dC_dB * B_se) ** 2))

    design_quad = sm.add_constant(np.column_stack([log_x, log_x**2]))
    quad_fit = sm.OLS(log_y, design_quad).fit()

    return PowerLawFit(
        E=float(E), C=float(C), E_se=float(B_se), C_se=C_se,
        A=float(A), A_se=float(A_se),
        r2_linear=float(linear_fit.rsquared),
        r2_adj_linear=float(linear_fit.rsquared_adj),
        r2_quadratic=float(quad_fit.rsquared),
        delta_r2=float(quad_fit.rsquared - linear_fit.rsquared),
        n_obs=int(x.shape[0]),
    )


def threshold_grid(
    n_points: int, min_pct: float, max_pct: float
) -> np.ndarray:
    """250 log-spaced threshold values Delta_x in percent (section 3.3)."""
    log_min, log_max = np.log(min_pct), np.log(max_pct)
    return np.exp(np.linspace(log_min, log_max, n_points))


def time_grid(n_points: int, min_seconds: float, max_seconds: float) -> np.ndarray:
    """245 log-spaced time-interval values Delta_t in seconds (section 3.3)."""
    log_min, log_max = np.log(min_seconds), np.log(max_seconds)
    return np.exp(np.linspace(log_min, log_max, n_points))
