"""Gaussian random walk (GRW) benchmark series, eq. (18) of Glattfelder,
Dupuis & Olsen, "Patterns in high-frequency FX data: Discovery of 12
empirical scaling laws" (arXiv:0809.1040), section 3.1.

The paper specifies one million ticks with unitary (one-second) time steps
and normally distributed relative price changes:

    x_0 = 1.336723, Delta t = 1 second, sigma = 1/6769.6,
    Delta x_i ~ N(0, sigma^2), x_{i+1} = x_i * (1 + Delta x_i)

The paper's formula is stated additively (Delta x_i = x(t_i+Delta t) -
x(t_i)); interpreted literally with a scale of sigma ~ 1.5e-4 this is
indistinguishable from a relative/multiplicative step for a FX rate near 1,
so the geometric form above is used, which also guarantees a strictly
positive price series (required for the paper's percentage-threshold
algorithms). This interpretation is recorded as an assumption in
results/RESULTS.md.
"""

from __future__ import annotations

import numpy as np


def generate_grw_series(
    n_ticks: int, x0: float, dt_seconds: float, sigma: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the GRW benchmark price series of eq. (18).

    Returns (prices, times) with ``n_ticks`` points, times in seconds at
    unitary spacing ``dt_seconds`` starting at 0.
    """
    if n_ticks < 2:
        raise ValueError("n_ticks must be at least 2")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    rng = np.random.default_rng(seed)
    relative_changes = rng.normal(loc=0.0, scale=sigma, size=n_ticks - 1)
    log_prices = np.log(x0) + np.concatenate(([0.0], np.cumsum(np.log1p(relative_changes))))
    prices = np.exp(log_prices)
    times = np.arange(n_ticks, dtype=np.float64) * dt_seconds
    return prices, times


def apply_constant_spread(prices: np.ndarray, spread_pct: float) -> np.ndarray:
    """Reduce each successive price move by a constant proportional spread,
    used for the "cumulative total move with costs" variant of law (12)
    (section 2.3: "we introduce a constant spread of 0.02%").

    Each step's absolute relative return is shrunk by ``spread_pct`` percent
    (floored at zero), emulating a round-trip transaction cost paid on every
    tick-to-tick move.
    """
    if spread_pct < 0:
        raise ValueError("spread_pct must be non-negative")
    spread = spread_pct / 100.0
    rel_change = np.diff(prices) / prices[:-1]
    shrunk = np.sign(rel_change) * np.maximum(np.abs(rel_change) - spread, 0.0)
    log_prices = np.log(prices[0]) + np.concatenate(([0.0], np.cumsum(np.log1p(shrunk))))
    return np.exp(log_prices)
