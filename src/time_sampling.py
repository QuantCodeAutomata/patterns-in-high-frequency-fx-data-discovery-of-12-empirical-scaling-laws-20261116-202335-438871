"""Fixed physical-time sampling for scaling laws (0a) and (3), section 2.1 of
Glattfelder, Dupuis & Olsen (arXiv:0809.1040).

The averaging operator used throughout section 2.1 is
    <x>_p = (1/n * sum_j x_j^p)^(1/p).
Prices are sampled at fixed intervals Delta_t using the last quoted price
before each sampling time ("last observation carried forward"), which the
paper describes as using "the last quoted price" before each time stamp.
"""

from __future__ import annotations

import numpy as np
from numba import njit


def sample_last_price(
    prices: np.ndarray, times: np.ndarray, dt_seconds: float
) -> np.ndarray:
    """Prices at fixed times t_i = times[0] + i * dt_seconds, using the most
    recent tick at or before t_i (last-observation-carried-forward)."""
    t0, t_end = times[0], times[-1]
    n_samples = int(np.floor((t_end - t0) / dt_seconds)) + 1
    if n_samples < 2:
        return np.empty(0, dtype=np.float64)
    sample_times = t0 + np.arange(n_samples, dtype=np.float64) * dt_seconds
    # searchsorted(side="right") - 1 gives the index of the last tick <= t.
    idx = np.searchsorted(times, sample_times, side="right") - 1
    idx = np.clip(idx, 0, times.shape[0] - 1)
    return prices[idx]


def mean_abs_change_p(sampled_prices: np.ndarray, p: int) -> float:
    """<|Delta x|>_p = (1/n * sum |Delta x_i|^p)^(1/p), law (0a).

    Returned in the same percent-number convention as every threshold grid
    in this project (``threshold_pct=0.01`` denotes a 0.01% move): the raw
    relative change is multiplied by 100, so this value is directly
    comparable with (and combinable into, eq. 17) the other scaling laws'
    Delta_x axis.
    """
    if sampled_prices.shape[0] < 2:
        raise ValueError("need at least 2 samples")
    rel_change_pct = 100.0 * np.diff(sampled_prices) / sampled_prices[:-1]
    return float(np.mean(np.abs(rel_change_pct) ** p) ** (1.0 / p))


@njit(cache=True)
def _rolling_max_range(prices: np.ndarray, times: np.ndarray, dt_seconds: float, p: int):
    """Non-overlapping windows [t0 + k*dt, t0 + (k+1)*dt] (section 2.1):
    relative price range (max-min)/start within each window, raised to the
    p-th power, computed directly over the raw tick data (not interpolated
    prices), per the methodology's instruction to scan the actual ticks."""
    n = prices.shape[0]
    t0 = times[0]
    t_end = times[n - 1]
    n_windows = int((t_end - t0) / dt_seconds)
    out = np.empty(n_windows, dtype=np.float64)
    j = 0  # pointer into prices/times
    for k in range(n_windows):
        w_start = t0 + k * dt_seconds
        w_end = w_start + dt_seconds
        while j < n and times[j] < w_start:
            j += 1
        j_start = j
        hi = -1.0e300
        lo = 1.0e300
        base = prices[j_start] if j_start < n else np.nan
        jj = j_start
        found = False
        while jj < n and times[jj] <= w_end:
            v = prices[jj]
            if v > hi:
                hi = v
            if v < lo:
                lo = v
            found = True
            jj += 1
        if found and base > 0:
            rng = 100.0 * (hi - lo) / base
            out[k] = rng**p
        else:
            out[k] = np.nan
    return out


def mean_max_range_p(
    prices: np.ndarray, times: np.ndarray, dt_seconds: float, p: int
) -> float:
    """<Delta x_max>_p for law (3): the p-th-power mean of the maximal
    relative price range observed within consecutive, non-overlapping
    windows of length Delta_t (section 2.1, windowing not further specified
    by the paper; non-overlapping windows are used here)."""
    values = _rolling_max_range(
        np.asarray(prices, dtype=np.float64), np.asarray(times, dtype=np.float64),
        float(dt_seconds), int(p),
    )
    values = values[np.isfinite(values)]
    if values.shape[0] < 2:
        raise ValueError("need at least 2 valid windows")
    return float(np.mean(values) ** (1.0 / p))
