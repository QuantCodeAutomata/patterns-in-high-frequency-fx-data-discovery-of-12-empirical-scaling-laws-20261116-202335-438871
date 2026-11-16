"""Price-move counting (Algorithm 1, section 3.2) of Glattfelder, Dupuis &
Olsen, "Patterns in high-frequency FX data: Discovery of 12 empirical scaling
laws" (arXiv:0809.1040).

Algorithm 1 (price-move count), reproduced from the paper:

    Require: n_up = n_down = 0, x_ext = x_0, Delta_x >= 0
    if (x - x_ext)/x_ext >= Delta_x:   n_up += 1;   x_ext <- x
    elif (x - x_ext)/x_ext <= -Delta_x: n_down += 1; x_ext <- x

Unlike Algorithm 2, there is no mode: any threshold crossing in either
direction counts and resets the running extreme. Each detected event marks
the end of one "price move" that started at the previous extreme.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _price_move_core(prices: np.ndarray, threshold: float):
    n = prices.shape[0]
    event_idx = np.empty(n, dtype=np.int64)
    event_price = np.empty(n, dtype=np.float64)
    event_direction = np.empty(n, dtype=np.int64)

    x_ext = prices[0]
    n_events = 0

    for i in range(1, n):
        x = prices[i]
        rel = (x - x_ext) / x_ext
        if rel >= threshold:
            event_idx[n_events] = i
            event_price[n_events] = x
            event_direction[n_events] = 1
            n_events += 1
            x_ext = x
        elif rel <= -threshold:
            event_idx[n_events] = i
            event_price[n_events] = x
            event_direction[n_events] = -1
            n_events += 1
            x_ext = x

    return event_idx[:n_events], event_price[:n_events], event_direction[:n_events]


def price_move_events(prices: np.ndarray, times: np.ndarray, threshold_pct: float) -> dict:
    """Run Algorithm 1 over ``prices`` at threshold ``threshold_pct`` (percent).

    Returns a dict with ``event_tick_index``, ``event_price``, ``event_time``
    and ``direction`` (+1 up, -1 down), one entry per detected price move.
    """
    if prices.shape[0] != times.shape[0]:
        raise ValueError("prices and times must have the same length")
    if threshold_pct <= 0:
        raise ValueError("threshold_pct must be positive")
    threshold = threshold_pct / 100.0
    prices = np.asarray(prices, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    idx, price, direction = _price_move_core(prices, threshold)
    return {
        "event_tick_index": idx,
        "event_price": price,
        "event_time": times[idx],
        "direction": direction,
    }


def count_events_between(
    reference_tick_index: np.ndarray, start_tick: np.ndarray, end_tick: np.ndarray
) -> np.ndarray:
    """Count of entries of the sorted ``reference_tick_index`` falling in each
    half-open interval (start_tick[k], end_tick[k]], for every k.

    Used to count 0.02%-threshold tick events (law 1's and law 11's "number
    of ticks") within arbitrary segments defined by their tick-index bounds.
    """
    left = np.searchsorted(reference_tick_index, start_tick, side="right")
    right = np.searchsorted(reference_tick_index, end_tick, side="right")
    return (right - left).astype(np.float64)


def segment_tick_counts(event_tick_index: np.ndarray, tick_event_tick_index: np.ndarray) -> np.ndarray:
    """Number of 0.02%-ticks (``tick_event_tick_index``, from a reference run
    of Algorithm 1 at the fixed tick threshold) falling within each price-move
    segment bounded by consecutive entries of ``event_tick_index``.

    Segment k (k = 1..n-1) runs from event_tick_index[k-1] (exclusive) to
    event_tick_index[k] (inclusive), matching the "number of ticks observed
    during a price move" of law (1).
    """
    if event_tick_index.shape[0] < 2:
        return np.empty(0, dtype=np.float64)
    return count_events_between(tick_event_tick_index, event_tick_index[:-1], event_tick_index[1:])


def segment_durations(event_time: np.ndarray) -> np.ndarray:
    """Duration Delta_t of each price-move segment (law 4), i.e. the time
    between consecutive detected events."""
    if event_time.shape[0] < 2:
        return np.empty(0, dtype=np.float64)
    return np.diff(event_time)
