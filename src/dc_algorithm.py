"""Directional-change event detection (Algorithm 2, section 3.2) and the
total-move / directional-change / overshoot decomposition (section 2.2) of
Glattfelder, Dupuis & Olsen, "Patterns in high-frequency FX data: Discovery
of 12 empirical scaling laws" (arXiv:0809.1040).

Algorithm 2 (direction-change count), reproduced from the paper:

    Require: n_up = n_down = 0, x_ext = x_0, mode is up, Delta_x_dc >= 0
    if mode is down:
        if x < x_ext:                       x_ext <- x
        elif (x - x_ext)/x_ext >= Delta_x:  n_up += 1; x_ext <- x; mode <- up
    elif mode is up:
        if x > x_ext:                       x_ext <- x
        elif (x - x_ext)/x_ext <= -Delta_x: n_down += 1; x_ext <- x; mode <- down

A directional-change (DC) event fires when the price reverses by the
threshold from the running extreme x_ext. Section 2.2 states: "At each
occurrence of a directional change, the overshoot associated with the
previous directional change is determined as the difference between the
price level at which the last directional change occurred and the extrema,
i.e., the high when in up mode or low when in down mode."

Writing E_k for the running extreme reached strictly between DC event k and
DC event k+1 (the value of x_ext right before it is reset by event k+1), and
D_k for the price at which DC event k fired:

    DC_k = |D_k - E_{k-1}|                  (the directional-change section)
    OS_k = |E_k - D_k|                      (the overshoot section)
    TM_k = DC_k + OS_k = |E_k - E_{k-1}|    (eq. 6)

E_{k-1} is recorded by the core loop as the value of x_ext immediately
before event k resets it; shifting that series by one event therefore also
gives E_k, which is all that is needed to compute OS_k and TM_k. DC_k is
available for every event k = 1..n; OS_k and TM_k require the following
event and are available only for k = 1..n-1. The segment before the first
event and the unfinished segment after the last event are dropped (a
boundary condition the paper does not specify).
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _directional_change_core(prices: np.ndarray, threshold: float):
    n = prices.shape[0]
    event_idx = np.empty(n, dtype=np.int64)
    event_price = np.empty(n, dtype=np.float64)
    event_direction = np.empty(n, dtype=np.int64)
    # Extreme reached in the mode preceding each event (E_{k-1} for event k).
    prior_extreme_idx = np.empty(n, dtype=np.int64)
    prior_extreme_price = np.empty(n, dtype=np.float64)

    mode_up = True  # Algorithm 2: "mode is up" initially
    x_ext = prices[0]
    x_ext_idx = 0
    n_events = 0

    for i in range(1, n):
        x = prices[i]
        if not mode_up:  # mode is down
            if x < x_ext:
                x_ext = x
                x_ext_idx = i
            elif (x - x_ext) / x_ext >= threshold:
                event_idx[n_events] = i
                event_price[n_events] = x
                event_direction[n_events] = 1
                prior_extreme_idx[n_events] = x_ext_idx
                prior_extreme_price[n_events] = x_ext
                n_events += 1
                x_ext = x
                x_ext_idx = i
                mode_up = True
        else:  # mode is up
            if x > x_ext:
                x_ext = x
                x_ext_idx = i
            elif (x - x_ext) / x_ext <= -threshold:
                event_idx[n_events] = i
                event_price[n_events] = x
                event_direction[n_events] = -1
                prior_extreme_idx[n_events] = x_ext_idx
                prior_extreme_price[n_events] = x_ext
                n_events += 1
                x_ext = x
                x_ext_idx = i
                mode_up = False

    return (
        event_idx[:n_events], event_price[:n_events], event_direction[:n_events],
        prior_extreme_idx[:n_events], prior_extreme_price[:n_events],
    )


def directional_change_events(
    prices: np.ndarray, times: np.ndarray, threshold_pct: float
) -> dict:
    """Run Algorithm 2 over ``prices`` at threshold ``threshold_pct`` (percent).

    Parameters
    ----------
    prices : mid-prices, free of consecutive duplicates.
    times : time in seconds aligned with ``prices``, monotonically increasing.
    threshold_pct : directional-change threshold Delta_x_dc, in percent
        (e.g. 0.1 for a 0.1% threshold).

    Returns
    -------
    dict with ``event_tick_index``, ``event_price``, ``event_time``,
    ``direction`` (+1 up, -1 down), and ``prior_extreme_price/time/tick_index``
    (E_{k-1} for each event k), one entry per detected event.
    """
    if prices.shape[0] != times.shape[0]:
        raise ValueError("prices and times must have the same length")
    if threshold_pct <= 0:
        raise ValueError("threshold_pct must be positive")
    threshold = threshold_pct / 100.0
    prices = np.asarray(prices, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    idx, price, direction, ext_idx, ext_price = _directional_change_core(prices, threshold)
    return {
        "event_tick_index": idx,
        "event_price": price,
        "event_time": times[idx],
        "direction": direction,
        "prior_extreme_tick_index": ext_idx,
        "prior_extreme_price": ext_price,
        "prior_extreme_time": times[ext_idx],
    }


def decompose_segments(events: dict) -> dict:
    """Decompose directional-change events into DC / OS / TM components.

    ``events`` must be the dict returned by :func:`directional_change_events`.
    DC_k = |D_k - E_{k-1}| is available for k = 1..n (n = number of events).
    OS_k = |E_k - D_k| and TM_k = DC_k + OS_k require the next event's E_k
    and are therefore only available for k = 1..n-1.

    Each component also carries its ``*_start_tick``/``*_end_tick`` tick-index
    bounds (not a raw-index tick count), so that the number of 0.02%-moves
    inside a segment (law 11's tick definition) can be obtained by counting
    the reference tick-event series within those bounds (see
    :func:`src.price_move_algorithm.count_events_between`).
    """
    event_price = events["event_price"]
    event_time = events["event_time"]
    event_tick = events["event_tick_index"]
    prior_price = events["prior_extreme_price"]
    prior_time = events["prior_extreme_time"]
    prior_tick = events["prior_extreme_tick_index"]
    n = event_price.shape[0]
    empty_f = np.empty(0, dtype=np.float64)
    empty_i = np.empty(0, dtype=np.int64)
    fields = (
        "dc_move", "dc_time", "dc_start_tick", "dc_end_tick",
        "os_move", "os_time", "os_start_tick", "os_end_tick",
        "tm_move", "tm_time", "tm_start_tick", "tm_end_tick",
    )
    if n < 2:
        return {k: (empty_i if "tick" in k else empty_f) for k in fields}

    # DC_k spans (E_{k-1} tick, D_k tick], k = 1..n.
    dc_move = np.abs(event_price - prior_price)
    dc_time = event_time - prior_time
    dc_start_tick = prior_tick
    dc_end_tick = event_tick

    # E_k (extreme after event k, before event k+1) = prior_extreme of event k+1.
    e_k_price = prior_price[1:]
    e_k_time = prior_time[1:]
    e_k_tick = prior_tick[1:]
    d_k_price = event_price[:-1]
    d_k_time = event_time[:-1]
    d_k_tick = event_tick[:-1]

    # OS_k spans (D_k tick, E_k tick], k = 1..n-1.
    os_move = np.abs(e_k_price - d_k_price)
    os_time = e_k_time - d_k_time
    os_start_tick = d_k_tick
    os_end_tick = e_k_tick

    # TM_k spans (E_{k-1} tick, E_k tick], k = 1..n-1.
    tm_move = dc_move[:-1] + os_move
    tm_time = dc_time[:-1] + os_time
    tm_start_tick = prior_tick[:-1]
    tm_end_tick = e_k_tick

    return {
        "dc_move": dc_move, "dc_time": dc_time,
        "dc_start_tick": dc_start_tick, "dc_end_tick": dc_end_tick,
        "os_move": os_move, "os_time": os_time,
        "os_start_tick": os_start_tick, "os_end_tick": os_end_tick,
        "tm_move": tm_move, "tm_time": tm_time,
        "tm_start_tick": tm_start_tick, "tm_end_tick": tm_end_tick,
    }
