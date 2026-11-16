"""Unit tests for Algorithm 1 (price-move count), section 3.2 of
Glattfelder, Dupuis & Olsen (arXiv:0809.1040)."""

from __future__ import annotations

import numpy as np
import pytest

from src.price_move_algorithm import (
    count_events_between,
    price_move_events,
    segment_durations,
    segment_tick_counts,
)


def test_algorithm1_has_no_mode_both_directions_count_immediately():
    # A pure up-move then a pure down-move, each >= threshold.
    prices = np.array([1.00, 1.02, 1.00])
    times = np.array([0.0, 1.0, 2.0])
    ev = price_move_events(prices, times, threshold_pct=1.0)
    assert ev["event_price"].shape[0] == 2
    assert list(ev["direction"]) == [1, -1]


def test_more_events_for_smaller_threshold(small_grw_series):
    prices, times = small_grw_series
    n_small = price_move_events(prices, times, 0.05)["event_price"].shape[0]
    n_large = price_move_events(prices, times, 0.5)["event_price"].shape[0]
    assert n_small > n_large > 0


def test_segment_tick_counts_are_nonnegative_and_consistent(small_grw_series):
    prices, times = small_grw_series
    tick_ev = price_move_events(prices, times, 0.02)
    ev = price_move_events(prices, times, 0.2)
    counts = segment_tick_counts(ev["event_tick_index"], tick_ev["event_tick_index"])
    assert counts.shape[0] == ev["event_tick_index"].shape[0] - 1
    assert np.all(counts >= 0)


def test_count_events_between_matches_brute_force():
    reference = np.array([2, 5, 9, 15, 20])
    start = np.array([0, 5, 10])
    end = np.array([5, 15, 20])
    result = count_events_between(reference, start, end)
    expected = np.array([
        np.sum((reference > 0) & (reference <= 5)),
        np.sum((reference > 5) & (reference <= 15)),
        np.sum((reference > 10) & (reference <= 20)),
    ])
    np.testing.assert_array_equal(result, expected)


def test_segment_durations_matches_diff():
    event_time = np.array([0.0, 1.5, 4.0, 10.0])
    out = segment_durations(event_time)
    np.testing.assert_allclose(out, np.diff(event_time))


def test_rejects_mismatched_lengths_and_nonpositive_threshold():
    prices = np.array([1.0, 1.01, 1.02])
    times = np.array([0.0, 1.0, 2.0])
    with pytest.raises(ValueError):
        price_move_events(prices, times[:-1], 0.1)
    with pytest.raises(ValueError):
        price_move_events(prices, times, -0.1)
