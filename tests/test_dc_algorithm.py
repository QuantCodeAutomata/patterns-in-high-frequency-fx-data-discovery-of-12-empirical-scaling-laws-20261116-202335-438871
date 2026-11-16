"""Unit tests for Algorithm 2 (directional change) and its decomposition,
section 3.2 / 2.2 of Glattfelder, Dupuis & Olsen (arXiv:0809.1040)."""

from __future__ import annotations

import numpy as np
import pytest

from src.dc_algorithm import decompose_segments, directional_change_events


def test_known_zigzag_sequence_detects_expected_events():
    # A hand-built zigzag with a 1% threshold: clear up/down swings.
    prices = np.array([1.00, 1.012, 1.020, 1.005, 0.995, 1.010, 1.025])
    times = np.arange(prices.shape[0], dtype=np.float64)
    ev = directional_change_events(prices, times, threshold_pct=1.0)
    # Price rises from 1.00 to 1.020 (+2%) -> up event once >=1% is crossed.
    assert ev["event_price"].shape[0] >= 1
    assert set(ev["direction"]).issubset({1, -1})


def test_small_threshold_detects_more_events_than_large_threshold():
    rng = np.random.default_rng(0)
    rel = rng.normal(0, 0.001, size=5000)
    prices = 1.3 * np.cumprod(1 + rel)
    times = np.arange(prices.shape[0], dtype=np.float64)
    n_small = directional_change_events(prices, times, 0.05)["event_price"].shape[0]
    n_large = directional_change_events(prices, times, 0.5)["event_price"].shape[0]
    assert n_small > n_large > 0


def test_decomposition_tm_equals_dc_plus_os(small_grw_series):
    prices, times = small_grw_series
    ev = directional_change_events(prices, times, threshold_pct=0.1)
    seg = decompose_segments(ev)
    assert seg["tm_move"].shape[0] == seg["dc_move"].shape[0] - 1
    np.testing.assert_allclose(seg["tm_move"], seg["dc_move"][:-1] + seg["os_move"])
    np.testing.assert_allclose(seg["tm_time"], seg["dc_time"][:-1] + seg["os_time"])


def test_all_moves_and_times_are_nonnegative(small_grw_series):
    prices, times = small_grw_series
    ev = directional_change_events(prices, times, threshold_pct=0.2)
    seg = decompose_segments(ev)
    for key in ("dc_move", "os_move", "tm_move"):
        assert np.all(seg[key] >= 0)
    for key in ("dc_time", "os_time", "tm_time"):
        assert np.all(seg[key] >= 0)


def test_decompose_segments_with_fewer_than_two_events_returns_empty():
    ev = {
        "event_price": np.array([1.0]), "event_time": np.array([0.0]),
        "event_tick_index": np.array([0]),
        "prior_extreme_price": np.array([1.0]), "prior_extreme_time": np.array([0.0]),
        "prior_extreme_tick_index": np.array([0]),
    }
    seg = decompose_segments(ev)
    assert seg["dc_move"].shape[0] == 0
    assert seg["tm_move"].shape[0] == 0


def test_rejects_mismatched_lengths_and_nonpositive_threshold():
    prices = np.array([1.0, 1.01, 1.02])
    times = np.array([0.0, 1.0, 2.0])
    with pytest.raises(ValueError):
        directional_change_events(prices, times[:-1], 0.1)
    with pytest.raises(ValueError):
        directional_change_events(prices, times, 0.0)
