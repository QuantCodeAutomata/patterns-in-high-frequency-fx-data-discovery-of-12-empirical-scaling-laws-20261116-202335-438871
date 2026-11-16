"""Unit tests for fixed physical-time sampling, section 2.1 of Glattfelder,
Dupuis & Olsen (arXiv:0809.1040)."""

from __future__ import annotations

import numpy as np
import pytest

from src.time_sampling import mean_abs_change_p, mean_max_range_p, sample_last_price


def test_sample_last_price_uses_last_observation_carried_forward():
    times = np.array([0.0, 2.0, 2.5, 7.0])
    prices = np.array([1.0, 1.1, 1.2, 1.3])
    sampled = sample_last_price(prices, times, dt_seconds=3.0)
    # sample times: 0, 3, 6 -> last tick at/before: idx0(1.0), idx2(1.2), idx2(1.2)
    np.testing.assert_allclose(sampled, [1.0, 1.2, 1.2])


def test_mean_abs_change_p1_vs_p2_power_mean_inequality():
    rng = np.random.default_rng(2)
    sampled = 1.3 * np.cumprod(1 + rng.normal(0, 0.001, size=2000))
    p1 = mean_abs_change_p(sampled, 1)
    p2 = mean_abs_change_p(sampled, 2)
    # Power mean inequality: <x>_1 <= <x>_2 for a nonnegative quantity.
    assert p1 <= p2
    assert p1 > 0 and p2 > 0


def test_mean_abs_change_p_requires_two_samples():
    with pytest.raises(ValueError):
        mean_abs_change_p(np.array([1.0]), 1)


def test_mean_max_range_p_is_nonnegative_and_grows_with_window(small_grw_series):
    prices, times = small_grw_series
    small = mean_max_range_p(prices, times, dt_seconds=10, p=1)
    large = mean_max_range_p(prices, times, dt_seconds=500, p=1)
    assert small > 0 and large > 0
    assert large > small  # a longer window observes a larger typical range


def test_mean_max_range_p_rejects_too_few_windows():
    prices = np.array([1.0, 1.01])
    times = np.array([0.0, 1.0])
    with pytest.raises(ValueError):
        mean_max_range_p(prices, times, dt_seconds=1_000_000, p=1)
