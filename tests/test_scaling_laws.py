"""Unit tests for log-log power-law fitting, section 3.3 of Glattfelder,
Dupuis & Olsen (arXiv:0809.1040)."""

from __future__ import annotations

import numpy as np
import pytest

from src.scaling_laws import fit_power_law, threshold_grid, time_grid


def test_fit_recovers_known_power_law_exactly():
    E_true, C_true = -2.0, 3.0
    x = np.linspace(0.5, 10.0, 50)
    y = (x / C_true) ** E_true
    fit = fit_power_law(x, y)
    assert fit.E == pytest.approx(E_true, rel=1e-6)
    assert fit.C == pytest.approx(C_true, rel=1e-6)
    assert fit.r2_linear == pytest.approx(1.0, abs=1e-8)
    assert fit.n_obs == 50


def test_fit_with_noise_has_high_r2_and_correct_sign():
    rng = np.random.default_rng(1)
    E_true, C_true = 2.0, 0.02
    x = np.exp(np.linspace(np.log(0.01), np.log(5.0), 200))
    y = (x / C_true) ** E_true * np.exp(rng.normal(0, 0.01, size=x.shape[0]))
    fit = fit_power_law(x, y)
    assert fit.r2_adj_linear > 0.99
    assert np.sign(fit.E) == np.sign(E_true)
    assert fit.E == pytest.approx(E_true, rel=0.05)


def test_fit_drops_nonpositive_and_nonfinite_points():
    x = np.array([1, 2, 3, 4, 5, -1, 0, np.nan, np.inf], dtype=np.float64)
    y = np.array([1, 4, 9, 16, 25, 1, 1, 1, 1], dtype=np.float64)
    fit = fit_power_law(x, y)
    assert fit.n_obs == 5


def test_fit_requires_at_least_three_points():
    with pytest.raises(ValueError):
        fit_power_law(np.array([1.0, 2.0]), np.array([1.0, 4.0]))


def test_threshold_grid_matches_paper_spec():
    grid = threshold_grid(250, 0.01, 5.05)
    assert grid.shape[0] == 250
    assert grid[0] == pytest.approx(0.01)
    assert grid[-1] == pytest.approx(5.05)
    assert np.all(np.diff(grid) > 0)
    log_step = np.diff(np.log(grid))
    assert log_step == pytest.approx(log_step[0], rel=1e-9)  # uniform log spacing


def test_time_grid_matches_paper_spec():
    grid = time_grid(245, 20, 3_975_783)
    assert grid.shape[0] == 245
    assert grid[0] == pytest.approx(20)
    assert grid[-1] == pytest.approx(3_975_783)
