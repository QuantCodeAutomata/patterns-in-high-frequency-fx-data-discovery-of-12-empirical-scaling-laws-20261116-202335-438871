"""Unit tests for the coastline evaluation of law (12), section 2.3 of
Glattfelder, Dupuis & Olsen (arXiv:0809.1040)."""

from __future__ import annotations

import numpy as np
import pytest

from src.coastline import coastline_at, coastline_ratio
from src.scaling_laws import PowerLawFit


def _fit(E: float, C: float) -> PowerLawFit:
    return PowerLawFit(E=E, C=C, E_se=0.0, C_se=0.0, A=0.0, A_se=0.0,
                        r2_linear=1.0, r2_adj_linear=1.0, r2_quadratic=1.0,
                        delta_r2=0.0, n_obs=250)


def test_coastline_decreases_as_threshold_increases_for_negative_exponent():
    fit = _fit(E=-1.04, C=0.51)
    values = coastline_at(fit, np.array([0.01, 0.1, 1.0, 5.0]))
    assert np.all(np.diff(values) < 0)


def test_coastline_ratio_matches_direct_evaluation():
    fit = _fit(E=-1.04, C=0.51)
    ratio = coastline_ratio(fit, low_pct=0.01, high_pct=5.0)
    direct = coastline_at(fit, np.array([0.01, 5.0]))
    assert ratio == pytest.approx(direct[0] / direct[1])


def test_coastline_ratio_is_greater_than_one_when_resolution_increases():
    fit = _fit(E=-1.04, C=0.51)
    ratio = coastline_ratio(fit, low_pct=0.01, high_pct=5.0)
    assert ratio > 1.0  # finer resolution (lower threshold) -> longer coastline
