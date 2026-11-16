"""Unit tests for the cross-law consistency checks, equations (14)-(17) of
Glattfelder, Dupuis & Olsen (arXiv:0809.1040)."""

from __future__ import annotations

import pytest

from src.consistency_checks import (
    check_14,
    check_15,
    check_16,
    check_decomposition_sum,
    combined_law_17,
)


def test_check_14_recovers_exact_relationship_by_construction():
    E_Nx, C_Nx, Y = -1.93, 9.5, 31_553_280.0
    E_tx = -E_Nx
    C_tx = Y ** (1.0 / E_Nx) * C_Nx
    e_res, c_res = check_14(E_Nx, C_Nx, Y, E_tx, C_tx)
    assert e_res.rel_error == pytest.approx(0.0, abs=1e-9)
    assert c_res.rel_error == pytest.approx(0.0, abs=1e-9)


def test_check_15_recovers_exact_relationship_by_construction():
    E_x1, C_x1 = 0.497, 6.632e5
    E_tx = 1.0 / E_x1
    C_tx = C_x1 ** (-E_tx)
    e_res, c_res = check_15(E_x1, C_x1, E_tx, C_tx)
    assert e_res.rel_error == pytest.approx(0.0, abs=1e-9)
    assert c_res.rel_error == pytest.approx(0.0, abs=1e-9)


def test_check_16_matches_decomposition_identity():
    result = check_16(mean_abs_move=0.2132, cum_move=1244.2, n_dc=5843.4)
    assert result.predicted == pytest.approx(1244.2 / 5843.4)
    assert result.rel_error < 0.01


def test_combined_law_17_matches_hand_computation():
    E_t_tck, C_t_tck = combined_law_17(E_x1=0.497, C_x1=6.632e5, E_Ntck=1.93, C_Ntck=9.5)
    assert E_t_tck == pytest.approx(0.497 * 1.93)
    assert C_t_tck == pytest.approx(6.632e5 * 9.5 ** (1.0 / 0.497))


def test_decomposition_sum_within_tolerance():
    result = check_decomposition_sum(cum_dc=600.0, cum_os=640.0, cum_tm=1244.2)
    assert result.rel_error < 0.01
