"""Cross-validation between the fitted scaling laws, as in the consistency
checks of Glattfelder, Dupuis & Olsen (arXiv:0809.1040), equations (14)-(17).

(14) Laws (2) and (4) are two views of the same price-move process, counted
     per unit time and per unit threshold respectively:
         E_t,x = -E_N,x        and   C_t,x = Y^(1/E_N,x) * C_N,x
     where Y = 31,553,280 seconds/year (section "Consistency checks").
(14b) The equivalent relationship between laws (0b) and (5):
         E_t,dc = -E_N,dc      and   C_t,dc = Y^(1/E_N,dc) * C_N,dc
(15) Laws (0a) p=1 and (4) relate the fixed-time and fixed-threshold views:
         E_t,x = 1 / E_x(1)    and   C_t,x = C_x(1)^(-E_x(1))
(16) Decomposition consistency: <|Delta x^*|> = Delta_x_cum^* / N(Delta_x_dc)
     for * in {tm, dc, os}.
(17) Combined law from (0a p=1) and (1):
         <N(Delta_x_tck)> = (Delta_t / C_t,tck)^E_t,tck
         E_t,tck = E_x(1) * E_N,tck     C_t,tck = C_x(1) * C_N,tck^(1/E_x(1))
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConsistencyResult:
    predicted: float
    measured: float
    rel_error: float


def _rel_error(measured: float, predicted: float) -> float:
    return abs(measured - predicted) / abs(predicted)


def check_14(E_Nx: float, C_Nx: float, seconds_per_year: float, E_tx: float, C_tx: float):
    """Laws (2)&(4), or equivalently (0b)&(5) with the same formula."""
    E_pred = -E_Nx
    C_pred = seconds_per_year ** (1.0 / E_Nx) * C_Nx
    return (
        ConsistencyResult(E_pred, E_tx, _rel_error(E_tx, E_pred)),
        ConsistencyResult(C_pred, C_tx, _rel_error(C_tx, C_pred)),
    )


def check_15(E_x1: float, C_x1: float, E_tx: float, C_tx: float):
    """Laws (0a p=1) and (4)."""
    E_pred = 1.0 / E_x1
    C_pred = C_x1 ** (-E_x1)
    return (
        ConsistencyResult(E_pred, E_tx, _rel_error(E_tx, E_pred)),
        ConsistencyResult(C_pred, C_tx, _rel_error(C_tx, C_pred)),
    )


def check_16(mean_abs_move: float, cum_move: float, n_dc: float) -> ConsistencyResult:
    """<|Delta x^*|> = Delta_x_cum^* / N(Delta_x_dc), for a single threshold."""
    predicted = cum_move / n_dc
    return ConsistencyResult(predicted, mean_abs_move, _rel_error(mean_abs_move, predicted))


def combined_law_17(E_x1: float, C_x1: float, E_Ntck: float, C_Ntck: float):
    """Derive law (17): E_t,tck = E_x(1) * E_N,tck, C_t,tck = C_x(1) * C_N,tck^(1/E_x(1))."""
    E_t_tck = E_x1 * E_Ntck
    C_t_tck = C_x1 * C_Ntck ** (1.0 / E_x1)
    return E_t_tck, C_t_tck


def check_decomposition_sum(cum_dc: float, cum_os: float, cum_tm: float) -> ConsistencyResult:
    """Delta_x_cum^tm = Delta_x_cum^dc + Delta_x_cum^os (within ~0.4% error)."""
    predicted = cum_dc + cum_os
    return ConsistencyResult(predicted, cum_tm, _rel_error(cum_tm, predicted))
