"""Shared computation of every scaling-law table from one price/time series.

This module is the single entry point every experiment script calls: it wires
together Algorithm 1 (:mod:`src.price_move_algorithm`), Algorithm 2
(:mod:`src.dc_algorithm`), the fixed-time sampling of :mod:`src.time_sampling`
and the log-log fitting of :mod:`src.scaling_laws`, following Glattfelder,
Dupuis & Olsen, "Patterns in high-frequency FX data: Discovery of 12 empirical
scaling laws" (arXiv:0809.1040), sections 2 and 3.

All public functions take ``prices`` (mid-prices, free of consecutive
duplicates) and ``times`` (seconds, strictly increasing, same length as
``prices``) and a grid of thresholds (percent) or time intervals (seconds).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.dc_algorithm import decompose_segments, directional_change_events
from src.price_move_algorithm import (
    count_events_between,
    price_move_events,
    segment_durations,
    segment_tick_counts,
)
from src.scaling_laws import PowerLawFit, fit_power_law
from src.time_sampling import mean_abs_change_p, mean_max_range_p, sample_last_price


def sample_years_from_times(times: np.ndarray, seconds_per_year: float) -> float:
    """Elapsed time of the series expressed in years, used as the
    annualisation divisor for laws (0b), (2) and (12).

    Section 3.1 annualises the 5-year empirical sample by dividing by 5; the
    general form used here divides by the series' own elapsed time so the
    same code annualises both the paper's 5-year FX sample and a
    shorter-lived series (e.g. the Gaussian random walk benchmark of
    eq. 18, section 4, whose unitary one-second ticks span about 11.6 days)
    to the same "events or price move per year" rate.
    """
    span_seconds = float(times[-1] - times[0])
    return span_seconds / seconds_per_year


def reference_tick_events(prices: np.ndarray, times: np.ndarray, tick_threshold_pct: float) -> dict:
    """Algorithm 1 run at the law-defined tick threshold (0.02%, section 3.2),
    used by every function below that needs to count "ticks" inside a
    segment."""
    return price_move_events(prices, times, tick_threshold_pct)


def law_0b_5_table(
    prices: np.ndarray, times: np.ndarray, thresholds_pct: np.ndarray, years_in_sample: float
) -> pd.DataFrame:
    """Law (0b): annualised directional-change count N(Delta_x_dc).
    Law (5): mean time between consecutive directional changes <Delta t_dc>.
    """
    rows = []
    for thr in thresholds_pct:
        ev = directional_change_events(prices, times, thr)
        n_events = ev["event_price"].shape[0]
        if n_events < 2:
            rows.append({"threshold_pct": thr, "n_dc_raw": n_events,
                          "n_dc_annualized": np.nan, "mean_dt_dc": np.nan})
            continue
        # <Delta t_dc> (law 5) is the time between consecutive directional-
        # change confirmations D_k -> D_{k+1}. In a long, stationary series
        # this converges to the same average as the tm-cycle time
        # E_{k-1} -> E_k (each D_k->D_{k+1} gap is OS_k + DC_{k+1}, and
        # stationarity makes E[OS_k] + E[DC_{k+1}] ~ E[OS] + E[DC] = E[TM]),
        # matching the paper's GRW table where law (5) sits next to law (10)
        # tm, not law (10) dc.
        dt = np.diff(ev["event_time"])
        rows.append({
            "threshold_pct": thr,
            "n_dc_raw": n_events,
            "n_dc_annualized": n_events / years_in_sample,
            "mean_dt_dc": float(np.mean(dt)),
        })
    return pd.DataFrame(rows)


def laws_9_10_11_12_table(
    prices: np.ndarray,
    times: np.ndarray,
    thresholds_pct: np.ndarray,
    tick_events: dict,
    years_in_sample: float,
) -> pd.DataFrame:
    """Laws (9)-(12): total-move / directional-change / overshoot decomposition.

    For each component ``* in {tm, dc, os}``: mean absolute move (9), mean
    duration (10), mean number of 0.02%-ticks (11), and the annualised
    cumulative move (12).
    """
    tick_idx = tick_events["event_tick_index"]
    rows = []
    for thr in thresholds_pct:
        ev = directional_change_events(prices, times, thr)
        seg = decompose_segments(ev)
        row = {"threshold_pct": thr, "n_dc": ev["event_price"].shape[0]}
        for comp in ("tm", "dc", "os"):
            move = seg[f"{comp}_move"]
            time_ = seg[f"{comp}_time"]
            if move.shape[0] == 0:
                row[f"mean_abs_{comp}"] = np.nan
                row[f"mean_dt_{comp}"] = np.nan
                row[f"mean_nticks_{comp}"] = np.nan
                row[f"cum_{comp}"] = np.nan
                continue
            nticks = count_events_between(tick_idx, seg[f"{comp}_start_tick"], seg[f"{comp}_end_tick"])
            row[f"mean_abs_{comp}"] = float(np.mean(move))
            row[f"mean_dt_{comp}"] = float(np.mean(time_))
            row[f"mean_nticks_{comp}"] = float(np.mean(nticks))
            row[f"cum_{comp}"] = float(np.sum(move)) / years_in_sample
        rows.append(row)
    return pd.DataFrame(rows)


def laws_1_2_4_table(
    prices: np.ndarray,
    times: np.ndarray,
    thresholds_pct: np.ndarray,
    tick_events: dict,
    years_in_sample: float,
) -> pd.DataFrame:
    """Law (1): mean number of 0.02%-ticks per price move <N(Delta_x_tck)>.
    Law (2): annualised count of price moves N(Delta_x).
    Law (4): mean time interval between consecutive price moves <Delta t_x>.
    """
    tick_idx = tick_events["event_tick_index"]
    rows = []
    for thr in thresholds_pct:
        ev = price_move_events(prices, times, thr)
        n_events = ev["event_price"].shape[0]
        if n_events < 2:
            rows.append({"threshold_pct": thr, "mean_nticks": np.nan,
                          "n_annualized": np.nan, "mean_dt": np.nan})
            continue
        nticks = segment_tick_counts(ev["event_tick_index"], tick_idx)
        durations = segment_durations(ev["event_time"])
        rows.append({
            "threshold_pct": thr,
            "mean_nticks": float(np.mean(nticks)) if nticks.shape[0] else np.nan,
            "n_annualized": n_events / years_in_sample,
            "mean_dt": float(np.mean(durations)) if durations.shape[0] else np.nan,
        })
    return pd.DataFrame(rows)


def laws_0a_3_table(
    prices: np.ndarray, times: np.ndarray, dt_grid_seconds: np.ndarray
) -> pd.DataFrame:
    """Law (0a): fixed-time mean (p=1) and quadratic-mean (p=2) absolute price
    change. Law (3): fixed-time mean (p=1, p=2) maximal price range.
    """
    rows = []
    for dt in dt_grid_seconds:
        row = {"dt_seconds": dt}
        try:
            sampled = sample_last_price(prices, times, dt)
            row["mean_abs_p1"] = mean_abs_change_p(sampled, 1)
            row["mean_abs_p2"] = mean_abs_change_p(sampled, 2)
        except ValueError:
            row["mean_abs_p1"] = np.nan
            row["mean_abs_p2"] = np.nan
        try:
            row["mean_max_p1"] = mean_max_range_p(prices, times, dt, 1)
            row["mean_max_p2"] = mean_max_range_p(prices, times, dt, 2)
        except ValueError:
            row["mean_max_p1"] = np.nan
            row["mean_max_p2"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def fit_columns(table: pd.DataFrame, x_col: str, columns: list[str]) -> pd.DataFrame:
    """Fit :func:`src.scaling_laws.fit_power_law` of each ``columns`` entry of
    ``table`` against ``table[x_col]`` and collect the parameters in one row
    per law."""
    x = table[x_col].to_numpy(dtype=np.float64)
    rows = []
    for col in columns:
        y = table[col].to_numpy(dtype=np.float64)
        try:
            fit = fit_power_law(x, y)
            rows.append({
                "law": col, "E": fit.E, "C": fit.C, "E_se": fit.E_se, "C_se": fit.C_se,
                "r2_linear": fit.r2_linear, "r2_adj_linear": fit.r2_adj_linear,
                "r2_quadratic": fit.r2_quadratic, "delta_r2": fit.delta_r2,
                "n_obs": fit.n_obs, "error": None,
            })
        except ValueError as exc:
            rows.append({
                "law": col, "E": np.nan, "C": np.nan, "E_se": np.nan, "C_se": np.nan,
                "r2_linear": np.nan, "r2_adj_linear": np.nan, "r2_quadratic": np.nan,
                "delta_r2": np.nan, "n_obs": 0, "error": str(exc),
            })
    return pd.DataFrame(rows)


def fit_lookup(fit_table: pd.DataFrame, law: str) -> PowerLawFit:
    """Reconstruct a :class:`PowerLawFit` from one row of a table produced by
    :func:`fit_columns`, for use by consistency checks across laws."""
    row = fit_table.loc[fit_table["law"] == law].iloc[0]
    return PowerLawFit(
        E=row["E"], C=row["C"], E_se=row["E_se"], C_se=row["C_se"],
        A=np.nan, A_se=np.nan, r2_linear=row["r2_linear"], r2_adj_linear=row["r2_adj_linear"],
        r2_quadratic=row["r2_quadratic"], delta_r2=row["delta_r2"], n_obs=int(row["n_obs"]),
    )
