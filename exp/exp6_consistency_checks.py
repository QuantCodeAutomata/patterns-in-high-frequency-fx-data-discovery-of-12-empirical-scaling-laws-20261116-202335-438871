"""Experiment 6: cross-validation between the fitted scaling laws of
Glattfelder, Dupuis & Olsen, "Patterns in high-frequency FX data: Discovery of
12 empirical scaling laws" (arXiv:0809.1040), equations (14)-(17).

Fits every law needed for the consistency checks on one price series and
verifies: the inverse relationship between laws (2) and (4) [eq. 14]; the
equivalent relationship between laws (0b) and (5) [eq. 14, same formula];
the relationship between laws (0a) p=1 and (4) [eq. 15]; the decomposition
<|Delta x^*|> = Delta_x_cum^* / N(Delta_x_dc) for * in {tm, dc, os} at
Delta_x_dc = 0.1% [eq. 16]; the cumulative decomposition Delta_x_cum^tm =
Delta_x_cum^dc + Delta_x_cum^os [section 2.2]; and the combined law (17)
derived from laws (0a p=1) and (1).

See results/RESULTS.md and src/config.DEMO_USES_GRW_ONLY for why this
experiment runs on the paper's own Gaussian random walk benchmark series
(eq. 18) instead of the paper's unavailable tick-by-tick FX quotes: the
consistency relations are properties of the fitting procedure and hold for
any well-fitted scaling-law set, independent of which price series they are
measured on.

Run: python -m exp.exp6_consistency_checks
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data.generate_synthetic import generate_grw_series
from src import config
from src.consistency_checks import check_14, check_15, check_16, check_decomposition_sum, combined_law_17
from src.pipeline import (
    fit_columns,
    law_0b_5_table,
    laws_0a_3_table,
    laws_1_2_4_table,
    laws_9_10_11_12_table,
    reference_tick_events,
    sample_years_from_times,
)
from src.scaling_laws import fit_power_law, threshold_grid, time_grid

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("exp6")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    prices, times_s = generate_grw_series(
        config.GRW_N_TICKS, config.GRW_X0, config.GRW_DT_SECONDS, config.GRW_SIGMA, config.GRW_SEED
    )
    span_s = times_s[-1] - times_s[0]
    thresholds = threshold_grid(config.N_THRESHOLD_POINTS, config.THRESHOLD_MIN_PCT, config.THRESHOLD_MAX_PCT)
    tick_events = reference_tick_events(prices, times_s, config.TICK_DEFINITION_PCT)

    sample_years = sample_years_from_times(times_s, config.SECONDS_PER_YEAR)
    log.info("Annualisation divisor (elapsed sample years): %.6f", sample_years)
    law0b5 = law_0b_5_table(prices, times_s, thresholds, sample_years)
    laws9_12 = laws_9_10_11_12_table(prices, times_s, thresholds, tick_events, sample_years)
    laws1_2_4 = laws_1_2_4_table(prices, times_s, thresholds, tick_events, sample_years)
    dt_grid = time_grid(config.N_TIME_POINTS, config.TIME_MIN_SECONDS, config.TIME_MAX_SECONDS)
    dt_grid = dt_grid[dt_grid < span_s / 4]
    laws0a3 = laws_0a_3_table(prices, times_s, dt_grid)

    fit_Nx = fit_power_law(laws1_2_4["threshold_pct"].to_numpy(), laws1_2_4["n_annualized"].to_numpy())
    fit_tx = fit_power_law(laws1_2_4["threshold_pct"].to_numpy(), laws1_2_4["mean_dt"].to_numpy())
    fit_Ndc = fit_power_law(law0b5["threshold_pct"].to_numpy(), law0b5["n_dc_annualized"].to_numpy())
    fit_tdc = fit_power_law(law0b5["threshold_pct"].to_numpy(), law0b5["mean_dt_dc"].to_numpy())
    fit_x1 = fit_power_law(laws0a3["dt_seconds"].to_numpy(), laws0a3["mean_abs_p1"].to_numpy())
    fit_Ntck = fit_power_law(laws1_2_4["threshold_pct"].to_numpy(), laws1_2_4["mean_nticks"].to_numpy())

    results = {}

    e14, c14 = check_14(fit_Nx.E, fit_Nx.C, config.SECONDS_PER_YEAR, fit_tx.E, fit_tx.C)
    results["eq14_laws_2_4"] = {"E": vars(e14), "C": vars(c14)}

    e14b, c14b = check_14(fit_Ndc.E, fit_Ndc.C, config.SECONDS_PER_YEAR, fit_tdc.E, fit_tdc.C)
    results["eq14_laws_0b_5"] = {"E": vars(e14b), "C": vars(c14b)}

    e15, c15 = check_15(fit_x1.E, fit_x1.C, fit_tx.E, fit_tx.C)
    results["eq15_laws_0a1_4"] = {"E": vars(e15), "C": vars(c15)}

    # eq. (16): decomposition at threshold closest to 0.1%.
    idx = int(np.argmin(np.abs(laws9_12["threshold_pct"].to_numpy() - 0.1)))
    row = laws9_12.iloc[idx]
    decomposition = {}
    for comp in ("tm", "dc", "os"):
        res = check_16(row[f"mean_abs_{comp}"], row[f"cum_{comp}"] * sample_years, row["n_dc"])
        decomposition[comp] = vars(res)
    results["eq16_decomposition_at_0.1pct"] = decomposition

    sum_check = check_decomposition_sum(row["cum_dc"], row["cum_os"], row["cum_tm"])
    results["cumulative_sum_dc_plus_os_eq_tm"] = vars(sum_check)

    e_t_tck, c_t_tck = combined_law_17(fit_x1.E, fit_x1.C, fit_Ntck.E, fit_Ntck.C)
    results["eq17_combined_law"] = {"E_t_tck": e_t_tck, "C_t_tck": c_t_tck}
    tick_every_s_law4 = float(np.interp(config.TICK_DEFINITION_PCT, laws1_2_4["threshold_pct"], laws1_2_4["mean_dt"]))
    tick_every_s_law17 = float(c_t_tck) if e_t_tck != 0 else np.nan
    results["tick_interval_comparison"] = {
        "law_4_direct_seconds": tick_every_s_law4,
        "law_17_combined_seconds": tick_every_s_law17,
    }

    with open(RESULTS_DIR / "exp6_consistency_checks.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    log.info("Consistency checks: %s", json.dumps(results, indent=2, default=float))

    log.info("exp6 finished in %.1f s", time.time() - t0)


if __name__ == "__main__":
    run()
