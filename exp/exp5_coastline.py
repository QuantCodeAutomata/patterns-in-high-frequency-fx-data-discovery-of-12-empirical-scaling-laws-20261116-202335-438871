"""Experiment 5: coastline-length estimation from scaling law (12) of
Glattfelder, Dupuis & Olsen, "Patterns in high-frequency FX data: Discovery of
12 empirical scaling laws" (arXiv:0809.1040), section 2.3.

The annualised cumulative total-move Delta_x_cum^{tm} is evaluated at the
resolutions 0.01%, 0.1%, 1%, 5% (section 2.3), both without transaction costs
and, using the "cumulative total move with costs" variant fitted only on the
[0.1%, 5%] interval where costs do not break the scaling (section 2.3), by
extrapolating to 0.05%.

See results/RESULTS.md and src/config.DEMO_USES_GRW_ONLY for why this
experiment runs on the paper's own Gaussian random walk benchmark series
(eq. 18) instead of the paper's unavailable tick-by-tick FX quotes; the paper
itself reports a GRW coastline for comparison (section 2.3).

Run: python -m exp.exp5_coastline
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.generate_synthetic import apply_constant_spread, generate_grw_series
from src import config
from src.coastline import coastline_at, coastline_ratio
from src.pipeline import laws_9_10_11_12_table, reference_tick_events, sample_years_from_times
from src.scaling_laws import fit_power_law, threshold_grid

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("exp5")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    prices, times_s = generate_grw_series(
        config.GRW_N_TICKS, config.GRW_X0, config.GRW_DT_SECONDS, config.GRW_SIGMA, config.GRW_SEED
    )
    thresholds = threshold_grid(config.N_THRESHOLD_POINTS, config.THRESHOLD_MIN_PCT, config.THRESHOLD_MAX_PCT)
    tick_events = reference_tick_events(prices, times_s, config.TICK_DEFINITION_PCT)

    sample_years = sample_years_from_times(times_s, config.SECONDS_PER_YEAR)
    log.info("Annualisation divisor (elapsed sample years): %.6f", sample_years)
    laws9_12 = laws_9_10_11_12_table(prices, times_s, thresholds, tick_events, sample_years)
    fit_tm = fit_power_law(laws9_12["threshold_pct"].to_numpy(), laws9_12["cum_tm"].to_numpy())

    coast_pcts = np.array(config.COASTLINE_THRESHOLDS_PCT)
    coast_values = coastline_at(fit_tm, coast_pcts)
    ratio = coastline_ratio(fit_tm, coast_pcts.min(), coast_pcts.max())
    log.info("Coastline (no costs) at %s%%: %s, ratio low/high = %.1f", coast_pcts, coast_values, ratio)

    # Cumulative total move with costs, fitted only on [0.1%, 5%] (section
    # 2.3: costs break the scaling for small thresholds).
    costed_prices = apply_constant_spread(prices, config.GRW_SPREAD_PCT)
    laws9_12_costs = laws_9_10_11_12_table(costed_prices, times_s, thresholds, tick_events, sample_years)
    lo, hi = config.COASTLINE_COST_FIT_RANGE_PCT
    fit_mask = (laws9_12_costs["threshold_pct"] >= lo) & (laws9_12_costs["threshold_pct"] <= hi)
    fit_tm_costs = fit_power_law(
        laws9_12_costs.loc[fit_mask, "threshold_pct"].to_numpy(),
        laws9_12_costs.loc[fit_mask, "cum_tm"].to_numpy(),
    )
    eval_pcts = np.array([config.COASTLINE_COST_ADJUSTED_EVAL_PCT, 0.1, 1.0, 5.0])
    coast_costs = coastline_at(fit_tm_costs, eval_pcts)
    daily_move = coast_costs[0] / config.TRADING_DAYS_PER_YEAR
    log.info("Coastline (with costs) at %s%%: %s; daily move at %.2f%% = %.3f%%",
              eval_pcts, coast_costs, config.COASTLINE_COST_ADJUSTED_EVAL_PCT, daily_move)

    result = {
        "no_costs": {f"{p}%": float(v) for p, v in zip(coast_pcts, coast_values)},
        "no_costs_ratio_low_over_high": float(ratio),
        "with_costs_fit_range_pct": list(config.COASTLINE_COST_FIT_RANGE_PCT),
        "with_costs": {f"{p}%": float(v) for p, v in zip(eval_pcts, coast_costs)},
        "daily_move_at_eval_pct": float(daily_move),
        "fit_tm_E": float(fit_tm.E), "fit_tm_C": float(fit_tm.C),
        "fit_tm_costs_E": float(fit_tm_costs.E), "fit_tm_costs_C": float(fit_tm_costs.C),
    }
    with open(RESULTS_DIR / "exp5_coastline.json", "w") as f:
        json.dump(result, f, indent=2)
    log.info("Coastline results: %s", result)

    fig, ax = plt.subplots(figsize=(7, 5))
    full_grid = threshold_grid(100, config.THRESHOLD_MIN_PCT, config.THRESHOLD_MAX_PCT)
    ax.loglog(full_grid, coastline_at(fit_tm, full_grid), label="no costs (fitted)")
    ax.loglog(laws9_12["threshold_pct"], laws9_12["cum_tm"], ".", ms=3, label="measured (no costs)")
    ax.loglog(full_grid, coastline_at(fit_tm_costs, full_grid), "--", label="with costs (fit on [0.1%,5%])")
    ax.set_xlabel("Delta_x_dc (%)")
    ax.set_ylabel("annualised cumulative total move (%)")
    ax.set_title("Experiment 5: coastline length vs resolution (GRW-benchmark demonstration)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp5_coastline.png", dpi=130)
    plt.close(fig)

    log.info("exp5 finished in %.1f s", time.time() - t0)


if __name__ == "__main__":
    run()
