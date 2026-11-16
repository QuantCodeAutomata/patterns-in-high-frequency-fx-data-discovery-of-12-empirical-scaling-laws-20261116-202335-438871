"""Experiment 1: directional-change event-based scaling laws (0b, 5, 9, 10,
11, 12) of Glattfelder, Dupuis & Olsen, "Patterns in high-frequency FX data:
Discovery of 12 empirical scaling laws" (arXiv:0809.1040), sections 2.2, 3.2
and 3.3.

Algorithm 2 (directional-change detection) is run over the price series at
250 log-spaced thresholds Delta_x_dc from 0.01% to 5.05% (section 3.3). Each
run yields the directional-change count (law 0b), the mean directional-change
duration (law 5), and the total-move / directional-change / overshoot
decomposition (laws 9-12). Every law is fit in log-log space as
Y = (Delta_x_dc / C)^E.

The paper's sample is tick-by-tick bid/ask quotes for 13 currency pairs from
December 1, 2002 to December 1, 2007 (section 3.1); see results/RESULTS.md
and src/config.DEMO_USES_GRW_ONLY for why this experiment instead runs on the
paper's own Gaussian random walk benchmark series (eq. 18).

Run: python -m exp.exp1_dc_scaling_laws
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.generate_synthetic import generate_grw_series
from src import config
from src.pipeline import (
    law_0b_5_table,
    laws_9_10_11_12_table,
    reference_tick_events,
    sample_years_from_times,
)
from src.scaling_laws import threshold_grid

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("exp1")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def fit_one(x: np.ndarray, y: np.ndarray, label: str) -> dict:
    from src.scaling_laws import fit_power_law

    fit = fit_power_law(x, y)
    return {
        "law": label, "E": fit.E, "C": fit.C, "E_se": fit.E_se, "C_se": fit.C_se,
        "r2_adj_linear": fit.r2_adj_linear, "delta_r2": fit.delta_r2, "n_obs": fit.n_obs,
    }


def run(full: bool) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    if full:
        log.warning(
            "The --full flag would require tick-by-tick bid/ask FX quotes for 13 "
            "currency pairs (2002-2007); see results/RESULTS.md for why this data "
            "is not obtainable through the QuantX Dataloader or Massive REST APIs. "
            "Running the GRW-benchmark demonstration configuration instead."
        )

    prices, times = generate_grw_series(
        config.GRW_N_TICKS, config.GRW_X0, config.GRW_DT_SECONDS, config.GRW_SIGMA, config.GRW_SEED
    )
    log.info("GRW series: %d ticks, span %.1f s (%.2f days)", prices.shape[0],
              times[-1] - times[0], (times[-1] - times[0]) / 86400)

    thresholds = threshold_grid(config.N_THRESHOLD_POINTS, config.THRESHOLD_MIN_PCT, config.THRESHOLD_MAX_PCT)
    tick_events = reference_tick_events(prices, times, config.TICK_DEFINITION_PCT)
    log.info("Reference 0.02%% tick events: %d", tick_events["event_price"].shape[0])

    # Section 3.1 annualises the 5-year empirical sample by dividing by 5.
    # The GRW benchmark's own published table (section 4) is not obtained
    # that way: its one-million-tick sample spans only ~11.6 days
    # (sample_years below), and dividing by that fraction of a year would
    # extrapolate the raw counts by a factor of ~31.5, inflating the fitted
    # constants of laws (0b) and (12) far past the paper's reported GRW
    # values. Using the raw totals (divisor 1) instead reproduces them, so
    # - matching experiment 4 - this run of the GRW series (the only series
    # available in place of the unobtainable tick-by-tick FX quotes) uses
    # divisor 1.
    sample_years = sample_years_from_times(times, config.SECONDS_PER_YEAR)
    log.info("Elapsed sample years of the GRW series: %.6f (not used as the divisor)", sample_years)
    grw_count_divisor = 1.0

    law0b5 = law_0b_5_table(prices, times, thresholds, grw_count_divisor)
    laws9_12 = laws_9_10_11_12_table(prices, times, thresholds, tick_events, grw_count_divisor)
    table = law0b5.merge(laws9_12, on="threshold_pct", suffixes=("", "_dup"))
    table.to_csv(RESULTS_DIR / "exp1_dc_event_table.csv", index=False)

    fits = []
    fits.append(fit_one(table["threshold_pct"].to_numpy(), table["n_dc_annualized"].to_numpy(), "law_0b_N_dc"))
    fits.append(fit_one(table["threshold_pct"].to_numpy(), table["mean_dt_dc"].to_numpy(), "law_5_t_dc"))
    for comp in ("tm", "dc", "os"):
        fits.append(fit_one(table["threshold_pct"].to_numpy(), table[f"mean_abs_{comp}"].to_numpy(), f"law_9_x_{comp}"))
        fits.append(fit_one(table["threshold_pct"].to_numpy(), table[f"mean_dt_{comp}"].to_numpy(), f"law_10_t_{comp}"))
        fits.append(fit_one(table["threshold_pct"].to_numpy(), table[f"mean_nticks_{comp}"].to_numpy(), f"law_11_N_{comp}"))
        fits.append(fit_one(table["threshold_pct"].to_numpy(), table[f"cum_{comp}"].to_numpy(), f"law_12_cum_{comp}"))
    fit_df = pd.DataFrame(fits)
    fit_df.to_csv(RESULTS_DIR / "exp1_dc_fits.csv", index=False)
    log.info("Fitted %d scaling laws, saved exp1_dc_fits.csv", len(fit_df))

    # Research-objective sanity checks (paper section 2.2): the total move is
    # approximately twice the directional-change threshold, and the overshoot
    # is approximately equal to it, at a representative mid-sample threshold.
    mid_idx = int(np.argmin(np.abs(table["threshold_pct"].to_numpy() - 0.5)))
    mid_row = table.iloc[mid_idx]
    checks = {
        "threshold_pct_used": float(mid_row["threshold_pct"]),
        "mean_abs_tm": float(mid_row["mean_abs_tm"]),
        "mean_abs_dc": float(mid_row["mean_abs_dc"]),
        "mean_abs_os": float(mid_row["mean_abs_os"]),
        "tm_over_2x_threshold": float(mid_row["mean_abs_tm"] / (2 * mid_row["threshold_pct"])),
        "os_over_threshold": float(mid_row["mean_abs_os"] / mid_row["threshold_pct"]),
        "decomposition_tm_minus_dc_plus_os": float(mid_row["mean_abs_tm"] - (mid_row["mean_abs_dc"] + mid_row["mean_abs_os"])),
    }
    with open(RESULTS_DIR / "exp1_decomposition_checks.json", "w") as f:
        json.dump(checks, f, indent=2)
    log.info("Decomposition checks: %s", checks)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (col, title) in zip(
        axes.flat,
        [("n_dc_annualized", "Law (0b): N(dc) vs threshold"),
         ("mean_dt_dc", "Law (5): <dt_dc> vs threshold"),
         ("mean_abs_tm", "Law (9): <|dx_tm|> vs threshold"),
         ("cum_tm", "Law (12): cumulative tm move vs threshold")],
    ):
        valid = table[["threshold_pct", col]].dropna()
        ax.loglog(valid["threshold_pct"], valid[col], ".", ms=3)
        ax.set_xlabel("Delta_x_dc (%)")
        ax.set_ylabel(col)
        ax.set_title(title)
    fig.suptitle("Experiment 1: directional-change scaling laws (GRW-benchmark demonstration)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_dc_scaling_laws.png", dpi=130)
    plt.close(fig)

    log.info("exp1 finished in %.1f s", time.time() - t0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                         help="attempt the paper's full 13-currency-pair tick data (documented as unavailable)")
    args = parser.parse_args()
    run(full=args.full)
