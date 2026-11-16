"""Experiment 2: price-move event-based scaling laws (1, 2, 4) of Glattfelder,
Dupuis & Olsen, "Patterns in high-frequency FX data: Discovery of 12 empirical
scaling laws" (arXiv:0809.1040), section 3.2.

Algorithm 1 (price-move counting, no up/down mode) is run at 250 log-spaced
thresholds Delta_x from 0.01% to 5.05% (section 3.3). Each run yields the
mean tick count per move (law 1), the annualised move count (law 2), and the
mean time interval between moves (law 4). The inverse relationship between
laws (2) and (4) is checked against Y = 31,553,280 seconds/year.

See results/RESULTS.md and src/config.DEMO_USES_GRW_ONLY for why this
experiment runs on the paper's own Gaussian random walk benchmark series
(eq. 18) instead of the paper's unavailable tick-by-tick FX quotes.

Run: python -m exp.exp2_price_move_scaling_laws
"""

from __future__ import annotations

import argparse
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
from src.consistency_checks import check_14
from src.pipeline import fit_columns, laws_1_2_4_table, reference_tick_events, sample_years_from_times
from src.scaling_laws import threshold_grid

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("exp2")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run(full: bool) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    if full:
        log.warning(
            "The --full flag would require tick-by-tick bid/ask FX quotes for 13 "
            "currency pairs (2002-2007); see results/RESULTS.md. Running the "
            "GRW-benchmark demonstration configuration instead."
        )

    prices, times = generate_grw_series(
        config.GRW_N_TICKS, config.GRW_X0, config.GRW_DT_SECONDS, config.GRW_SIGMA, config.GRW_SEED
    )
    log.info("GRW series: %d ticks, span %.2f days", prices.shape[0], (times[-1] - times[0]) / 86400)

    thresholds = threshold_grid(config.N_THRESHOLD_POINTS, config.THRESHOLD_MIN_PCT, config.THRESHOLD_MAX_PCT)
    tick_events = reference_tick_events(prices, times, config.TICK_DEFINITION_PCT)

    sample_years = sample_years_from_times(times, config.SECONDS_PER_YEAR)
    log.info("Annualisation divisor (elapsed sample years): %.6f", sample_years)
    table = laws_1_2_4_table(prices, times, thresholds, tick_events, sample_years)
    table.to_csv(RESULTS_DIR / "exp2_price_move_table.csv", index=False)

    fit_df = fit_columns(table, "threshold_pct", ["mean_nticks", "n_annualized", "mean_dt"])
    fit_df["law"] = ["law_1_N_tck", "law_2_N_x", "law_4_t_x"]
    fit_df.to_csv(RESULTS_DIR / "exp2_price_move_fits.csv", index=False)
    log.info("Fitted laws (1,2,4):\n%s", fit_df.to_string(index=False))

    # Research objective: inverse relationship between laws (2) and (4),
    # eq. (14): E_t,x = -E_N,x, C_t,x = Y^(1/E_N,x) * C_N,x.
    law2 = fit_df.loc[fit_df["law"] == "law_2_N_x"].iloc[0]
    law4 = fit_df.loc[fit_df["law"] == "law_4_t_x"].iloc[0]
    e_check, c_check = check_14(law2["E"], law2["C"], config.SECONDS_PER_YEAR, law4["E"], law4["C"])
    consistency = {
        "E_predicted": e_check.predicted, "E_measured": e_check.measured, "E_rel_error": e_check.rel_error,
        "C_predicted": c_check.predicted, "C_measured": c_check.measured, "C_rel_error": c_check.rel_error,
    }
    pd.DataFrame([consistency]).to_csv(RESULTS_DIR / "exp2_law2_law4_consistency.csv", index=False)
    log.info("Law (2)-(4) consistency check: %s", consistency)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (col, title) in zip(
        axes,
        [("mean_nticks", "Law (1): <N(tick)> vs threshold"),
         ("n_annualized", "Law (2): N(dx) vs threshold"),
         ("mean_dt", "Law (4): <dt_x> vs threshold")],
    ):
        valid = table[["threshold_pct", col]].dropna()
        ax.loglog(valid["threshold_pct"], valid[col], ".", ms=3)
        ax.set_xlabel("Delta_x (%)")
        ax.set_ylabel(col)
        ax.set_title(title)
    fig.suptitle("Experiment 2: price-move scaling laws (GRW-benchmark demonstration)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_price_move_scaling_laws.png", dpi=130)
    plt.close(fig)

    log.info("exp2 finished in %.1f s", time.time() - t0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                         help="attempt the paper's full 13-currency-pair tick data (documented as unavailable)")
    args = parser.parse_args()
    run(full=args.full)
