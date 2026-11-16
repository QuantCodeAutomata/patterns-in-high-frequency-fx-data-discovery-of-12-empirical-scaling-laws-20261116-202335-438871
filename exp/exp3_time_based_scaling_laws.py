"""Experiment 3: physical time-based scaling laws (0a, 3) of Glattfelder,
Dupuis & Olsen, "Patterns in high-frequency FX data: Discovery of 12 empirical
scaling laws" (arXiv:0809.1040), section 2.1.

Prices are sampled at 245 log-spaced fixed time intervals Delta_t from 20
seconds to 3,975,783 seconds (~47 days, section 3.3), using the last quoted
price before each sampling time. Law (0a) fits the mean (p=1) and quadratic
mean (p=2) absolute price change; law (3) fits the mean (p=1, p=2) maximal
price range within non-overlapping windows of length Delta_t, computed over
the raw ticks inside each window.

See results/RESULTS.md and src/config.DEMO_USES_GRW_ONLY for why this
experiment runs on the paper's own Gaussian random walk benchmark series
(eq. 18) instead of the paper's unavailable tick-by-tick FX quotes. The
Gaussian-random-walk comparison is itself one of the paper's stated
objectives for law (0a) p=2 (theoretical exponent 0.5).

Run: python -m exp.exp3_time_based_scaling_laws
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.generate_synthetic import generate_grw_series
from src import config
from src.pipeline import fit_columns, laws_0a_3_table
from src.scaling_laws import time_grid

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("exp3")

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

    prices, times_s = generate_grw_series(
        config.GRW_N_TICKS, config.GRW_X0, config.GRW_DT_SECONDS, config.GRW_SIGMA, config.GRW_SEED
    )
    span_s = times_s[-1] - times_s[0]
    log.info("GRW series: %d ticks, span %.1f s (%.2f days)", prices.shape[0], span_s, span_s / 86400)

    dt_grid_full = time_grid(config.N_TIME_POINTS, config.TIME_MIN_SECONDS, config.TIME_MAX_SECONDS)
    # The GRW series spans only ~11.6 days (1e6 one-second ticks); time
    # intervals longer than the series span cannot yield any sampled window
    # and are dropped (not a reduction of the paper's grid definition, only
    # of which points of it can be evaluated on this shorter demonstration
    # series).
    dt_grid = dt_grid_full[dt_grid_full < span_s / 4]
    log.info("Using %d of %d grid points with Delta_t < span/4 (%.1f s)",
             dt_grid.shape[0], dt_grid_full.shape[0], span_s / 4)

    table = laws_0a_3_table(prices, times_s, dt_grid)
    table.to_csv(RESULTS_DIR / "exp3_time_based_table.csv", index=False)

    fit_df = fit_columns(table, "dt_seconds", ["mean_abs_p1", "mean_abs_p2", "mean_max_p1", "mean_max_p2"])
    fit_df["law"] = ["law_0a_p1", "law_0a_p2", "law_3_p1", "law_3_p2"]
    fit_df.to_csv(RESULTS_DIR / "exp3_time_based_fits.csv", index=False)
    log.info("Fitted laws (0a,3):\n%s", fit_df.to_string(index=False))

    e_0a_p2 = fit_df.loc[fit_df["law"] == "law_0a_p2", "E"].iloc[0]
    grw_check = {
        "E_0a_p2_measured": float(e_0a_p2),
        "E_0a_p2_theoretical_GRW": 0.5,
        "abs_diff": float(abs(e_0a_p2 - 0.5)),
    }
    log.info("GRW theoretical check for law (0a) p=2: %s", grw_check)
    import json
    with open(RESULTS_DIR / "exp3_grw_theoretical_check.json", "w") as f:
        json.dump(grw_check, f, indent=2)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (col, title) in zip(
        axes.flat,
        [("mean_abs_p1", "Law (0a) p=1: <|dx|>_1 vs Delta_t"),
         ("mean_abs_p2", "Law (0a) p=2: <dx^2>_2 vs Delta_t"),
         ("mean_max_p1", "Law (3) p=1: <dx_max>_1 vs Delta_t"),
         ("mean_max_p2", "Law (3) p=2: <dx_max^2>_2 vs Delta_t")],
    ):
        valid = table[["dt_seconds", col]].dropna()
        ax.loglog(valid["dt_seconds"], valid[col], ".", ms=3)
        ax.set_xlabel("Delta_t (s)")
        ax.set_ylabel(col)
        ax.set_title(title)
    fig.suptitle("Experiment 3: fixed-time scaling laws (GRW-benchmark demonstration)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp3_time_based_scaling_laws.png", dpi=130)
    plt.close(fig)

    log.info("exp3 finished in %.1f s", time.time() - t0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                         help="attempt the paper's full 13-currency-pair tick data (documented as unavailable)")
    args = parser.parse_args()
    run(full=args.full)
