"""Parameters for the replication of Glattfelder, Dupuis & Olsen (2010/2008),
"Patterns in high-frequency FX data: Discovery of 12 empirical scaling laws"
(arXiv:0809.1040). Every value below cites the paper section/equation it comes
from, or states that the paper does not specify it. No classes or dicts: every
experiment parameter is a plain module-level variable.
"""

# ---------------------------------------------------------------------------
# Data set definition (paper section 3.1)
# ---------------------------------------------------------------------------

# The 13 exchange rates studied in the paper (section 3.1), with the exact
# tick counts reported there (parentheses, page 10-11) kept for reference.
CURRENCY_PAIRS = [
    "AUDJPY", "AUDUSD", "CHFJPY", "GBPCHF", "GBPJPY", "GBPUSD", "EURAUD",
    "EURGBP", "EURCHF", "EURJPY", "EURUSD", "USDCHF", "USDJPY",
]
TICK_COUNTS_PAPER = {
    "AUDJPY": 15_286_858, "AUDUSD": 7_037_203, "CHFJPY": 17_081_987,
    "GBPCHF": 27_141_146, "GBPJPY": 26_423_199, "GBPUSD": 13_918_523,
    "EURAUD": 19_111_129, "EURGBP": 13_847_688, "EURCHF": 9_912_921,
    "EURJPY": 22_594_396, "EURUSD": 13_093_081, "USDCHF": 13_812_055,
    "USDJPY": 13_507_173,
}  # section 3.1, page 10-11

SAMPLE_START_DATE = "2002-12-01"  # section 3.1: "from December 1, 2002"
SAMPLE_END_DATE = "2007-12-01"  # section 3.1: "to December 1, 2007"
YEARS_IN_SAMPLE = 5  # section 3.1: annualisation divisor for laws (0b) and (2)
SECONDS_PER_YEAR = 31_553_280  # consistency-check section, page 9 ("Y")

# Only two of the thirteen series are published by the authors, at
# www.olsen.ch/more/datasets/ (section 3.1); the other eleven have no stated
# public source.
PUBLIC_PAIRS_AT_OLSEN = ["USDJPY", "EURUSD"]

# ---------------------------------------------------------------------------
# Threshold and time-interval grids (section 3.3 "Data fitting")
# ---------------------------------------------------------------------------

N_THRESHOLD_POINTS = 250  # section 3.3: "We select 250 measurement points"
THRESHOLD_MIN_PCT = 0.01  # section 3.3: "range is from 0.01% to 5.05%"
THRESHOLD_MAX_PCT = 5.05
THRESHOLD_LOG_STEP = 0.025  # section 3.3: "in log-space ... always 0.025"
# ln(5.05/0.01) / 0.025 = 249.0 -> 250 points, confirming natural-log spacing.

N_TIME_POINTS = 245  # section 3.3: "245 observations"
TIME_MIN_SECONDS = 20  # section 3.3: "range is from 20 ... seconds"
TIME_MAX_SECONDS = 3_975_783  # section 3.3: "46 days, 23 minutes and 3 seconds"
TIME_LOG_STEP = 0.05  # section 3.3: "logarithmic steps are always 0.05"
# ln(3975783/20) / 0.05 = 243.98 -> 245 points, confirming natural-log spacing.

TICK_DEFINITION_PCT = 0.02  # law (1): "a tick ... price move larger than 0.02%"

# ---------------------------------------------------------------------------
# Gaussian random walk (GRW) benchmark model (section 3.1, equation 18)
# ---------------------------------------------------------------------------

GRW_N_TICKS = 1_000_000  # eq. (18): "one million ticks"
GRW_X0 = 1.336723  # eq. (18): initial price
GRW_DT_SECONDS = 1  # eq. (18): "Delta t = 1 second"
GRW_SIGMA = 1.0 / 6769.6  # eq. (18): "sigma = 1/6769.6"
GRW_SEED = 42  # not specified in the paper; fixed for reproducibility

# Constant spread used for the "cumulative total move with costs" variant of
# law (12) (section 2.3: "we introduce a constant spread of 0.02%").
GRW_SPREAD_PCT = 0.02

# ---------------------------------------------------------------------------
# Coastline analysis (section 2.3)
# ---------------------------------------------------------------------------

COASTLINE_THRESHOLDS_PCT = [0.01, 0.1, 1.0, 5.0]  # section 2.3
COASTLINE_COST_ADJUSTED_EVAL_PCT = 0.05  # section 2.3: "measuring it for 0.05%"
COASTLINE_COST_FIT_RANGE_PCT = (0.1, 5.0)  # section 2.3: fit interval [0.1%,5%]
TRADING_DAYS_PER_YEAR = 252  # not specified in the paper; standard convention,
# used only to convert the annualised coastline into an average daily move.

# ---------------------------------------------------------------------------
# Demonstration tick-series size (data unavailability, see results/RESULTS.md)
# ---------------------------------------------------------------------------

# Genuine tick-by-tick FX bid/ask data for 2002-2007 is not obtainable through
# the QuantX Dataloader or Massive REST APIs (Massive's forex quotes endpoint
# states "Records date back to September 25, 2009", after the paper's sample
# window) and the authors' own public archive covers only 2 of the 13 pairs as
# uncompressed multi-million-tick files. Experiments 1-3, 5 and 6 therefore run
# their methodology on the paper's own Gaussian random walk benchmark series
# (section 3.1, eq. 18), which the paper itself uses as a point of comparison
# for every scaling law, and compare the fitted values against the GRW
# parameter values the paper reports. This is a deviation from the paper's
# headline (empirical, 13-currency-pair) results; see results/RESULTS.md.
DEMO_USES_GRW_ONLY = True
