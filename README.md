# Patterns in high-frequency FX data: Discovery of 12 empirical scaling laws

Replication of J. B. Glattfelder, A. Dupuis and R. B. Olsen,
"Patterns in high-frequency FX data: Discovery of 12 empirical scaling laws"
(arXiv:0809.1040).

See `results/RESULTS.md` for the full findings, deviations from the paper and
what could not be reproduced.

Genuine tick-by-tick FX bid/ask data for the paper's December 2002 -
December 2007 window could not be obtained (Massive's forex quotes start
2009-09-25 and the authors' public archive covers only 2 of the 13 pairs as
very large uncompressed files; see `results/RESULTS.md`). All six
experiments below therefore run their methodology on the paper's own
Gaussian random walk (GRW) benchmark series (section 3.1, eq. 18), which the
paper itself reports as a point of comparison for every scaling law, and
compare the fitted exponents and constants against the GRW values the paper
tabulates. This reproduces the paper's internal GRW-vs-empirical validation,
not the 13-currency-pair empirical headline tables, which are listed under
"Not reproduced" in `results/RESULTS.md`.

## Layout

```
src/        Directional-change (Algorithm 2) and price-move (Algorithm 1)
            event detection, fixed-time sampling, log-log scaling-law
            fitting, coastline evaluation, cross-law consistency checks,
            and src/config.py with every paper-cited parameter.
exp/        One script per experiment (python -m exp.<name>).
data/       data/data_loader.py (QuantX Dataloader probe for FX bars) and
            data/generate_synthetic.py (the paper's Gaussian random walk
            benchmark, section 3.1, eq. 18).
tests/      pytest tests for src/ (synthetic fixtures) and the data path.
results/    Saved tables, plots and RESULTS.md.
```

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY / MASSIVE_API_KEY to run the market_data tests
```

## Tests

```
python -m pytest -m "not market_data"
```

## Experiments

```
python -m exp.exp1_dc_scaling_laws
python -m exp.exp2_price_move_scaling_laws
python -m exp.exp3_time_based_scaling_laws
python -m exp.exp4_grw_benchmark
python -m exp.exp5_coastline
python -m exp.exp6_consistency_checks
```

See `results/RESULTS.md` for what each run produced and how it compares with
the paper.
