# AGENTS.md

Guide for coding agents and contributors working on this repository.

## What this implements

A replication of J. B. Glattfelder, A. Dupuis and R. B. Olsen, "Patterns in
high-frequency FX data: Discovery of 12 empirical scaling laws"
(arXiv:0809.1040): the directional-change event detection (Algorithm 2) and
price-move counting (Algorithm 1) procedures, the fixed-time sampling scheme,
the twelve log-log scaling laws the paper fits on top of them, the coastline
evaluation of law (12), and the paper's cross-law consistency checks
(eqs. 14-17). See `results/RESULTS.md` for what was run, the comparison with
the paper's reported values, a verdict for every metric, and the full list of
deviations and non-reproduced items.

The paper's tick-by-tick FX bid/ask data for 13 currency pairs (December 2002
- December 2007) could not be obtained through the QuantX Dataloader or the
Massive REST API, nor downloaded in full from the authors' public archive
(only 2 of the 13 pairs, as multi-gigabyte uncompressed files). All six
experiments therefore run the full methodology on the paper's own Gaussian
random walk (GRW) benchmark series (section 3.1, eq. 18) and compare against
the GRW parameter values the paper itself tabulates, rather than against its
13-pair empirical tables. Details are in `results/RESULTS.md`.

## Layout

- `src/` - flat modules: `dc_algorithm.py` (Algorithm 2), `price_move_algorithm.py`
  (Algorithm 1), `time_sampling.py` (fixed-interval sampling for laws 0a/3),
  `scaling_laws.py` (log-log OLS fitting, linear and quadratic), `coastline.py`
  (law 12 coastline evaluation), `consistency_checks.py` (eqs. 14-17),
  `pipeline.py` (shared per-experiment orchestration), and `config.py`
  (every parameter as a module-level constant with a paper citation or a note
  that the paper does not specify it).
- `exp/` - one script per experiment, run as `python -m exp.<name>`. Each
  calls only the `src/` functions with their documented signatures.
- `data/` - `data_loader.py` (QuantX Dataloader probe for FX OHLCV bars and
  the Massive forex-quotes coverage check) and `generate_synthetic.py` (the
  paper's GRW benchmark generator, section 3.1, eq. 18). Downloaded or
  generated data is never committed.
- `tests/` - pytest tests for every `src/` module (synthetic fixtures with
  known properties) and for the data-loading path (`market_data`-marked,
  skipped without API keys).
- `results/` - tables (CSV), fitted-parameter and consistency-check summaries
  (JSON), plots (PNG) and `RESULTS.md`, all overwritten by the experiment
  commands below.

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY / MASSIVE_API_KEY to run the market_data tests
```

API keys are read only from the environment (via `python-dotenv`) and are
never written to source, config, logs or results.

## Tests

```
python -m pytest -m "not market_data"
```

Run the full suite, including the `market_data`-marked data-path tests
(require `DATALOADER_API_KEY` and `MASSIVE_API_KEY`), with:

```
python -m pytest
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

Each command overwrites its corresponding files in `results/`. The GRW
benchmark is generated at the full scale the paper specifies
(1,000,000 ticks); no experiment parameter was reduced for runtime.

## Conventions

- All experiment and model parameters live only in `src/config.py` as
  module-level `UPPER_CASE` variables, each with a comment citing the
  paper section/equation or stating that the paper does not specify it.
  No parameter is hardcoded elsewhere.
- Market data passes only through `data/data_loader.py`; synthetic data only
  through `data/generate_synthetic.py`. No `src/` or `exp/` module fetches or
  generates data on its own.
- After rerunning the experiment commands, the files in `results/` are
  overwritten; the numbers and verdicts written by hand into
  `results/RESULTS.md` must then be checked against the new files and updated
  if they changed.
- Deviations from the paper and anything not reproduced are recorded only in
  `results/RESULTS.md`, never as silent substitutions in code.
