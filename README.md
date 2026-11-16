# Patterns in high-frequency FX data: Discovery of 12 empirical scaling laws

Replication of J. B. Glattfelder, A. Dupuis and R. B. Olsen,
"Patterns in high-frequency FX data: Discovery of 12 empirical scaling laws"
(arXiv:0809.1040).

Work in progress: this README is updated as experiments are completed. See
`results/RESULTS.md` for the current findings, deviations from the paper and
what could not be reproduced.

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
python -m exp.exp3_physical_time_scaling_laws
python -m exp.exp4_grw_benchmark
python -m exp.exp5_coastline
python -m exp.exp6_consistency_checks
```

See `results/RESULTS.md` for what each run produced and how it compares with
the paper.
