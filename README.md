# Alpha Research Framework

A modular quantitative research framework built from scratch to reproduce the workflow used by professional systematic investment firms.

The purpose of this repository is **not** to build a single profitable strategy, but to build a reusable infrastructure capable of supporting many different alpha ideas.

The project emphasizes:

- reproducibility
- modularity
- computational efficiency
- statistical rigor
- realistic quantitative research workflows

---

# Objectives

The long-term objective is to reproduce the complete research pipeline used by quantitative researchers.

The framework will eventually support

- downloading raw market data
- data cleaning and validation
- feature engineering
- alpha generation
- signal evaluation
- false discovery control (FDR)
- portfolio construction
- backtesting
- robustness analysis
- comparison of multiple signals under a common framework

The guiding principle is that every new research idea should only require implementing the **signal itself**. Everything else should already exist.

---

# Technologies

## DuckDB + SQL

Used whenever operations involve large datasets.

Typical tasks:

- imports
- joins
- filtering
- aggregations
- quality checks

DuckDB works directly on parquet files and avoids loading unnecessary data into memory.

---

## Polars

Used for feature engineering and signal construction.

Reasons:

- lazy evaluation
- automatic parallelization
- very fast rolling computations
- memory efficiency

The objective is to keep transformations lazy until the final `collect()`.

---

## Pandas

Only used once the dataset has become sufficiently small.

Typical uses:

- plotting
- exploratory analysis
- debugging

---

## Matplotlib

Used for visual sanity checks.

Every important transformation should be inspected visually before moving to the next stage.

---

# Repository structure

```
.
├── config/
│   Project-wide configuration.
│
│   paths.py
│       Centralized filesystem paths.
│
│   constants.py
│       Research constants
│       (trading days, rolling windows, thresholds...)
│
│   settings.py
│       Global project settings.
│
├── data/
│
│   raw/
│       Original downloaded datasets.
│
│       prices/
│       fundamentals/
│       macro/
│       sectors/
│
│   processed/
│       Clean datasets ready for research.
│
│   intermediate/
│       Temporary datasets created during processing.
│
│   exports/
│       figures/
│       reports/
│
├── notebooks/
│
│   Used only for
│       - exploration
│       - plotting
│       - debugging
│       - hypothesis generation
│
│   Every notebook starts with
│
│       from setup import *
│
│   which
│       - adds the project root to Python's path
│       - imports common libraries
│       - imports plotting helpers
│       - defines plotting defaults
│
├── scripts/
│
│   Executable scripts.
│
│   Examples
│
│       download_data.py
│       clean_prices.py
│       build_features.py
│       inspect_prices.py
│       build_signals.py
│       run_backtest.py
│
├── src/
│
│   Reusable Python code, mainly called in scripts.
│
│   features/
│       Feature engineering.
│
│   signals/
│       Alpha signal construction.
│
│   portfolio/
│       Portfolio construction.
│
│   evaluation/
│       IC, Sharpe, diagnostics, FDR.
│
│   models/
│       Machine learning models.
│
│   utils/
│       Shared helper functions.
│
├── sql/
│
│   SQL queries used by DuckDB.
│
│   Empty for now, but in the future, it will be more readable to keep SQL separate from Python.
│
├── tests/
│
│   Unit tests.
│
└── README.md
```

---

# Workflow

```
Download data

↓

Clean data

↓

Inspect prices

↓

Build features

↓

Inspect features

↓

Generate alpha signal

↓

Evaluate predictive power

↓

Control false discoveries

↓

Portfolio construction

↓

Backtest

↓

Performance attribution

↓

Robustness analysis
```

---

# Development philosophy

The repository is intentionally written manually.

The goal is to understand every design decision rather than maximizing coding speed.

Large language models are used only for

- discussion
- technical guidance
- debugging
- software engineering advice
- code review

Every implementation decision is understood before being added to the project.

---

# Known limitations

## Survivorship bias

The ticker universe is built from **today's** S&P 500 constituents, applied
retroactively across the full historical period. Concretely:

- companies removed from the index since (bankruptcy, acquisition, delisting)
  are absent from the dataset entirely
- companies added to the index more recently are included with history that
  predates their actual index membership

As a result, every metric derived from this dataset — returns, Sharpe ratios,
hit rates, factor exposures — is biased upward relative to what an investor
could have realistically captured at the time, since the universe only
contains firms that survived to today's index composition.

This is a deliberate scope decision for this project, not an oversight:
point-in-time constituent history requires a paid data vendor (e.g. CRSP,
Compustat) that isn't accessible outside an institutional setting. It's
documented here so results are interpreted with that caveat in mind. A
free, approximate point-in-time constituent list is on the roadmap as a
partial mitigation.

## Data provenance and corporate actions

Price data comes from Yahoo Finance via `yfinance`, an unofficial API rather
than a licensed vendor. Two consequences worth knowing:

- `Adj Close` values are not point-in-time stable — Yahoo retroactively
  restates historical adjusted prices whenever a new corporate action
  (split, dividend) occurs, so re-downloading the same history today vs.
  a year ago can produce slightly different values for the same date.
- there's no guarantee of the same accuracy, completeness, or update
  timeliness as a licensed vendor (e.g. CRSP, Bloomberg), so occasional
  gaps or restatements in the raw data are possible and not separately
  validated here.

## No transaction costs or market impact (yet)

The pipeline currently has no cost model: no bid-ask spread, commissions,
slippage, or market impact. Any Sharpe ratio or return figure produced
before this is added should be read as a pre-cost, idealized upper bound,
not a realistic backtest result.

## Single asset class, single frequency

The project currently covers US equities only, at daily bar frequency.
No intraday data, no other asset classes (rates, FX, credit), and no
fundamental/earnings data — all features so far are derived purely from
price and volume.

## Market proxy for beta

beta.py uses the equal-weighted cross-sectional mean return of the
universe as a market proxy, not a real capitalization-weighted benchmark
(e.g. SPY). This is a simplification, and will diverge from a "true"
market beta, especially since the universe itself is survivorship-biased
(see above).

## Test coverage

`tests/` is currently present but not implemented. Correctness of the
feature pipeline currently relies on manual inspection notebooks
(`inspect_*.ipynb`) rather than automated regression tests. More 
systematic tests will come in the future.

---

# Current Progress

- ✔ Download historical prices
- ✔ Clean price data
- ✔ Price inspection
- ✔ Feature engineering
- ⬜ Signal library
- ⬜ Portfolio construction
- ⬜ Evaluation framework
- ⬜ FDR
- ⬜ Machine learning models
- ⬜ Robustness analysis