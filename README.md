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
│   Reusable Python code.
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
│   Keeping SQL separate from Python improves readability for complex queries.
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

# Current Progress

- ✔ Download historical prices
- ✔ Clean price data
- ✔ Price inspection
- ⬜ Feature engineering
- ⬜ Signal library
- ⬜ Portfolio construction
- ⬜ Evaluation framework
- ⬜ FDR
- ⬜ Machine learning models
- ⬜ Robustness analysis