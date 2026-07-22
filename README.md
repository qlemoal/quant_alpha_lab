# Alpha Research Framework

A modular quantitative research framework built from scratch to reproduce the workflow used by professional systematic investment firms.

The purpose of this repository is **not** to build a single profitable strategy, but to build reusable infrastructure capable of supporting many different alpha ideas — data ingestion, feature engineering, signal generation, statistical evaluation, portfolio construction, and backtesting, all under one consistent framework.

The project emphasizes:

- reproducibility
- modularity
- computational efficiency
- statistical rigor
- realistic quantitative research workflows, including the limitations that come with them

---

# About

I hold a PhD in Mathematics (Quantitative Finance) and am currently looking for a Quantitative Researcher role, ideally in Switzerland. This repository is both a personal research sandbox and a demonstration of how I structure, validate, and reason about a research codebase end to end.

I've tried to be as transparent as possible about what's genuinely finished, what's a simplification, and what's still missing — a "Known Limitations" section is included below deliberately, rather than presenting the framework as more complete than it is.

*(Contact: qs.lemoal@gmail.com / [LinkedIn](https://www.linkedin.com/in/qlemoal/))*

---

# Objectives

The long-term objective is to reproduce the complete research pipeline used by quantitative researchers:

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

The guiding principle is that every new research idea should only require implementing the **signal itself**. Everything else — data plumbing, evaluation, portfolio construction, backtesting — should already exist and be reusable.

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

Every important transformation is inspected visually before moving to the next stage — see `notebooks/inspect_*.ipynb`.

---

# Setup

This project is installed as an editable local package via `pyproject.toml`, so `src/` can be imported from anywhere (notebooks, scripts, tests) without manual `sys.path` manipulation.

```bash
# clone the repo
git clone https://github.com/<your-username>/trading_strategies_01.git
cd trading_strategies_01

# create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# install the project in editable mode
pip install -e .
```

After this, `import src.features.beta` (etc.) works from any notebook or script, as long as its kernel/interpreter points at this environment.

A frozen, fully pinned environment snapshot is also available in `requirements.txt` for exact reproducibility of results.

---

# Repository structure

```
.
├── pyproject.toml
│   Project metadata, dependencies, and editable install configuration.
│
├── requirements.txt
│   Frozen environment snapshot for exact reproducibility.
│
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
├── data/                        (git-ignored)
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
├── scripts/
│
│   Executable pipeline scripts.
│
│       download_data.py
│       clean_data.py
│       build_features.py
│
├── src/
│
│   Reusable, importable Python code.
│
│   features/
│       Feature engineering — one file per feature family.
│
│   signals/
│       Alpha signal construction.
│
│   portfolio/
│       Portfolio construction and cost modeling.
│
│   evaluation/
│       IC, Sharpe, diagnostics, FDR.
│
│   models/
│       Machine learning models.
│
│   utils/
│       Shared helper functions (plotting, etc.)
│
├── sql/
│
│   SQL queries used by DuckDB.
│
│   Keeping SQL separate from Python improves readability for complex queries.
│
├── tests/
│
│   Unit tests (pytest).
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

Portfolio construction (incl. transaction costs)

↓

Backtest

↓

Performance attribution

↓

Robustness analysis
```

---

# Implemented features

All features live in `src/features/`, are built with Polars, operate lazily, and are grouped by ticker (`.over('ticker')`) on data explicitly sorted by `['ticker', 'date']`.

| Feature | File | Description |
|---|---|---|
| `logret` | `returns.py` | Daily log return: `log(close).diff()` per ticker. |
| `mom{w}` | `momentum.py` | Percentage price change over a trailing `w`-day window. |
| `std{w}` | `volatility.py` | Rolling standard deviation over a trailing `w`-day window. |
| `adv{w}` | `adv.py` | Average dollar volume over a trailing `w`-day window. |
| `beta{w}` | `beta.py` | Rolling market beta over a trailing `w`-day window, via `Cov(r_i, r_mkt) / Var(r_mkt)`, using the cross-sectional equal-weighted return as the market proxy. |
| `seasonality.py` | *(in progress)* | Calendar-effect features. |

---

# Known limitations

Documented deliberately, so results are interpreted correctly rather than assumed to be production-grade.

## Survivorship bias

The ticker universe is built from **today's** S&P 500 constituents, applied retroactively across the full historical period. Concretely:

- companies removed from the index since (bankruptcy, acquisition, delisting) are absent from the dataset entirely
- companies added to the index more recently are included with history that predates their actual index membership

As a result, every metric derived from this dataset — returns, Sharpe ratios, hit rates, factor exposures — is biased upward relative to what an investor could have realistically captured at the time, since the universe only contains firms that survived to today's index composition.

This is a deliberate scope decision for this project, not an oversight: point-in-time constituent history requires a paid data vendor (e.g. CRSP, Compustat) that isn't accessible outside an institutional setting. A free, approximate point-in-time constituent list is on the roadmap as a partial mitigation.

## Data provenance and corporate actions

Price data comes from Yahoo Finance via `yfinance`, an unofficial API rather than a licensed vendor. Two consequences worth knowing:

- `Adj Close` values are not point-in-time stable — Yahoo retroactively restates historical adjusted prices whenever a new corporate action (split, dividend) occurs, so re-downloading the same history today vs. a year ago can produce slightly different values for the same date.
- there's no guarantee of the same accuracy, completeness, or update timeliness as a licensed vendor (e.g. CRSP, Bloomberg), so occasional gaps or restatements in the raw data are possible and not separately validated here.

## No transaction costs (yet)

The pipeline currently has no cost model — no bid-ask spread, commissions, slippage, or market impact. Any Sharpe ratio or return figure produced before this is added should be read as a pre-cost, idealized upper bound, not a realistic backtest result. A cost model (spread + commission + square-root market impact, using `adv` as the participation-rate basis) is planned in `src/portfolio/costs.py` once portfolio construction is in place.

## Single asset class, single frequency

The project currently covers US equities only, at daily bar frequency. No intraday data, no other asset classes (rates, FX, credit), and no fundamental/earnings data — all features so far are derived purely from price and volume.

## Market proxy for beta

`beta.py` uses the equal-weighted cross-sectional mean return of the universe as a market proxy, not a real capitalization-weighted benchmark (e.g. SPY). This is a simplification, and will diverge from a "true" market beta, especially since the universe itself is survivorship-biased (see above).

## Test coverage

`tests/` is scaffolded but not yet fully populated. Correctness of the feature pipeline currently relies on manual inspection notebooks (`inspect_*.ipynb`) as well as targeted unit tests as they're added — automated regression coverage across the full feature library is a work in progress.

---

# Development philosophy

The repository is intentionally written manually. The goal is to understand every design decision rather than maximizing coding speed.

Large language models are used for:

- discussion and technical guidance
- debugging
- software engineering advice
- code review

Every implementation decision is understood before being added to the project — LLM-suggested code is read, questioned, and verified (including against a second implementation, e.g. pandas, where correctness isn't obvious) before it's kept.

---

# Current progress

- ✔ Download historical prices
- ✔ Clean price data
- ✔ Price inspection
- ✔ Feature engineering — returns, momentum, volatility, ADV, beta implemented and validated; seasonality not started
- 🔶 Signal library
- ⬜ Portfolio construction
- ⬜ Transaction cost modeling
- ⬜ Evaluation framework
- ⬜ FDR
- ⬜ Machine learning models
- ⬜ Robustness analysis

---

# Roadmap

- [ ] Finish `beta.py` and `seasonality.py`
- [ ] Populate `tests/` with unit tests for each feature (synthetic data, hand-computed expected values)
- [ ] Build `src/signals/` — first alpha signal on top of the current feature set
- [ ] Build `src/evaluation/` — Information Coefficient, FDR-controlled significance testing
- [ ] Build `src/portfolio/` — position sizing, transaction cost model
- [ ] Build a full backtest loop with performance attribution
- [ ] Approximate point-in-time constituent list, to partially address survivorship bias