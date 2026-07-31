# Alpha Research Framework

A quant research framework built from scratch, modeled on the workflow used by systematic investment firms.

Not trying to find one profitable strategy. Trying to build reusable infrastructure: data, features, signals, evaluation, portfolio construction, backtesting, all under one framework. New idea = implement the signal, everything else already exists.

---

# About

PhD in Mathematics (Quantitative Finance chair) from EPFL. Looking for a Quantitative Researcher role in Switzerland.

This repo is a real research sandbox, not a polished demo. I show what's finished, what's simplified, and what's missing. See "Known Limitations" below, it's not decoration.

Contact: qs.lemoal@gmail.com / [LinkedIn](https://www.linkedin.com/in/qlemoal/)

---

# Objectives

Full pipeline:

- download raw market data
- clean and validate it
- engineer features
- generate alpha signals
- evaluate them properly
- control false discoveries (FDR)
- build a portfolio
- backtest it
- check robustness
- compare multiple signals in the same framework

---

# Stack

**DuckDB + SQL:**
For anything touching large datasets: imports, joins, filtering, aggregations, quality checks. Works directly on parquet, no need to load everything into memory.

**Polars:**
For features and signals. Lazy evaluation, fast rolling computations, low memory footprint. Stays lazy until the final `collect()`.

**Pandas:**
Only once data is small. Plotting, exploration, quick debugging.

**Matplotlib:**
Visual checks. Every transformation gets eyeballed before moving to the next stage, see `notebooks/inspect_*.ipynb`.

---

# Setup

```bash
git clone https://github.com/qlemoal/quant_alpha_lab.git
cd quant_alpha_lab

python -m venv .venv
source .venv/bin/activate

pip install -e .
```

`pip install -e .` installs the project as an editable package, so `import src.features.beta` works from any notebook or script, no manual path hacking.

`requirements.txt` is a frozen, pinned snapshot for exact reproducibility. `pyproject.toml` lists the actual direct dependencies.

---

# Repository structure

```
.
├── pyproject.toml          project metadata, dependencies, editable install
├── requirements.txt        frozen environment snapshot
│
├── docs/
│   methodology.md           evaluation, correlation cleaning, signal construction
│   tips_and_tricks.md       small practical stuff worth remembering
│
├── config/
│   paths.py                 centralized filesystem paths
│   constants.py             trading days, rolling windows, thresholds
│   settings.py              global settings
│
├── data/                    git-ignored
│   raw/                     original downloads (prices, fundamentals, macro, sectors)
│   processed/                clean, research-ready data
│   intermediate/             scratch data
│   exports/                  figures, reports
│
├── notebooks/                exploration, plotting, debugging, hypothesis generation only
│
├── scripts/                  executable pipeline scripts
│   download_data.py
│   clean_data.py             ETL: extract raw CSVs, transform, load prices.parquet
│   build_features.py
│
├── src/                      reusable, importable code
│   features/                 feature engineering, one file per family
│   signals/                  turning features into tradeable signals
│   portfolio/                position sizing, transaction costs
│   evaluation/               IC, Sharpe, FDR, diagnostics
│   risk/                     correlation matrix cleaning, factor diagnostics
│   models/                   ML models
│   utils/                    plotting helpers etc
│
├── sql/                      DuckDB queries, kept separate for readability
├── tests/                    pytest unit tests
└── README.md
```

---

# Workflow

```
Download data -> Clean data -> Inspect prices -> Build features -> Inspect features
-> Generate signal -> Evaluate predictive power -> Control false discoveries
-> Portfolio construction (incl. costs) -> Backtest -> Attribution -> Robustness
```

---

# Implemented features

`src/features/`, all Polars, all grouped by ticker (`.over('ticker')`), all computed on data sorted by `['ticker', 'date']`.

| Feature | File | What it is |
|---|---|---|
| `logret` | `returns.py` | daily log return |
| `mom{w}` | `momentum.py` | % price change over trailing `w` days |
| `std{w}` | `volatility.py` | rolling std over trailing `w` days |
| `log_adv{w}` | `adv.py` | average (log-) dollar volume over trailing `w` days |
| `beta{w}` | `beta.py` | rolling market beta, `Cov(r_i, r_mkt) / Var(r_mkt)` |
| `seasonality.py` | not started | calendar effects |

Full methodology behind evaluation, signal construction, and correlation cleaning: see [`docs/methodology.md`](docs/methodology.md).

---

# Known limitations

## Survivorship bias

The universe is today's S&P 500 constituents, applied retroactively. Delisted/removed companies are missing entirely. Recently added companies have history predating their actual index membership. Every return, Sharpe ratio, and factor exposure in this repo is biased upward as a result.

Not an oversight, a scope decision: true point-in-time constituent history needs a paid vendor (CRSP, Compustat) I don't have access to outside an institutional setting. A free approximate point-in-time list is on the roadmap.

**This bias also corrupts the market proxy used for beta.** `beta.py` currently derives "market return" as the equal-weighted average across this same biased, thin-in-early-years universe. In the early 2000s, few of today's constituents have history yet, so the market proxy is built from a small, unstable sample, which makes its variance swing around and produces spuriously extreme beta values pre-2003. Two fixes: a minimum-ticker-count guard to null out beta until the universe is wide enough (quick, implemented as a stopgap), and replacing the proxy with a real benchmark index (S&P 500 / SPY) entirely (the actual fix, planned).

## Data provenance

Prices come from `yfinance`, unofficial, not a licensed feed. `Adj Close` gets retroactively restated by Yahoo whenever a new corporate action happens, so re-downloading the same history later can give slightly different values for the same date. No guarantee on completeness or timeliness compared to a real vendor.

## No transaction costs yet

No spread, commission, or slippage modeled. Any return or Sharpe figure right now is a pre-cost upper bound, not realistic. Planned in `src/portfolio/costs.py`: spread + commission + square-root market impact, using `adv` as the participation-rate basis, once portfolio construction exists.

## Single asset class, single frequency

US equities, daily bars only. No intraday, no other asset classes, no fundamentals. Everything so far comes from price and volume alone.

## Test coverage

`tests/` exists but isn't fully populated yet. Correctness currently leans on the `inspect_*.ipynb` notebooks plus targeted unit tests as they get added.

---

# Current progress

- [x] Download historical prices
- [x] Clean price data
- [x] Price inspection
- [X] Feature engineering, returns/momentum/volatility/ADV done and validated, beta in progress, seasonality not started
- [~] Signal library
- [ ] Portfolio construction
- [ ] Transaction cost modeling
- [ ] Evaluation framework
- [ ] FDR
- [ ] ML models
- [ ] Robustness analysis

---

# TODO

- [ ] fix market proxy, add `seasonality.py`
- [ ] Fill `tests/` with real unit tests
- [ ] `src/signals/`, first alpha signal on top of current features
- [ ] `src/evaluation/`, Rank IC, FDR
- [ ] `src/portfolio/`, position sizing, transaction costs
- [ ] `src/risk/covariance_cleaning.py`, RMT denoising, absorption ratio
- [ ] Full backtest loop with attribution
- [ ] Approximate point-in-time constituent list

---

# Development philosophy

Written manually, on purpose. Point is understanding every decision, not maximizing coding speed.

LLMs get used for discussion, debugging, and code review, not for writing implementation decisions I don't understand. Anything an LLM suggests gets read, questioned, and checked (often against a second implementation) before it stays in the codebase.