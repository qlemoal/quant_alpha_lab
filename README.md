# Alpha Research Framework

Hi! This is a quant research framework built from scratch, modeled on the workflow used by systematic investment firms.

I'm not trying to find one profitable strategy. I'm trying to build reusable infrastructure: data, features, signals, evaluation, portfolio construction, backtesting, all under one framework. New idea = implement the signal, everything else already exists.

---

# About

PhD in Mathematics (Quantitative Finance chair) from EPFL, I'm looking for a Quantitative Researcher role in Switzerland.

This repo is a real research sandbox, not a polished demo. I show what's finished, what's simplified, and what's missing. See "Known Limitations" below before trusting any number above it.

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
For anything touching large datasets: imports, joins, filtering, aggregations, quality checks. Works directly on parquet, no need to lowad everything into memory.

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
│   constants.py             trading days, rolling windows, thresholds, feature/label column rules
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
│   inspect_signal.py         standalone diagnostic runner, no notebook needed
│
├── src/                      reusable, importable code
│   features/                 feature engineering, one file per family
│   signals/                  transform.py, turning features into tradeable signals
│   evaluation/
│       signals/              IC, significance, stability, one-call report and plots
│       portfolio/            not started, real turnover, drawdown, cost-adjusted returns
│       models/               not started, log loss, Brier score, calibration
│   portfolio/                not started, position sizing, transaction costs
│   risk/                     correlation matrix cleaning, factor diagnostics (stubbed)
│   models/                   not started, ML models
│   utils/                    panel reshaping, plotting helpers
│
├── sql/                      DuckDB queries, kept separate for readability
├── tests/                    pytest unit tests, conftest.py holds the edge-case panel fixtures
└── README.md
```

---

# Workflow

```
-> Download data -> ETL data -> Inspect data 
-> Build features -> Inspect features 
-> Generate signal -> Inspect signals -> Evaluate predictive power -> Control false discoveries
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

# Signals and evaluation
 
`src/signals/transform.py` turns a feature into a tradeable signal, cross-sectionally, per date. Four construction methods: `zscore_tanh`, `zscore_clip`, `rank`, `decile`. Which one fits a given feature depends on how much its raw magnitude is trusted, see `docs/methodology.md`.
 
`src/evaluation/signals/` scores a signal once it exists:
 
| File | What it gives you |
|---|---|
| `ic/core.py` | Rank IC per date, summary stats (mean, IR, hit rate) |
| `ic/metrics.py` | Newey-West corrected significance, IC decay across forward horizons |
| `ic/plots.py` | IC time series, cumulative IC, IC distribution, decile spread, comparing several methods at once |
| `performance.py` | signal stability (day-over-day change), naive long-short paper Sharpe |
| `report.py` | one call, numeric-only health check, no plots |
| `signal_plots.py` | one call, visual health check, coverage/moments/heatmap/sample tickers |
 
Every new signal gets both a `report.signal_report()` call and a `signal_plots.one_pager()` call before it's trusted for anything further, this is checked, not assumed, `tests/` has a deliberately awkward set of edge-case panels (`conftest.py`) every new function gets run against.
 
---


# Design philosophy: no free parameters, by construction

A recurring principle across this repo: prefer methods with no parameter to hand-pick over methods that need one, even when the tunable version is more powerful. A parameter chosen by checking what it does to the results, however innocently, is a parameter chosen to fit this specific dataset, and that's a direct path to overfitting the research process itself, not just a model. 

This shows up in several independent decisions, e.g.:

- **Rank IC over raw-return backtests.** A single backtest's Sharpe ratio is trivial to overfit by adjusting almost anything about it. Rank correlation against forward returns is harder to game by construction.
- **Rank transform preferred over zscore-tanh for untrusted features.** No clip bound, no tanh scale to pick, ordinal position only.
- **Louvain over k-means for clustering.** k-means needs k chosen in advance, a free parameter with no principled answer from the data itself. Louvain finds the number of communities by maximizing modularity, nothing to feed it beforehand.
- **Newey-West's lag length uses the standard 1994 rule of thumb**, not a value searched over to produce a preferred t-stat.
- **FDR: Storey's q-value over a fixed BH/BY alpha**, wherever it's usable. BH and BY both require declaring a significance level up front, a small number written by a human, and a different number will validate a different, cherry-pickable set of signals. Storey's pi0 is estimated from the p-value distribution itself. See docs/methodology.md for the mechanics.

The distinction that matters isn't "no parameters anywhere," some numbers are unavoidable (a significance convention, a kernel bandwidth rule). It's whether a parameter was fixed by a defensible convention before looking at results, or picked, however implicitly, because it produced a result that looked good. Every parameter in this repo should be defensible as the former. If it can't be, it's a bug in my research process, not just in the code.

---

# Known limitations

## Survivorship bias

The universe is today's S&P 500 constituents, applied retroactively. Delisted/removed companies are missing entirely. Recently added companies have history predating their actual index membership. Every return, Sharpe ratio, and factor exposure in this repo is biased upward as a result.
 
Not an oversight, a scope decision: true point-in-time constituent history needs a paid vendor (CRSP, Compustat) I don't have access to outside an institutional setting. A free approximate point-in-time list is on the roadmap.
 
**This bias also corrupts the market proxy used for beta.** `beta.py` currently derives "market return" as the equal-weighted average across this same biased, thin-in-early-years universe. In the early 2000s, few of today's constituents have history yet, so the market proxy is built from a small, unstable sample, which makes its variance swing around and produces spuriously extreme beta values pre-2003. Two fixes: a minimum-ticker-count guard to null out beta until the universe is wide enough (quick, implemented as a stopgap), and replacing the proxy with a real benchmark index (S&P 500 / SPY) entirely (the actual fix, planned).

## Data provenance

Prices come from `yfinance`, unofficial, not a licensed feed. `Adj Close` gets retroactively restated by Yahoo whenever a new corporate action happens, so re-downloading the same history later can give slightly different values for the same date. No guarantee on completeness or timeliness compared to a real vendor.

Also worth flagging alongside price reversal: `Adj Close` gets retroactively restated whenever a new corporate action happens, so re-downloaded history can carry slightly different daily values than the original download. That's a second, harder-to-rule-out source of artificial day-to-day return noise, no clean way to separate it from genuine short-term reversal without a licensed feed to compare against.

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
- [X] Feature engineering, returns/momentum/volatility/ADV done and validated, beta implemented (market proxy has a known thin-history issue, see limitations), seasonality not started
- [x] Signal construction, four methods, tested against edge-case panels
- [x] Signal evaluation, Rank IC, Newey-West significance, IC decay, stability, paper Sharpe, numeric and visual one-call checks
- [X] FDR across multiple candidate signals
- [X] Rolling cross-validation, with purge and embargo buffer dates
- [ ] Elastic Net signal combiner
- [ ] Portfolio construction
- [ ] Transaction cost modeling
- [ ] RMT correlation cleaning
- [ ] GBM comparison model
- [ ] Robustness analysis

---

# Roadmap
 
- [ ] Fix beta's market proxy (real benchmark index instead of the biased equal-weighted universe average)
- [ ] `seasonality.py`
- [ ] Elastic Net combiner across FDR-surviving signals
- [ ] `src/portfolio/`, position sizing, transaction costs
- [ ] `src/risk/covariance_cleaning.py`, RMT denoising, absorption ratio
- [ ] GBM comparison model, purged/embargoed walk-forward CV
- [ ] Full backtest loop with attribution
- [ ] Approximate point-in-time constituent list

---

# Development philosophy

Written manually, on purpose. Point is understanding every decision, not maximizing coding speed.

LLMs get used for discussion, debugging, and code review, not for writing implementation decisions I don't understand. Anything an LLM suggests gets read, questioned, and checked (often against a second implementation) before it stays in the codebase.