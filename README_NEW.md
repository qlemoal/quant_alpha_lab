# Alpha Research Framework

> A production-grade quantitative research pipeline built from scratch, implementing the full workflow used at systematic investment firms (Millennium, Citadel, Two Sigma, AQR).

---

## 🎯 Objective

Build **reusable infrastructure** — not a single strategy. Every component (data, features, signals, evaluation, portfolio construction, backtesting) is modular, tested, and replaceable. New alpha idea = implement the signal; everything else already exists.

---

## 👤 About

**Quentin Le Moal**  
PhD in Mathematics (Quantitative Finance) — EPFL  
Seeking **Quantitative Researcher** role in Switzerland

📧 qs.lemoal@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/qlemoal/)  
🐙 [GitHub](https://github.com/qlemoal)

> This repo is a real research sandbox, not a polished demo. I show what's finished, what's simplified, and what's missing. See **Known Limitations** before trusting any number.

---

## 🏗️ Full Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   DOWNLOAD  │───▶│     ETL     │───▶│  FEATURES   │───▶│  SIGNALS    │───▶│  EVALUATE   │
│  (yfinance) │    │  (DuckDB)   │    │  (Polars)   │    │ (transform) │    │   (IC/FDR)  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                       │
                                                                       ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  BACKTEST   │◀───│   COSTS     │◀───│ PORTFOLIO   │◀───│  COMBINE    │◀───│   FDR       │
│  (walk-fwd) │    │  (sqrt)     │    │  (HRP/ERC)  │    │  (EN/GBM)   │    │  (q-value)  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

---

## 🛠️ Stack

| Layer | Technology | Why |
|-------|------------|-----|
| **Data & ETL** | DuckDB + SQL | Works directly on Parquet, no memory loading, fast joins/aggregations |
| **Features & Signals** | Polars | Lazy evaluation, fast rolling ops, low memory, stays lazy until `collect()` |
| **Evaluation & Plotting** | Pandas + Matplotlib | Small data only, exploration/debugging |
| **Optimization** | NumPy + SciPy (+ CVXPY optional) | Linear algebra, convex optimization |
| **Validation** | Custom CPCV | Purge/embargo, combinatorial, no leakage |

**Dependencies (pinned in `requirements.txt`):**
`polars`, `duckdb`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `networkx`, `yfinance`, `pytest`

---

## 📁 Repository Structure

```
quant_alpha_lab/
├── pyproject.toml              # Package metadata, editable install
├── requirements.txt            # Frozen environment snapshot
├── config/
│   ├── paths.py                # Centralized filesystem paths
│   ├── constants.py            # Trading days, windows, thresholds
│   └── settings.py             # Global settings
├── data/                       # Git-ignored
│   ├── raw/                    # Original downloads (prices, fundamentals)
│   ├── processed/              # Clean, research-ready data
│   ├── intermediate/           # Scratch data
│   └── exports/                # Figures, backtest results
├── notebooks/                  # Exploration, plotting, hypothesis generation ONLY
├── scripts/
│   ├── pipeline/               # Executable pipeline steps
│   │   ├── 01_download_data.py
│   │   ├── 02_etl_prices.py
│   │   ├── 03_build_features.py
│   │   └── 04_build_portfolio.py    # NEW: full portfolio + backtest
│   ├── research/               # Standalone research scripts
│   │   ├── embargo_selection.py
│   │   └── inspect_signal.py
│   └── demos/                  # Comparison/validation demos
│       ├── compare_gbm_vs_en_pbo.py
│       └── plot_en_combiner_coef_stability.py
├── src/
│   ├── features/               # Feature engineering (one file per family)
│   │   ├── returns.py          # logret, fwdret (open-to-open)
│   │   ├── momentum.py         # mom{w}: % price change over w days
│   │   ├── volatility.py       # std{w}: rolling std over w days
│   │   ├── adv.py              # log_adv{w}: avg dollar volume
│   │   ├── beta.py             # beta{w}: rolling market beta
│   │   └── seasonality.py      # Calendar effects (planned)
│   ├── signals/                # Signal construction
│   │   ├── combine.py          # make_signal(): 4 transform methods
│   │   ├── transforms.py       # zscore, tanh, clip
│   │   ├── ranking.py          # rank, decile
│   │   ├── normalize.py        # zscore_expr
│   │   └── thresholds.py       # thresholding
│   ├── evaluation/
│   │   ├── signals/
│   │   │   ├── ic/             # Rank IC, Newey-West, IC decay
│   │   │   ├── fdr.py          # BH, BY, Storey q-value
│   │   │   ├── performance.py  # Stability, long-short Sharpe
│   │   │   ├── report.py       # Numeric health check
│   │   │   └── signal_plots.py # Visual health check
│   │   └── portfolio/          # Attribution (planned)
│   ├── models/
│   │   ├── elastic_net_combiner.py   # EN with CPCV-tuned α, l1_ratio
│   │   └── gbm_combiner.py         # HistGradientBoostingRegressor
│   ├── risk/
│   │   ├── correlation_cleaning.py # RMT / Marchenko-Pastur denoising
│   │   ├── clustering.py           # Louvain community detection
│   │   ├── covariance.py           # Ledoit-Wolf, OAS, Factor model
│   │   └── portfolio_construction.py # Signal→weights, cluster caps, vol target
│   ├── portfolio/
│   │   ├── costs.py              # Spread, commission, √-impact
│   │   ├── optimization.py       # HRP, ERC, BL, cost-aware MVO
│   │   └── backtest.py           # Walk-forward loop + attribution
│   ├── validation/
│   │   ├── walk_forward_cv.py    # Purged walk-forward CV
│   │   ├── cpcv.py               # Combinatorial Purged CV + PBO
│   │   └── cpcv_visualization.py # Fold geometry plots
│   └── utils/
│       ├── stats.py              # Autocorrelation, p-values
│       ├── helpers.py            # as_list, add_expr
│       ├── panels.py             # Panel reshaping
│       └── my_plotting.py        # Plot helpers
├── tests/
│   ├── conftest.py               # "Musée des horreurs" edge-case fixtures
│   ├── test_signals.py
│   ├── test_cpcv.py
│   ├── test_fdr.py
│   ├── test_beta.py
│   └── test_en_combiner.py
└── docs/
    ├── methodology.md            # Detailed methodology (this file's twin)
    ├── findings.md               # Dated research log
    └── tips_and_tricks.md        # Practical notes
```

---

## ⚡ Quick Start

```bash
git clone https://github.com/qlemoal/quant_alpha_lab.git
cd quant_alpha_lab

python -m venv .venv
source .venv/bin/activate

pip install -e .        # Installs package + dependencies
# or: pip install -r requirements.txt  # Exact frozen env
```

Run the pipeline:
```bash
# 1. Download S&P 500 prices from Yahoo
python scripts/pipeline/01_download_data.py

# 2. ETL: clean, validate, write prices.parquet
python scripts/pipeline/02_etl_prices.py

# 3. Build features: returns, momentum, vol, ADV, beta
python scripts/pipeline/03_build_features.py

# 4. Generate signals & evaluate (via notebooks or scripts)
#    See notebooks/inspect_signal.ipynb

# 5. Combine signals (Elastic Net + GBM) and run backtest
python scripts/pipeline/04_build_portfolio.py
```

---

## ✅ Implemented Features

### Data & Features
- [x] **Download**: S&P 500 constituents from Wikipedia + yfinance
- [x] **ETL**: DuckDB aggregation, cleaning, deduplication → `prices.parquet`
- [x] **Features** (all Polars, per-ticker, sorted by `[ticker, date]`):

| Feature | File | Formula |
|---------|------|---------|
| `logret` | `returns.py` | $\log(P_t) - \log(P_{t-1})$ |
| `mom{w}` | `momentum.py` | $\frac{P_t}{P_{t-w}} - 1$ |
| `std{w}` | `volatility.py` | $\sqrt{\frac{1}{w-1}\sum_{i=1}^w (r_{t-i} - \bar{r})^2}$ |
| `log_adv{w}` | `adv.py` | $\log\left(\frac{1}{w}\sum_{i=1}^w \text{Volume}_{t-i} \times \text{Close}_{t-i}\right)$ |
| `beta{w}` | `beta.py` | $\frac{\text{Cov}(r_i, r_{mkt})}{\text{Var}(r_{mkt})}$ |

### Signals & Evaluation
- [x] **4 Transform Methods** (`signals/combine.py`):
  - `zscore_tanh`: $z \mapsto \tanh(\text{scale} \cdot z)$ — well-behaved features
  - `zscore_clip`: $z \mapsto \text{clip}(z, -3, 3)$ — hard bound
  - `rank`: $\text{rank}(x) \mapsto [-1, 1]$ — robust, no magnitude
  - `decile`: Bucket into 10 quantiles — academic baseline
- [x] **Evaluation** (`evaluation/signals/`):
  - Rank IC (Spearman) per date + summary (mean, std, IR, hit rate)
  - **Newey-West HAC t-stat** (L=4·(N/100)^(2/9)) — corrects for overlapping windows
  - IC decay across horizons (1, 5, 10, 20 days)
  - Stability (day-over-day turnover proxy)
  - Naive long-short paper Sharpe (pre-cost)
- [x] **FDR Control** (`fdr.py`):
  - Benjamini-Hochberg (BH)
  - Benjamini-Yekutieli (BY) — valid under arbitrary dependence
  - **Storey q-value** (preferred) — estimates π₀ from p-value distribution, no α needed
- [x] **Validation** (`validation/`):
  - Walk-forward CV with purge + embargo
  - **CPCV** (Combinatorial Purged CV) — all combinations of test blocks
  - **PBO** (Probability of Backtest Overfitting) — Bailey et al. (2017)

### Signal Combination
- [x] **Elastic Net** (`elastic_net_combiner.py`):
  - Pooled panel regression (not Fama-MacBeth)
  - q-value weighting: $X_j \leftarrow (1 - q_j) X_j$ (Adaptive Lasso equivalent)
  - CPCV-tuned α, l1_ratio (nested: search on CPCV, eval on untouched holdout)
- [x] **GBM** (`gbm_combiner.py`): `HistGradientBoostingRegressor` with same CPCV
- [x] **PBO Comparison**: GBM vs EN on identical folds, shared holdout

### Risk & Portfolio Construction (NEW)
- [x] **RMT Correlation Cleaning** (`correlation_cleaning.py`):
  - Marchenko-Pastur bounds: $\lambda_{\max} = (1 + \sqrt{q})^2$, $q = N/T$
  - Eigenvalues ≤ λₘₐₓ → replace with mean (preserves trace)
- [x] **Louvain Clustering** (`clustering.py`):
  - Graph from positive correlations only (negative = diversification, not risk)
  - Modularity maximization → no *k* to choose
- [x] **Covariance Estimation** (`covariance.py`):
  - **Ledoit-Wolf (2004)**: Analytical shrinkage to identity
  - **OAS (2012)**: Finite-sample optimal shrinkage
  - **Factor Model (PCA)**: Low-rank + diagonal residual
- [x] **Portfolio Construction** (`portfolio_construction.py`):
  - Signal → weights: proportional / inverse-vol / rank
  - Cluster caps: max 2× equal share per Louvain cluster
  - Vol targeting: scale to 10% annualized using cleaned Σ
- [x] **Transaction Costs** (`costs.py`):
  - Spread: power-law in ADV (Bacry et al. 2015)
  - Commission: fixed bps
  - **Market Impact**: Almgren-Chriss √-model: $\kappa \sigma \sqrt{\frac{|\Delta w| \cdot \text{NAV}}{\text{ADV}}}$
- [x] **Optimization** (`optimization.py`):
  - Inverse Volatility (naive risk parity)
  - **Equal Risk Contribution (ERC)**: $RC_i = w_i (\Sigma w)_i / (w'\Sigma w) = 1/P$
  - **Hierarchical Risk Parity (HRP)**: López de Prado (2016)
  - **Clustered Risk Parity**: Raffinot (2018) — uses Louvain clusters directly
  - **Black-Litterman**: Market prior + signal views
  - **Cost-Aware MVO**: Gârleanu & Pedersen (2013) — $\max w'\mu - \frac{1}{2}w'(\Sigma+\Lambda)w + w'\Lambda w_{\text{prev}}$
- [x] **Backtest Engine** (`backtest.py`):
  - Walk-forward rebalancing (daily/weekly/monthly)
  - Realistic execution lag (signal at close *t* → trade at open *t+1*)
  - Cost-adjusted returns, turnover, attribution framework

---

## 📊 Current Progress

| Component | Status | Notes |
|-----------|--------|-------|
| Data download & ETL | ✅ Complete | |
| Feature engineering | ✅ Complete | 5 families, 20+ features |
| Signal transforms | ✅ Complete | 4 methods, tested |
| Signal evaluation (IC, NW, decay) | ✅ Complete | |
| FDR (BH, BY, q-value) | ✅ Complete | |
| Walk-forward CV + CPCV | ✅ Complete | Purge/embargo |
| Elastic Net combiner | ✅ Complete | CPCV-tuned, q-weighted |
| GBM combiner | ✅ Complete | Same CV framework |
| PBO (GBM vs EN) | ✅ Complete | Identical folds, shared holdout |
| RMT correlation cleaning | ✅ Complete | Validated on synthetic |
| Louvain clustering | ✅ Complete | Validated on synthetic |
| Covariance shrinkage | ✅ Complete | LW, OAS, Factor |
| Portfolio construction | ✅ Complete | 3-layer pipeline |
| Transaction costs | ✅ Complete | Spread + commission + √-impact |
| Optimization solvers | ✅ Complete | HRP, ERC, BL, cost-aware |
| Backtest engine | ✅ Complete | Walk-forward + attribution |
| **Seasonality features** | 🔲 Planned | |
| **Point-in-time constituents** | 🔲 Planned | Need CRSP/Compustat approx |
| **Fundamentals** | 🔲 Not started | |
| **Intraday / other assets** | 🔲 Not started | |

---

## 🧪 Testing Philosophy

> **Every function is tested against the "Musée des Horreurs"** — a zoo of deliberately awkward panels in `tests/conftest.py`:

| Fixture | Tests |
|---------|-------|
| `panel_normal` | Baseline: 4 tickers, 1 date, distinct values |
| `panel_normal_multidate` | Real baseline: 4 tickers, 3 dates |
| `panel_unsorted` | Row order shuffled — catches missing `.sort()` |
| `panel_ragged` | IPOs/delistings — tickers not on every date |
| `panel_single_ticker` | Group size = 1 (std=0, rank degenerate) |
| `panel_identical_values` | All equal — std=0, all ranks tied |
| `panel_with_nulls` | Missing features (insufficient history) |
| `panel_uneven_group` | 7 tickers → doesn't divide into 10 deciles |
| `panel_with_ties` | Tied values — tests `qcut` behavior |

**Rule**: Every new feature/signal function must pass relevant fixtures before touching real data.

---

## ⚠️ Known Limitations

### 1. Survivorship Bias (Critical)
**Universe = today's S&P 500 applied retroactively.**  
Delisted/removed companies missing entirely. Recent additions have history pre-dating index membership. All returns, Sharpe ratios, factor exposures biased upward.

> **Not an oversight — scope decision.** True point-in-time constituents need CRSP/Compustat (paid). Free approximate PIT list is on roadmap.

**Also corrupts market proxy for beta:** Equal-weighted average over this biased universe. Pre-2003: few constituents → unstable variance → extreme beta values.  
*Fixes*: (1) Minimum ticker count guard (implemented), (2) Replace with SPY/SP500 index (planned).

### 2. Data Provenance
- Prices from **yfinance** (unofficial, not licensed)
- `Adj Close` retroactively restated by Yahoo on new corporate actions
- Re-downloading same history → slightly different values
- No completeness/timeliness guarantees vs real vendor

### 3. No Transaction Costs in Early Stages
- Signal evaluation Sharpe = **pre-cost upper bound**
- Costs modeled only in final backtest (`04_build_portfolio.py`)

### 4. Single Asset Class / Frequency
- US equities, daily bars only
- No intraday, no other asset classes, no fundamentals

### 5. Test Coverage
- `tests/` exists but not fully populated
- Correctness leans on `inspect_*.ipynb` notebooks + targeted unit tests

---

## 🗺️ Roadmap

### High Priority
- [ ] **Fix beta market proxy** → use SPY/SP500 index returns
- [ ] **Seasonality features** (`seasonality.py`)
- [ ] **Approximate point-in-time constituent list** (free sources)
- [ ] **Factor attribution** in backtest (style, sector, country)

### Medium Priority
- [ ] **Fundamentals integration** (earnings, book value, etc.)
- [ ] **Multi-asset class** (futures, FX, crypto)
- [ ] **Intraday features** (if data available)
- [ ] **Real-time paper trading** simulation

### Research Ideas
- [ ] **Nonlinear signal transforms** (splines, trees)
- [ ] **Regime-aware portfolio construction** (vol regime, correlation regime)
- [ ] **Alternative risk models** (GARCH, stochastic vol)
- [ ] **Alternative clustering** (HDBSCAN, spectral)

---

## 💡 Design Philosophy

> **Written manually, on purpose.** The point is understanding every decision, not maximizing coding speed.

- **No free parameters by construction** (see Methodology):
  - Rank IC over raw backtest Sharpe (harder to overfit)
  - Rank transform over zscore-tanh for untrusted features (no clip bound to pick)
  - Louvain over k-means (no *k* to choose)
  - Newey-West lag = standard rule of thumb (not tuned per signal)
  - Storey q-value over fixed α (estimates π₀ from data)
- **LLMs used for discussion, debugging, code review — NOT for implementation decisions I don't understand.** Anything suggested gets read, questioned, checked against a second implementation.

---

## 📚 Key References

| Area | Reference |
|------|-----------|
| **Backtest Overfitting** | Bailey, Borwein, López de Prado & Zhu (2017), *J. Comput. Finance* 20(4) |
| **CPCV / Purged CV** | López de Prado (2018), *Advances in Financial ML*, Ch. 7, 12 |
| **RMT / Marchenko-Pastur** | Laloux, Cizeau, Bouchaud, Potters (1999), *PRL* 83(7) |
| **Louvain Clustering** | Blondel, Guillaume, Lambiotte, Lefebvre (2008), *J. Stat. Mech.* |
| **HRP** | López de Prado (2016), *J. Portfolio Mgmt* 42(4) |
| **Clustered Risk Parity** | Raffinot (2018), *SSRN* |
| **Ledoit-Wolf Shrinkage** | Ledoit & Wolf (2004), *J. Multivariate Anal.* 88(2) |
| **OAS** | Ledoit & Wolf (2012), *Ann. Statist.* 40(2) |
| **Almgren-Chriss Impact** | Almgren & Chriss (2001), *J. Risk* 3(2) |
| **Cost-Aware Optimization** | Gârleanu & Pedersen (2013), *J. Finance* 68(6) |
| **Black-Litterman** | Black & Litterman (1992), *FAJ* 48(5) |
| **FDR / q-value** | Storey (2002), *JRSS-B* 64(3); Benjamini & Yekutieli (2001) |
| **Newey-West** | Newey & West (1987, 1994), *Econometrica* / *Rev. Econ. Stud.* |

---

## 📄 License

MIT License — see `LICENSE` file.