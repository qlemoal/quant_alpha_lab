# Methodology

**Version:** 1.0  
**Last Updated:** 2026-10-20  
**Status:** Living document — updated as components are implemented and validated

---

## Overview

This document describes the **full quantitative research pipeline** implemented in this repository, from raw data to backtested portfolio. Every component is modular, tested, and replaceable — the goal is reusable infrastructure, not a single strategy.

The methodology follows practices used at systematic investment firms (Millennium, Citadel, Two Sigma, AQR) but built from scratch to ensure full understanding of every decision.

---

## Pipeline Architecture

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

## 1. Data Layer

### 1.1 Universe & Download

- **Source:** S&P 500 constituents from Wikipedia + daily OHLCV from yfinance
- **Frequency:** Daily, adjusted close (`Adj Close`)
- **Period:** 2000-01-01 to present (configurable)

### 1.2 ETL & Cleaning (DuckDB)

| Step | Description |
|------|-------------|
| Deduplication | Drop exact duplicate rows |
| Missing data | Forward-fill `Adj Close` within ticker (max 5 days) |
| Returns | Compute log returns: $r_t = \log(P_t) - \log(P_{t-1})$ |
| Validation | Flag tickers with $< 60$ observations, extreme returns $> 50\%$/day |
| Output | `prices.parquet`: [date, ticker, open, high, low, close, adj_close, volume, logret] |

### 1.3 Known Data Limitations

> **Survivorship Bias (Critical):** Universe = today's S&P 500 applied retroactively. Delisted companies missing entirely. Returns, Sharpe ratios, factor exposures biased upward. **Not an oversight — scope decision.** True point-in-time constituents need CRSP/Compustat.

> **Beta Market Proxy:** Equal-weighted average over biased universe. Pre-2003: few constituents $\to$ unstable variance $\to$ extreme beta values. Fixes planned: minimum ticker guard + SPY index proxy.

> **yfinance Provenance:** Unofficial, not licensed. `Adj Close` retroactively restated on corporate actions. Re-downloading yields slightly different values.

---

## 2. Feature Engineering

All features computed per-ticker, sorted by `[ticker, date]`, using **Polars lazy evaluation** (memory efficient, no collect until needed).

| Feature | File | Formula | Lookback Windows |
|---------|------|---------|------------------|
| **Log Return** | `returns.py` | $r_t = \log P_t - \log P_{t-1}$ | — |
| **Forward Return** | `returns.py` | $R_{t \to t+h} = \sum_{k=1}^h r_{t+k}$ | 1, 5, 10, 20 |
| **Momentum** | `momentum.py` | $\text{mom}_w = \frac{P_t}{P_{t-w}} - 1$ | 5, 20, 60, 126, 252 |
| **Volatility** | `volatility.py` | $\sigma_w = \sqrt{\frac{1}{w-1}\sum_{i=1}^w (r_{t-i} - \bar{r})^2}$ | 20, 60, 126, 252 |
| **Avg Dollar Volume** | `adv.py` | $\text{ADV}_w = \frac{1}{w}\sum_{i=1}^w V_{t-i} \cdot P_{t-i}$ | 20, 60, 126 |
| **Beta** | `beta.py` | $\beta_w = \frac{\text{Cov}(r_i, r_{mkt})}{\text{Var}(r_{mkt})}$ | 60, 126, 252 |

**Output:** `features.parquet` with all features joined on `[date, ticker]`.

---

## 3. Signal Construction

A raw feature is not a signal. It must be transformed cross-sectionally (per date, across all tickers) to become comparable.

### 3.1 Transform Methods (`src/signals/combine.py`)

| Method | Formula | Properties | Best For |
|--------|---------|------------|----------|
| **zscore_tanh** | $z \mapsto \tanh(c \cdot z)$ | Smooth, bounded, preserves magnitude near 0 | Validated features (mom, adv, std) |
| **zscore_clip** | $z \mapsto \text{clip}(z, -c, c)$ | Hard bound, discontinuous at boundary | Quick prototyping |
| **rank** | $\text{rank}(x) \mapsto \frac{2\cdot\text{rank} - N - 1}{N - 1} \in [-1, 1]$ | Bounded, outlier-proof, discards magnitude | Untrusted features (beta) |
| **decile** | $x \mapsto \text{bucket}_{10}(x)$ | Coarse, academic baseline | Comparison vs literature |

Where $z = \frac{x - \mu_{\text{cs}}}{\sigma_{\text{cs}}}$ (cross-sectional z-score), $c=3$ default.

### 3.2 Design Principle

- **No free parameters by construction:** Rank transform needs no clip bound. Louvain needs no $k$. Storey q-value needs no $\alpha$.
- **Different features $\to$ different transforms:** Not mutually exclusive, not usually worth stacking.

---

## 4. Signal Evaluation

**Primary metric = Rank IC (Spearman), not backtest Sharpe.**  
Backtest Sharpe is far easier to overfit by adjusting construction details. IC measures predictive power directly.

### 4.1 Rank IC

For each date $t$:
$$
\text{IC}_t = \rho_S(s_t, R_{t \to t+h})
$$
where $\rho_S$ = Spearman rank correlation, $s_t$ = signal values, $R_{t \to t+h}$ = forward returns.

Summary statistics:
- $\text{Mean IC}$, $\text{Std IC}$
- **Information Ratio:** $\text{IR} = \frac{\text{Mean IC}}{\text{Std IC}}$
- **Hit Rate:** $\frac{1}{T}\sum_{t=1}^T \mathbb{1}[\text{IC}_t > 0]$

### 4.2 Newey-West HAC t-statistic

Overlapping windows (e.g., mom20 uses 20-day window) induce serial correlation in IC series. Naive t-stat assumes independence $\to$ wrong.

**Long-run variance:**
$$
\hat{\Omega} = \gamma_0 + 2\sum_{k=1}^L \left(1 - \frac{k}{L+1}\right)\gamma_k
$$
where $\gamma_k$ = lag-$k$ autocovariance of demeaned IC series, Bartlett kernel weights.

**Standard error:** $\text{SE} = \sqrt{\hat{\Omega} / T}$  
**t-stat:** $t = \frac{\text{Mean IC}}{\text{SE}}$

**Lag $L$:** Newey-West (1994) rule of thumb $L = 4(T/100)^{2/9}$ — fixed convention, **not tuned per signal**.

> **Direction not fixed:** Positive autocorrelation $\to$ NW inflates SE $\to$ lower t-stat (typical). Negative autocorrelation $\to$ NW tightens SE $\to$ higher t-stat. Both are legitimate.

### 4.3 IC Decay

Rank IC vs forward horizon $h \in \{1, 5, 10, 20\}$:
$$
\text{IC}(h) = \rho_S(s_t, R_{t \to t+h})
$$
Informs rebalance frequency: IC only at $h=1$ $\to$ daily rebalancing (expensive); IC holds to $h=20$ $\to$ weekly/monthly feasible.

### 4.4 Naive Long-Short Paper Sharpe

Pre-cost upper bound. Decile long-short, equal-weight, rebalanced daily:
$$
\text{Sharpe}_{\text{paper}} = \frac{\mathbb{E}[R_{\text{LS}}]}{\text{Std}[R_{\text{LS}}]} \times \sqrt{252}
$$
**Uses simple returns for cross-sectional aggregation** (log returns not additive across assets — see Section 12.1).

### 4.5 Stability

Day-over-day turnover proxy: $\frac{1}{T}\sum_t \|\mathbf{w}_t - \mathbf{w}_{t-1}\|_1$ where $\mathbf{w}_t$ = signal ranks.

---

## 5. False Discovery Rate Control

Testing $m$ candidate signals $\to$ multiple testing problem. Fixed $\alpha = 0.05$ per test $\to$ false positives by construction.

### 5.1 Benjamini-Hochberg (BH)

Sort p-values: $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$  
Find max $k$: $p_{(k)} \le \alpha \cdot \frac{k}{m}$  
Reject $H_0$ for $i \le k$. Valid under independence / PRDS.

### 5.2 Benjamini-Yekutieli (BY)

Same mechanics, threshold divided by harmonic number:
$$
p_{(k)} \le \frac{\alpha \cdot k}{m \cdot H_m}, \quad H_m = \sum_{i=1}^m \frac{1}{i}
$$
Valid under **arbitrary dependence** (our case: correlated signals from overlapping history). ~3.6$\times$ more conservative than BH at $m=20$.

### 5.3 Storey's q-value (Preferred)

Flips the question: *If this p-value were the cutoff, what FDR would result?*

1. Estimate $\pi_0$ = proportion of true nulls from p-value right tail:
   $$
   \hat{\pi}_0(\lambda) = \frac{\#\{p_i > \lambda\}}{m(1-\lambda)}, \quad \lambda \in \Lambda
   $$
   Cubic spline fit over fixed grid $\Lambda$, evaluate at $\lambda \to 1$.

2. Compute q-values (backward pass, monotonic):
   $$
   q_{(i)} = \min\left(\hat{\pi}_0 \cdot m \cdot \frac{p_{(i)}}{i},\; q_{(i+1)}\right), \quad q_{(m)} = \hat{\pi}_0 \cdot p_{(m)}
   $$

**Advantages:**
- No $\alpha$ needed to compute
- More powerful when $\pi_0 < 1$ (real signals present)
- Continuous weight for combiner, not hard gate

**Usage:** q-values feed into Elastic Net as continuous weights $(1 - q_j)$, not as survives/doesn't-survive filter.

---

## 6. Validation: Leakage-Safe Cross-Validation

### 6.1 Purged Walk-Forward CV

Train strictly before test. Purge = drop training rows whose labels overlap test period. Embargo = gap after test before next training starts.

```
Train ────────┐     ┌──────── Test ────────┐     ┌──── Next Train
              └─Purge─┘                     └─Embargo─┘
```

**Embargo = Structural + Empirical:**
- Structural: $h + 1$ days (forward label reaches $h+1$ days past test end)
- Empirical: Largest lag with FDR-validated autocorrelation in market returns (currently 16 days, BY-corrected)
- **Total embargo = 37 days for $h=20$** (sum convention, López de Prado 2018)

### 6.2 Combinatorial Purged CV (CPCV)

True K-fold: train can surround test chronologically $\to$ embargo needed for **single-fold validity** (not just cross-fold).

CPCV: Evaluate all $\binom{n_{\text{blocks}}}{n_{\text{test}}}$ combinations of held-out blocks. More folds, more robust against path-dependent overfitting.

### 6.3 Probability of Backtest Overfitting (PBO)

Bailey et al. (2017): Given performance matrix across candidates, repeatedly split into IS/OOS halves. PBO = fraction where IS winner ranks below OOS median.

**Used for:** GBM vs Elastic Net comparison (Section 8).

---

## 7. Signal Combination

### 7.1 Elastic Net Combiner (`src/models/elastic_net_combiner.py`)

Pooled panel regression (not Fama-MacBeth):
$$
\min_{\beta} \frac{1}{2N}\|\mathbf{y} - \mathbf{X}\beta\|_2^2 + \lambda\left[\alpha\|\beta\|_1 + \frac{1-\alpha}{2}\|\beta\|_2^2\right]
$$
with **q-value column weighting**: $\mathbf{X}_j \leftarrow (1 - q_j)\mathbf{X}_j$

**Why Elastic Net?**
- Candidate signals correlated by construction (mom20/mom60/mom252 share history)
- Pure Lasso: arbitrarily picks one from correlated group (unstable)
- Pure Ridge: no sparsity, keeps noise
- EN L1+L2: standard answer for correlated predictors (Zou & Hastie 2005)

**Why pooled, not Fama-MacBeth?**
- Fama-MacBeth = inference tool (time-series of cross-sectional $\beta$)
- We need **predictive combination for deployment**
- Single date: ~500 names, too thin for stable fit; pooled across fold: ~40,000 rows

**Hyperparameter search:** CPCV-tuned $(\alpha, l1\_ratio)$ on identical folds for fair comparison.

### 7.2 GBM Combiner (`src/models/gbm_combiner.py`)

`HistGradientBoostingRegressor` — nonlinear counterpoint to EN.
- If GBM outperforms $\to$ real interaction effects exist
- If not $\to$ flexibility adds variance, PBO should reflect this (Friedman 2001)

### 7.3 Missing Data Policy

Current: **Drop row if any signal null** (all-or-nothing).  
Cost: Long-lookback signals (mom252) disproportionately drop early/recent tickers.

Alternative (planned): Cross-sectional median imputation. Since signals are normalized, median $\approx 0$ $\to$ "no conviction" on that signal for that ticker.

---

## 8. Model Comparison: GBM vs Elastic Net via PBO

**Protocol (`scripts/compare_gbm_vs_en_pbo.py`):**
1. Identical CPCV folds for both models (fairness)
2. Hyperparameter search on CPCV
3. Fixed, chosen models evaluated on **shared untouched holdout**
4. Holdout split into contiguous sub-blocks for PBO matrix
5. Compute PBO: fraction of splits where IS winner fails OOS

**Sanity checks (passing):**
- Synthetic linear: High PBO (no real difference to detect) ✓
- Synthetic interaction: Low PBO (GBM genuinely better) ✓

---

## 9. Risk Model: Correlation & Covariance

### 9.1 RMT Correlation Cleaning (`src/risk/correlation_cleaning.py`)

Empirical correlation matrix $\mathbf{C}$ over $N$ assets, $T$ observations $\to$ mostly noise.

**Marchenko-Pastur (MP) bounds:** For pure noise, eigenvalues lie in
$$
\lambda \in \left[(1 - \sqrt{q})^2,\; (1 + \sqrt{q})^2\right], \quad q = \frac{N}{T}
$$

**Cleaning:** Eigenvalues $\le \lambda_{\max} = (1 + \sqrt{q})^2$ replaced by their mean (preserves trace = $N$). Only eigenvalues $> \lambda_{\max}$ treated as genuine factor structure.

**Why upper bound only:** Financial $\mathbf{C}$ is PSD by construction; MP lower bound can dip below 0 for large $q$ $\to$ not a real constraint.

**Validated:** Synthetic 3-cluster panel (sizes 15/10/8 + 12 noise, $T=500$) $\to$ exactly 3 eigenvalues above $\lambda_{\max}$, gap between 3rd (3.39) and 4th (1.27, at noise boundary 1.69), trace preserved exactly (45.0 $\to$ 45.0).

### 9.2 Louvain Community Detection (`src/risk/clustering.py`)

Graph from **positive correlations only** (negative = diversification, not risk clustering). Edges: $w_{ij} = \max(C_{ij}, 0)$.

Modularity maximization:
$$
Q = \frac{1}{2m}\sum_{ij} \left[A_{ij} - \frac{k_i k_j}{2m}\right]\delta(c_i, c_j)
$$
No $k$ to choose $\to$ number of clusters falls out automatically (no-free-parameters).

**Validated:** Same synthetic panel $\to$ perfect recovery of 3 planted clusters on RMT-cleaned matrix.

### 9.3 Covariance Estimation (`src/risk/covariance.py`)

$\mathbf{\Sigma} = \mathbf{D}\mathbf{C}\mathbf{D}$ where $\mathbf{D} = \text{diag}(\sigma_i)$

| Method | Description | Reference |
|--------|-------------|-----------|
| **Ledoit-Wolf (LW)** | Analytical shrinkage to identity/single-factor | Ledoit & Wolf (2004) |
| **Oracle Approximating Shrinkage (OAS)** | Finite-sample optimal shrinkage | Ledoit & Wolf (2012) |
| **Factor Model (PCA)** | Low-rank $\mathbf{B}\mathbf{B}'$ + diagonal residual | Bai & Ng (2002) for factor selection |

---

## 10. Portfolio Construction (`src/risk/portfolio_construction.py`)

One cross-section at a time. Re-estimation frequency belongs to backtest loop.

### 10.1 Score $\to$ Raw Weights

| Method | Formula | Notes |
|--------|---------|-------|
| **Proportional** (default) | $w_i = \frac{s_i}{\sum_j |s_j|}$ | Dollar-neutral, gross-normalized, continuous |
| **Inverse Volatility** | $w_i \propto \frac{s_i}{\sigma_i}$ | Risk-equalized signal tilt |
| **Rank** | $w_i \propto \frac{\text{rank}(s_i)}{\sum_j |\text{rank}(s_j)|}$ | Robust to outliers |

**Proportional preferred:** No bucket-count parameter, uses continuous magnitude. Tradeoff: likely higher turnover than decile buckets (not yet measured).

### 10.2 Cluster Exposure Caps

Cap each cluster's absolute exposure at $\text{max\_multiple} \times \text{equal\_share}$ (default 2.0).

**Verified behavior:** Caps relative to *original* total gross, does **not** redistribute freed capacity. A capped cluster's share of the *shrunken* total can exceed nominal cap (e.g., capped to 66.7% $\to$ ends at 93% of final total). Iterative redistribution would give intuitive "no cluster exceeds X% of final portfolio" — not implemented, revisit if binding.

### 10.3 Volatility Targeting

Scale weights uniformly:
$$
\mathbf{w} \leftarrow \mathbf{w} \cdot \frac{\sigma_{\text{target}}}{\sqrt{\mathbf{w}'\mathbf{\Sigma}\mathbf{w}}}
$$
Applied **after** cluster capping (target vol of portfolio actually held). $\mathbf{\Sigma}$ built from **RMT-cleaned correlation**, not raw.

### 10.4 Scope Boundary

Drawdown constraints = path-dependent $\to$ belong in backtest loop, not here.

---

## 11. Transaction Costs (`src/portfolio/costs.py`)

Realistic cost model for optimization and backtesting.

### 11.1 Spread Cost

Power-law in ADV (Bacry et al. 2015):
$$
\text{spread}_i = \text{spread}_{\text{ref}} \cdot \left(\frac{\text{ADV}_{\text{ref}}}{\text{ADV}_i}\right)^\beta, \quad \beta \approx 0.25
$$
Half-spread paid per trade (crossing spread).

### 11.2 Commission

Fixed per notional:
$$
\text{comm}_i = \text{commission\_bps} \cdot 10^{-4} \cdot |\Delta w_i| \cdot \text{NAV}
$$

### 11.3 Market Impact (Almgren-Chriss Square-Root)

$$
\text{impact}_i = \kappa \cdot \sigma_i \cdot \sqrt{\frac{|\Delta w_i| \cdot \text{NAV}}{\text{ADV}_i}}
$$
where $\kappa \approx 0.1$–$0.3$ (calibrated), $\sigma_i$ = daily volatility.

### 11.4 Total Cost per Rebalance

$$
\text{Cost} = \sum_i \left(\frac{\text{spread}_i}{2} + \text{comm}_i + \text{impact}_i\right) \cdot |\Delta w_i| \cdot \text{NAV}
$$

---

## 12. Optimization (`src/portfolio/optimization.py`)

### 12.1 Hierarchical Risk Parity (HRP)

López de Prado (2016). Uses correlation hierarchy (our Louvain clusters) for recursive bisection allocation. Solves Markowitz instability without ad-hoc constraints.

### 12.2 Clustered Risk Parity

Raffinot (2018). Equal risk contribution **within and across** Louvain clusters directly.

### 12.3 Equal Risk Contribution (ERC)

$RC_i = w_i (\mathbf{\Sigma}\mathbf{w})_i / (\mathbf{w}'\mathbf{\Sigma}\mathbf{w}) = 1/P$ for all $i$.

### 12.4 Black-Litterman

Market equilibrium prior + signal views:
$$
\mathbf{\mu}_{\text{BL}} = [(\tau\mathbf{\Sigma})^{-1} + \mathbf{P}'\mathbf{\Omega}^{-1}\mathbf{P}]^{-1}[(\tau\mathbf{\Sigma})^{-1}\mathbf{\Pi} + \mathbf{P}'\mathbf{\Omega}^{-1}\mathbf{Q}]
$$
Signal = views $\mathbf{Q}$, confidence $\mathbf{\Omega}$.

### 12.5 Cost-Aware MVO

Gârleanu & Pedersen (2013):
$$
\max_{\mathbf{w}} \mathbf{w}'\boldsymbol{\mu} - \frac{1}{2}\mathbf{w}'(\mathbf{\Sigma} + \mathbf{\Lambda})\mathbf{w} + \mathbf{w}'\mathbf{\Lambda}\mathbf{w}_{\text{prev}}
$$
where $\mathbf{\Lambda}$ = diagonal cost matrix from Section 11.

---

## 13. Backtest Engine (`src/portfolio/backtest.py`)

### 13.1 Walk-Forward Loop

For each rebalance date $t$:
1. **Lookback window** $[t-W, t]$: Estimate $\mathbf{\Sigma}$, clusters, ADV, vol
2. **Signal at close $t$** $\to$ construct weights
3. **Execute at open $t+1$** (1-day execution lag)
4. **Hold until next rebalance** $\to$ realize returns
5. **Apply costs** based on weight change $\Delta \mathbf{w}$

### 13.2 Key Realism Features

- Execution lag: Signal at close $t$ $\to$ trade at open $t+1$
- Costs deducted from gross returns $\to$ net returns
- Vol targeting uses cleaned $\mathbf{\Sigma}$ at each rebalance
- Turnover, costs, attribution tracked per period

### 13.3 Performance Metrics

| Metric | Formula |
|--------|---------|
| Net Sharpe | $\frac{\mathbb{E}[r_{\text{net}}]}{\text{Std}[r_{\text{net}}]}\sqrt{252}$ |
| Gross Sharpe | Same, pre-cost returns |
| Annual Return | $(1 + \text{mean daily})^{252} - 1$ |
| Max Drawdown | $\max_{\tau} \frac{\text{peak}_\tau - \text{trough}_\tau}{\text{peak}_\tau}$ |
| Hit Rate | $\frac{1}{T}\sum \mathbb{1}[r_t > 0]$ |
| Cost Drag | $\text{Gross Sharpe} - \text{Net Sharpe}$ (annualized bps) |
| Turnover | $\frac{1}{T}\sum_t \|\mathbf{w}_t - \mathbf{w}_{t-1}\|_1$ |

---

## 14. Known Limitations & What Doesn't Work

| Area | Limitation | Impact | Mitigation |
|------|------------|--------|------------|
| **Survivorship Bias** | Today's S&P 500 applied retroactively | Upward bias in all returns/Sharpe | Scope decision; PIT list planned |
| **Beta Proxy** | EW average over biased universe | Extreme pre-2003 betas | Min ticker guard + SPY index (planned) |
| **Data Vendor** | yfinance unofficial | Retroactive Adj Close changes | Documented, not fixable free |
| **Single Asset Class** | US equities daily only | No diversification benefit | Roadmap: futures, FX, crypto |
| **No Fundamentals** | Price/volume only | Misses value/quality factors | Roadmap: earnings, book value |
| **Pre-Cost Evaluation** | Signal IC/Sharpe = upper bound | Overstates tradability | Costs modeled only in final backtest |
| **Test Coverage** | Not all functions tested | Reliance on notebook inspection | `tests/` expansion planned |
| **Cluster Cap Redistribution** | No reallocation of capped capital | Final portfolio may be concentrated | Iterative scheme if binding |

---

## 15. Design Philosophy (No Free Parameters)

| Decision | Principle | Alternative Rejected |
|----------|-----------|---------------------|
| Rank IC > Paper Sharpe | Harder to overfit | Backtest Sharpe (too many degrees of freedom) |
| Rank transform | No clip bound to pick | zscore+clip (free parameter $c$) |
| Louvain clustering | No $k$ to choose | k-means (free parameter $k$) |
| NW lag = rule of thumb | Not tuned per signal | Cross-validated lag (overfits) |
| Storey q-value > BH/BY | No $\alpha$ needed | Fixed $\alpha$ (free parameter) |
| q-value as weight | Continuous, not binary | Hard filter (second free parameter) |
| Pooled panel EN | Predictive, not inferential | Fama-MacBeth (inference tool) |

---

## 16. Key References

| Area | Reference |
|------|-----------|
| **Backtest Overfitting / PBO** | Bailey, Borwein, López de Prado & Zhu (2017), *J. Comput. Finance* 20(4) |
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
| **Elastic Net** | Zou & Hastie (2005), *JRSS-B* 67(2) |
| **Adaptive Lasso** | Zou (2006), *JASA* 101(476) |

---

## 17. Reproducibility

```bash
# Full pipeline
python scripts/pipeline/01_download_data.py
python scripts/pipeline/02_etl_prices.py
python scripts/pipeline/03_build_features.py
# Inspect signals via notebooks/inspect_signal.ipynb
python scripts/pipeline/04_build_portfolio.py  # Combines signals, builds portfolio, backtests
```

All randomness seeded. Configuration via environment variables (see `04_build_portfolio.py`).

---

*End of Methodology*