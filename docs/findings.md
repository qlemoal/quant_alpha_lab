# Research Findings Log

**Format:** Dated, append-only entries. Each entry: what was tested, what was found, statistical status, robustness checks. Cross-references to config/commit for reproducibility.

---

## 2026-08-20: Momentum/Reversal Dead Zone Across Horizons

### Test
Rank IC of `mom` feature across lookback horizons: 5, 10, 20, 60, 126, 252 days. Forward return horizon: 20 days. FDR correction: Benjamini-Yekutieli (BY), $\alpha = 0.2$.

### Finding
Two distinct, literature-consistent regimes with a **dead zone** between them:

| Horizon | IC Sign | BY-Significant? | Interpretation |
|---------|---------|-----------------|----------------|
| 5 days | Negative | ✓ | Short-term reversal (Jegadeesh 1990) |
| 10 days | Negative | ✓ | Short-term reversal |
| 20 days | Negative | ✓ | Short-term reversal |
| 60 days | ~Zero | ✗ | **Dead zone** — transition region |
| 126 days | ~Zero | ✗ | **Dead zone** |
| 252 days | Positive | ✓ | Medium-term momentum (Jegadeesh & Titman 1993) |

### Robustness
- Confirmed via **naive long-short Sharpe** sign consistency: negative at short horizons, positive at 252 days, near-zero in dead zone
- Not an artifact of signal transform — tested across zscore_tanh, rank, decile
- Dead zone aligns with known literature: reversal and momentum are separate phenomena with opposite-signed mechanisms (liquidity provision vs. behavioral underreaction)

### Implication
Do not blend all momentum horizons into one signal. Treat short-horizon (reversal) and long-horizon (momentum) as **separate signal families** with different economic drivers. The 60-126 day range adds noise, not signal.

### Files
- `notebooks/inspect_signal.ipynb` (IC decay plots)
- `src/evaluation/signals/IC/metrics.py` (`ic_decay()`)
- Commit: `843e7a2` (correlation_cleaning.py edits)

---

## 2026-09-02: Isolated Autocorrelation at Lags ~15-16 and ~32 in Market Returns

### Test
Lag-wise autocorrelation of equal-weighted market return (S&P 500 universe), lags 1–60. FDR correction: BY, $\alpha = 0.05$ (distinct from signal FDR's $\alpha = 0.2$, chosen independently).

### Finding
Beyond the expected **lag-1 reversal** (strong, well-documented), two **isolated significant lags** at **lag 15–16** and **lag 32**:

- Not part of a contiguous decay from lag 1
- Disconnected from the boundary (no significant lags between 2–14 or 17–31)
- Survive BY correction (valid under arbitrary dependence)

### Leading Explanations (Not Yet Verified)
| Lag | Candidate Mechanism | Status |
|-----|---------------------|--------|
| 1 | Nonsynchronous trading (Lo & MacKinlay 1990) — some stocks trade before close, information diffuses next day | **Well-established** |
| 15–16 | Biweekly rebalancing flows? Options-expiration-adjacent effects? | **Unverified** |
| 32 | Monthly rebalancing? End-of-month effects? | **Unverified** |

### Lag-1 Reversal Deep Dive
**Mechanism:** Not every stock trades up to the close. Less liquid names print last trade minutes/hours early. Cross-sectional average return today partly reflects stale prices. Tomorrow, those names "catch up," inducing positive autocorrelation in the portfolio average even if no individual stock is predictable.

**Formalization:** Lo & MacKinlay (1990), *An Econometric Analysis of Nonsynchronous Trading*, J. Econometrics 45(1-2).

**Compounding factor:** Information diffusion speed varies by coverage — large caps react faster, small caps lag, creating apparent "continuation."

### Impact on Embargo
Used for empirical embargo component in CPCV (see Methodology §6.1):
- Structural embargo: $h + 1 = 21$ days (for $h=20$ forward horizon)
- Empirical embargo: **16 days** (largest BY-surviving lag)
- **Total embargo = 37 days** (sum convention, López de Prado 2018)

### Files
- `scripts/embargo_selection.py`
- `src/validation/cpcv.py` (`total_embargo_window()`)
- Commit: `67e19ec` (pipeline scripts)

---

## 2026-09-10: GBM vs. Elastic Net — PBO Comparison Sanity Checks

### Test
Probability of Backtest Overfitting (PBO) comparison between Elastic Net and GBM (`HistGradientBoostingRegressor`) on **synthetic data** with known ground truth. Protocol: identical CPCV folds, shared untouched holdout, PBO computed on holdout sub-blocks.

### Check 1: Linear Relationship (Null Case — No Real Difference)
**DGP:** $y = 0.8 \cdot s_1 + \epsilon$, where $s_1$ = linear signal, $\epsilon \sim \mathcal{N}(0,1)$. No interaction structure for GBM to exploit.

| Metric | Elastic Net | GBM |
|--------|-------------|-----|
| Holdout Block Sharpes | Interleaved, no pattern | Interleaved, no pattern |
| **PBO** | **0.629** (high) | — |

**Interpretation:** High PBO = IS winner fails OOS frequently. Correctly indicates "no real difference to detect, don't trust either preference."

### Check 2: Sign-Flip Interaction (Alternative Case — GBM Should Win)
**DGP:** $y = 1.2 \cdot s_a \cdot \text{sign}(s_b) + \epsilon$. **Zero marginal linear correlation** — linear model cannot see this structure by construction.

| Metric | Elastic Net | GBM |
|--------|-------------|-----|
| EN Coefficients | Collapse to ~0 (correct) | N/A |
| Holdout Long-Short Sharpe | 0.72 (noise) | **38.28** |
| **PBO** | **0.000** | — |

**Interpretation:** PBO = 0.000 = GBM wins all 70 IS/OOS splits. Full confidence in genuine difference.

### Conclusion
PBO **is not defaulting** to a fixed answer. It swings correctly:
- High PBO (0.629) when no real difference exists
- Low PBO (0.000) when genuine nonlinear structure exists

### Next Step
Re-run both sanity checks on **real signals** before trusting PBO verdict on actual GBM vs EN comparison. Confirms tool behaves correctly before it matters.

### Files
- `scripts/compare_gbm_vs_en_pbo.py`
- `src/validation/cpcv.py` (`probability_of_backtest_overfitting()`)
- `src/models/elastic_net_combiner.py`, `src/models/gbm_combiner.py`

---

## 2026-09-15: RMT Correlation Cleaning — Synthetic Validation

### Test
Marchenko-Pastur denoising on synthetic correlation matrix with **known planted structure**:
- 3 clusters: sizes 15, 10, 8 (total 33 structured assets)
- 12 pure-noise assets
- $N = 45$, $T = 500$ $\to$ $q = 0.09$, $\lambda_{\max} = (1 + \sqrt{q})^2 = 1.69$

### Finding
| Eigenvalue | Value | Status |
|------------|-------|--------|
| $\lambda_1$ | 12.84 | > $\lambda_{\max}$ ✓ (Cluster 1) |
| $\lambda_2$ | 7.21 | > $\lambda_{\max}$ ✓ (Cluster 2) |
| $\lambda_3$ | 3.39 | > $\lambda_{\max}$ ✓ (Cluster 3) |
| $\lambda_4$ | 1.27 | ≤ $\lambda_{\max}$ ✓ (Noise) |
| ... | ... | ... |
| **Trace** | 45.0 → 45.0 | **Preserved exactly** |

- Exactly 3 eigenvalues above $\lambda_{\max}$ — matches planted clusters
- Clear gap: $\lambda_3 = 3.39$ vs $\lambda_4 = 1.27$ (boundary = 1.69)
- Trace preservation confirms cleaning doesn't shrink total variance

### Validation Implication
RMT cleaning correctly separates signal from noise on ground-truth data. Safe to apply to real correlation matrices.

### Files
- `src/risk/correlation_cleaning.py` (`clean_correlation_rmt()`)
- `tests/test_correlation_cleaning.py` (synthetic validation)
- Commit: `ba7319e` (networkx for Louvain)

---

## 2026-09-18: Louvain Clustering — Synthetic Validation

### Test
Louvain community detection on the **RMT-cleaned** synthetic correlation matrix from 2026-09-15.

### Finding
| Planted Cluster | Size | Recovered Members | Purity |
|-----------------|------|-------------------|--------|
| Cluster 1 | 15 | 15/15 | 100% |
| Cluster 2 | 10 | 10/10 | 100% |
| Cluster 3 | 8 | 8/8 | 100% |
| Noise | 12 | 0 assigned to clusters | 100% |

- **Perfect recovery** of all three planted clusters
- No mixing between clusters
- Noise assets correctly left unclustered (or as singletons)
- Modularity maximization finds correct $k=3$ without being told

### Edge-Weight Convention Verified
Graph built from **positive correlations only** ($w_{ij} = \max(C_{ij}, 0)$). Negative correlations dropped — correct for risk clustering (negative correlation = diversification benefit, not shared risk).

### Files
- `src/risk/clustering.py` (`louvain_clustering()`)
- `tests/test_clustering.py`
- Commit: `ba7319e`

---

## 2026-09-25: Newey-West t-stat Direction Can Go Either Way

### Context
Earlier assumption: "Overlapping windows $\to$ positive autocorrelation $\to$ NW inflates SE $\to$ lowers t-stat."

### Finding
**Direction is not fixed.** Observed cases where NW t-stat > naive t-stat:

- **Slow-moving feature** (e.g., `mom252`): Cross-sectional ranking barely changes day-to-day (especially after decile bucketing)
- IC day-to-day variation comes almost entirely from **forward-return side**, not signal side
- If short-horizon returns have mild **negative** serial correlation (short-term reversal, bid-ask bounce, or `Adj Close` restatement artifacts), $\gamma_1 < 0$
- Negative $\gamma_1$ $\to$ NW **tightens** SE $\to$ **higher** NW t-stat than naive

### Historical Bug (For Context)
Earlier `newey_west_ic_tstat` silently accepted 2D input (full DataFrame instead of IC column). Corrupted mean with date values cast to float. Produced absurd t-stats (observed: 3.8 naive vs 194 NW — ratio impossible from real autocorrelation). Fixed by validating input shape (`if x.ndim != 1: raise ValueError`).

**Lesson:** NW t-stat moving "wrong" direction $\to$ check input shape first, then actual autocorrelation.

### Files
- `src/evaluation/signals/IC/metrics.py` (`newey_west_ic_tstat()`, `autocorrelation()`)
- `src/utils/stats.py`

---

## 2026-10-01: Log Returns Not Additive Across Assets

### Bug Found
`decile_longshort_returns()` and `get_decile_longshort_returns_DF()` were averaging **log returns** directly across assets within each basket.

### Why It's Wrong
Log returns are additive across **time**, not across **assets**:
- Basket simple return = weight-average of constituents' simple returns
- Same statement is **false** for log returns
- $\log(1 + R_{\text{basket}}) \neq \sum w_i \log(1 + R_i)$

### Impact
- ~10% relative error in single-day basket return (measured on 20-name basket with dispersed daily log returns)
- Compounds across every day in the series
- **Every `long_short_sharpe` reported before this fix inherited this bias**

### Fix
Convert to simple return before cross-sectional aggregation:
$$
R_{\text{simple}} = \exp(r_{\text{log}}) - 1
$$
Aggregate in simple-return space (where equal-weight average = valid P&L arithmetic). Combine long/short legs (each already basket-level simple returns; dollar-neutral $+1/-1$ combination of simple returns is additive).

### Applies To
- Signal evaluation (decile long-short Sharpe)
- Future backtest loop: portfolio return = $\sum w_i R_i$ in **simple-return space**, not $\sum w_i r_i$ in log space

### Files
- `src/evaluation/signals/performance.py`
- `src/portfolio/backtest.py` (future implementation)
- Commit: `ece9d4a` (build_portfolio.py comments)

---

## 2026-10-10: Cluster Cap Behavior — Verified, Not Assumed

### Test
`apply_cluster_caps()` with `max_multiple = 2.0`, 3 clusters, equal initial allocation.

### Finding
**Caps are relative to original gross, not redistributed.**

Example:
- 3 clusters, gross = 1.0 $\to$ equal share = 1/3 ≈ 33.3% per cluster
- Cap = $2.0 \times 33.3\% = 66.7\%$ absolute
- Cluster 1 capped to 66.7% (original weight was 80%)
- **Freed 13.3% is NOT redistributed** to other clusters
- Final gross = 86.7%
- Cluster 1 share of **final** portfolio = 66.7% / 86.7% ≈ **77%** (not 66.7%!)
- Can exceed nominal `max_multiple / n_clusters` fraction significantly

### Current Behavior
```python
# Does NOT do this:
# 1. Cap clusters
# 2. Redistribute freed capacity proportionally
# 3. Renormalize to target gross
```

### Implication
Final portfolio can be more concentrated than `max_multiple` suggests. Iterative redistribution scheme would guarantee "no cluster exceeds X% of **final** portfolio" — not implemented. Revisit if backtest shows binding clusters often.

### Files
- `src/risk/portfolio_construction.py` (`apply_cluster_caps()`)
- `tests/test_portfolio_construction.py`

---

## 2026-10-15: Missing Data Policy in Elastic Net — Drop vs Impute

### Current Policy
**Drop entire (date, ticker) row if ANY signal is null.** All-or-nothing per row.

### Cost
Long-lookback signals (e.g., `mom252` needs 252 days) disproportionately drop:
- Early-history tickers (first 252 days of data)
- Recently listed tickers (IPOs)
- Interacts with survivorship bias (delisted tickers already missing)

### Alternative: Cross-Sectional Median Imputation
Since signals are pre-normalized (z-scored/ranked), cross-sectional median $\approx 0$.
- Impute missing signal $s_{it}$ with $\text{median}_j(s_{jt})$ for that date
- Effectively: "No conviction on this signal for this ticker"
- Keeps row's **other signals** and forward-return label as training data

### Status
Not yet tested against "musée des horreurs" fixtures (`tests/conftest.py`):
- `panel_ragged` (IPOs/delistings)
- `panel_with_nulls` (missing features)
- `panel_uneven_group` (doesn't divide into 10 deciles)

**Next:** Test both policies on fixtures before choosing production default.

### Files
- `src/models/elastic_net_combiner.py` (`build_design_matrix()`)
- `tests/conftest.py` (fixtures)

---

## 2026-10-20: Elastic Net CPCV Search Not Yet Default

### Current State
`fit_elastic_net_combiner()` uses **walk-forward CV only** (consecutive mode). CPCV-tuned version exists in:
- `scripts/en_cpcv_demo.py`
- `scripts/compare_gbm_vs_en_pbo.py`

### Gap
Nested design (CPCV for hyperparameter search + untouched holdout for final eval) **not folded back** into main combiner function. Two divergent code paths.

### Plan
Once design settled, make CPCV search the default in `fit_elastic_net_combiner()` with walk-forward as fallback option. Single code path, configurable.

### Files
- `src/models/elastic_net_combiner.py`
- `scripts/en_cpcv_demo.py`
- `scripts/compare_gbm_vs_en_pbo.py`

---

## Summary of Open Questions

| # | Question | Priority | Owner |
|---|----------|----------|-------|
| 1 | What drives lags 15–16 and 32 autocorrelation? | Medium | Research |
| 2 | Median imputation vs drop for missing signals in EN? | High | Engineering |
| 3 | Iterative cluster cap redistribution needed? | Low | Research (wait for backtest) |
| 4 | CPCV as default for EN hyperparameter search? | High | Engineering |
| 5 | Point-in-time constituent list (free approx)? | High | Data |
| 6 | Beta market proxy: SPY index vs EW biased universe? | High | Data |
| 7 | Seasonality features (turn-of-month, earnings)? | Medium | Research |
| 8 | Fundamentals integration (earnings, book value)? | Medium | Data/Engineering |

---
