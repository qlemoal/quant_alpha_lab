# Methodology

This document explains each stage of the pipeline: **what** it does, **why**, the **formulas**, the **checks**, and the **limits**.
The README gives the overview. This file gives the reasoning.
Math is written in plain text (no LaTeX), so it reads the same everywhere.

**Contents**
1. [Conventions](#1-conventions)
2. [Data layer (01, 02)](#2-data-layer-stages-01-and-02)
3. [Features (03)](#3-features-stage-03)
4. [Signals (04a)](#4-signals-stage-04a)
5. [Signal evaluation (04b)](#5-signal-evaluation-stage-04b)
6. [False discovery rate](#6-false-discovery-rate)
7. [Validation: CPCV, purge, embargo, holdout](#7-validation)
8. [Combination: Elastic Net (04c)](#8-combination-stage-04c)
9. [Risk model (04d)](#9-risk-model-stage-04d)
10. [Portfolio construction (04e)](#10-portfolio-construction-stage-04e)
11. [Backtest (04f)](#11-backtest-stage-04f)
12. [Metrics: definitions](#12-metrics-definitions)
13. [Library modules not in the pipeline](#13-library-modules-not-in-the-pipeline)
14. [Limitations](#14-limitations)
15. [Reproducibility](#15-reproducibility)
16. [References](#16-references)

---

## 1. Conventions

### 1.1 The clock

```mermaid
flowchart LR
    A["Close of t<br/>last data used"] --> B["Signal S(t)<br/>weights w(t)"] --> C["Open of t+1<br/>entry price"] --> D["Open of t+1+h<br/>exit price"]
```

- Signal at the close of day t. Trade at the open of day t+1. This is the earliest trade that is possible.
- Label (forward return): `fwdret_h(t) = log(open_adj[t+1+h]) − log(open_adj[t+1])`.
- Default h = 20 trading days. Horizons 1, 5, 10 and 20 are saved.
- A label of day t and a label of day t+1 share h − 1 days. They overlap. Every test below accounts for this.

### 1.2 Units

| Quantity | Unit |
|---|---|
| Weight | fraction of NAV. Gross = Σ|w|. Gross 1 means 50% long and 50% short. |
| Volatility, covariance | annualised (daily × 252) |
| Costs | basis points of traded dollars, then returned as fractions of NAV |
| Sharpe | annualised. For h-day returns: mean / std × √(252 / h) |

### 1.3 Periods

```mermaid
flowchart LR
    S["SEARCH<br/>select, tune,<br/>out-of-fold scores"] --> G["GAP<br/>embargo"] --> H["HOLDOUT<br/>last 200 dates<br/>judge here"]
```

- **Holdout:** the last 200 dates that have a label.
- **Gap:** `embargo` dates between the end of the search period and the start of the holdout, so no search label reaches into the holdout.
- 04b and 04c use the search period only. 04f reports the holdout separately.

---

## 2. Data layer (stages 01 and 02)

```mermaid
flowchart LR
    W["Wikipedia<br/>S&P 500 table"] --> T["to_yahoo_symbol<br/>BRK.B → BRK-B"] --> Y["Yahoo download<br/>auto_adjust = False"]
    Y --> R["raw CSV<br/>per ticker"] --> E["ETL<br/>clean, dedupe, adjust open"] --> P["ohlcv.parquet"]
```

### 2.1 Download (01)

- Universe: the current S&P 500 members from Wikipedia. Yahoo uses `-` where Wikipedia uses `.` for share classes.
- Each ticker gets 3 tries. A download is accepted only if it has the expected columns and at least 20 rows.
- Failures are saved in `config.json`. The script stops if more than 5% of tickers fail.
- `auto_adjust = False`. We keep both the price as traded (`close_raw`) and the adjusted close (`close`).

### 2.2 ETL (02)

| Step | Why |
|---|---|
| Read every column as text, then cast | A bad value becomes null and is counted. It does not crash the read. |
| Drop rows with unreadable date, price or volume ≤ 0 | These are data errors. The count per reason is saved. |
| One row per (ticker, date) | Duplicates that differ in one value are removed by a fixed rule, not by chance. |
| `open_adj = open × close / close_raw` | The open must be on the same adjusted scale as the close. |

**Why `open_adj` matters.** We enter and exit at the open. If the open were not adjusted, the open-to-open return would show a fake drop at each ex-dividend date and at each split.

### 2.3 Known data limits

- **Survivorship bias:** only today's members are in the universe (section 14).
- Yahoo restates the adjusted close at each corporate action. A new download can differ in the last digits.

---

## 3. Features (stage 03)

```mermaid
flowchart TB
    O["ohlcv.parquet"] --> LR["logret = log(close_t / close_t−1)"]
    LR --> M["mom_w = sum of logret over w days<br/>w = 5, 10, 20, 60, 120, 252"]
    LR --> V["vol_w = std of logret over w days<br/>w = 5 … 252"]
    LR --> B["beta_w = cov(r, market) / var(market)<br/>w = 60, 252"]
    O --> D["dollar_volume = close_raw × volume"] --> A["log_adv_w = log(mean dollar volume over w days)"]
```

- **Momentum** `mom_w`: cumulative log return over the last w days.
- **Volatility** `vol_w`: standard deviation of daily log returns over the last w days.
- **Liquidity** `log_adv_w`: log of the average dollar volume. It uses `close_raw`, because dollars traded = price as traded × shares.
- **Beta** `beta_w`: regression slope on the equal-weighted market return (a proxy; section 14).

**Why windows from 5 to 252:** short windows capture reversal, long windows capture trend. The data decides which is useful (stage 04b). The number of candidates is counted in the false discovery rate.

**Check:** all features are computed per ticker with trailing windows. A test cuts the prices at a date and confirms that no earlier feature changes (no look-ahead).

---

## 4. Signals (stage 04a)

```mermaid
flowchart LR
    F["feature<br/>(mom20, vol60, …)"] --> C["clean<br/>NaN, inf → null"] --> X["transform<br/>across stocks<br/>of the SAME date"] --> S["signal<br/>'{feature}_{method}'"]
    O["open_adj"] --> L["labels fwdret_h<br/>per ticker"]
```

### 4.1 Transforms

| Method | Formula (per date, across stocks) | Use |
|---|---|---|
| `zscore` | z = (x − mean) / std | scale-free comparison |
| `zscore_tanh` | tanh(z / scale) | bounded in (−1, 1), outliers pushed in |
| `zscore_clip` | clip(z, −3, 3) | bounded, linear inside |
| `rank` | 2 × (rank − 1) / (N − 1) − 1 | uniform in [−1, 1] |
| `decile` | +1 top 10%, −1 bottom 10%, 0 otherwise | sparse long-short |

**Why across stocks of the same date:** the strategy bets on which stock is better *today*, not on whether the market is high. A cross-sectional transform removes the market level and the time trend from the signal.

**Why `clean_features` first:** a null is skipped by the mean of a date. A NaN is not. One NaN would turn the mean of the whole date into NaN.

### 4.2 Labels

- `fwdret_h` is computed **per ticker**. The last h + 1 rows of each ticker have no label (the prices do not exist).
- **Check `check_labels`:** if any of those rows has a value, the label crossed a ticker boundary and the script stops. This guards bug 1 of the README table.
- **Check `calendar_gaps`:** labels shift by rows. A row shift equals trading days only if the ticker has a row for every date. The script reports the missing share.

### 4.3 What 04a never does

It never computes a return-based statistic of a signal. Looking at IC over all dates here would let the holdout influence which signals you decide to build. The descriptive statistics (null share, mean, std, names per date) use no returns.

---

## 5. Signal evaluation (stage 04b)

```mermaid
flowchart LR
    S["signals + labels<br/>search period only"] --> I["daily IC<br/>Spearman(signal, fwdret)"] --> N["IC series"]
    N --> T["Newey-West<br/>t-stat"] --> P["p-value"]
    N --> NO["non-overlapping<br/>t-stat (cross-check)"]
    P --> F["FDR<br/>BH, BY, Storey q"] --> SEL["selected: q < 0.2"]
```

### 5.1 Information coefficient

For each date d: `IC(d) = Spearman rank correlation between signal(d, i) and fwdret_h(d, i), over stocks i`.

**Why Spearman (rank):** it ignores outliers and the scale of returns. It measures only the *order* of stocks, which is what a long-short portfolio uses.

**Why IC and not Sharpe:** the IC has fewer free choices (no weights, no costs, no leverage). It is harder to overfit.

### 5.2 Newey-West t-statistic

The IC series has autocorrelation by construction: IC(d) and IC(d+1) use labels that share h − 1 days.
A plain t-statistic `mean(IC) / (std(IC) / √n)` would be too large.

- Newey-West uses a long-run variance with Bartlett weights over `L` lags.
- `L = max(1, 4 × (n / 100)^(2/9), 2 × (h − 1))`. The term 2(h − 1) comes from the overlap. The first term is the standard rule.
- `t = mean(IC) / sqrt(LRV / n)`, p-value from the normal law (two-sided).

**Cross-check:** keep one IC every h days (no overlap) and compute a plain t-statistic. The two should agree in sign and order of size. A signal that passes only the Newey-West test is suspicious.

### 5.3 Outputs

`ic_summary` (mean IC, t-stat, p-value, hit rate), `ic_decay` (IC at each horizon), `fdr_report` (BH, BY, Storey q), `selected_signals.json` (names, q-values, search period, horizon).

---

## 6. False discovery rate

We test many signals. At a 5% level, 1 in 20 pure-noise signals passes. The false discovery rate (FDR) controls the *share of false signals among those we select*.

```mermaid
flowchart LR
    P["p-values<br/>m signals"] --> SORT["sort p(1) ≤ … ≤ p(m)"]
    SORT --> BH["BH: largest k with<br/>p(k) ≤ k/m × α"]
    SORT --> BY["BY: same, with α / c(m)<br/>(any dependence)"]
    SORT --> ST["Storey q-value:<br/>estimate π₀, then q"]
```

| Method | Rule | When |
|---|---|---|
| BH (Benjamini-Hochberg) | reject p(1..k), k the largest with p(k) ≤ (k / m) × α | independent or positively dependent tests |
| BY (Benjamini-Yekutieli) | BH with α divided by c(m) = Σ 1/j | any dependence. More strict. |
| Storey q-value | q(i) = π₀ × m × p(i) / i, made monotone. π₀ = share of true nulls, estimated from large p-values. | the selection rule used here: q < α (default 0.2) |

**Selection weights.** 04c uses `(1 − q)` as a prior weight for each selected signal. A weaker signal enters with a smaller weight.

**Honest warning.** Transforms of one feature (mom20 with tanh and with rank) are not independent tests. BH can then be too permissive. The BY column and the non-overlapping t-statistic are there for this reason.

---

## 7. Validation

### 7.1 Why plain K-fold fails here

Labels overlap in time. A training row at day t and a test row at day t + 5 share 15 days of return. A random split leaks the answer.

### 7.2 Combinatorial Purged CV (CPCV)

```mermaid
flowchart TB
    A["search dates cut into 10 blocks"] --> B["choose 2 blocks as TEST<br/>45 combinations = 45 folds"]
    B --> C["PURGE: remove from training the dates<br/>whose label overlaps a test block (h + 1 = 21)"]
    C --> D["EMBARGO: remove the dates just after a test block<br/>(21 + 16 = 37)"]
    D --> E["fit on the rest, score the test blocks"]
```

```
 blocks:   1   2   3   4   5   6   7   8   9  10
 fold k:   .   T   .   .   .   T   .   .   .   .       T = test, . = train
                  ^^^                  ^^^
                  purge before, embargo after each test block
```

- Each date is in the test set of 9 of the 45 folds. Its out-of-fold score is the **average** of its 9 test predictions.
- Purge = h + 1 = 21 dates. Embargo = purge + 16 = 37 dates (16 = `EMBARGO_WINDOW`, the length of the longest feature memory that matters after a block).
- **Holdout:** the last 200 labelled dates. Not in the 10 blocks. A gap of `embargo` dates separates the search period from it.

### 7.3 What the out-of-fold score means

A score for date d comes from models that never saw d (nor its overlapping neighbours). It is an honest estimate of how the model behaves on new dates **for fixed hyperparameters**. The hyperparameters themselves (alpha, l1_ratio) are chosen on the same folds, so search-period scores are still optimistic. Only the holdout is clean.

---

## 8. Combination (stage 04c)

```mermaid
flowchart TB
    SEL["selected signals<br/>(from 04b)"] --> STD["standardise each signal<br/>(mean and std from the training rows)"]
    STD --> PW["multiply by (1 − q)"]
    Y["target = fwdret_h minus its<br/>cross-sectional mean per date"] --> EN
    PW --> EN["Elastic Net<br/>grid over alpha, l1_ratio"]
    EN --> CV["CPCV: 45 folds"]
    CV --> OOF["out-of-fold score<br/>(search period)"]
    CV --> BEST["best alpha, l1_ratio"] --> FIN["final fit on the whole search period"]
    FIN --> GH["gap and holdout score"]
    OOF --> OUT["en_scores.parquet<br/>period = search | gap | holdout"]
    GH --> OUT
```

### 8.1 Model

`score(i, d) = Σ_k beta_k × x_k(i, d)`, where x_k is the standardised signal k, multiplied by (1 − q_k).
Elastic Net minimises: `mean squared error + alpha × (l1_ratio × Σ|beta| + (1 − l1_ratio) / 2 × Σ beta²)`.

**Why Elastic Net:**
- L1 removes signals that add nothing.
- L2 shares weight among correlated signals (many signals are transforms of the same feature). Pure Lasso would pick one at random.
- The model is linear. Every coefficient can be read and plotted (`en_combiner_coef_stability.png`).

**Why a demeaned target:** the portfolio is dollar-neutral. The market return of the day is not what we predict. We predict the order of stocks.

### 8.2 Output layout

| `period` | Meaning |
|---|---|
| `search` | out-of-fold score (average of the folds where the date was test) |
| `gap` | final model on dates between search and holdout |
| `holdout` | final model on the last 200 labelled dates |

### 8.3 Checks (tests)

- A planted signal in a synthetic panel is recovered (coefficient has the right sign and size).
- Every search date has an out-of-fold score.
- Purge and embargo dates are respected.
- If the holdout labels are destroyed, the model does not change.
- The holdout IC on the planted signal is positive.

---

## 9. Risk model (stage 04d)

**Goal:** an estimate of the covariance Σ of the stock returns, good enough to size the book and to cap clusters.

```mermaid
flowchart TB
    R["returns, last 252 days<br/>N stocks (about 500)"] --> ST["standardise each stock"]
    ST --> C["correlation matrix C (pairwise)"]
    C --> MP["Marchenko-Pastur<br/>keep eigenvalues above the noise edge"]
    MP --> LW["Ledoit-Wolf shrinkage<br/>toward the identity, in correlation space"]
    LW --> SIG["Σ = D · C_clean · D<br/>D = diag(daily vol) × √252"]
    C --> LV["Louvain communities<br/>(graph of correlations)"] --> CL["clusters"]
```

### 9.1 The problem

With N = 500 stocks and T = 252 days, T / N ≈ 0.5. The sample covariance has rank at most T − 1. Its small eigenvalues are close to 0 and its large ones are too large. An optimiser would use this noise.

### 9.2 Marchenko-Pastur cleaning

For pure noise with ratio q = N / T, the eigenvalues of the correlation matrix fall in `[(1 − √q)², (1 + √q)²]`.

- Eigenvalues above the upper edge carry structure (market, sectors). They are kept.
- Eigenvalues inside are noise. They are replaced by their average (the trace is kept).
- Result: a cleaned correlation matrix with a clear factor part.

### 9.3 Ledoit-Wolf shrinkage

`C_final = (1 − rho) × C_clean + rho × I`. The intensity `rho` is estimated from the data (no free parameter) in correlation space, from the standardised returns.
Check: a simulation with a known covariance (error against the truth: shrunk ≈ 45–55, sample 56, a broken version 848).

### 9.4 From correlation to covariance

`Σ = D × C × D` with `D = diag(daily vol) × √252`. The diagonal of Σ equals the annualised stock variances. **A test checks this** (bug 7 returned C as if it were Σ).

### 9.5 Louvain clusters

Build a graph where the stocks are nodes and the edges are positive cleaned correlations. Louvain maximises modularity. There is **no k to choose**. The clusters are used in 04e as exposure caps.

### 9.6 Dates

The model is estimated on **weekly rebalance dates**, each with the trailing 252 days. Nothing after the date is used.

---

## 10. Portfolio construction (stage 04e)

```mermaid
flowchart LR
    S["score<br/>(04c)"] --> W["1. raw weights<br/>proportional to the<br/>demeaned score"]
    W --> CAP["2. cluster caps<br/>limit gross exposure<br/>of each cluster"]
    CAP --> NEU["3. re-neutralise<br/>long = short"]
    NEU --> VT["4. volatility target<br/>scale so w'Σw = 10%²"]
    VT --> LEV["5. leverage cap<br/>gross ≤ 4"]
    LEV --> CHK["6. checks<br/>neutral, finite,<br/>capped"]
```

| Step | What | Why |
|---|---|---|
| 1 | `w_i ∝ score_i − mean(score)` on the stocks that have a score and a risk estimate, then Σ|w| = 1 | simple and transparent. Score size = conviction. |
| 2 | Cap the gross exposure of each Louvain cluster at 2 × (gross / number of clusters). Overweight clusters are scaled down. The freed capacity is not redistributed. | a book concentrated in one correlated group is a sector bet, not a stock-picking bet |
| 3 | Subtract the mean again so Σ long = Σ short | capping breaks neutrality |
| 4 | `scale = target_vol / sqrt(w' Σ w)` | the book takes the same risk in calm and in stormy markets |
| 5 | If gross > 4, scale down | an assumption (limits borrowing). It rarely binds with a 10% target. |
| 6 | Checks at each date | the stage stops on a book that is not neutral or not finite |

**Weights are fractions of NAV.** Gross 1 = 50% long, 50% short. After step 4, the gross can be above or below 1.

**Output:** `weights.parquet` (date, ticker, weight) and `weights_diag.parquet` (gross, long, short, ex-ante vol, leverage, number of names, per date).

---

## 11. Backtest (stage 04f)

```mermaid
flowchart LR
    W["weights at close t"] --> E["entry at open t+1<br/>costs on the change<br/>in position"]
    E --> H["hold: weights drift<br/>with the stock returns"]
    H --> NX["next rebalance:<br/>trade only the difference"]
    NX --> E
    H --> R["daily gross return"]
    E --> C["daily cost"]
    R --> N["net = gross − cost"]
    C --> N
```

### 11.1 Mechanics

- A weight set from date t is traded at the **open of t+1** (the earliest possible fill).
- Between rebalances the positions are **bought and held**. Weights drift with price moves. We do not rebalance to the target every day (that would hide turnover).
- At each rebalance only the difference between the drifted weights and the new target is traded.

### 11.2 Costs

For each stock traded on a rebalance, with `trade$ = |Δw| × AUM` and `ADV$` its average daily dollar volume:

```
 cost_bps = half_spread + impact + commission
 half_spread = 3 bps                         (fixed, an assumption)
 impact      = kappa × daily_vol × sqrt(participation)    per dollar traded (kappa = 0.5)
               participation = min(1, trade$ / ADV$)
 commission  = 1 bp
 cost        = Σ |Δw| × cost_bps
```

- The square-root law says: impact grows with the *square root* of the trade size relative to the volume.
- AUM matters. A bigger book pays more per dollar.
- The cost is charged on the entry date of the trade.

**Not modelled:** borrow cost of shorts, financing cost, delisting returns, taxes, market-closure effects. These make the result **optimistic**.

### 11.3 Reports

The backtest prints three blocks: whole period, before the holdout, **holdout**. Read the holdout.

| Number | How to read it |
|---|---|
| Sharpe (net) and 95% interval | the interval is wide. A few years cannot pin a Sharpe down. |
| Sharpe gross | the same, without costs. The gap shows the cost drag. |
| Market beta | should be near 0 (dollar-neutral book) |
| Realised vol / ex-ante vol | near 1 means the risk model is right. Far from 1 means it is not. |
| Annual turnover, annual cost (bps) | a signal that trades too much dies by costs |
| Sharpe at 1×, 1.5×, 2×, 3× costs | if the Sharpe falls to 0 at 2×, the strategy is fragile |

---

## 12. Metrics: definitions

| Metric | Definition |
|---|---|
| IC | Spearman correlation of signal and forward return, per date |
| IC t-stat | mean IC / Newey-West standard error |
| Sharpe (daily returns) | mean / std × √252 |
| Sharpe (h-day returns) | mean / std × √(252 / h). **Not** √252: the returns overlap. |
| Sharpe 95% interval (Lo, 2002) | SR ± 1.96 × sqrt((1 + SR² / 2) / n_years-like) with the autocorrelation correction |
| Max drawdown | on the **compounded** curve of non-overlapping returns: min over t of (equity_t / max_{s ≤ t} equity_s − 1) |
| Market beta | slope of the strategy return on the equal-weighted market return |
| Turnover | one-way: Σ |Δw| / 2 per rebalance, summed over a year (multiples of NAV) |
| Realised / ex-ante vol | std of net daily returns × √252 divided by the mean ex-ante vol of 04d |

**Why compounded drawdown:** a cumulated sum of overlapping h-day returns counts each day h times. The drawdown looked about 20 times too large in the first version.

---

## 13. Library modules not in the pipeline

These modules are tested, but 04a to 04f do not call them.

| Module | What it offers | Why not wired yet |
|---|---|---|
| `src/portfolio/optimization.py` | HRP, ERC, Black-Litterman, cost-aware mean-variance | the main pipeline uses the simple proportional chain on purpose, as a baseline |
| `src/models/gbm_combiner.py` + PBO | gradient boosting against Elastic Net, compared by the probability of backtest overfitting | needs a decision on how to report it |
| `src/validation/walk_forward_cv.py` | walk-forward splits | CPCV is the main scheme |
| `src/portfolio/backtest.py` | older backtest engine | replaced by `weights_backtest.py` |
| `scripts/pipeline/04_build_portfolio.py` | legacy script | to be archived |

**PBO (probability of backtest overfitting).** Split the data into blocks. For each combination, pick the best strategy on half the blocks and check its rank on the other half. PBO is the share of cases where the in-sample winner ranks below the median out-of-sample. A value near 0.5 means that selection is not better than chance.

---

## 14. Limitations

1. **Survivorship bias.** Current S&P 500 members only. Stocks that fell out of the index (often after bad returns) are missing. The long side looks better than it was.
2. **Data quality.** Free data. Splits and dividends are adjusted by the vendor. Errors are possible. The label check counts moves beyond ±2.
3. **Cost model.** The inputs (AUM, spread, kappa, commission) are assumptions. Borrow cost, financing and delisting are absent.
4. **A single holdout path.** 200 dates is about 9 months. The Sharpe interval is wide.
5. **Dependent tests.** Candidate signals are transforms of few features. FDR assumes less dependence than we have.
6. **Beta proxy.** The market is the equal-weighted average of the stocks in the file.
7. **Capacity.** No volume constraint on positions beyond the impact term.
8. **Scope.** US equities, daily bars.

---

## 15. Reproducibility

- Every stage reads files and writes files under `data/clean/.../{stage}/{run_id}/`.
- `config.json` in each stage stores: parameters, a hash of the parameters, a hash of the input file, the git commit, and whether the working tree had uncommitted changes.
- Random seeds are fixed where randomness exists (cross-validation, Louvain).
- `04_run_all.py` runs the chain and stops at the first failure.
- Tests: `pytest -q`.

---

## 16. References

| Area | Reference |
|---|---|
| FDR | Benjamini & Hochberg (1995) · Benjamini & Yekutieli (2001) · Storey (2002), Storey & Tibshirani (2003) |
| Autocorrelation | Newey & West (1987, 1994) |
| Purged CV, CPCV | López de Prado (2018), *Advances in Financial Machine Learning*, ch. 7 and 12 |
| Backtest overfitting | Bailey, Borwein, López de Prado & Zhu (2017), *J. Computational Finance* 20(4) |
| Random matrix theory | Marchenko & Pastur (1967) · Laloux, Cizeau, Bouchaud & Potters (1999), *PRL* 83(7) |
| Shrinkage | Ledoit & Wolf (2003, 2004) · Chen, Wiesel, Eldar & Hero (2010), *IEEE TSP* 58(10) (OAS) |
| Number of factors | Bai & Ng (2002), *Econometrica* 70(1) |
| Clustering | Blondel, Guillaume, Lambiotte & Lefebvre (2008), *J. Stat. Mech.* (Louvain) |
| Risk parity | López de Prado (2016), *J. Portfolio Management* 42(4) (HRP) |
| Trading costs | Almgren & Chriss (2000), *J. Risk* 3(2) · Gârleanu & Pedersen (2013), *J. Finance* 68(6) |
| Sharpe uncertainty | Lo (2002), *Financial Analysts Journal* 58(4) |
| Multiple testing in finance | Harvey, Liu & Zhu (2016), *Review of Financial Studies* 29(1) |