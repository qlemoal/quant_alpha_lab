# Methodology notes

Write-ups on the parts I want to get right, not just implement. Linked from the main README.

---

## Evaluating a signal properly

- **Information Coefficient (IC):** correlation between a signal's values and forward returns. Main metric, not raw returns from one backtest, which is too easy to overfit to.
- **Spearman, not Pearson.** IC here means Spearman rank correlation ("Rank IC"), not Pearson. Pearson is sensitive to outliers, a handful of extreme return days can dominate it and give a misleading picture. Spearman only cares about ranking, which is what actually matters when the signal is used to rank and pick stocks, not predict exact magnitudes.
- **Walk-forward, purged, embargoed CV, not random k-fold.** Random k-fold leaks future information into training folds in a time series. Walk-forward keeps training strictly before testing, chronologically. Purging drops training rows whose label or feature window overlaps the test period. Embargo adds a small gap after the test period before the next fold's training starts, since return autocorrelation can leak information across a hard boundary even after purging. Reference: López de Prado, *Advances in Financial Machine Learning*.
- **FDR before trusting anything.** Testing many signals, or many parameter windows of the same signal, is a multiple-testing problem. A 5% significance threshold applied naively across dozens of tests produces false positives by construction. Benjamini-Hochberg (or similar) gets applied before calling anything "significant."
- **Newey-West correction can increase the t-stat, not just decrease it.**
The naive t-stat assumes independent daily IC observations. Overlapping-window features like `mom20` violate that with positive serial correlation, so the usual expectation is NW correction shrinks the t-stat. It can legitimately go the other way for a slow-moving feature like `mom252`, whose cross-sectional ranking barely changes day to day: the day-to-day variation in Rank IC then comes almost entirely from the forward-return side, not the signal side. Short-horizon (1-day) equity returns carry mild negative serial correlation (short-term reversal, bid-ask bounce), and with a near-static signal ranking, that telescopes directly into negative autocorrelation of the daily IC series. Negative serial correlation genuinely tightens the NW standard error relative to the naive one, successive noise partially cancels, so a higher NW t-stat in this case is a legitimate finding, not a bug (distinct from the earlier shape-mismatch bug that produced spurious inflation from a silently 2D array).
- **FDR: BH and BY, reported side by side, not collapsed to one verdict.**
Testing many candidate signals (or many parameter variants of the same signal) at a fixed 5% threshold produces false positives by construction, this is the multiple-testing problem. Benjamini-Hochberg (BH) is the standard first choice, but it assumes independence or positive regression dependence among the test statistics. Candidate signals here are built from the same overlapping return panel, different lookback windows of the same feature, or different transform methods of the same underlying signal, so that assumption is questionable. 

Benjamini-Yekutieli (BY) is valid under arbitrary dependence but strictly more conservative (its threshold shrinks by a factor of H_m = sum(1/i, i=1..m), so at 20 candidates roughly 3.6x more conservative than BH). Both get reported together rather than picking one silently, following Harvey, Liu & Zhu (2016, "...and the Cross-Section of Expected Returns," Review of Financial Studies), who use this exact approach for the same reason: candidate return-predictive signals in finance are rarely independent tests. A signal surviving BH but not BY isn't rejected outright, it's flagged as significant only under an assumption the data likely violates, and gets a closer look rather than an automatic pass or fail.

---

## Turning a feature into a signal

A raw feature (e.g. `mom20`) isn't directly usable to rank stocks against each other, different tickers live on different scales. Needs a cross-sectional transform first, per date, across all tickers.

**Z-score, then bound the tails.**
Standardize the feature within each date (subtract the cross-sectional mean, divide by the cross-sectional std). Preserves relative magnitude: a stock two standard deviations above average looks meaningfully different from one that's five above. Needs a follow-up step to handle outliers, since z-scores are unbounded:
- **Hard clip** (`clip(-3, 3)`), simple, standard default, but discontinuous at the boundary.
- **tanh**, smooth alternative, near-linear close to 0, saturates past roughly `|x| = 3`. Preferred when the discontinuity of a hard clip feels wrong for the use case.

**Rank transform.**
Rank the feature within each date, rescale to `[-1, 1]`. Bounded and outlier-proof by construction, no separate clipping step needed, since it only uses ordinal position, not magnitude. Trade-off: discards magnitude information, a stock ranked #1 and #2 get treated as equally far apart as #50 and #51, even if their actual feature values are wildly different.

**Decile / quantile bucketing.**
Split the cross-section into buckets (deciles are the classic academic-paper convention) and assign each stock a bucket label instead of a continuous score. Coarser than either of the above, but this is the standard way academic factor papers report results (long top decile, short bottom decile) and it's an easy, standard baseline to benchmark a more continuous signal against.

**Which one fits which feature, roughly:**
- Well-behaved, already-validated features (`mom`, `adv`, `std`): z-score + tanh by default. Magnitude is trustworthy, worth keeping.
- Features still noisy or not fully trusted yet (`beta`, until the market-proxy fix lands): rank transform. Robust to whatever extreme values the underlying computation is still producing, and doesn't require deciding on clip/tanh parameters for a feature whose scale isn't stable yet.
- Building a simple, explainable baseline strategy, or comparing against an academic long/short factor result: decile bucketing.

None of these are mutually exclusive across the project, different features can use different methods depending on how much the magnitude is trusted. Not usually worth stacking two on the same feature.

---

## Evaluating a new signal, the fixed checklist

Run in this order, every time, before a signal is trusted for anything downstream:

1. Build it with `make_signal()`.
2. `signal_plots.one_pager()`, visual check: coverage, per-date mean/std, ticker x date heatmap, sample ticker lines. Catches structural bugs before anything else does.
3. `report.signal_report()`, numeric check: IC mean/std/IR, Newey-West corrected t-stat, hit rate, stability, naive long-short paper Sharpe.
4. `ic.metrics.ic_decay()`, how far out the predictive power actually holds, informs a sensible rebalance frequency.
5. `report.compare_reports()` against other candidates, side by side.
6. FDR across everything tested this round, not yet built, this is the next real piece of infrastructure.
7. Survivors move to the signal combiner.

---

## Cleaning the correlation matrix

Planned: `src/risk/covariance_cleaning.py`.

- An empirical correlation matrix over hundreds of tickers is mostly noise. Random Matrix Theory gives a theoretical eigenvalue range (Marchenko-Pastur) expected from pure noise, given the number of assets and observations. Eigenvalues inside that range get shrunk or removed, only eigenvalues above it are treated as genuine structure.
- **Absorption Ratio** (Kritzman & Li): share of total variance explained by the top few eigenvalues. When it rises, the market's correlation structure is collapsing onto fewer common factors, historically associated with periods preceding market stress. Planned as a rolling diagnostic once the risk pipeline is further along.