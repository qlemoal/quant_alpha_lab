# Methodology notes

Detailed write-ups on the parts of this project worth getting right, not just
implementing: what each method does, the formula, why it's the right tool
here rather than a common alternative, and what its actual failure modes are.
Written to be read back later, not just once. Linked from the main README.

---

## 1. Evaluating a signal properly

### 1.1 Information Coefficient (IC)

The Information Coefficient is the cross-sectional correlation, per date,
between a signal's values and the forward return realized after that date.
It's the main evaluation metric here, not a single backtest's return or
Sharpe ratio, which is far easier to overfit by adjusting almost anything
about how the backtest is constructed.

**Spearman, not Pearson.** IC here always means Spearman rank correlation
("Rank IC"), computed via `pl.corr(signal, fwd_ret, method='spearman')`.
Pearson correlation is sensitive to outliers, a handful of extreme return
days can dominate the whole day's IC value. Spearman only depends on
ranking, which is what actually matters when the signal is used to rank and
select stocks, not predict their exact return magnitude.

`src/evaluation/signals/IC/core.py`: `compute_ic()` returns one IC value per
date. `summarize_ic()` aggregates that series into `mean`, `std`,
`ir = mean/std` (information ratio, the main number for comparing signals),
`hit_rate` (fraction of days with IC > 0), and a naive t-stat.

### 1.2 Naive significance vs Newey-West HAC correction

**The naive t-stat.** `mean / (std / sqrt(n))`, treating each day's IC as an
independent draw. This is the textbook formula, and it's wrong here by
construction: overlapping-window features (`mom20` uses a 20-day window, so
consecutive days' feature values, and therefore consecutive days' IC, are
mechanically correlated) violate the independence assumption the naive
formula needs.

**Newey-West (HAC) correction.** `IC/metrics.py`, `newey_west_ic_tstat()`.
Instead of treating the IC series as independent draws, it estimates the
long-run variance directly, accounting for serial correlation up to a chosen
number of lags:

```
long_run_var = gamma_0 + 2 * sum_{k=1}^{L} (1 - k/(L+1)) * gamma_k
```

where `gamma_k` is the lag-k autocovariance of the demeaned IC series, and
the `(1 - k/(L+1))` weight is the Bartlett kernel, it down-weights higher
lags smoothly rather than cutting them off abruptly. The standard error of
the mean is then `sqrt(long_run_var / n)`, and the t-stat is `mean / se`.
`L`, the max lag, defaults to the Newey-West (1994) rule of thumb,
`4 * (n/100)^(2/9)`, a fixed convention, not tuned per signal.

**Which direction the correction goes is not fixed.** The common intuition
is "overlapping windows mean positive autocorrelation, so NW inflates the
standard error and lowers the t-stat." That's the typical case, but not
guaranteed. NW responds to whatever autocorrelation is actually present, and
that can be negative: a slow-moving feature like `mom252`, whose
cross-sectional ranking barely changes day to day (especially after decile
bucketing), produces an IC series whose day-to-day *variation* comes almost
entirely from the forward-return side, not the signal side. If short-horizon
returns carry mild negative serial correlation (short-term reversal,
bid-ask bounce, or artifacts from `Adj Close` being retroactively restated,
see README's data provenance limitation), that negative correlation
telescopes into the IC series, and a negative `gamma_1` genuinely tightens
the NW standard error relative to the naive one, successive noise partially
cancels. A higher NW t-stat than the naive one is a legitimate result in
that case, not a bug, distinct from a real bug that also produces this
symptom (see next paragraph).

**A distinct historical bug, for context.** An earlier version of
`newey_west_ic_tstat` silently accepted a 2D input (a full `date, ic`
DataFrame instead of the `ic` column alone), which corrupted the mean with
date values cast to float and produced nonsensical, wildly inflated t-stats
(observed: 3.8 naive vs 194 NW, a ratio no realistic autocorrelation
structure can produce, verified by simulation: even an unrealistic
lag-1 autocorrelation of -0.9 over 1250 observations only inflates the
t-stat by about 3.3x). Fixed by validating input shape at the top of the
function (`if x.ndim != 1: raise ValueError(...)`) rather than coercing
silently. Kept here as a reminder: a NW t-stat moving in the "wrong"
direction is worth checking the input shape first, then the actual
autocorrelation, in that order.

### 1.3 Autocorrelation diagnostics

`src/utils/stats.py`, `autocorrelation()`. General-purpose lag-k
autocorrelation, usable on a flat single time series (e.g. a daily IC
series, one row per date) or on a ticker panel (per-group autocorrelation,
aggregated to a cross-group mean and std, since a single "the"
autocorrelation doesn't exist across a panel of different tickers). This is
the direct diagnostic for the NW question above: print `gammas` from
`newey_west_ic_tstat`, or call `autocorrelation(ic_df, 'ic', lags=(1,2,3))`
directly, to see the sign and size of the serial dependence actually driving
a given correction, rather than inferring it indirectly from IC decay
(decay measures the *level* of predictive power at different horizons, a
different question, conflating the two was an earlier mistake in this
project's own reasoning, worth remembering not to repeat).

### 1.4 False discovery rate control

Testing many candidate signals, or many parameter variants of the same
signal, in the same round is a multiple-testing problem. A 5% significance
threshold applied independently to each one produces false positives by
construction. `src/evaluation/signals/fdr.py` implements three approaches,
deliberately kept side by side rather than collapsing to one:

**Benjamini-Hochberg (BH).** Sort p-values ascending, `p_(1) <= p_(2) <=
... <= p_(m)`. Find the largest `k` such that `p_(k) <= alpha * k / m`,
reject (call significant) every hypothesis up to and including that `k`.
Valid under independence or positive regression dependence (PRDS) among the
test statistics.

**Benjamini-Yekutieli (BY).** Identical mechanics, but the threshold is
divided by the harmonic number `H_m = sum_{i=1}^{m} 1/i`:
`p_(k) <= alpha * k / (m * H_m)`. Valid under arbitrary dependence, no PRDS
assumption needed, at the cost of a threshold that shrinks as `H_m` grows
with `m` (roughly 3.6x more conservative than BH at m=20). The right choice
when candidates are correlated, which is the normal case here: different
lookback windows of the same feature, or different transform methods of the
same underlying signal, are built from the same overlapping return history,
so BH's independence/PRDS assumption is questionable. Harvey, Liu & Zhu
(2016, "...and the Cross-Section of Expected Returns", Review of Financial
Studies) use exactly this reasoning for the same problem in finance.

**BH and BY both fix `alpha` before producing an answer**, and both
implicitly assume the worst case, that every single candidate could be a
true null (`pi0 = 1`), which is exactly why they're conservative even when
most candidates are obviously real signals.

**Storey's q-value.** Flips the question: instead of "is this significant
at a chosen alpha," it asks "if this p-value were used as the cutoff, what
FDR would that produce?" That minimum achievable FDR, per candidate, is its
q-value, the FDR analogue of a p-value. Two differences from BH/BY:

- `pi0` (the true proportion of null candidates) is *estimated* from the
  data instead of assumed to be 1. Genuine null p-values are uniform on
  [0,1]; genuine signals pile up near 0. Looking at the right tail of the
  p-value distribution (p-values near 1, where real signals essentially
  never land) estimates what fraction is pure noise. Implementation:
  compute `pi0_hat(lambda) = mean(p > lambda) / (1 - lambda)` across a
  fixed grid of `lambda` values (a standard, dataset-independent grid, not
  tuned per batch), fit a cubic smoothing spline across the grid, evaluate
  at the right edge. This is the estimator itself, no alpha involved.
- q-values need no `alpha` to compute. Every candidate gets a number,
  monotonically enforced via a backward pass,
  `q_(i) = min(pi0 * m * p_(i) / i, q_(i+1))`, starting from
  `q_(m) = pi0 * p_(m)`. Any cutoff applied afterward is a visibly separate
  decision from the significance computation itself.

Net effect: when `pi0 < 1` (expected whenever a batch actually contains real
signals, not pure noise), q-values are strictly more powerful than BH/BY at
the same nominal FDR, because they aren't spending budget pretending
everything might be null. The cost is one extra estimation step BH/BY don't
need. This is preferred going forward per this project's no-free-parameters
principle (see README, "Design philosophy"): BH/BY require a human-chosen
`alpha` to produce any output at all; q-values don't.

*Reading a q-value, concretely.* Test 10 candidate signals, sorted
p-values `0.001, 0.004, 0.006, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.5`, and
suppose the data suggests about half this batch is genuinely null
(`pi0 = 0.5`). The resulting q-values: `0.005, 0.01, 0.01, 0.0125, 0.02,
0.025, 0.036, 0.05, 0.056, 0.25`.

Read the 3rd one: q=0.01 means "if I declared the 3 smallest p-values
significant, I'd expect 1% of that set of 3 to be false discoveries." Not
"this single signal has a 1% chance of being false", that's not what any
p-value-derived number means, it's a statement about the batch you get if
you drew the line at this candidate's rank.

Notice p and q move in different directions relative to each other: the
smallest p-value (0.001) gets a *larger* q (0.005), penalized for having
been tested alongside 9 other candidates. The largest p-value (0.5) gets a
*smaller* q (0.25) than its own p, because `pi0=0.5` means the procedure
isn't assuming, the way BH's threshold implicitly does, that every last
candidate might be null.

Which number to trust when reporting one figure per signal: the q-value.
A p-value in isolation says nothing about how many other candidates were
tested alongside it, quoting it alone silently implies it's the only test
run, which it never is here. The q-value carries that context built in.

### 1.5 IC decay across forward-return horizons

`ic_decay()`, Rank IC of a signal against forward returns at several
horizons (1, 5, 10, 20 days by default). Shows how long predictive power
actually persists, which directly informs a sensible rebalance frequency: a
signal with IC only at horizon 1 wants daily rebalancing (expensive), one
that holds to horizon 20 can rebalance far less often. Depends entirely on
forward returns being computed correctly as the *cumulative* return over the
horizon, `log(P_{t+h}/P_t) = sum_{k=1}^{h} logret_{t+k}`, not the single
day's return h days out, an earlier bug (`shift(-h)` instead of
`rolling_sum(h).shift(-h)`) computed the latter, which produces a decay
curve that looks artificially flat because every horizon was really
measuring close to the same 1-day relationship at different offsets.

### 1.6 Walk-forward, purged, embargoed cross-validation

Random k-fold leaks future information into training folds in a time
series, a model trained partly on data chronologically after its test set
has seen the future. Walk-forward keeps training strictly before testing.
Purging drops training rows whose label or feature window overlaps the test
period (a `mom252`-based label computed near the fold boundary uses
information technically inside the test window). Embargo adds a small gap
after the test period before the next fold's training starts, since return
autocorrelation can leak information across a hard boundary even after
purging alone. Reference: López de Prado, *Advances in Financial Machine
Learning*. Not yet implemented, needed once the GBM comparison model exists
and cross-validation choices actually matter.

---

## 2. Turning a feature into a signal

A raw feature (`mom20`) isn't directly usable to rank stocks against each
other, different tickers live on different scales. Needs a cross-sectional
transform first, per date, across all tickers. `src/signals/combine.py`,
`make_signal()`.

### 2.1 Z-score, then bound the tails

Standardize within each date: subtract the cross-sectional mean, divide by
the cross-sectional std. Preserves relative magnitude, a stock two standard
deviations above average looks meaningfully different from one five above.
Unbounded by construction, needs a follow-up step:

- **Hard clip** (`clip(-3, 3)`), simple, standard default, discontinuous at
  the boundary.
- **tanh**, smooth alternative, near-linear close to 0, saturates past
  roughly `|x| = 3`. Preferred when the clip's discontinuity feels wrong.

### 2.2 Rank transform

Rank the feature within each date, rescale to `[-1, 1]`. Bounded and
outlier-proof by construction, since it only uses ordinal position, no
separate clipping step needed. Trade-off: discards magnitude, a stock ranked
#1 and #2 are treated as equally far apart as #50 and #51, even if their raw
values differ wildly.

### 2.3 Decile / quantile bucketing

Split the cross-section into buckets, assign a bucket label instead of a
continuous score. Coarser than either of the above, but this is the
standard way academic factor papers report results (long top decile, short
bottom decile), useful as an easy baseline to benchmark a continuous signal
against.

### 2.4 Which one fits which feature

- Well-behaved, already-validated features (`mom`, `adv`, `std`): z-score +
  tanh by default. Magnitude is trustworthy, worth keeping.
- Features not fully trusted yet (`beta`, until the market-proxy fix
  lands): rank transform, robust to whatever extreme values the underlying
  computation still produces.
- A simple, explainable baseline, or comparison against an academic
  long/short factor result: decile bucketing.

Not mutually exclusive across the project, different features can use
different methods. Not usually worth stacking two on the same feature.

---

## 3. Correlation structure

Two genuinely separate questions get asked about the same correlation
matrix, kept as separate functions on purpose, never chained
automatically under the hood:

### 3.1 Cleaning the correlation matrix (RMT / Marchenko-Pastur)

`src/risk/correlation_cleaning.py`, not yet implemented. An empirical
correlation matrix over hundreds of tickers is mostly noise. Random Matrix
Theory gives a theoretical eigenvalue range (Marchenko-Pastur) expected from
a pure-noise correlation matrix, given the assets-to-observations ratio
`q = n_assets / n_obs`. Eigenvalues inside that range get shrunk or
replaced (commonly with their average, to preserve the trace), only
eigenvalues above the range are treated as genuine common-factor structure.
Answers: "how much of this correlation matrix is real." Reference: Laloux,
Cizeau, Bouchaud, Potters (1999/2000).

### 3.2 Community detection (Louvain)

`src/risk/clustering.py`, not yet implemented. Groups tickers into clusters
that maximize modularity on the correlation graph. Answers a different
question from RMT cleaning: "which stocks move together," not "how much of
the correlation structure is real." Louvain over k-means deliberately, no
`k` to choose in advance, the number of clusters falls out of the
modularity-maximization itself, consistent with this project's
no-free-parameters preference (README, "Design philosophy"). Operates on
whatever matrix it's given, raw or RMT-cleaned; chaining the two is a
separate, explicit orchestration step, not automatic inside
`louvain_clustering()` itself, so cleaning is never silently applied or
silently skipped. Reference: Blondel et al. (2008), "Fast unfolding of
communities in large networks."

Kritzman & Li's **Absorption Ratio** (`absorption_ratio.py`, not yet
implemented), share of total variance explained by the top few eigenvalues,
is a related but distinct diagnostic: a rising ratio means the correlation
structure is collapsing onto fewer common factors, historically associated
with periods preceding market stress. A rolling summary statistic, not a
clustering or cleaning method.

---

## 4. Evaluating a new signal: the fixed checklist

Run in this order, every time, before a signal is trusted for anything
downstream:

1. Build it with `make_signal()`.
2. `signal_plots.one_pager()`, visual check: coverage, per-date mean/std,
   ticker x date heatmap, sample ticker lines. Catches structural bugs
   before anything else does.
3. `report.signal_report()`, numeric check: IC mean/std/IR, Newey-West
   corrected t-stat, hit rate, stability, naive long-short paper Sharpe.
4. `IC/metrics.py`'s `ic_decay()`, how far out predictive power actually
   holds, informs rebalance frequency.
5. `report.compare_reports()` against other candidates, side by side.
6. `fdr.py`'s `fdr_report()` across everything tested this round: BH, BY,
   and q-value reported together, no single verdict picked automatically.
7. Survivors move to the signal combiner.