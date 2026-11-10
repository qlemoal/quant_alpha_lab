'''
Statistical refinements on the daily IC series (built from core.py).

THE PROBLEM THIS FILE SOLVES
----------------------------
The label is a 20-day forward return. The label of day t and the label of day t+1
share 19 of their 20 days. So the daily IC values are strongly autocorrelated,
up to lag h - 1 = 19, even when the signal is pure noise.

The naive t-stat assumes independent days. It is far too large.
Newey-West (HAC) fixes this, but ONLY if its max lag covers the overlap.
The old default lag was int(4 * (n/100) ** (2/9)), about 5 for n = 6500.
That is much shorter than 19, so the t-stat stayed too large.

THE FIX
-------
    max_lag = max(rule_of_thumb, 2 * (horizon - 1))

The horizon is given by the caller (`horizon=`). If the caller does not give it,
DEFAULT_HORIZON is used (the project label is fwdret20).
'''

import numpy as np
import polars as pl
from scipy.stats import norm

from src.evaluation.signals.ic.core import compute_ic, summarize_ic

# Label horizon of the project (fwdret20). Used when the caller does not pass `horizon`.
DEFAULT_HORIZON = 20


# =============================================================================
# 1. Lag choice
# =============================================================================

def default_nw_lag(n: int, horizon: "int | None" = None) -> int:
    '''
    Choose the Newey-West max lag.

    Two lower bounds, we keep the larger one:
      - Newey-West (1994) rule of thumb:  4 * (n / 100) ** (2/9)
        This covers generic autocorrelation.
      - 2 * (horizon - 1):  windows of h days overlap by h - 1 days, so the IC
        series is autocorrelated up to lag h - 1 by construction.
        Why twice? With the Bartlett kernel, a lag of exactly h - 1 still gives
        about 12% false positives at the 5% level (simulation, n = 6500, h = 20,
        pure noise). A lag of 2 * (h - 1) brings it to about 7-8%.
        The non-overlapping check below gives the exact 5%.

    The lag is capped at n // 4 so that very short series stay usable.
    '''
    h = DEFAULT_HORIZON if horizon is None else int(horizon)
    rule_of_thumb = int(4 * (n / 100) ** (2 / 9))
    lag = max(1, rule_of_thumb, 2 * (h - 1))
    return int(min(lag, max(1, n // 4)))


# =============================================================================
# 2. Newey-West t-stat
# =============================================================================

def newey_west_ic_tstat(
    ic_values: "pl.LazyFrame|pl.DataFrame|list|np.ndarray",
    max_lag: "int | None" = None,
    horizon: "int | None" = None,
) -> dict:
    '''
    Newey-West (HAC) t-stat for the MEAN of a daily IC series.

    Idea in one line: the variance of the mean is larger when neighbours are
    positively correlated, so we add the weighted autocovariances to the variance.

    Steps:
      1. Remove NaN. n = number of values.
      2. mean = average IC. demeaned = IC - mean.
      3. gamma_0 = variance of the series (divide by n).
      4. For each lag k = 1..L:
           gamma_k = sum(demeaned[k:] * demeaned[:-k]) / n       (autocovariance)
           w_k     = 1 - k / (L + 1)                              (Bartlett weight)
      5. long_run_var = gamma_0 + 2 * sum_k w_k * gamma_k
         Bartlett weights keep long_run_var >= 0.
      6. se = sqrt(long_run_var / n).  t_stat = mean / se.

    Parameters:
      ic_values: only the IC numbers, NOT the dates (1-D).
      max_lag:   L. If None, default_nw_lag(n, horizon) is used.
      horizon:   label horizon h in days. Sets L >= 2 * (h - 1). Default DEFAULT_HORIZON.

    Returns dict: mean, se, t_stat, max_lag (None values if n < 2 or variance <= 0).
    '''
    x = np.asarray(ic_values, dtype=float)
    if x.ndim != 1:
        raise ValueError(f'newey_west_ic_tstat expects 1D IC values, got shape {x.shape}')
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return {'mean': None, 'se': None, 't_stat': None, 'max_lag': None}

    if max_lag is None:
        max_lag = default_nw_lag(n, horizon)
    max_lag = int(min(max_lag, n - 1))   # a lag cannot exceed n - 1

    mean = x.mean()
    demeaned = x - mean
    long_run_var = np.sum(demeaned ** 2) / n      # gamma_0

    for k in range(1, max_lag + 1):
        w = 1 - k / (max_lag + 1)                                    # Bartlett weight
        gamma_k = np.sum(demeaned[k:] * demeaned[:-k]) / n           # autocovariance at lag k
        long_run_var += 2 * w * gamma_k

    se = np.sqrt(long_run_var / n) if long_run_var > 0 else None
    t_stat = mean / se if se else None
    return {'mean': mean, 'se': se, 't_stat': t_stat, 'max_lag': max_lag}


# =============================================================================
# 3. Non-overlapping cross-check
# =============================================================================

def ic_tstat_non_overlapping(
    ic_values: "list|np.ndarray",
    horizon: "int | None" = None,
    offset: int = 0,
) -> dict:
    '''
    Simple robustness check, no HAC needed.

    Keep one IC every h days (days offset, offset + h, offset + 2h, ...).
    These labels do not overlap, so the plain t-stat is valid.
    The price is power: we keep only n / h points.

    Use it next to the Newey-West t-stat. The two should be close.
    If NW says "significant" and this says "no", distrust the signal.

    Returns dict: mean, t_stat, n_used.
    '''
    h = DEFAULT_HORIZON if horizon is None else int(horizon)
    x = np.asarray(ic_values, dtype=float)
    x = x[~np.isnan(x)][offset::h]
    n = len(x)
    if n < 3 or x.std(ddof=1) == 0:
        return {'mean': None, 't_stat': None, 'n_used': n}
    mean = x.mean()
    t_stat = mean / (x.std(ddof=1) / np.sqrt(n))
    return {'mean': float(mean), 't_stat': float(t_stat), 'n_used': n}


# =============================================================================
# 4. p-value
# =============================================================================

def ic_pvalue_nw(nw_result: dict) -> "float | None":
    '''
    Two-sided p-value from the Newey-West t-stat, normal approximation
    (the NW t-stat is asymptotically normal).
    Takes the dict returned by newey_west_ic_tstat.
    '''
    t = nw_result.get('t_stat')
    if t is None:
        return None
    return 2 * (1 - norm.cdf(abs(t)))


# =============================================================================
# 5. IC decay
# =============================================================================

def ic_decay(lf, signal_col, horizons=(1, 5, 10, 20)):
    '''
    IC of a signal against forward returns at several horizons.

    It shows how long the signal keeps its predictive power.
    A signal with IC only at h = 1 needs fast (costly) rebalancing.
    A signal that holds up to h = 20 can rebalance less often.

    Uses the pre-computed columns fwdret{h}.
    For each horizon the Newey-West t-stat uses lag >= 2 * (h - 1) (overlap of THAT horizon),
    returned in the column `t_stat_nw`.
    '''
    available_cols = set(lf.collect_schema().names())
    rows = []
    for h in horizons:
        fwd_col = f'fwdret{h}'
        if fwd_col not in available_cols:
            continue
        ic_df = compute_ic(lf, signal_col, fwd_col).collect()   # one collect, reused below
        stats = summarize_ic(ic_df)
        nw = newey_west_ic_tstat(ic_df['ic'].to_numpy(), horizon=h)
        stats['t_stat_nw'] = nw['t_stat']
        stats['horizon'] = h
        rows.append(stats)

    return pl.DataFrame(rows).select(['horizon', 'mean', 'std', 'ir', 'hit_rate', 'n_days', 't_stat_nw'])


if __name__ == '__main__':
    from src.signals.combine import make_signal
    with pl.Config(tbl_cols=-1):
        lf = pl.scan_parquet('data/processed/features.parquet')
        lf = make_signal(lf, ['mom5', 'mom10', 'mom20', 'mom60', 'mom120', 'mom252'], method='decile')
        print(ic_decay(lf, 'mom60'))