#  Heavier, occasionally-invoked statistical refinements, built from core.py

import numpy as np
import polars as pl


def newey_west_ic_tstat(ic_values, max_lag=None):
    """
    Newey-West (HAC) corrected t-stat for the mean of a daily IC series.

    The naive t-stat in summarize_ic() assumes independent daily
    observations, which overlapping-window features violate (mom20's
    20-day window means consecutive days' IC values are correlated).
    This corrects for that autocorrelation instead of just flagging it
    as a caveat.

    max_lag defaults to the standard Newey-West (1994) rule of thumb,
    4 * (n/100)^(2/9), if not given explicitly.
    """
    x = np.asarray(ic_values, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return {'mean': None, 'se': None, 't_stat': None, 'max_lag': None}

    if max_lag is None:
        max_lag = max(1, int(4 * (n / 100) ** (2 / 9)))

    mean = x.mean()
    demeaned = x - mean
    gamma0 = np.sum(demeaned ** 2) / n

    var = gamma0
    for k in range(1, max_lag + 1):
        w = 1 - k / (max_lag + 1)
        gamma_k = np.sum(demeaned[k:] * demeaned[:-k]) / n
        var += 2 * w * gamma_k

    se = np.sqrt(var / n) if var > 0 else None
    t_stat = mean / se if se else None

    return {'mean': mean, 'se': se, 't_stat': t_stat, 'max_lag': max_lag}


def ic_decay(lf, signal_col, horizons=(1, 5, 10, 20), group='date'):
    """
    Rank IC of a signal against forward returns at several horizons.
    Shows how long the signal's predictive power actually persists,
    directly informs what rebalance frequency makes sense: a signal
    that only shows IC at horizon 1 wants daily rebalancing (expensive),
    one that holds up to horizon 20 can rebalance far less often.
    """
    from src.features.labels import add_forward_return
    from src.evaluation.ic.core import compute_ic, summarize_ic

    rows = []
    for h in horizons:
        lf_h = add_forward_return(lf, horizon=h)
        ic_lf = compute_ic(lf_h, signal_col, f'fwd_ret_{h}', group)
        stats = summarize_ic(ic_lf)
        stats['horizon'] = h
        rows.append(stats)

    return pl.DataFrame(rows).select(['horizon', 'mean', 'std', 'ir', 'hit_rate', 'n_days'])