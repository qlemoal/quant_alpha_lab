# Basic building blocks of IC statistics

# The IC observations have finite variance, so by CLT the average becomes approximately Gaussian ===> we can use t-stat on the IC.
# However there is serial dependence so we might turn to: Newey-West t-stat, block bootstrap, White Reality Check, SPA tests


import numpy as np
import polars as pl
from polars import col as c
from src.utils.my_plotting import *
from src.signals.combine import make_signal



def compute_ic(lf:pl.LazyFrame, signal_col, forward_return_col='fwdret', method='spearman') -> pl.LazyFrame:
    '''
    Cross-sectional information coefficient, per date.

    For each date, correlation btw today's signal against tomorrow's returns across all tickers. 
        Returns one row per date. 
        'spearman' is the default (Rank IC) since it only depends on ranking, not raw magnitude, 
        less sensitive to a handful of extreme return days than Pearson would be.
    '''
    return (
        lf
        .drop_nulls([signal_col, forward_return_col])
        .group_by('date')
        .agg(
                pl.corr(signal_col, forward_return_col, method=method).alias('ic')
            )
        .sort('date')
    )


def summarize_ic(ic_lf:pl.LazyFrame|pl.DataFrame, ic_col='ic') -> dict:
    '''
    Returns dict of IC stats from a per-date IC series:
        mean       -- average predictive power
        std        -- how much that power varies day to day
        ir         -- information ratio, mean / std. Risk-adjusted signal quality, the main number to compare across different signals/methods.
        t_stat     -- mean / (std / sqrt(n)). Rough significance check, treats daily ICs as independent, 
                        which they likely aren't given overlapping-window features, so read as indicative, not exact.
        hit_rate   -- fraction of days with IC > 0
        n_days     -- number of dates with a non-null IC
    '''
    ic = ic_lf.collect() if isinstance(ic_lf, pl.LazyFrame) else ic_lf
    vals = ic[ic_col].drop_nulls()

    n = vals.len()
    mean = vals.mean()
    std = vals.std()
    ir = mean / std if std else None
    t_stat = mean / (std / np.sqrt(n)) if std and n > 0 else None
    hit_rate = (vals > 0).mean()

    return {
        'mean': mean,
        'std': std,
        'ir': ir,
        't_stat': t_stat,
        'hit_rate': hit_rate,
        'n_days': n,
    }


def compare_methods_ic(lf, feature_col, methods, **kwargs):
    '''
    Build a signal from 'feature_col' under each method in 'methods' (via src.signals.transform.make_signal), compute its IC time series,
        and return a summary table, one row per method, for quick comparison.

    methods: list of method names, e.g. ['zscore_tanh', 'zscore_clip', 'rank', 'decile']
    '''

    rows = []
    for method in methods:
        signal_lf = make_signal(lf, feature_col, method=method, **kwargs)
        signal_col = f'{feature_col}_{method}'

        ic_lf = compute_ic(signal_lf, signal_col, forward_return_col='fwdret')
        stats = summarize_ic(ic_lf)
        stats['method'] = method
        rows.append(stats)

    return pl.DataFrame(rows).select(
        ['method', 'mean', 'std', 'ir', 't_stat', 'hit_rate', 'n_days']
    )

