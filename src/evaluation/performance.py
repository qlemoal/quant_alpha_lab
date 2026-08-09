import numpy as np
import polars as pl
from polars import col as c

def compute_turnover(lf, signal_col):
    """
    Average day-over-day absolute change in signal value, per ticker,
    aggregated across the cross-section for each date. A cheap proxy
    for how much trading a signal implies once turned into portfolio
    weights, higher turnover means more transaction cost exposure once
    src/portfolio/costs.py exists.
    """
    return (
        lf.sort(['ticker', 'date'])
        .with_columns(
            (c(signal_col) - c(signal_col).shift(1).over('ticker'))
            .abs()
            .alias('abs_change')
        )
        .group_by('date')
        .agg(c('abs_change').mean().alias('turnover'))
        .sort('date')
    )
 
 
def quantile_spread_returns(lf, signal_col, fwd_ret_col='fwd_ret', n_buckets=10, group='date'):
    """
    Naive long-short paper portfolio: equal-weight long the top bucket,
    equal-weight short the bottom bucket, each date. No costs, no real
    position sizing, this is a quick 'does the direction of this bet
    make money' check, not a real backtest.
    """
    bucketed = lf.with_columns(
        c(signal_col)
        .qcut(n_buckets, labels=[str(i) for i in range(n_buckets)], allow_duplicates=True)
        .over(group)
        .cast(pl.Utf8)
        .cast(pl.Int32)
        .alias('bucket')
    )
 
    long_ret = (
        bucketed.filter(c('bucket') == n_buckets - 1)
        .group_by(group)
        .agg(c(fwd_ret_col).mean().alias('long_ret'))
    )
    short_ret = (
        bucketed.filter(c('bucket') == 0)
        .group_by(group)
        .agg(c(fwd_ret_col).mean().alias('short_ret'))
    )
 
    return (
        long_ret.join(short_ret, on=group)
        .with_columns((c('long_ret') - c('short_ret')).alias('spread_ret'))
        .sort(group)
    )
 
 
def sharpe_ratio(returns, periods_per_year=252):
    """Annualized Sharpe of a return series, no risk-free rate adjustment."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) == 0 or r.std() == 0:
        return None
    return r.mean() / r.std() * np.sqrt(periods_per_year)