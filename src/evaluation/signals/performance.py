#  Performance metrics of a signal

#  Latest modification corrected the decile_longshort_returns which used log-returns, 
#    which are not cross-sectionally additive => transform back to returns before computing.


import numpy as np
import polars as pl
import pandas as pd
from polars import col as c

from src.signals.combine import make_signal
from src.utils.helpers import as_list
from src.utils.my_plotting import *




def max_drawdown(returns):
    '''
    Max peak-to-trough drawdown of the cumulative long-short paper return
        series. Same pre-cost caveat as long_short_sharpe, directional
        sanity check only, not a real risk figure.
    '''
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return None
    cum = np.cumsum(r)
    running_max = np.maximum.accumulate(cum)
    return float((cum - running_max).min())


def compute_stability(lf:pl.LazyFrame, signal_col:str) -> pl.LazyFrame:
    '''
    ~ Turnover
    Average day-over-day absolute change in signal value, per ticker, aggregated across the cross-section for each date. 
        A cheap proxy for how much trading a signal implies once turned into portfolio weights, 
        higher turnover means more transaction cost exposure once src/portfolio/costs.py exists.
    signal_col: str

    returns pl.LazyFrame of 'turnover'
    '''

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


def get_stability_DF(lf:pl.LazyFrame, signal_col:str|list, _plot=False, _print=False) -> pd.DataFrame :
    '''
    ~ Turnover
    Average day-over-day absolute change in signal value, per ticker, aggregated across the cross-section for each date. 
        A cheap proxy for how much trading a signal implies once turned into portfolio weights, 
        higher turnover means more transaction cost exposure once src/portfolio/costs.py exists.
    signal_col: str or list if multiple signals must be tested.

    if _plot: plots the signal stability for each signal.
    if _print: prints the mean, std and number of observation for stability.
    returns pd.DataFrame of 'turnover' columns for each signal.
    '''
    signal_col = as_list(signal_col)
    exprs = [ (c(sc) - c(sc).shift(1).over('ticker')).abs().alias(f'{sc}_abs_change') for sc in signal_col ]
    DF_stability = (
        lf.sort(['ticker', 'date'])
        .with_columns(exprs)
        .group_by('date')
        .agg([ c(f'{sc}_abs_change').mean().alias(f'{sc}_turnover') for sc in signal_col ])
        .sort('date')
    ).collect().to_pandas().set_index('date')

    if _print:
        print('----> Mean: <----')
        print(DF_stability.mean())
        print('----> Std: <----')
        print(DF_stability.std())
        print('----> n obs: <----')
        print(DF_stability.count())
    if _plot: 
        _, ax = new_fig(fs=(12, 6))
        DF_stability.plot(ax=ax)
        finish_fig(ax=ax, yl='Turnover', title='Average day-over-day absolute change in signal')

    return DF_stability


 
def decile_longshort_returns(lf, signal_col, fwd_ret_col='fwdret', n_buckets=10) -> pl.LazyFrame:
    '''
    Naive long-short paper portfolio: equal-weight long the top bucket, equal-weight short the bottom bucket, each date. 
    No costs, no real position sizing, this is a quick 'does the direction of this bet make money' check, not a real backtest.

    fwd_ret_col is a LOG return throughout this project (see src/features/returns.py). Log returns are additive ACROSS
    TIME (that's exactly why rolling_sum() works for multi-day horizons), they are NOT additive ACROSS ASSETS at a point
    in time: a basket's simple return is the mean of its constituents' simple returns, the same statement is false for
    log returns, Sigma w_i*log(1+r_i) != log(1 + Sigma w_i*r_i) in general. Averaging fwd_ret_col directly within a
    basket (the previous version of this function) silently computed the wrong basket return, convert to simple return
    first (exp(x)-1), aggregate across assets in simple-return space, that's the only space in which "equal-weight
    average across a basket" is actually valid P&L arithmetic.
    '''
    lf = make_signal(lf, signal_col, method='decile', n_buckets=n_buckets)
    lf = lf.with_columns( c(fwd_ret_col).exp().sub(1).alias('_simple_ret') )

    long_ret = (
        lf.filter( c(f'{signal_col}_decile') == 1 )
        .group_by('date')
        .agg( c('_simple_ret').mean().alias('long_ret') )
    )
 
    short_ret = (
        lf.filter( c(f'{signal_col}_decile') == -1 )
        .group_by('date')
        .agg( (-c('_simple_ret')).mean().alias('short_ret') )
    )

    return (
        long_ret.join(short_ret, on='date')
            .with_columns(
                (c('long_ret') + c('short_ret')).alias('spread_ret')
                # valid here: long and short legs are each already basket-level SIMPLE returns,
                # a dollar-neutral (+1/-1 notional) combination of simple returns is itself additive
            )
            .sort('date')
        )



 
def get_decile_longshort_returns_DF(lf, signal_col, fwd_ret_col='fwdret', n_buckets=10, _plot=False, _print=False) -> pd.DataFrame:
    '''
    Naive long-short paper portfolio: equal-weight long the top bucket, equal-weight short the bottom bucket, each date. 
    No costs, no real position sizing, this is a quick 'does the direction of this bet make money' check, not a real backtest.

    Same simple-return conversion as decile_longshort_returns(), see its docstring, log returns aren't additive across
    assets, only across time, a basket average must happen in simple-return space.
    '''
    lf = make_signal(lf, signal_col, method='decile', n_buckets=n_buckets)
    lf = lf.with_columns( c(fwd_ret_col).exp().sub(1).alias('_simple_ret') )

    long_ret = (
        lf.filter( c(f'{signal_col}_decile') == 1 )
        .group_by('date')
        .agg( c('_simple_ret').mean().alias('long_ret') )
    )
 
    short_ret = (
        lf.filter( c(f'{signal_col}_decile') == -1 )
        .group_by('date')
        .agg( (-c('_simple_ret')).mean().alias('short_ret') )
    )

    DF_perf = (
        long_ret.join(short_ret, on='date')
            .with_columns(
                (c('long_ret') + c('short_ret')).alias('spread_ret')
            )
            .sort('date')
        ).collect().to_pandas().set_index('date')

    if _print:
        print('----> Mean: <----')
        print(DF_perf.mean())
        print('----> Std: <----')
        print(DF_perf.std())
    if _plot: 
        _, ax = new_fig(fs=(12, 6))
        DF_perf.iloc[:, :2].plot(ax=ax)
        DF_perf.spread_ret.cumsum().plot(label=f'Cumulative return of {signal_col}')
        finish_fig(ax=ax, yl='simple return', title=f'Average simple return of top-bottom decile investing - {n_buckets} buckets')

    return DF_perf


 
def sharpe_ratio(returns, periods_per_year=252):
    '''
    Annualized Sharpe of a return series, no risk-free rate adjustment.
    '''
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) == 0 or r.std() == 0:
        return None
    return r.mean() / r.std() * np.sqrt(periods_per_year)