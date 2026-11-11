'''
Quick performance checks of a signal: a naive decile long-short "paper" portfolio, its Sharpe,
its drawdown, and the signal turnover.

THESE ARE DIAGNOSTICS, NOT A BACKTEST. No costs, no position sizing, no risk model.
They answer: "does the direction of this bet make money, and is it stable?"

THE OVERLAP TRAP (why most functions here take a `horizon`)
-----------------------------------------------------------
The label is an h-day forward return (fwdret20 means h = 20). The paper portfolio earns, on date t,
the h-day return of the trade opened on t. So on consecutive dates the returns OVERLAP:
the return of date t and the return of date t+1 share h - 1 days.

    day:        1  2  3  4  5  6 ...
    return t=1: [--------- 20 days ---------]
    return t=2:    [--------- 20 days ---------]       <- 19 days shared with t=1

Two classic mistakes follow, both made by the previous version of this file:

  1. SHARPE annualised with sqrt(252).
     Each value is an h-day return, so there are only 252 / h independent periods per year.
     Correct: Sharpe = mean / std * sqrt(252 / h).
     The old formula was too large by a factor sqrt(h): about 4.5 for h = 20.

  2. DRAWDOWN from a cumulative SUM of the overlapping returns.
     Each day's return is counted h times, so the curve is h times too steep.
     Correct: build the equity curve from NON-overlapping h-day returns (every h-th date) and compound them.

If a function takes `horizon`, pass the h of the label. `horizon_from_col('fwdret20')` gives 20.
With horizon = 1 (default) the series is treated as plain daily returns and nothing is rescaled.
'''

import re

import numpy as np
import polars as pl
import pandas as pd
from polars import col as c

from src.signals.combine import make_signal
from src.utils.helpers import as_list
from src.utils.my_plotting import new_fig, finish_fig


# =============================================================================
# 1. HORIZON HELPERS
# =============================================================================

def horizon_from_col(fwd_ret_col: str) -> int:
    '''
    Horizon (in days) written in a forward-return column name.
        'fwdret20' -> 20,   'fwdret5' -> 5,   'fwdret' -> 1 (legacy name: one-day forward return).
    '''
    match = re.search(r'(\d+)$', fwd_ret_col)
    return int(match.group(1)) if match else 1


def _check_horizon(horizon) -> int:
    if horizon is None:
        return 1
    h = int(horizon)
    if h < 1:
        raise ValueError(f'horizon must be >= 1, got {horizon}')
    return h


def _clean(returns) -> np.ndarray:
    '''Return a 1-D float array without NaN.'''
    r = np.asarray(returns, dtype=float).reshape(-1)
    return r[~np.isnan(r)]


# =============================================================================
# 2. SHARPE AND DRAWDOWN
# =============================================================================

def sharpe_ratio(returns, periods_per_year: int = 252, horizon: int = 1):
    '''
    Annualised Sharpe ratio of a return series. No risk-free rate.

        Sharpe = mean(r) / std(r) * sqrt(periods_per_year / horizon)

    returns:  one value per date. Each value is a return over `horizon` days.
              (overlapping windows are fine: the mean and the std of an h-day return are still valid)
    horizon:  h of the label. 1 = daily returns (no rescaling).
              For fwdret20 spreads pass horizon=20, otherwise the result is about 4.5 times too large.

    Uses the sample std (ddof = 1). Returns None if fewer than 2 values or zero variance.

    Note: only the POINT ESTIMATE is corrected here. Overlap still makes the estimate noisier than
    the number of dates suggests; do not read a standard error off it.
    '''
    h = _check_horizon(horizon)
    r = _clean(returns)
    if len(r) < 2:
        return None
    if r.max() == r.min():          # constant series: zero variance (checked exactly, a float std can be 1e-17)
        return None
    std = r.std(ddof=1)
    return float(r.mean() / std * np.sqrt(periods_per_year / h))


def max_drawdown(returns, horizon: int = 1, offset: "int | None" = None):
    '''
    Maximum peak-to-trough drawdown of the compounded equity curve, as a NEGATIVE fraction.
    Example: -0.25 means the curve fell 25% from a previous peak.

    How it works:
      1. Keep NON-overlapping returns: every `horizon`-th value (see the overlap trap in the file header).
      2. Equity curve = 1, then cumulative product of (1 + r).
      3. Drawdown at each point = equity / running peak - 1. The result is the minimum.

    horizon: h of the label. 1 = daily returns (all values used).
    offset:  which of the h possible sub-series to use (0 .. h-1).
             None (default) = compute the drawdown of ALL h sub-series and return their average,
             so the answer does not depend on the arbitrary start date.
             Caution: a curve sampled every h days hides moves inside the h days, so the drawdown is
             slightly UNDERSTATED (in a test: -0.21 instead of the true daily -0.23).

    The series must be in date order, with one value per trading day (NaN values are dropped).
    Returns None if there is no data.
    '''
    h = _check_horizon(horizon)
    r = _clean(returns)
    if len(r) == 0:
        return None

    def one_path(sub: np.ndarray) -> float:
        equity = np.concatenate([[1.0], np.cumprod(1.0 + sub)])
        peak = np.maximum.accumulate(equity)
        return float((equity / peak - 1.0).min())

    if h == 1:
        return one_path(r)
    if offset is not None:
        if not 0 <= offset < h:
            raise ValueError(f'offset must be in [0, {h - 1}], got {offset}')
        sub = r[offset::h]
        return one_path(sub) if len(sub) else None

    paths = [one_path(r[o::h]) for o in range(h) if len(r[o::h])]
    return float(np.mean(paths)) if paths else None


# =============================================================================
# 3. SIGNAL TURNOVER
# =============================================================================

def compute_stability(lf: pl.LazyFrame, signal_col: str) -> pl.LazyFrame:
    '''
    Turnover proxy of ONE signal.

    For each ticker: absolute change of the signal value from the previous day.
    For each date: the mean of that change across tickers. Columns: date, turnover.

    A cheap indicator of how much trading the signal implies: higher turnover means more transaction costs.
    '''
    return (
        lf.sort(['ticker', 'date'])
        .with_columns(
            (c(signal_col) - c(signal_col).shift(1).over('ticker')).abs().alias('abs_change')
        )
        .group_by('date')
        .agg(c('abs_change').mean().alias('turnover'))
        .sort('date')
    )


def get_stability_DF(lf: pl.LazyFrame, signal_col: "str|list", _plot=False, _print=False) -> pd.DataFrame:
    '''
    Same as compute_stability for one OR several signals; returns a pandas DataFrame
    (index = date, one '{signal}_turnover' column per signal).

    _plot:  plot the turnover of each signal.
    _print: print mean, std and count of each turnover column.
    '''
    signal_cols = as_list(signal_col)
    exprs = [(c(sc) - c(sc).shift(1).over('ticker')).abs().alias(f'{sc}_abs_change') for sc in signal_cols]
    df = (
        lf.sort(['ticker', 'date'])
        .with_columns(exprs)
        .group_by('date')
        .agg([c(f'{sc}_abs_change').mean().alias(f'{sc}_turnover') for sc in signal_cols])
        .sort('date')
    ).collect().to_pandas().set_index('date')

    if _print:
        print('----> Mean: <----')
        print(df.mean())
        print('----> Std: <----')
        print(df.std())
        print('----> n obs: <----')
        print(df.count())
    if _plot:
        _, ax = new_fig(fs=(12, 6))
        df.plot(ax=ax)
        finish_fig(ax=ax, yl='Turnover', title='Average day-over-day absolute change in signal')
    return df


# =============================================================================
# 4. DECILE LONG-SHORT PAPER PORTFOLIO
# =============================================================================

def _bucket_column(lf: pl.LazyFrame, signal_col: str, n_buckets: int) -> "tuple[pl.LazyFrame, str]":
    '''
    Make sure there is a column holding +1 (top bucket), -1 (bottom bucket) or 0 (middle).

    - If signal_col is ALREADY such a column (its name ends with '_decile', made by make_signal),
      use it as it is. Bucketing it again would be wrong: it only holds -1 / 0 / +1.
    - Otherwise create '{signal_col}_decile' with make_signal.
    '''
    if signal_col.endswith('_decile'):
        return lf, signal_col
    lf = make_signal(lf, signal_col, method='decile', n_buckets=n_buckets)
    return lf, f'{signal_col}_decile'


def decile_longshort_returns(lf, signal_col, fwd_ret_col='fwdret', n_buckets=10) -> pl.LazyFrame:
    '''
    Naive long-short paper portfolio, one value per date:
        long  = equal-weight the top bucket,
        short = equal-weight the bottom bucket,
        spread_ret = long_ret + short_ret   (+1 / -1 notional, so gross exposure = 2).

    Columns: date, long_ret, short_ret, spread_ret, n_long, n_short.
    n_long / n_short = number of stocks in each leg that day (check them: a leg with 3 stocks is noise).

    WHICH RETURN IS AVERAGED: fwd_ret_col is a LOG return. Log returns add over TIME but not across ASSETS
    (sum of w_i * log(1 + r_i) is not log(1 + sum of w_i * r_i)). So each stock's log return is converted to a
    simple return, exp(x) - 1, BEFORE averaging over the basket. Only then is "equal-weight average" real P&L.

    EACH VALUE IS AN h-DAY RETURN (h = horizon_from_col(fwd_ret_col)). The values overlap.
    Pass horizon=h to sharpe_ratio() and max_drawdown() (see the file header).
    '''
    lf, bucket = _bucket_column(lf, signal_col, n_buckets)
    lf = lf.with_columns(c(fwd_ret_col).exp().sub(1).alias('_simple_ret'))

    long_leg = (
        lf.filter(c(bucket) == 1)
        .group_by('date')
        .agg(c('_simple_ret').mean().alias('long_ret'), c('_simple_ret').count().alias('n_long'))
    )
    short_leg = (
        lf.filter(c(bucket) == -1)
        .group_by('date')
        .agg((-c('_simple_ret')).mean().alias('short_ret'), c('_simple_ret').count().alias('n_short'))
    )

    return (
        long_leg.join(short_leg, on='date')          # inner join: a date needs both legs
        .with_columns((c('long_ret') + c('short_ret')).alias('spread_ret'))
        .select(['date', 'long_ret', 'short_ret', 'spread_ret', 'n_long', 'n_short'])
        .sort('date')
    )


def get_decile_longshort_returns_DF(lf, signal_col, fwd_ret_col='fwdret', n_buckets=10,
                                    _plot=False, _print=False) -> pd.DataFrame:
    '''
    decile_longshort_returns() collected into a pandas DataFrame (index = date).

    _print: print mean and std of each column.
    _plot:  plot the average return of each leg, and the equity curve of the spread built from
            NON-overlapping h-day returns (every h-th date, compounded), h = horizon_from_col(fwd_ret_col).
    '''
    df = decile_longshort_returns(lf, signal_col, fwd_ret_col, n_buckets).collect().to_pandas().set_index('date')

    if _print:
        print('----> Mean: <----')
        print(df.mean())
        print('----> Std: <----')
        print(df.std())
    if _plot:
        h = horizon_from_col(fwd_ret_col)
        _, ax = new_fig(fs=(12, 6))
        df[['long_ret', 'short_ret']].plot(ax=ax, alpha=0.5)
        equity = (1.0 + df['spread_ret'].iloc[::h]).cumprod()
        equity.plot(ax=ax, label=f'Equity of {signal_col} ({h}-day non-overlapping)')
        finish_fig(ax=ax, yl='simple return / equity',
                   title=f'Top-bottom bucket investing - {n_buckets} buckets')
    return df