'''
Plumbing for stage 03 (features). The script 03_build_features.py only calls these functions.
The feature formulas are in the other files of src/features/.

THE CHAIN
---------

    ohlcv.parquet  (from 02: one row per ticker and date, adjusted and raw prices, volume)
          |
          |  check_ohlcv         raises on duplicates and bad prices, reports suspicious returns
          v
    [1] logret                   daily log-return of the adjusted close
    [2] beta{w}                  needs logret and the market return
    [3] mom{w}                   past return over w days
    [4] vol{w}                   std of logret over w days
    [5] dollar_volume, adv, log_adv{w}
    [6] fwdret                   LABEL (legacy, for the research scripts). Uses FUTURE prices.
          |
          |  check_features      no inf, warm-up nulls where expected
          v
    features.parquet

THE RULE OF THE FILE: every FEATURE at date t uses data up to the close of t. The tests prove it:
tests/test_features.py cuts the prices at a date and checks that no feature of the earlier dates changes.
'''

import numpy as np
import polars as pl

from src.features.adv import add_adv, add_log_adv
from src.features.beta import add_beta
from src.features.liquidity import add_dollar_volume
from src.features.momentum import add_momentum
from src.features.returns import add_log_returns, add_fwd_returns
from src.features.volatility import add_volatility

# Columns that use FUTURE prices. They are labels, not features: never build a signal from them.
LABEL_COLS = ['fwdret']

REQUIRED_COLS = ['ticker', 'date', 'open', 'close', 'close_raw', 'open_adj', 'volume']


def build_features(lf: pl.LazyFrame, windows: dict, min_tickers: int = 100) -> pl.LazyFrame:
    '''
    Add all the features to the price table. Lazy: nothing is computed until .collect().

    windows: {'beta': [...], 'momentum': [...], 'volatility': [...], 'adv': [...]}  (trading days)
    min_tickers: a date needs this many stocks to have a market return (and so a beta).
    '''
    lf = lf.sort(['ticker', 'date'])
    lf = add_log_returns(lf)
    lf = add_beta(lf, window=windows['beta'], min_tickers=min_tickers)
    lf = add_momentum(lf, window=windows['momentum'])
    lf = add_volatility(lf, window=windows['volatility'])
    lf = add_dollar_volume(lf)
    lf = add_log_adv(lf, window=windows['adv'])
    lf = add_fwd_returns(lf)                                   # LABEL, see LABEL_COLS
    return lf.sort(['ticker', 'date'])


# =============================================================================
# CHECKS
# =============================================================================

def check_ohlcv(lf: pl.LazyFrame, extreme_logret: float = 0.4) -> dict:
    '''
    Check the price table before the features are built.

    Raises ValueError for: a missing column, two rows for the same (ticker, date), a non-positive
    price or volume. These make every later number wrong.
    Reports (does not raise): the largest daily moves of the adjusted close. A move beyond +-40%
    (log) is rare for an S&P 500 stock. A split that was not adjusted shows as log(2) = 0.69,
    log(3) = 1.10, log(1/2) = -0.69, and so on.
    '''
    names = lf.collect_schema().names()
    missing = [c for c in REQUIRED_COLS if c not in names]
    if missing:
        raise ValueError(f'ohlcv.parquet lacks the columns {missing}. Rerun 02_etl_prices.py.')

    stats = lf.select(
        pl.len().alias('n_rows'),
        pl.struct(['ticker', 'date']).n_unique().alias('n_unique'),
        pl.col('ticker').n_unique().alias('n_tickers'),
        pl.col('date').n_unique().alias('n_dates'),
        pl.col('date').min().alias('first_date'),
        pl.col('date').max().alias('last_date'),
        ((pl.col('close') <= 0) | (pl.col('close_raw') <= 0) | (pl.col('open_adj') <= 0)).sum().alias('bad_price'),
        (pl.col('volume') <= 0).sum().alias('bad_volume'),
    ).collect().row(0, named=True)

    if stats['n_unique'] != stats['n_rows']:
        raise ValueError(f'{stats["n_rows"] - stats["n_unique"]} duplicate (ticker, date) rows in ohlcv.parquet. '
                         f'Rerun 02_etl_prices.py.')
    if stats['bad_price'] or stats['bad_volume']:
        raise ValueError(f'{stats["bad_price"]} non-positive prices and {stats["bad_volume"]} non-positive volumes.')

    moves = (
        lf.sort(['ticker', 'date'])
        .select(['ticker', 'date', pl.col('close').log().diff().over('ticker').alias('logret')])
        .filter(pl.col('logret').abs() > extreme_logret)
        .sort(pl.col('logret').abs(), descending=True)
        .collect()
    )
    stats['n_extreme_moves'] = moves.height
    stats['largest_moves'] = [
        {'ticker': r['ticker'], 'date': str(r['date']), 'logret': round(r['logret'], 3)}
        for r in moves.head(5).iter_rows(named=True)
    ]
    return stats


def check_features(df: pl.DataFrame, windows: dict) -> dict:
    '''
    Check the finished feature table.

    Raises ValueError if a feature holds NaN or +-inf (a single one would spoil the cross-sectional
    mean of its date in 04). Returns the share of nulls per feature; the nulls at the start of each
    ticker (warm-up of the window) are normal.
    '''
    feature_cols = [c for c in df.columns if c not in REQUIRED_COLS + LABEL_COLS + ['mkt_logret']]
    float_cols = [c for c in feature_cols if df.schema[c] in (pl.Float32, pl.Float64)]
    bad = {c: int((~df[c].drop_nulls().is_finite()).sum()) for c in float_cols}
    bad = {c: n for c, n in bad.items() if n}
    if bad:
        raise ValueError(f'features hold NaN or inf: {bad}')
    return {'null_pct': {c: round(df[c].null_count() / df.height * 100, 2) for c in feature_cols}}