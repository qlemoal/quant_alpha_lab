'''
Market beta of each stock.

THE IDEA
    market return of a day = average log-return of all the stocks that traded that day (equal weight).
    beta{w}[t] = covariance(stock, market) / variance(market), over the last w days of the stock.

    stock return  ----+
                      |--> rolling covariance --+
    market return ----+                         |--> beta
                      |--> rolling variance ----+
                      (both over the SAME w rows of the stock)

CAUTIONS
    - The market is built from the stocks in the file. With today's S&P 500 members only, the early years
      hold few stocks. A date with fewer than `min_tickers` stocks has no market return (null), so no beta.
    - The market includes the stock itself (1/N of it): negligible with 500 stocks.
    - Uses day t at the latest. The first w rows of each ticker are null.
'''

import polars as pl
from polars import col as c

from src.utils.helpers import as_list


def add_market_return(lf: pl.LazyFrame, min_tickers: int = 100) -> pl.LazyFrame:
    '''
    Add mkt_logret (equal-weight average of logret over the tickers of the date) to every row.
    It is null on dates with fewer than `min_tickers` stocks: such a market is too noisy to be a market.
    '''
    market = (
        lf.group_by('date')
        .agg(c('logret').mean().alias('mkt_logret'), c('logret').count().alias('n_tickers'))
        .with_columns(
            pl.when(c('n_tickers') >= min_tickers).then(c('mkt_logret')).otherwise(None).alias('mkt_logret')
        )
        .select(['date', 'mkt_logret'])
    )
    return lf.join(market, on='date', how='left').sort(['ticker', 'date'])


def add_beta(lf: pl.LazyFrame, window=60, min_tickers: int = 100) -> pl.LazyFrame:
    '''
    Add beta{w} for one window or a list of windows. Adds mkt_logret first if it is missing.
    A window with a zero market variance gives null, not inf.
    '''
    if 'mkt_logret' not in lf.collect_schema().names():
        lf = add_market_return(lf, min_tickers)
    lf = lf.sort(['ticker', 'date'])
    exprs = []
    for w in as_list(window):
        market_var = c('mkt_logret').rolling_var(w)
        beta = pl.when(market_var > 0).then(pl.rolling_cov(c('logret'), c('mkt_logret'), window_size=w) / market_var)
        exprs.append(beta.over('ticker').alias(f'beta{w}'))
    return lf.with_columns(exprs)