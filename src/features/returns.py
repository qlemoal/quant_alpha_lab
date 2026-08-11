import polars as pl
from polars import col as c





def add_log_returns(lf:pl.LazyFrame) -> pl.LazyFrame:
    return lf.sort(['ticker', 'date']).with_columns(
                logret = c('close').log().diff().over('ticker')
            )


def add_fwd_returns(lf:pl.LazyFrame) -> pl.LazyFrame:
    return lf.sort(['ticker', 'date']).with_columns(
                fwdret = c('logret').shift(-1).over('ticker')
            )

def add_fwdret_horizon(lf:pl.LazyFrame, horizon=1) -> pl.LazyFrame:
    '''
    Adds forward log-returns, horizon trading days ahead, per ticker.
    '''
    return lf.sort(['ticker', 'date']).with_columns(
        c('logret').shift(-horizon).over('ticker').alias(f'fwd_ret_{horizon}')
    )