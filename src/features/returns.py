import polars as pl
from polars import col as c





def add_log_returns(lf:pl.LazyFrame) -> pl.LazyFrame:
    '''
    Computes log-returns 'logret' from 'close' prices column.
    '''
    return lf.sort(['ticker', 'date']).with_columns(
                logret = c('close').log().diff().over('ticker')
            )


def add_fwd_returns(lf:pl.LazyFrame) -> pl.LazyFrame:
    '''
    Computes 'fwdret' column with tomorrow's log-returns. Used to compute the performance or IC, this is not a feature obviously.
    '''
    return lf.sort(['ticker', 'date']).with_columns(
                fwdret = c('logret').shift(-1).over('ticker')
            )

def add_fwdret_horizon(lf:pl.LazyFrame, horizon=1) -> pl.LazyFrame:
    '''
    Adds forward log-returns, horizon trading days ahead, per ticker.
    TODO: take the sum of log-returns until horizon days ahead ?
    '''
    return lf.sort(['ticker', 'date']).with_columns(
        c('logret').shift(-horizon).over('ticker').alias(f'fwd_ret_{horizon}')
    )