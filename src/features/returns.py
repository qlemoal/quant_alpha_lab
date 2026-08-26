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
    Instead of using Close(t+1) relative to Close(t), we choose to use Open(t+2) relative to Open(t+1), 
        to represent more realistically what we can effectively trade in practice.
        This forces to use 2 days in the future for the next fwdret.
    '''
    open_logret = c('open').log().diff().over('ticker')
    return lf.sort(['ticker', 'date']).with_columns(
                fwdret = open_logret.shift(-2).over('ticker')
            )
    # Previous version using Close prices
    # '''
    # Computes 'fwdret' column with tomorrow's log-returns. Used to compute the performance or IC, this is not a feature obviously.
    # '''
    # return lf.sort(['ticker', 'date']).with_columns(
    #             fwdret = c('logret').shift(-1).over('ticker')
    #         )
    


def add_fwdret_horizon(lf:pl.LazyFrame, horizon=1) -> pl.LazyFrame:
    '''
    Same thing for fwdret at longer horizons, we use the open prices at t+2 relative to t+1, 
        to make our results more realistic in terms of trading capabilities. 
        Using log-returns, we just need to add the log-returns until horizon days in the future.
    '''
    open_logret = c('open').log().diff().over('ticker')
    return lf.sort(['ticker', 'date']).with_columns(
        open_logret.rolling_sum(horizon).over('ticker').shift(-(horizon+1)).alias(f'fwd_ret_{horizon}')
    )
    # Previous version using Close prices
    # '''
    # Adds forward log-returns, horizon trading days ahead, per ticker.
    #     Using log-returns, it's simply a sum of the 'horizon' next days. 
    # '''
    # return lf.sort(['ticker', 'date']).with_columns(
    #     c('logret').rolling_sum(horizon).over('ticker').shift(-horizon).alias(f'fwd_ret_{horizon}')
    # )