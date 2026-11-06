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
    open_logret = c('open_adj').log().diff().over('ticker')
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
    Forward log-returns using adjusted open prices (total return series).
    We use open_adj(t+1+horizon) relative to open_adj(t+1) - skip execution day (t+1).
    So fwdret{h} = log(open_adj[t+1+h]) - log(open_adj[t+1])
    '''
    log_open = c('open_adj').log().over('ticker')
    fwd_expr = log_open.shift(-(horizon + 1)) - log_open.shift(-1)
    return lf.sort(['ticker', 'date']).with_columns(**{f'fwdret{horizon}': fwd_expr})
    # Previous version using Close prices
    # '''
    # Adds forward log-returns, horizon trading days ahead, per ticker.
    #     Using log-returns, it's simply a sum of the 'horizon' next days. 
    # '''
    # return lf.sort(['ticker', 'date']).with_columns(
    #     c('logret').rolling_sum(horizon).over('ticker').shift(-horizon).alias(f'fwd_ret_{horizon}')
    # )