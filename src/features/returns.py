import polars as pl
from polars import col as c





def _add_log_returns_per_ticker(df: pl.DataFrame) -> pl.DataFrame:
    '''Compute log returns per ticker (eager).'''
    return df.sort('date').with_columns(
        logret=c('close').log().diff()
    )


def _add_fwd_returns_per_ticker(df: pl.DataFrame) -> pl.DataFrame:
    '''Compute forward returns per ticker using adjusted open (eager).'''
    return df.sort('date').with_columns(
        fwdret=c('open_adj').log().diff().shift(-2)
    )


def add_log_returns(lf: pl.LazyFrame) -> pl.LazyFrame:
    '''
    Computes log-returns 'logret' from 'close' prices column.
    Uses map_groups to avoid window expressions in LazyFrame query plan.
    '''
    schema = dict(lf.collect_schema())
    schema['logret'] = pl.Float64
    return lf.sort(['ticker', 'date']).group_by('ticker', maintain_order=True).map_groups(_add_log_returns_per_ticker, schema=schema)


def add_fwd_returns(lf: pl.LazyFrame) -> pl.LazyFrame:
    '''
    Forward returns using adjusted open prices (open_adj).
    fwdret = log(open_adj[t+2]) - log(open_adj[t+1]) — models next-day execution at open.
    Uses map_groups to avoid window expressions in LazyFrame query plan.
    '''
    schema = dict(lf.collect_schema())
    schema['fwdret'] = pl.Float64
    return lf.sort(['ticker', 'date']).group_by('ticker', maintain_order=True).map_groups(_add_fwd_returns_per_ticker, schema=schema)
    


def add_fwdret_horizon(lf:pl.LazyFrame, horizon=1) -> pl.LazyFrame:
    '''
    Forward log-returns using adjusted open prices (total return series).
    We use open_adj(t+1+horizon) relative to open_adj(t+1) - skip execution day (t+1).
    So fwdret{h} = log(open_adj[t+1+h]) - log(open_adj[t+1])
    '''
    # The shift MUST happen inside the per-ticker window: `.over('ticker')` is applied to the whole
    # expression. (Applied to the log only, the shift runs over the full frame and the last rows of a ticker
    # read the prices of the NEXT ticker: wrong labels, not null, in the last horizon+1 rows of every ticker.)
    log_open = c('open_adj').log()
    fwd_expr = (log_open.shift(-(horizon + 1)) - log_open.shift(-1)).over('ticker')
    return lf.sort(['ticker', 'date']).with_columns(**{f'fwdret{horizon}': fwd_expr})
    # Previous version using Close prices
    # '''
    # Adds forward log-returns, horizon trading days ahead, per ticker.
    #     Using log-returns, it's simply a sum of the 'horizon' next days. 
    # '''
    # return lf.sort(['ticker', 'date']).with_columns(
    #     c('logret').rolling_sum(horizon).over('ticker').shift(-horizon).alias(f'fwd_ret_{horizon}')
    # )