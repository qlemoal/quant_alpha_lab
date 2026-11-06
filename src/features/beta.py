import polars as pl
from polars import col as c


def test_func():
    print('The test works')



def add_market_return(lf):
    '''
    Cross-sectional equal-weighted market return per date,
    joined back onto every row of the ticker panel.
    Update: Added a fitler to have at least 100 tickers to compute the market returns.
        Since I am computing the market erturns using today's constituents of the SP500, 
        there is much less around 2000, bringing in a huge bias. That will be fixed later on.
    '''
    mkt = (
        lf
        .group_by('date')
        .agg( 
            c('logret').mean().alias('mkt_logret') ,
            c('logret').count().alias('n_tickers') 
        )
        .with_columns(
            pl.when(c('n_tickers') >= 100)
            .then(c('mkt_logret'))
            .otherwise(None)
            .alias('mkt_logret')
        )
        .sort('date')
    )
    return lf.join(mkt, on='date', how='left').sort('ticker', 'date')


def _market_var(lf, window):
    '''
    Rolling market variance, computed ONCE on a one-row-per-date frame, then joined back.
    Note: if computed that in add_beta directly, we would end up with different 
        values for the market variance since each ticker has different dates (IPO, listing, missing days, ...).
    Note: you need to sort the lf after using .unique() because that messes up the order.
    Note: the choice was made to add a column with the market returns, mean, variance, etc. I could 
        have added the market as an extra ticker, so only rows, without duplicating the values for each ticker,
        but I would have needed to filter it out for all consecutive stats, correlations, etc. 
    '''
    return (
        lf
        .select(['date', 'mkt_logret'])
        .unique(subset = 'date')
        .sort('date')
        .with_columns(
            pl.rolling_cov(c('mkt_logret'), c('mkt_logret'), window_size=window).alias(f'mkt_var{window}')
        )
        .select(['date', f'mkt_var{window}'])
    )


def add_beta(lf, window=60):
    
    if 'mkt_logret' not in lf.collect_schema().names():
        lf = add_market_return(lf)

    def _add_single(lf, w):
        mkt_var = _market_var(lf, w)
        lf = lf.sort(['ticker', 'date']).join(mkt_var, on='date', how='left')
        
        # Polars 1.36+: compute rolling_cov per ticker using map_batches on struct
        # The issue is rolling_cov cannot be used as window expression (.over('ticker'))
        # Solution: collect per ticker, compute, then reassemble
        # But we need to stay lazy - use map_groups on partition_by approach
        
        # First collect the lazy frame to work with DataFrame
        df = lf.collect()
        
        def compute_beta(partition):
            # partition is a DataFrame for one ticker
            # Compute rolling covariance for this ticker
            cov = pl.rolling_cov(partition['logret'], partition['mkt_logret'], window_size=w)
            beta = cov / partition[f'mkt_var{w}']
            return partition.with_columns(pl.Series(beta).alias(f'beta{w}')).drop(f'mkt_var{w}')
        
        # Partition by ticker and apply
        df = df.partition_by('ticker', as_dict=False).map_groups(compute_beta)
        return df.lazy()

    if isinstance(window, list):
        for w in window:
            lf = _add_single(lf, w)
        return lf
    return _add_single(lf, window)