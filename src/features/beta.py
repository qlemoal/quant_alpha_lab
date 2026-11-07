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
    # Compute eagerly to avoid window expression conflicts with existing .over() in query plan
    df = lf.collect()
    mkt = (
        df
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
    return df.join(mkt, on='date', how='left').sort('ticker', 'date').lazy()


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
    # Compute eagerly to avoid window expression issues in lazy joins
    mkt_df = (
        lf
        .select(['date', 'mkt_logret'])
        .unique(subset='date')
        .sort('date')
        .collect()
    )
    return mkt_df.with_columns(
        pl.rolling_cov(c('mkt_logret'), c('mkt_logret'), window_size=window).alias(f'mkt_var{window}')
    ).select(['date', f'mkt_var{window}'])


def add_beta(lf, window=60):
    
    # Check if market return exists by collecting schema (cheap)
    schema = lf.collect_schema()
    if 'mkt_logret' not in schema.names():
        lf = add_market_return(lf)
    
    # Collect once at the start to avoid any lazy query plan issues
    df = lf.collect()

    def _add_single(df, w):
        mkt_var = _market_var(df.lazy(), w)
        
        # Join eagerly
        df = df.sort(['ticker', 'date']).join(mkt_var, on='date', how='left')
        
        # Use partition_by + expression API in with_columns
        partitions = df.partition_by('ticker', as_dict=False)
        result_parts = []
        for part in partitions:
            # Compute rolling covariance using expression API
            part = part.with_columns(
                (pl.rolling_cov(c('logret'), c('mkt_logret'), window_size=w) / c(f'mkt_var{w}')).alias(f'beta{w}')
            ).drop(f'mkt_var{w}')
            result_parts.append(part)
        
        return pl.concat(result_parts, how='vertical_relaxed').sort(['ticker', 'date'])

    if isinstance(window, list):
        for w in window:
            df = _add_single(df, w)
        return df.lazy()
    return _add_single(df, window).lazy()