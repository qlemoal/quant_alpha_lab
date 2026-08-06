import polars as pl



def pivot_wide(df, value_col, index='date', on='ticker'):
    '''
    Long (ticker, date, value) panel -> wide (date rows, ticker columns). Requires an already-collected pl.DataFrame. 
        This is useful for a per-ticker time series plot, or a ticker-ticker correlation matrix of a signal.
    '''
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    return df.pivot(index=index, on=on, values=value_col, aggregate_function='first')

