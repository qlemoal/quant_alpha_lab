import polars as pl
from polars import col as c



def rank_into_minus1_1(lf: pl.LazyFrame, feature:str, descending=True):
    '''
    Rank feature at each date across tickers --> normalise into (0,1) --> transform into (-1, 1).
    Use descending=False if lower feature is better.
    '''
    rank = ( c(feature)
             .rank(descending=descending)
             .over('date') )

    n = pl.len().over('date')

    return lf.with_columns(
        (2 * ((rank - 1) / (n-1)) - 1).alias(feature + '_signal')
    )



