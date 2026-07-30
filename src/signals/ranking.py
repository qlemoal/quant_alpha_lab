import polars as pl
from polars import col as c

from src.utils.helpers import as_list


def rank_into_minus1_1(lf: pl.LazyFrame, feature:str|list, descending=True) -> pl.LazyFrame:
    '''
    Rank feature at each date across tickers --> normalise into (0,1) --> transform into (-1, 1).
    Use descending=False if lower feature is better.
    Using helper function as_list, to make a string a list of strings.
    '''
    feature = as_list(feature)

    exprs = []
    for feat in feature:
        rank = ( c(feat)
                .rank(descending=descending)
                .over('date') )
        n = pl.len().over('date')
        exprs.append(
            (2 * ((rank - 1) / (n-1)) - 1).alias(feat + '_rank')
        )
    return lf.with_columns(exprs).sort(['ticker', 'date'])




def rank(lf: pl.LazyFrame, feature:str|list, descending=True) -> pl.LazyFrame:
    '''
    Rank feature at each date across tickers .
    Use descending=False if lower feature is better.
    '''
    feature = as_list(feature)
    
    exprs = []
    for feat in feature:
        exprs.append(( 
                        c(feat)
                        .rank(descending=descending)
                        .over('date') 
                        ).alias(feat + '_rank')
                    )
    return lf.with_columns(exprs).sort(['ticker', 'date'])


