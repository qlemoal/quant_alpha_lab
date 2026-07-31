import polars as pl
from polars import col as c

from src.utils.helpers import as_list


def rank_scaled_expr(feature:str, descending=True) -> pl.LazyFrame:
    '''
    Expr version of rank_into_minus1_1, to chain expressions without touching the frame. 
    '''
    return (2 * ( (c(feature)
                 .rank(descending=descending)
                 .over('date') - 1) / (pl.len().over('date')-1)
                ) - 1)


def rank_scaled(lf: pl.LazyFrame, feature:str|list, descending=True) -> pl.LazyFrame:
    '''
    Rank feature at each date across tickers --> normalise into (0,1) --> transform into (-1, 1).
    Use descending=False if lower feature is better.
    Using helper function as_list, to make a string a list of strings.
    '''
    feature = as_list(feature)
    
    return lf.with_columns( [
                                rank_scaled_expr(lf, feat, descending=descending).alias(f'{feat}_rank')
                                for feat in feature
                            ] ).sort(['ticker', 'date'])


    


def rank_expr(feature:str, descending=True) -> pl.LazyFrame:
    '''
    Expr version of rank, to chain expressions without touching the frame. 
    '''
    return ( 
            c(feature)
            .rank(descending=descending)
            .over('date') 
            )

def rank(lf: pl.LazyFrame, feature:str|list, descending=True) -> pl.LazyFrame:
    '''
    Rank feature at each date across tickers .
    Use descending=False if lower feature is better.
    '''
    feature = as_list(feature)
    return lf.with_columns( [ 
                                rank_expr(lf, feat, descending=descending).alias(f'{feat}_rank') 
                                for feat in feature 
                            ] ).sort(['ticker', 'date'])

