import polars as pl
from polars import col as c

from src.utils.helpers import as_list, add_expr


def rank_scaled_expr(expr:pl.Expr, descending=False) -> pl.Expr:
    '''
    Expr version of rank_into_minus1_1, to chain expressions without touching the frame. 
    '''
    return (2 * ( (expr
                 .rank(descending=descending)
                 .over('date') - 1) / (pl.len().over('date')-1)
                ) - 1)


def rank_scaled(lf: pl.LazyFrame, feature:str|list, descending=False) -> pl.LazyFrame:
    '''
    Rank feature at each date across tickers --> normalise into (0,1) --> transform into (-1, 1).
    Use descending=False if lower feature is better.
    Using helper function as_list, to make a string a list of strings.
    '''
    return add_expr(lf, rank_scaled_expr, feature, suffix='_rank', descending=descending).sort(['ticker', 'date'])


    



def rank_expr(expr:pl.Expr, descending=False) -> pl.Expr:
    '''
    Expr version of rank, to chain expressions without touching the frame. 
    '''
    return ( 
            expr
            .rank(descending=descending)
            .over('date') 
            )

def rank(lf: pl.LazyFrame, feature:str|list, descending=False) -> pl.LazyFrame:
    '''
    Rank feature at each date across tickers .
    Use descending=False if lower feature is better.
    '''
    return add_expr(lf, rank_expr, feature, suffix='_rank', descending=descending)





def decile_expr(expr:pl.Expr, n_buckets=10, descending=False) -> pl.Expr:
    if descending==True: inversion = -1 
    else: inversion = 1  
    deciles = (
        ( expr
         .qcut(n_buckets, 
               labels=[str(ii) for ii in range(n_buckets)],
               allow_duplicates=True)
         .over('date')
         .cast(pl.Utf8)  # You apparently need to cast from categorical to string first in polars. Plus labels only accept strings in qcut
         .cast(pl.Float64) ) 
        / (n_buckets - 1) * 2 - 1
        )
    return (
        pl.when(expr.is_not_null())
        .then(
            pl.when(deciles == 1).then(inversion * 1)
            .when(deciles == -1).then(inversion * (-1))
            .otherwise(0)
        ).otherwise(None)
    )

def decile(lf:pl.LazyFrame, colname:str|list, n_buckets=10, descending=False) -> pl.LazyFrame:
    '''
    Categorizes data in colname into n_buckets.
    It should then be coupled into a long-short position in the top and bottom deciles.

    n_buckets: number of categories to sort the data into. 10 buckets corresponds to sorting into deciles.
    '''

    return add_expr(lf, decile_expr, colname, suffix='_decile', n_buckets=n_buckets, descending=descending)