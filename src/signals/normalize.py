import polars as pl
from polars import col as c

from src.utils.helpers import as_list




def zscore_expr(colname:str) -> pl.LazyFrame:
    '''
    Expr version of z_score, to chain computations together without touching the frame.
    '''
    return ( (c(colname) - c(colname).mean().over('date')) / c(colname).std().over('date') )


def z_score(lf:pl.LazyFrame, colname:str|list):
    '''
    Normalizes a column cross-sectionally, i.e., over dates.
    If colname is a list, then it normalizes all columns in the list, using the helper function as_list.
    Preserves relative magnitude, sensitive to outliers unless paired with winsorize() or apply_tanh() afterwards.
    '''
    colname = as_list(colname)
    return lf.with_columns([
                                zscore_expr(lf, cn).alias(f'{cn}_z') 
                                for cn in colname
                            ]).sort(['ticker', 'date'])



def decile_bucket(lf:pl.LazyFrame, colname:str|list, n_buckets=10) -> pl.LazyFrame:
    '''
    Categorizes data in colname into n_buckets.
    It should then be coupled into a long-short position in the top and bottom deciles.

    n_buckets: number of categories to sort the data into. 10 buckets corresponds to sorting into deciles.
    '''
    colname = as_list(colname)

    exprs = [
        (
            c(cn)
            .qcut(n_buckets, labels=[str(ii) for ii in range(n_buckets)])
            .over('date')
            .cast(pl.Int32)
            .alias(cn + '_decile')
        )
        for cn in colname
    ]
    return lf.with_columns(exprs).sort(['ticker', 'date'])


