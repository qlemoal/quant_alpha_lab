import polars as pl
from polars import col as c

from src.utils.helpers import as_list



def tanh_expr(colname:str, scale=1):
    '''
    Expr version of tanh, to chain computations in memory without touching the frame.
    '''
    return (c(colname) * scale).tanh()

def tanh(lf:pl.LazyFrame, colname:str|list, scale=1):
    '''
    Applies tanh function to column called colname.
    If colname is a list, then tanh is applied to each column name in it.
    '''
    colname = as_list(colname)

    return lf.with_columns([
                                tanh_expr(lf, cn, scale=scale).alias(f'{cn}_tanh')
                                for cn in colname
                            ])




def clip_expr(colname:str, low=-3, high=3) -> pl.LazyFrame:
    '''
    Expr version of apply_clip, to chain computations in memory without touching the frame.
    '''
    return c(colname).clip(lower_bound=low, upper_bound=high)

def clip(lf:pl.LazyFrame, colname:str|list, low=-3, high=3) -> pl.LazyFrame:
    '''
    Clips values of colname, between low and high.
    If colname is a llist, it clips all the names in colname.

    Note: it is called clipping when we set the low/high values, but winsorizing when the values correspond to some quantile.
    '''
    colname = as_list(colname)

    return lf.with_columns([
                                clip_expr(lf, cn, low=low, high=high).alias(f'{cn}_clip')
                                for cn in colname
                            ])



