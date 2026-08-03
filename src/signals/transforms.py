import polars as pl
from polars import col as c

from src.utils.helpers import as_list, add_expr



def tanh_expr(expr:pl.Expr, scale=1) -> pl.Expr:
    '''
    Expr version of tanh, to chain computations in memory without touching the frame.
    '''
    return (expr * scale).tanh()

def tanh(lf:pl.LazyFrame, colname:str|list, scale=1):
    '''
    Applies tanh function to column called colname.
    If colname is a list, then tanh is applied to each column name in it.
    '''
    return add_expr(lf, tanh_expr, colname, suffix='_tanh', scale=scale)



def clip_expr(expr:pl.Expr, low=-3, high=3) -> pl.Expr:
    '''
    Expr version of apply_clip, to chain computations in memory without touching the frame.
    '''
    return expr.clip(lower_bound=low, upper_bound=high)

def clip(lf:pl.LazyFrame, colname:str|list, low=-3, high=3) -> pl.LazyFrame:
    '''
    Clips values of colname, between low and high.
    If colname is a llist, it clips all the names in colname.

    Note: it is called clipping when we set the low/high values, but winsorizing when the values correspond to some quantile.
    '''
    return add_expr(lf, clip_expr, colname, suffix='_clip', low=low, high=high)



