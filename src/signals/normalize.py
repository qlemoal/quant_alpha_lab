import polars as pl
from polars import col as c

from src.utils.helpers import as_list, add_expr






def zscore_expr(expr:pl.Expr, descending=False) -> pl.Expr:
    '''
    Expr version of z_score, to chain computations together without touching the frame.
    '''
    if descending==True: inversion = -1 
    else: inversion = 1 

    mean = (expr - expr.mean().over('date'))
    var = expr.std().over('date') 
    return (
        pl.when(var > 0)
        .then(inversion * mean / var)
        .otherwise(None)
    )


def z_score(lf:pl.LazyFrame, colname:str|list, descending=False):
    '''
    Normalizes a column cross-sectionally, i.e., over dates.
    If colname is a list, then it normalizes all columns in the list, using the helper function as_list.
    Preserves relative magnitude, sensitive to outliers unless paired with winsorize() or apply_tanh() afterwards.
    '''
    return add_expr(lf, zscore_expr, colname, suffix='_z', descending=descending)



