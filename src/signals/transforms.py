import polars as pl
from polars import col as c

from src.utils.helpers import as_list


def apply_tanh(lf:pl.LazyFrame, colname:str|list, scale=1):
    '''
    Applies tanh function to column called colname.
    If colname is a list, then tanh is applied to each column name in it.
    '''
    colname = as_list(colname)

    exprs = []
    for cn in colname:
        exprs.append((c(cn) * scale).tanh().alias(cn))
    return lf.with_columns(exprs)




def apply_clip(lf:pl.LazyFrame, colname:str|list, low=-3, high=3) -> pl.LazyFrame:
    '''
    Clips values of colname, between low and high.
    If colname is a llist, it clips all the names in colname.
    '''
    colname = as_list(colname)

    exprs = []
    for cn in colname:
        exprs.append(c(cn).clip(lower_bound=low, upper_bound=high).alias(cn))
    return lf.with_columns(exprs)



