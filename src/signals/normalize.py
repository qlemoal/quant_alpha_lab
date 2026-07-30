import polars as pl
from polars import col as c

from src.utils.helpers import as_list



def zscore(lf:pl.LazyFrame, colname:str|list) -> pl.LazyFrame:
    '''
    Normalizes a column cross-sectionally, i.e., over dates.
    If colname is a list, then it normalizes all columns in the list, using the helper function as_list.
    Preserves relative magnitude, sensitive to outliers unless paired with winsorize() or apply_tanh() afterwards.
    '''
    colname = as_list(colname)

    exprs = []
    for cn in colname:
        mean = c(cn).mean().over('date')
        std = c(cn).std().over('date')
        exprs.append(((c(cn) - mean) / std).alias(cn + '_z'))
    return lf.with_columns(exprs).sort(['ticker', 'date'])


df = pl.LazyFrame({
    'ticker': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
    'date': [1, 2, 3, 1, 2, 3, 1, 2, 3],
    'logret': [0.02, 0.04, -0.02, 0.00, 0.01, 0.01, 1, 1, 1],
    'logret2': [0.02, 0.04, -0.02, 0.00, 0.01, 0.01, 1, 1, 1],
})

res = df.select(
    c('logret').qcut(2).over('date')
).collect()

print(res)