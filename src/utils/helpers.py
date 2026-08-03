from collections.abc import Iterable
from polars import col as c

def as_list(x):
    if isinstance(x, str):
        return [x]
    elif isinstance(x, Iterable):
        return list(x)
    else:
        raise TypeError("features must be str or iterable")


def add_expr(lf, expr_func, colname, suffix=None, **kwargs):
    colname = as_list(colname)
    return lf.with_columns(
        [
            expr_func(c(cn), **kwargs).alias(f'{cn}{suffix}')
            for cn in colname
        ]
    )