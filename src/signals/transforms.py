import polars as pl
from polars import col as c
from src.features.beta import test_func


def tanh_signal(lf:pl.LazyFrame, signal:str):
    '''
    Applies tanh function to column called signal.
    '''
    return lf.with_columns(
        c(signal).tanh().alias(signal)
    )


test_func()


lf = pl.LazyFrame({
    'ticker': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
    'date': [1, 2, 3, 1, 2, 3, 1, 2, 3],
    'logret': [0.02, 0.04, -0.02, 0.00, 0.02, 0.02, .2, -.2, .2],
})
print(tanh_signal(lf, 'logret').sort('date').collect())