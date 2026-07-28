import polars as pl
from polars import col as c


def tanh_signal(lf:pl.LazyFrame, signal:str):
    '''
    Applies tanh function to column called signal.
    '''
    return lf.with_columns(
        c(signal).tanh().alias(signal)
    )

