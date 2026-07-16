import polars as pl
from polars import col as c








lf = pl.LazyFrame({
    'ticker': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
    'date': [1, 2, 3, 1, 2, 3, 1, 2, 3],
    'logret': [0.02, 0.04, -0.02, 0.00, 0.02, 0.02, .2, -.2, .2],
})
print(feature_to_signal(lf, 'logret').sort('date').collect())