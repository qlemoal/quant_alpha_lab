# tests/test_beta.py
import setup
import polars as pl
import pytest
from polars import col as c
from src.features.beta import add_beta

# def test_add_beta_window2():


df = pl.LazyFrame({
    'ticker': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
    'date': [1, 2, 3, 1, 2, 3, 1, 2, 3],
    'logret': [0.02, 0.04, -0.02, 0.00, 0.01, 0.01, 1, 1, 1],
    'logret2': [0.02, 0.04, -0.02, 0.00, 0.01, 0.01, 1, 1, 1],
})


result = add_beta(df.lazy().sort(['ticker', 'date']), window=[2, 3]).collect()

print(result)
# hand-computed: mkt_logret = [0.01, 0.03, 0.00] for dates 1,2,3
# ticker A, date 2 window=[date1,date2]: cov=0.0002, var_m = 0.0002 -> beta=1.0
# ticker A, date 3 window=[date2,date3]: cov=0.0009, var_m =0.00045 -> beta=2.0
# ticker B, date 3 window=[date2,date3]: cov=0.0 (B constant) -> beta=0.0

got = result.sort(['ticker', 'date'])['beta2'].to_list()
expected = [None, 1.0, 2.0, None, 1.0, 0.0]

print(got)
print(expected)

for g, e in zip(got, expected):
    if e is None:
        assert g is None
    else:
        assert g == pytest.approx(e, rel=1e-6)