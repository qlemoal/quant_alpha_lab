'''Tests for src/features/beta.py: the market return and the rolling beta, against a plain numpy computation.'''

import numpy as np
import polars as pl
import pytest

from src.features.beta import add_beta, add_market_return


def _panel(seed=0, n_days=120, n_tickers=6):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.01, (n_days, n_tickers)) + 0.5 * rng.normal(0, 0.01, (n_days, 1))
    return pl.DataFrame({
        'ticker': [f'T{j}' for j in range(n_tickers) for _ in range(n_days)],
        'date': list(range(n_days)) * n_tickers,
        'logret': r.T.ravel(),
    }), r


def test_market_return_is_equal_weight_average():
    df, r = _panel()
    out = add_market_return(df.lazy(), min_tickers=1).collect().sort(['ticker', 'date'])
    mkt = out.filter(pl.col('ticker') == 'T0')['mkt_logret'].to_numpy()
    assert np.allclose(mkt, r.mean(axis=1))


def test_market_return_is_null_when_too_few_tickers():
    df, _ = _panel(n_tickers=4)
    out = add_market_return(df.lazy(), min_tickers=5).collect()
    assert out['mkt_logret'].null_count() == out.height


def test_beta_matches_numpy():
    df, r = _panel()
    w = 60
    out = add_beta(df.lazy(), window=w, min_tickers=1).collect().sort(['ticker', 'date'])
    mkt = r.mean(axis=1)
    for j in (0, 3):
        got = out.filter(pl.col('ticker') == f'T{j}')['beta60'].to_numpy()
        assert np.isnan(got[:w - 1]).all() or (got[:w - 1] == None).all()      # warm-up: null
        for t in (w - 1, 80, 119):
            x, m = r[t - w + 1:t + 1, j], mkt[t - w + 1:t + 1]
            assert got[t] == pytest.approx(np.cov(x, m)[0, 1] / np.var(m, ddof=1), rel=1e-9)


def test_beta_of_the_market_itself_is_one_and_constant_market_gives_null():
    n = 80
    mkt = np.random.default_rng(1).normal(0, 0.01, n)
    one = pl.DataFrame({'ticker': ['A'] * n, 'date': range(n), 'logret': mkt})
    out = add_beta(one.lazy(), window=20, min_tickers=1).collect()
    assert out['beta20'].drop_nulls().to_numpy() == pytest.approx(1.0)
    flat = pl.DataFrame({'ticker': ['A'] * n, 'date': range(n), 'logret': np.zeros(n)})
    assert add_beta(flat.lazy(), window=20, min_tickers=1).collect()['beta20'].null_count() == n


def test_two_windows_are_added():
    df, _ = _panel()
    out = add_beta(df.lazy(), window=[20, 60], min_tickers=1).collect()
    assert {'beta20', 'beta60'} <= set(out.columns)