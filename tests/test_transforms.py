# tests/test_transform.py
import polars as pl
import pytest
from src.signals.combine import make_signal

def test_make_signal_rank():
    # 4 tickers, one date, feature values 10/20/30/40
    df = pl.DataFrame({
        'ticker': ['A', 'B', 'C', 'D'],
        'date':   [1, 1, 1, 1],
        'feat':   [10.0, 20.0, 30.0, 40.0],
    })
    result = make_signal(df.lazy(), 'feat', method='rank', descending=False).collect()
    got = result.sort('ticker')['feat_rank'].to_list()

    # rank transform, N=4: rank 1,2,3,4 -> (rank-1)/(N-1)*2-1
    expected = [-1.0, -1/3, 1/3, 1.0]
    for g, e in zip(got, expected):
        assert g == pytest.approx(e, rel=1e-6)

def test_make_signal_zscore_tanh():
    df = pl.DataFrame({
        'ticker': ['A', 'B', 'C', 'D'],
        'date':   [1, 1, 1, 1],
        'feat':   [10.0, 20.0, 30.0, 40.0],
    })
    result = make_signal(df.lazy(), 'feat', method='zscore_tanh').collect()
    got = result.sort('ticker')['feat_zscore_tanh'].to_list()

    # mean=25, sample std (ddof=1) = sqrt(500/3) ~= 12.9099
    # z = [-1.1619, -0.3873, 0.3873, 1.1619], then tanh
    import math
    expected = [math.tanh(z) for z in [-1.16190, -0.38730, 0.38730, 1.16190]]
    for g, e in zip(got, expected):
        assert g == pytest.approx(e, rel=1e-3)

def test_make_signal_decile():
    df = pl.DataFrame({
        'ticker': ['A', 'B', 'C', 'D'],
        'date':   [1, 1, 1, 1],
        'feat':   [10.0, 20.0, 30.0, 40.0],
    })
    result = make_signal(df.lazy(), 'feat', method='decile').collect()
    got = result.sort('ticker')['feat_decile'].to_list()

    # mean=25, sample std (ddof=1) = sqrt(500/3) ~= 12.9099
    # z = [-1.1619, -0.3873, 0.3873, 1.1619], then tanh
    import math
    expected = [-1, 0, 0, 1]
    for g, e in zip(got, expected):
        assert g == pytest.approx(e, rel=1e-3)

def test_make_signal_never_touches_original():
    df = pl.DataFrame({
        'ticker': ['A', 'B'], 'date': [1, 1], 'feat': [10.0, 20.0],
    })
    result = make_signal(df.lazy(), 'feat', method='rank').collect()
    # original column untouched, both present
    assert 'feat' in result.columns
    assert 'feat_rank' in result.columns
    assert result['feat'].to_list() == [10.0, 20.0]