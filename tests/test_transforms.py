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



























#  Add fixture decorator for pytest to remember the "variable". It also works for multiple tests.
@pytest.fixture
def panel_normal():
    '''
    1 date, 4 tickers, distinct values, the 'nothing weird' baseline case.
    '''
    return pl.DataFrame({
        'ticker': ['A', 'B', 'C', 'D'],
        'date':   [1, 1, 1, 1],
        'feat':   [10.0, 20.0, 30.0, 40.0],
    })


@pytest.fixture
def panel_single_ticker():
    '''
    Group of size 1. std() is undefined (or 0), rank has nothing to rank against.
    '''
    return pl.DataFrame({
        'ticker': ['A'],
        'date':   [1],
        'feat':   [10.0],
    })


@pytest.fixture
def panel_identical_values():
    '''
    All tickers share the same value. std = 0, every rank tied.
    '''
    return pl.DataFrame({
        'ticker': ['A', 'B', 'C', 'D'],
        'date':   [1, 1, 1, 1],
        'feat':   [5.0, 5.0, 5.0, 5.0],
    })


@pytest.fixture
def panel_with_nulls():
    '''
    Some tickers missing the feature on this date (e.g. not enough history yet for a rolling window).
    '''
    return pl.DataFrame({
        'ticker': ['A', 'B', 'C', 'D'],
        'date':   [1, 1, 1, 1],
        'feat':   [10.0, None, 30.0, None],
    })


@pytest.fixture
def panel_uneven_group():
    '''
    7 tickers, doesn't divide evenly into 10 deciles.
    '''
    return pl.DataFrame({
        'ticker': list('ABCDEFG'),
        'date':   [1] * 7,
        'feat':   [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    })


def test_zscore_single_ticker_does_not_crash(panel_single_ticker):
    result = make_signal(panel_single_ticker.lazy(), 'feat', method='zscore_tanh').collect()
    # std of a single value is 0 or null depending on ddof; either way,
    # dividing by it should not silently produce inf, check it's null instead
    val = result['feat_zscore_tanh'][0]
    assert val is None or not (val == float('inf') or val == float('-inf'))


def test_zscore_identical_values_no_inf(panel_identical_values):
    result = make_signal(panel_identical_values.lazy(), 'feat', method='zscore_tanh').collect()
    vals = result['feat_zscore_tanh'].to_list()
    # std = 0 here -> division by zero. Should not produce inf/nan silently.
    for v in vals:
        assert v is None or (v == v and abs(v) != float('inf'))  # v == v filters NaN


def test_rank_with_nulls_preserves_null_and_ranks_rest(panel_with_nulls):
    result = make_signal(panel_with_nulls.lazy(), 'feat', method='rank').collect().sort('ticker')
    got = dict(zip(result['ticker'], result['feat_rank']))
    # B and D had null feat, their signal should stay null, not silently become 0
    assert got['B'] is None
    assert got['D'] is None
    # A and C should be ranked against each other only (A=10 < C=30)
    assert got['A'] < got['C']


def test_decile_uneven_group_does_not_crash(panel_uneven_group):
    result = make_signal(panel_uneven_group.lazy(), 'feat', method='decile').collect()
    # just checking it runs and produces a value for every row, not the
    # exact bucket assignment, since 7 tickers into 10 buckets is a
    # degenerate case by construction
    assert result['feat_decile'].null_count() == 0


def test_normal_case_matches_hand_computation(panel_normal):
    result = make_signal(panel_normal.lazy(), 'feat', method='rank').collect().sort('ticker')
    got = result['feat_rank'].to_list()
    expected = [-1.0, -1 / 3, 1 / 3, 1.0]
    for g, e in zip(got, expected):
        assert g == pytest.approx(e, rel=1e-6)