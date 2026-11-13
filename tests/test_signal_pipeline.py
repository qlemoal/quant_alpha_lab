'''Tests for the forward-return labels and the 04a helpers.'''

import numpy as np
import polars as pl
import pytest

from src.features.returns import add_fwdret_horizon
from src.signals.signal_pipeline import check_labels, clean_features, validate_specs, signal_warnings


def _prices(n=10):
    rows = [(t, i, base * 1.01 ** i) for t, base in (('A', 100.0), ('B', 50.0)) for i in range(n)]
    return pl.DataFrame(rows, schema=['ticker', 'date', 'open_adj'], orient='row')


def test_label_never_crosses_tickers():
    '''Regression test: the shift used to run over the whole frame, so the tail of ticker A read ticker B.'''
    out = add_fwdret_horizon(_prices().lazy(), horizon=2).collect().sort(['ticker', 'date'])
    for t in ('A', 'B'):
        lab = out.filter(pl.col('ticker') == t)['fwdret2'].to_list()
        assert lab[-3:] == [None, None, None]                       # last h+1 rows: no price yet
        assert all(abs(x - 2 * np.log(1.01)) < 1e-12 for x in lab[:-3])


def test_label_is_open_to_open_with_one_day_delay():
    df = pl.DataFrame({'ticker': ['A'] * 6, 'date': range(6), 'open_adj': [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]})
    lab = add_fwdret_horizon(df.lazy(), horizon=2).collect()['fwdret2'].to_list()
    assert lab[0] == pytest.approx(np.log(8.0 / 2.0))               # log(open[t+3]) - log(open[t+1])


def test_check_labels_catches_a_cross_ticker_label():
    df = add_fwdret_horizon(_prices().lazy(), horizon=2).collect()
    check_labels(df, [2])                                           # correct labels pass
    broken = df.sort(['ticker', 'date']).with_columns(
        pl.when((pl.col('ticker') == 'A') & (pl.col('date') == 9)).then(0.5).otherwise(pl.col('fwdret2')).alias('fwdret2'))
    with pytest.raises(ValueError, match='crosses a ticker boundary'):
        check_labels(broken, [2])


def test_clean_features_turns_nan_and_inf_into_null():
    df = pl.DataFrame({'x': [1.0, float('nan'), float('inf'), -float('inf'), 2.0]})
    out = clean_features(df.lazy(), ['x']).collect()['x']
    assert out.null_count() == 3 and out.is_nan().sum() == 0


def test_validate_specs():
    usable, skipped = validate_specs([('a', 'zscore_tanh', {}), ('b', 'rank', {})], {'a'})
    assert len(usable) == 1 and skipped[0]['spec'][0] == 'b'
    with pytest.raises(ValueError):
        validate_specs([('a', 'nope', {})], {'a'})
    with pytest.raises(ValueError):
        validate_specs([('a', 'rank', {}), ('a', 'rank', {})], {'a'})


def test_signal_warnings():
    stats = {'s1': {'null_pct': 100.0, 'std': None}, 's2': {'null_pct': 0.0, 'std': 0.0},
             's3': {'null_pct': 70.0, 'std': 1.0}, 's4': {'null_pct': 1.0, 'std': 1.0}}
    w = signal_warnings(stats)
    assert len(w) == 3 and not any('s4' in x for x in w)