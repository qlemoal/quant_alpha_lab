'''
Tests for the feature library. The most important one: NO LOOK-AHEAD.
Cut the prices at a date, rebuild the features, and the features of all earlier dates must be identical.
'''

import numpy as np
import polars as pl
import pytest

from src.features.feature_pipeline import build_features, check_ohlcv, check_features, LABEL_COLS
from src.features.liquidity import add_dollar_volume

WINDOWS = {'beta': [20], 'momentum': [5, 20], 'volatility': [5, 20], 'adv': [5, 20]}


def make_ohlcv(n_days=150, n_tickers=8, seed=0):
    rng = np.random.default_rng(seed)
    close = 50 * np.exp(np.cumsum(rng.normal(0, 0.015, (n_days, n_tickers)), axis=0))
    raw = close * rng.uniform(1.0, 1.3, (1, n_tickers))              # raw close above adjusted close
    vol = rng.integers(1_000, 100_000, (n_days, n_tickers))
    return pl.DataFrame({
        'ticker': [f'T{j}' for j in range(n_tickers) for _ in range(n_days)],
        'date': list(range(n_days)) * n_tickers,
        'open': (close * 0.999).T.ravel(), 'open_adj': (close * 0.999).T.ravel(),
        'close': close.T.ravel(), 'close_raw': raw.T.ravel(), 'volume': vol.T.ravel(),
    })


def feature_columns(df):
    return [c for c in df.columns if c not in LABEL_COLS + ['ticker', 'date']]


def test_no_look_ahead():
    full = make_ohlcv()
    cut = 100
    a = build_features(full.lazy(), WINDOWS, min_tickers=1).collect().sort(['ticker', 'date'])
    b = build_features(full.filter(pl.col('date') <= cut).lazy(), WINDOWS, min_tickers=1).collect().sort(['ticker', 'date'])
    a = a.filter(pl.col('date') <= cut)
    assert a.height == b.height
    for col in feature_columns(a):
        x, y = a[col].to_numpy().astype(float), b[col].to_numpy().astype(float)
        assert np.allclose(x, y, equal_nan=True, rtol=1e-12, atol=1e-14), f'{col} changes when later prices are removed'


def test_the_label_does_look_ahead():
    '''The control of the test above: the label MUST change when later prices are removed.'''
    full = make_ohlcv()
    a = build_features(full.lazy(), WINDOWS, 1).collect().sort(['ticker', 'date']).filter(pl.col('date') <= 100)
    b = build_features(full.filter(pl.col('date') <= 100).lazy(), WINDOWS, 1).collect().sort(['ticker', 'date'])
    assert not np.allclose(a['fwdret'].to_numpy().astype(float), b['fwdret'].to_numpy().astype(float), equal_nan=True)


def test_logret_momentum_volatility_by_hand():
    df = make_ohlcv()
    out = build_features(df.lazy(), WINDOWS, 1).collect().sort(['ticker', 'date'])
    g = out.filter(pl.col('ticker') == 'T2')
    close = g['close'].to_numpy()
    logret = np.diff(np.log(close))
    assert g['logret'][0] is None and g['logret'].to_numpy()[1:] == pytest.approx(logret)
    assert g['mom5'].to_numpy()[50] == pytest.approx(close[50] / close[45] - 1)
    assert g['vol20'].to_numpy()[50] == pytest.approx(np.std(np.diff(np.log(close))[31:50], ddof=1) if False else
                                                      np.std(logret[50 - 20:50], ddof=1))   # logret[k-1] is the return of day k
    assert g['mom5'].null_count() == 5 and g['vol20'].null_count() == 20


def test_dollar_volume_uses_the_raw_close():
    df = make_ohlcv()
    out = add_dollar_volume(df.lazy()).collect()
    assert np.allclose(out['dollar_volume'].to_numpy(), (df['close_raw'] * df['volume']).to_numpy())
    with pytest.raises(ValueError, match='close_raw'):
        add_dollar_volume(df.drop('close_raw').lazy())


def test_adv_is_a_rolling_mean_of_dollar_volume():
    df = make_ohlcv()
    out = build_features(df.lazy(), WINDOWS, 1).collect().sort(['ticker', 'date'])
    g = out.filter(pl.col('ticker') == 'T1')
    dv = g['dollar_volume'].to_numpy()
    assert g['log_adv5'].to_numpy()[30] == pytest.approx(np.log(dv[26:31].mean()))


def test_features_do_not_mix_tickers():
    '''The features of a ticker must not depend on the other tickers (except the market return in beta).'''
    full = make_ohlcv()
    alone = full.filter(pl.col('ticker') == 'T3')
    a = build_features(full.lazy(), WINDOWS, 1).collect().filter(pl.col('ticker') == 'T3').sort('date')
    b = build_features(alone.lazy(), WINDOWS, 1).collect().sort('date')
    for col in ['logret', 'mom5', 'mom20', 'vol5', 'vol20', 'dollar_volume', 'log_adv5', 'log_adv20', 'fwdret']:
        assert np.allclose(a[col].to_numpy().astype(float), b[col].to_numpy().astype(float), equal_nan=True)


def test_check_ohlcv():
    df = make_ohlcv()
    report = check_ohlcv(df.lazy())
    assert report['n_tickers'] == 8 and report['n_extreme_moves'] == 0
    with pytest.raises(ValueError, match='duplicate'):
        check_ohlcv(pl.concat([df, df.head(1)]).lazy())
    with pytest.raises(ValueError, match='non-positive'):
        check_ohlcv(df.with_columns(pl.when(pl.col('date') == 3).then(-1.0).otherwise(pl.col('close')).alias('close')).lazy())
    with pytest.raises(ValueError, match='lacks the columns'):
        check_ohlcv(df.drop('close_raw').lazy())


def test_check_ohlcv_flags_an_unadjusted_split():
    df = make_ohlcv()
    split = df.with_columns(
        pl.when((pl.col('ticker') == 'T0') & (pl.col('date') >= 70)).then(pl.col('close') / 2).otherwise(pl.col('close')).alias('close'))
    report = check_ohlcv(split.lazy())
    assert report['n_extreme_moves'] == 1 and report['largest_moves'][0]['ticker'] == 'T0'
    assert report['largest_moves'][0]['logret'] == pytest.approx(-np.log(2), abs=0.05)


def test_check_features_rejects_inf():
    out = build_features(make_ohlcv().lazy(), WINDOWS, 1).collect()
    check_features(out, WINDOWS)
    bad = out.with_columns(pl.when(pl.col('date') == 60).then(float('inf')).otherwise(pl.col('mom5')).alias('mom5'))
    with pytest.raises(ValueError, match='inf'):
        check_features(bad, WINDOWS)