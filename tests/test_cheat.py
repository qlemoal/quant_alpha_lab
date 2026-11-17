'''Tests of the cheat signals: the IC must be the IC we ask for, the noise must be independent and repeatable.'''

import numpy as np
import polars as pl
import pytest
from scipy.stats import spearmanr

from src.signals.cheat import add_cheat_signals, cheat_name, validate_cheat_specs


@pytest.fixture(scope='module')
def panel():
    rng = np.random.default_rng(1)
    n_dates, n_tickers = 250, 400
    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 1) + pl.duration(days=n_dates - 1), eager=True)
    fwd = rng.standard_t(3, size=(n_dates, n_tickers)) * 0.05          # fat tails on purpose
    df = pl.DataFrame({
        'date': np.repeat(dates.to_numpy(), n_tickers),
        'ticker': [f'T{i}' for i in range(n_tickers)] * n_dates,
        'fwdret20': fwd.ravel(),
    })
    # T0 has no label on any date (like the end of a history)
    return df.with_columns(pl.when(pl.col('ticker') == 'T0').then(None).otherwise(pl.col('fwdret20')).alias('fwdret20'))


def mean_rank_ic(df, col):
    ics = []
    for _, g in df.group_by('date'):
        g = g.drop_nulls([col, 'fwdret20'])
        ics.append(spearmanr(g[col], g['fwdret20']).statistic)
    return float(np.mean(ics))


@pytest.mark.parametrize('ic', [0.02, 0.05, 0.1, 0.3, 0.6, 0.9])
def test_rank_ic_is_the_requested_ic(panel, ic):
    lf, names = add_cheat_signals(panel.lazy(), [(20, ic)])
    assert mean_rank_ic(lf.collect(), names[0]) == pytest.approx(ic, abs=0.012)


def test_no_label_gives_no_signal(panel):
    lf, names = add_cheat_signals(panel.lazy(), [(20, 0.1)])
    out = lf.collect()
    assert out.filter(pl.col('ticker') == 'T0')[names[0]].null_count() == 250
    assert out.filter(pl.col('ticker') != 'T0')[names[0]].null_count() == 0


def test_noise_is_not_a_fixed_bias_of_the_stock(panel):
    '''The signal of one stock must not repeat from one day to the next (the noise changes every day).'''
    lf, names = add_cheat_signals(panel.lazy(), [(20, 0.05)])
    out = lf.collect()
    s = out.filter(pl.col('ticker') == 'T5').sort('date')[names[0]].to_numpy()
    assert abs(np.corrcoef(s[1:], s[:-1])[0, 1]) < 0.15
    assert abs(out[names[0]].mean()) < 0.05 and out[names[0]].std() == pytest.approx(1.0, abs=0.05)


def test_same_seed_same_result_other_seed_other_noise(panel):
    a = add_cheat_signals(panel.lazy(), [(20, 0.05)], seed=0)[0].collect()
    b = add_cheat_signals(panel.lazy(), [(20, 0.05)], seed=0)[0].collect()
    c = add_cheat_signals(panel.lazy(), [(20, 0.05)], seed=1)[0].collect()
    assert a.equals(b)
    assert not a.equals(c)


def test_validation():
    assert validate_cheat_specs([(20, 0.05), (20, 0.9)], [1, 5, 10, 20]) == [(20, 0.05), (20, 0.9)]
    with pytest.raises(ValueError, match='not in HORIZONS'):
        validate_cheat_specs([(7, 0.05)], [1, 5, 10, 20])
    with pytest.raises(ValueError, match='IC must be'):
        validate_cheat_specs([(20, 0.0)], [20])
    with pytest.raises(ValueError, match='IC must be'):
        validate_cheat_specs([(20, 1.0)], [20])
    with pytest.raises(ValueError, match='twice'):
        validate_cheat_specs([(20, 0.05), (20, 0.05)], [20])


def test_missing_label_column_is_an_error(panel):
    with pytest.raises(ValueError, match='fwdret5'):
        add_cheat_signals(panel.lazy(), [(5, 0.05)])


def test_names():
    assert cheat_name(20, 0.05) == 'cheat_h20_ic050'
    assert cheat_name(5, 0.3) == 'cheat_h5_ic300'
    assert cheat_name(20, 0.9) == 'cheat_h20_ic900'


def test_signals_do_not_share_their_noise(panel):
    '''
    Four cheat signals together must not allow a linear model to rebuild the future return.
    With ONE shared noise, least squares cancels it and finds IC = 1. With independent noises the best
    combination of ICs 0.04, 0.05, 0.09, 0.10 is about 0.14 (square root of the sum of the squares).
    '''
    lf, names = add_cheat_signals(panel.lazy(), [(20, 0.04), (20, 0.05), (20, 0.09), (20, 0.1)])
    out = lf.collect().drop_nulls(names + ['fwdret20'])
    x = out.select(names).to_numpy()
    y = out['fwdret20'].to_numpy()
    beta = np.linalg.lstsq(np.c_[x, np.ones(len(x))], y, rcond=None)[0]
    fit = np.c_[x, np.ones(len(x))] @ beta
    assert spearmanr(fit, y).statistic < 0.3