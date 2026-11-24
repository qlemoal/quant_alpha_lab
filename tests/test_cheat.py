'''Tests of the cheat signals: the IC must be the IC we ask for, the noise must be independent and repeatable.'''

import numpy as np
import polars as pl
import pytest
from scipy.stats import spearmanr

from src.signals.cheat import (add_cheat_signals, add_placebo_signals, cheat_name, validate_cheat_specs, placebo_names,
                               signal_group, check_benchmark_tag)


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


# ---------------------------------------------------------------- placebo signals

def test_placebo_names_and_groups():
    assert placebo_names(3) == ['placebo_01', 'placebo_02', 'placebo_03']
    assert placebo_names(120)[0] == 'placebo_001'
    assert signal_group('placebo_01') == 'placebo'
    assert signal_group('cheat_h20_ic050') == 'cheat'
    assert signal_group('mom20_zscore_tanh') == 'real'


def test_placebo_uses_no_label(panel):
    '''A placebo must be buildable on a frame WITHOUT any return column: it uses no return.'''
    no_label = panel.select(['date', 'ticker'])
    lf, names = add_placebo_signals(no_label.lazy(), 5)
    out = lf.collect()
    assert out.columns == ['date', 'ticker'] + names
    assert out[names[0]].null_count() == 0                 # also the rows that have no label


def test_placebo_has_zero_ic_and_unit_noise(panel):
    lf, names = add_placebo_signals(panel.lazy(), 12, noise_days=1)    # fast noise: independent dates, plain t-test valid
    out = lf.collect()
    t_stats = []
    for nm in names:
        ics = []
        for _, g in out.group_by('date'):
            g = g.drop_nulls(['fwdret20'])
            ics.append(spearmanr(g[nm], g['fwdret20']).statistic)
        t_stats.append(np.mean(ics) / (np.std(ics, ddof=1) / np.sqrt(len(ics))))
    assert abs(np.mean(t_stats)) < 1.0 and np.std(t_stats) < 1.6        # t-statistics of a null: about N(0, 1)
    assert out[names[0]].mean() == pytest.approx(0.0, abs=0.03)
    assert out[names[0]].std() == pytest.approx(1.0, abs=0.03)


def test_placebos_are_independent_and_repeatable(panel):
    # One-day noise: 100,000 rows, so the sampling error of a correlation is about 0.003. A tight bound is fine.
    lf, names = add_placebo_signals(panel.lazy(), 6, noise_days=(1,))
    out = lf.collect()
    c = np.corrcoef(out.select(names).to_numpy().T)
    assert np.abs(c - np.eye(6)).max() < 0.02
    # Slow noise (5, 20, 60 days) is autocorrelated, so a correlation between two independent slow series has a
    # much larger sampling error (up to about 0.04 depending on the hash of the Polars version). The bound is loose
    # on purpose. The real protection is that each placebo has its own seed; a SHARED noise gives a correlation near 1.
    lf, names = add_placebo_signals(panel.lazy(), 6)
    out = lf.collect()
    c = np.corrcoef(out.drop_nulls().select(names).to_numpy().T)   # slow placebos have no value at the start of a ticker
    assert np.abs(c - np.eye(6)).max() < 0.12
    again = add_placebo_signals(panel.lazy(), 6)[0].collect()
    other_seed = add_placebo_signals(panel.lazy(), 6, seed=1)[0].collect()
    assert out.equals(again) and not out.equals(other_seed)


def test_placebo_noise_is_not_the_cheat_noise(panel):
    '''A placebo must not be a copy of the noise of a cheat signal (else the combiner could subtract it).'''
    cheat, cn = add_cheat_signals(panel.lazy(), [(20, 0.05)])
    both, pn = add_placebo_signals(cheat, 3)
    out = both.collect().drop_nulls()
    c = np.corrcoef(out.select(cn + pn).to_numpy().T)
    assert np.abs(c[0, 1:]).max() < 0.05


def test_placebo_count_must_be_positive():
    with pytest.raises(ValueError, match='at least 1'):
        add_placebo_signals(pl.DataFrame({'date': [], 'ticker': []}).lazy(), 0)


def test_tag_rules():
    check_benchmark_tag('default', False, False)
    check_benchmark_tag('cheat', True, False)
    check_benchmark_tag('placebo_v2', False, True)
    check_benchmark_tag('cheat_placebo', True, True)
    with pytest.raises(ValueError, match='cheat'):
        check_benchmark_tag('default', True, False)
    with pytest.raises(ValueError, match='placebo'):
        check_benchmark_tag('cheat', True, True)


# ---------------------------------------------------------------- persistent noise

@pytest.mark.parametrize('ic', [0.03, 0.1])
def test_rank_ic_is_still_the_requested_ic_with_persistent_noise(panel, ic):
    lf, names = add_cheat_signals(panel.lazy(), [(20, ic)], noise_days=20)
    out = lf.collect().drop_nulls(names + ['fwdret20'])
    assert mean_rank_ic(out, names[0]) == pytest.approx(ic, abs=0.015)


def test_persistent_noise_changes_slowly_and_has_unit_variance(panel):
    lf, names = add_cheat_signals(panel.lazy(), [(20, 0.02)], noise_days=20)
    out = lf.collect()
    s = out.filter(pl.col('ticker') == 'T5').sort('date')[names[0]].drop_nulls().to_numpy()
    assert np.corrcoef(s[1:], s[:-1])[0, 1] == pytest.approx(19 / 20, abs=0.03)
    assert out[names[0]].std() == pytest.approx(1.0, abs=0.05)
    assert out.filter(pl.col('ticker') == 'T5').sort('date')[names[0]].head(19).null_count() == 19   # no history yet


def test_persistent_noise_works_on_an_unsorted_frame(panel):
    shuffled = panel.sample(fraction=1.0, shuffle=True, seed=3)
    a = add_cheat_signals(panel.lazy(), [(20, 0.05)], noise_days=10)[0].collect().sort(['ticker', 'date'])
    b = add_cheat_signals(shuffled.lazy(), [(20, 0.05)], noise_days=10)[0].collect().sort(['ticker', 'date'])
    assert a.equals(b)


def test_placebos_mix_fast_and_slow_noise(panel):
    lf, names = add_placebo_signals(panel.lazy(), 4, noise_days=(1, 20))
    out = lf.collect()
    def lag1(name):
        s = out.filter(pl.col('ticker') == 'T7').sort('date')[name].drop_nulls().to_numpy()
        return np.corrcoef(s[1:], s[:-1])[0, 1]
    assert abs(lag1(names[0])) < 0.15                # 1 day: new noise every date
    assert lag1(names[1]) > 0.85                     # 20 days: slow
    assert abs(lag1(names[2])) < 0.15 and lag1(names[3]) > 0.85


def test_noise_days_must_be_a_positive_integer(panel):
    with pytest.raises(ValueError, match='noise_days'):
        add_cheat_signals(panel.lazy(), [(20, 0.05)], noise_days=0)
    with pytest.raises(ValueError, match='noise_days'):
        add_placebo_signals(panel.lazy(), 2, noise_days=(5, 2.5))