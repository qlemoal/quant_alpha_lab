'''
Tests for the family combiner (src/models/family_combiner.py) and the positive / one-SE options of the
Elastic Net (src/models/elastic_net_combiner.py).

Synthetic truth: two momentum-type signals (mom_a, mom_b) are noisy copies of one hidden factor that drives the return.
rev is sign-reversed in the data: its prior sign is -1, so aligned (sign x rev) it predicts positively.
junk has no effect.

What is checked:
  1. group_by_family and add_family_scores: signs applied, mean taken, nulls ignored, missing family -> error
  2. positive=True gives no negative coefficient even when the data push one negative
  3. the one-SE rule never picks a smaller alpha than the minimum-error rule
  4. the drop-one test: removing the real family lowers the OOF IC, removing the junk family does not
'''

import datetime as dt

import numpy as np
import polars as pl
import pytest

from src.models.family_combiner import group_by_family, add_family_scores, family_column, drop_one_oof_ic
from src.models.elastic_net_combiner import run_en_combiner, fit_elastic_net_combiner, apply_one_se_rule

N_DATES, N_TICKERS, H = 500, 30, 5
KW = dict(horizon=H, embargo_window=3, n_blocks=6, n_test_blocks=2, holdout_dates=60, l1_ratio_grid=[0.5, 1.0])


def make_lf(seed=0) -> pl.LazyFrame:
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(N_DATES, N_TICKERS))                  # the hidden factor
    mom_a = f + rng.normal(0, 1, f.shape)
    mom_b = f + rng.normal(0, 1, f.shape)
    rev = -f + rng.normal(0, 1, f.shape)                       # sign -1 in the priors
    junk = rng.normal(size=f.shape)
    y = 0.02 * f + rng.normal(0, 0.05, f.shape)
    days = [dt.date(2018, 1, 1) + dt.timedelta(days=i) for i in range(N_DATES)]
    return pl.DataFrame({
        'date': [d for d in days for _ in range(N_TICKERS)],
        'ticker': [f't{j}' for j in range(N_TICKERS)] * N_DATES,
        'mom_a': mom_a.ravel(), 'mom_b': mom_b.ravel(), 'rev': rev.ravel(), 'junk': junk.ravel(),
        f'fwdret{H}': y.ravel(),
    }).lazy()


META = {'mom_a': {'family': 'momentum', 'sign': 1}, 'mom_b': {'family': 'momentum', 'sign': 1},
        'rev': {'family': 'reversal', 'sign': -1}, 'junk': {'family': 'noise', 'sign': 1}}


def test_group_by_family_is_sorted_and_signed():
    fam = group_by_family(META)
    assert list(fam) == ['momentum', 'noise', 'reversal']
    assert fam['momentum'] == [('mom_a', 1), ('mom_b', 1)]
    assert fam['reversal'] == [('rev', -1)]


@pytest.mark.parametrize('bad', [{'x': None}, {'x': {'family': None, 'sign': 1}}, {'x': {'family': 'f', 'sign': 0}}])
def test_group_by_family_rejects_missing_prior(bad):
    with pytest.raises(ValueError):
        group_by_family(bad)


def test_family_score_applies_sign_and_mean():
    lf = pl.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0], 'r': [1.0, -2.0]}).lazy()
    out = add_family_scores(lf, {'m': [('a', 1), ('b', 1)], 'rv': [('r', -1)]}).collect()
    assert out[family_column('m')].to_list() == [2.0, 3.0]
    assert out[family_column('rv')].to_list() == [-1.0, 2.0]


def test_family_score_ignores_a_null_member():
    lf = pl.DataFrame({'a': [1.0, None], 'b': [3.0, 4.0]}).lazy()
    out = add_family_scores(lf, {'m': [('a', 1), ('b', 1)]}).collect()
    assert out['fam_m'].to_list() == [2.0, 4.0]


@pytest.fixture(scope='module')
def fam_result():
    lf = add_family_scores(make_lf(), group_by_family(META))
    cols = ['fam_momentum', 'fam_noise', 'fam_reversal']
    return run_en_combiner(lf, cols, positive=True, one_se=True, **KW), cols


def test_positive_coefficients_only(fam_result):
    res, _ = fam_result
    assert (res['model'].coef_ >= 0).all()


def test_planted_family_gets_weight_junk_does_not(fam_result):
    res, cols = fam_result
    coef = dict(zip(cols, res['model'].coef_))
    assert coef['fam_momentum'] > 0 and coef['fam_reversal'] > 0
    assert coef['fam_noise'] <= 0.1 * max(coef['fam_momentum'], coef['fam_reversal'])


def test_positive_blocks_a_negative_weight():
    '''A column that is anti-correlated with the target must get 0, not a negative weight.'''
    lf = make_lf()
    lf = lf.with_columns((-pl.col('mom_a')).alias('anti'))
    res = run_en_combiner(lf, ['anti', 'junk'], positive=True, **KW)
    assert (res['model'].coef_ >= 0).all()
    assert res['model'].coef_[0] == 0.0
    free = run_en_combiner(lf, ['anti', 'junk'], positive=False, **KW)
    assert free['model'].coef_[0] != 0.0                        # without the constraint it would use the sign


def test_one_se_alpha_is_at_least_the_best_alpha():
    from sklearn.linear_model import ElasticNetCV
    rng = np.random.default_rng(5)
    X = rng.normal(size=(600, 4))
    y = 0.3 * X[:, 0] + rng.normal(size=600)
    folds = [(np.setdiff1d(np.arange(600), np.arange(i, 600, 5)), np.arange(i, 600, 5)) for i in range(5)]
    m = ElasticNetCV(l1_ratio=[0.5, 1.0], cv=folds).fit(X, y)
    a_min = m.alpha_
    a_se = apply_one_se_rule(m, X, y, [0.5, 1.0], cv_folds=folds)
    assert a_se >= a_min and m.alpha_ == a_se
    assert abs(m.coef_[0]) <= 0.3 + 1e-6


def test_drop_one_ic_real_family_matters_junk_does_not(fam_result):
    res, cols = fam_result
    d = drop_one_oof_ic(res['search_panel'], res)
    assert d['fam_momentum']['delta'] > 0
    assert d['fam_reversal']['delta'] > 0
    assert d['fam_noise']['delta'] <= 0.002                    # the junk family adds (almost) nothing
    assert d['fam_momentum']['ic_full'] == pytest.approx(d['fam_reversal']['ic_full'])


def test_drop_one_with_one_column_gives_nan():
    lf = add_family_scores(make_lf(), {'momentum': [('mom_a', 1)]})
    res = run_en_combiner(lf, ['fam_momentum'], positive=True, **KW)
    d = drop_one_oof_ic(res['search_panel'], res)
    assert np.isnan(d['fam_momentum']['delta'])


def test_one_se_keeps_a_weak_real_signal_and_zeroes_pure_noise():
    '''The MSE version zeroed a weak real signal. The correlation version must keep it, and drop noise.'''
    rng = np.random.default_rng(7)
    n = 20000
    folds = [(np.setdiff1d(np.arange(n), np.arange(i, n, 5)), np.arange(i, n, 5)) for i in range(5)]
    from sklearn.linear_model import ElasticNetCV
    x = rng.normal(size=(n, 1))
    y = 0.03 * x[:, 0] + rng.normal(size=n)                       # correlation about 0.03
    m = ElasticNetCV(l1_ratio=[0.5], cv=folds, positive=True).fit(x, y)
    apply_one_se_rule(m, x, y, [0.5], positive=True, cv_folds=folds)
    assert m.coef_[0] > 0
    y0 = rng.normal(size=n)
    m0 = ElasticNetCV(l1_ratio=[0.5], cv=folds, positive=True).fit(x, y0)
    apply_one_se_rule(m0, x, y0, [0.5], positive=True, cv_folds=folds)
    assert m0.coef_[0] == 0.0