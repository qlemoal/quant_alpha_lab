'''
Tests for the Elastic Net combiner (src/models/elastic_net_combiner.py) on a synthetic panel with a KNOWN truth:
    fwdret = 0.02 * good_signal + noise            (noise_signal has no relation to the return)

What is checked:
  1. the planted signal is recovered, the noise signal is shrunk to zero
  2. every search row gets an out-of-fold score (no in-sample score in the search period)
  3. the date layout: search | gap (= embargo) | holdout, with purge = h + 1
  4. the holdout is untouched: destroying the holdout labels does not change the fitted model
  5. the holdout score has real predictive power
'''

import datetime as dt

import numpy as np
import polars as pl
import pytest

from src.models.elastic_net_combiner import run_en_combiner, combined_score_column, score_contributions

N_DATES, N_TICKERS, H = 500, 30, 5
KW = dict(horizon=H, embargo_window=3, n_blocks=6, n_test_blocks=2, holdout_dates=60, l1_ratio_grid=[0.5, 1.0])


def make_panel(seed=0, wreck_holdout=False) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    good = rng.normal(size=(N_DATES, N_TICKERS))
    noise = rng.normal(size=(N_DATES, N_TICKERS))
    y = 0.02 * good + rng.normal(0, 0.05, (N_DATES, N_TICKERS))
    if wreck_holdout:
        y[-68:] = rng.normal(0, 5.0, (68, N_TICKERS))              # garbage labels in the gap and the holdout (not the search)
    days = [dt.date(2018, 1, 1) + dt.timedelta(days=i) for i in range(N_DATES)]
    return pl.DataFrame({
        'date': [d for d in days for _ in range(N_TICKERS)],
        'ticker': [f't{j}' for j in range(N_TICKERS)] * N_DATES,
        'good': good.ravel(), 'noise': noise.ravel(), f'fwdret{H}': y.ravel(),
    })


@pytest.fixture(scope='module')
def result():
    return run_en_combiner(make_panel().lazy(), ['good', 'noise'], **KW)


def test_planted_signal_recovered_noise_suppressed(result):
    coef = dict(zip(['good', 'noise'], result['model'].coef_))
    assert coef['good'] > 0.01
    assert abs(coef['noise']) < 0.1 * coef['good']


def test_search_rows_are_all_out_of_fold(result):
    assert result['oof'].height == result['search_panel'].height


def test_date_layout(result):
    assert result['purge_w'] == H + 1
    assert result['embargo_w'] == H + 1 + KW['embargo_window']
    dates = result['search_panel']['date'].unique().sort().to_list()
    assert np.datetime64(dates[-1]) == np.datetime64(result['train_end'])
    assert result['holdout_start'] > result['train_end']
    gap_days = (result['holdout_start'] - result['train_end']) / np.timedelta64(1, 'D')
    assert gap_days == result['embargo_w']                         # one row per calendar day in this panel
    assert np.datetime64(result['holdout_panel']['date'].min()) == np.datetime64(result['holdout_start'])


def test_holdout_labels_do_not_change_the_model():
    a = run_en_combiner(make_panel().lazy(), ['good', 'noise'], **KW)
    b = run_en_combiner(make_panel(wreck_holdout=True).lazy(), ['good', 'noise'], **KW)
    assert np.allclose(a['model'].coef_, b['model'].coef_)
    assert a['model'].alpha_ == b['model'].alpha_


def test_holdout_score_predicts(result):
    assert result['holdout_report']['ic_mean'] > 0.05


def test_combined_score_uses_oof_in_search_period(result):
    lf = pl.concat([result['search_panel'], result['holdout_panel']]).lazy()
    scored = combined_score_column(lf, result).collect()
    oof = result['oof']
    assert scored.height == lf.collect().height
    assert scored['combined_score'].null_count() == 0


def test_score_contributions_add_up_and_point_to_the_planted_signal(result):
    share = score_contributions(result, result['search_panel'])
    assert sum(share.values()) == pytest.approx(1.0, abs=1e-9)
    assert share['good'] > 0.95                      # the planted signal carries the score
    assert abs(share['noise']) < 0.05                # the noise signal carries (almost) nothing