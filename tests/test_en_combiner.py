'''
Real pytest assertions (unlike the deleted tests/test_en_combiner.py,
which was a print-only driver, no assert anywhere, despite living in
tests/). Exercises run_en_combiner() end to end: internal fwdret
computation from raw open prices (not a precomputed column, the actual
new behavior worth testing, not the old shortcut), CPCV search, holdout
evaluation, coefficient stability, all through the one orchestrated call.

Synthetic panel with a KNOWN planted relationship, same principle as
test_elastic_net_combiner.py: constructs actual `open` price paths such
that the realized 1-day forward open-to-open return (exactly what
add_fwd_returns() computes) equals TRUE_COEF * good_signal + noise, by
construction, not fit after the fact. good_signal's coefficient recovery,
noise_signal's suppression to ~0, and the holdout report's IC/Sharpe are
all checked against this known ground truth, not just "did it run."
'''

import numpy as np
import polars as pl
import pytest
import datetime

from src.models.elastic_net_combiner import run_en_combiner, coefficient_stability_by_fold


N_DATES = 900
N_TICKERS = 30
TRUE_COEF = 0.8
NOISE_STD = 3.0
EMBARGO_WINDOW = 16  # matches config.constants.EMBARGO_WINDOW at time of writing, not imported directly so this test doesn't silently drift if that constant is retuned without a deliberate check here


def build_synthetic_price_panel(seed: int = 0) -> pl.LazyFrame:
    rng = np.random.default_rng(seed)
    calendar_dates = [datetime.date(2015, 1, 1) + datetime.timedelta(days=i) for i in range(N_DATES)]

    rows_date, rows_ticker, rows_open, rows_good, rows_noise = [], [], [], [], []
    for tkr in range(N_TICKERS):
        good_signal = rng.normal(0, 1, N_DATES)
        # open_logret(s) is realized FROM day s-1 TO day s. add_fwd_returns()
        # computes fwdret(t) = open_logret(t+2) = log(open(t+2)) - log(open(t+1)),
        # so planting the relationship at open_logret(t+2) plants it exactly
        # at fwdret(t), matching the real execution-lag construction in
        # returns.py, not an approximation of it.
        open_logret = np.zeros(N_DATES)
        open_logret[2:] = TRUE_COEF * good_signal[:-2] + rng.normal(0, NOISE_STD, N_DATES - 2)
        open_price = 100 * np.exp(np.cumsum(open_logret))

        rows_date.extend(calendar_dates)
        rows_ticker.extend([f'T{tkr:02d}'] * N_DATES)
        rows_open.extend(open_price)
        rows_good.extend(good_signal)
        rows_noise.extend(rng.normal(0, 1, N_DATES))

    return pl.DataFrame({
        'date': rows_date, 'ticker': rows_ticker, 'open': rows_open,
        'good_signal': rows_good, 'noise_signal': rows_noise,
    }).lazy()


@pytest.fixture(scope='module')
def result():
    lf = build_synthetic_price_panel()
    return run_en_combiner(
        lf, ['good_signal', 'noise_signal'], horizon=1, embargo_window=EMBARGO_WINDOW,
        q_values={'good_signal': 0.01, 'noise_signal': 0.6},
    )


def test_fwdret_computed_internally_not_passed_in(result):
    # the whole point of the refactor: the caller never names or builds
    # fwdret, run_en_combiner derives it from horizon alone
    assert result['fwdret_col'] == 'fwdret'  # horizon=1 -> add_fwd_returns()'s own literal name
    assert 'fwdret' in result['search_panel'].columns
    assert 'fwdret' in result['holdout_panel'].columns


def test_no_manual_dict_patching_needed(result):
    # coefficient_stability_by_fold used to require the caller to manually
    # patch fit_result['fwd_ret_col'] afterward (underscore mismatch bug,
    # now fixed), confirm it just works off what run_en_combiner returns
    stability = coefficient_stability_by_fold(result, result['search_panel'])
    assert stability.height == len(result['cv_folds'])


def test_good_signal_recovered_noise_suppressed(result):
    coefs = dict(zip(result['signal_cols'], result['model'].coef_))
    assert np.sign(coefs['good_signal']) == np.sign(TRUE_COEF)
    assert abs(coefs['good_signal'] - TRUE_COEF) < 0.15  # loose tolerance, regularized estimate, not exact recovery
    assert abs(coefs['noise_signal']) < 0.05


def test_coefficient_stability_consistent_sign_across_folds(result):
    stability = coefficient_stability_by_fold(result, result['search_panel'])
    good_signs = np.sign(stability['good_signal'].to_numpy())
    assert (good_signs == np.sign(TRUE_COEF)).all()


def test_holdout_shows_real_predictive_power(result):
    # genuinely out-of-sample: holdout_panel was never touched during the
    # CPCV search, see run_en_combiner()'s gap_cutoff construction
    report = result['holdout_report']
    assert report['ic_mean'] > 0.1  # strong given the planted relationship, would be near 0 on real markets
    assert report['ic_pval_nw'] < 0.01
    assert report['long_short_sharpe'] > 0


def test_purge_and_embargo_derived_correctly(result):
    assert result['horizon'] == 1
    assert result['purge_w'] == 2  # horizon + 1
    assert result['embargo_w'] == 2 + EMBARGO_WINDOW  # sum convention, methodology.md Section 1.7