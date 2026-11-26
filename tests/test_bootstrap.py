'''
Tests for the bootstrap test and Romano-Wolf (src/evaluation/signals/bootstrap.py),
and for the simulated IC series (src/evaluation/signals/calibration.py).

  block length    1. no memory -> short block      2. long memory -> long block
  resampling      3. indices in range, mean block length as asked     4. align_ic_series puts NaN where a date is missing
                  4b. batch-means se equals the plain se for independent data
  one test        5. SIZE: true IC = 0 -> about 5% rejections        6. POWER: a real IC is found
                  7. a wrong-sign signal is not significant (one-sided)  8. no prior -> two-sided
  Romano-Wolf     9. FWER: 10 correlated null signals -> about 5% runs with ANY rejection
                  10. adjusted p >= single p                          11. a real signal survives among nulls
  simulation      12. the simulated series has the asked mean, std and memory
  comparison      13. Newey-West over-rejects on slow signals, the bootstrap does not (the reason for this module)
'''

import numpy as np
import pytest

from src.evaluation.signals.bootstrap import (
    align_ic_series, politis_white_block_length, stationary_bootstrap_indices, bootstrap_test_table, batch_means_se,
)
from src.evaluation.signals.calibration import simulate_ic_series, rejection_rates, p_bootstrap, p_newey_west

H = 20


def _series(ic_list):
    return {f's{i}': (list(range(len(x))), x) for i, x in enumerate(ic_list)}


# ----------------------------------------------------------------------------- block length

def test_block_length_is_short_without_memory():
    x = np.random.default_rng(0).normal(size=3000)
    assert politis_white_block_length(x) < 5


def test_block_length_is_long_with_memory():
    x = simulate_ic_series(3000, H, persistence=20, rng=np.random.default_rng(0))
    assert politis_white_block_length(x) > 2 * H


# ----------------------------------------------------------------------------- resampling

def test_indices_are_in_range_with_the_asked_mean_block():
    idx = stationary_bootstrap_indices(5000, 25.0, 20, seed=1)
    assert idx.shape == (20, 5000) and idx.min() >= 0 and idx.max() < 5000
    jumps = np.mean(np.diff(idx, axis=1) != 1)        # a jump starts a new block (wrap-around counts as one, rare)
    assert 1 / jumps == pytest.approx(25.0, rel=0.1)


def test_align_puts_nan_on_missing_dates():
    dates, names, M = align_ic_series({'a': ([1, 2, 3], [0.1, 0.2, 0.3]), 'b': ([2, 3, 4], [1.0, 2.0, 3.0])})
    assert list(dates) == [1, 2, 3, 4] and names == ['a', 'b']
    assert np.isnan(M[3, 0]) and np.isnan(M[0, 1]) and M[2, 1] == 2.0


def test_batch_means_se_matches_plain_se_without_memory():
    x = np.random.default_rng(9).normal(size=(40_000, 1))
    assert batch_means_se(x, 100)[0] == pytest.approx(x.std() / np.sqrt(len(x)), rel=0.15)


# ----------------------------------------------------------------------------- one test

def test_size_is_about_five_percent():
    rate = rejection_rates(200, 1500, H, 0.0, persistence=5, seed=11, tests={'b': p_bootstrap})['b']
    assert 0.01 <= rate <= 0.10


def test_power_finds_a_real_signal():
    rate = rejection_rates(60, 2500, H, 0.05, persistence=5, seed=12, tests={'b': p_bootstrap})['b']
    assert rate > 0.8


def test_wrong_sign_is_not_significant():
    ic = simulate_ic_series(2500, H, mean_ic=0.04, rng=np.random.default_rng(3))
    tab = bootstrap_test_table(_series([ic]), {'s0': -1}, H, n_boot=199)
    assert tab['signals']['s0']['p_single'] > 0.5


def test_no_prior_is_two_sided():
    ic = simulate_ic_series(2500, H, mean_ic=-0.04, rng=np.random.default_rng(4))
    tab = bootstrap_test_table(_series([ic]), {'s0': None}, H, n_boot=199)
    s = tab['signals']['s0']
    assert s['sided'] == 'two-sided' and s['p_single'] < 0.05


# ----------------------------------------------------------------------------- Romano-Wolf

def _correlated_nulls(rng, k=10, n=1500, rho=0.6, mean_first=0.0):
    common = simulate_ic_series(n, H, rng=rng)
    out = []
    for i in range(k):
        own = simulate_ic_series(n, H, rng=rng)
        x = np.sqrt(rho) * common + np.sqrt(1 - rho) * own
        out.append(x + (mean_first if i == 0 else 0.0))
    return out


def test_romano_wolf_controls_the_family_wise_error():
    rng = np.random.default_rng(5)
    n_runs, any_reject = 100, 0
    for r in range(n_runs):
        tab = bootstrap_test_table(_series(_correlated_nulls(rng)), {f's{i}': +1 for i in range(10)}, H,
                                   n_boot=199, seed=r)
        any_reject += any(v['p_rw'] < 0.05 for v in tab['signals'].values())
    assert any_reject / n_runs <= 0.12


def test_adjusted_p_is_never_below_single_p():
    tab = bootstrap_test_table(_series(_correlated_nulls(np.random.default_rng(6))), {}, H, n_boot=199)
    assert all(v['p_rw'] >= v['p_single'] - 1e-12 for v in tab['signals'].values())


def test_a_real_signal_survives_among_nulls():
    sig = _series(_correlated_nulls(np.random.default_rng(7), mean_first=0.10, n=2500))
    tab = bootstrap_test_table(sig, {f's{i}': +1 for i in range(10)}, H, n_boot=499)
    assert tab['signals']['s0']['p_rw'] < 0.05


# ----------------------------------------------------------------------------- simulation

def test_simulated_series_has_the_asked_mean_std_and_memory():
    x = simulate_ic_series(200_000, H, mean_ic=0.02, ic_std=0.15, persistence=10, rng=np.random.default_rng(8))
    assert x.mean() == pytest.approx(0.02, abs=0.003) and x.std() == pytest.approx(0.15, rel=0.03)
    acf = lambda k: np.corrcoef(x[:-k], x[k:])[0, 1]
    assert acf(H // 2) > 0.3 and abs(acf(H + 10 + 5)) < 0.02


# ----------------------------------------------------------------------------- the reason for this module

def test_newey_west_over_rejects_on_slow_signals_and_the_bootstrap_does_not():
    rates = rejection_rates(150, 2500, H, 0.0, persistence=40, seed=13,
                            tests={'nw': p_newey_west, 'boot': p_bootstrap})
    assert rates['nw'] > 0.10 and rates['boot'] < rates['nw']
