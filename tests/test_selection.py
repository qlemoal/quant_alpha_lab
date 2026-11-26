'''
Tests for the strict selection: priors, deflated Sharpe, gates.

What is checked:
  1. priors: names map to the right family and sign; unknown names have no prior
  2. DSR: N = 1 equals PSR; more trials raise SR0; a Monte Carlo of skill-less trials shows that the
     best raw Sharpe is often "significant" while the DSR of the best is rarely above 0.95
  3. gates (prior, sign, Romano-Wolf with earlier trials): each gate stops a signal built to fail only that gate;
     a missing statistic fails; the cross-checks (Newey-West, q-value, DSR) decide nothing
'''

import numpy as np
import pytest
from scipy.stats import norm

from src.signals.priors import prior_for, strip_transform, priors_table
from src.evaluation.signals.dsr import (
    sharpe_stats, expected_max_sharpe, probabilistic_sharpe, deflated_sharpe, deflated_sharpe_table,
)
from src.evaluation.signals.selection import SelectionConfig, apply_gates, gate_summary, GATES


# ----------------------------------------------------------------------------- priors

@pytest.mark.parametrize('name, family, sign', [
    ('mom5_zscore_tanh', 'reversal', -1),
    ('mom20_zscore_tanh', 'reversal', -1),
    ('mom60_zscore_tanh', 'momentum', +1),
    ('mom252_rank', 'momentum', +1),
    ('vol60_zscore_clip', 'low_volatility', -1),
    ('log_adv20_zscore_tanh', 'illiquidity', -1),
    ('beta60_rank', 'low_beta', -1),
    ('cheat_h20_ic030', 'cheat', +1),
    ('placebo_007', 'placebo', +1),
])
def test_prior_for(name, family, sign):
    p = prior_for(name)
    assert p.family == family and p.sign == sign and p.rationale and p.reference


def test_unknown_signal_has_no_prior():
    assert prior_for('size_rank') is None
    assert priors_table(['size_rank'])['size_rank'] is None


def test_strip_transform():
    assert strip_transform('log_adv20_zscore_tanh') == 'log_adv20'
    assert strip_transform('mom20') == 'mom20'


# ----------------------------------------------------------------------------- DSR

def test_one_trial_dsr_equals_psr():
    assert expected_max_sharpe(1, 0.01) == 0.0
    assert deflated_sharpe(0.2, 100, 0.0, 3.0, n_trials=1, var_sr=0.01) == pytest.approx(
        probabilistic_sharpe(0.2, 100, 0.0, 3.0, 0.0))


def test_psr_matches_normal_formula():
    # skew 0, kurt 3: PSR = Phi((SR - SR0) sqrt(T-1) / sqrt(1 + SR^2 / 2))
    sr, n = 0.15, 200
    assert probabilistic_sharpe(sr, n, 0.0, 3.0, 0.05) == pytest.approx(
        norm.cdf((sr - 0.05) * np.sqrt(n - 1) / np.sqrt(1 + sr ** 2 / 2)))


def test_more_trials_raise_the_bar():
    assert expected_max_sharpe(10, 0.01) < expected_max_sharpe(100, 0.01) < expected_max_sharpe(1000, 0.01)


def test_sharpe_stats_step_and_bad_input():
    x = np.arange(100, dtype=float)
    assert sharpe_stats(x, step=10)['n'] == 10
    assert np.isnan(sharpe_stats([1.0, 1.0, 1.0, 1.0, 1.0])['sr'])
    assert np.isnan(sharpe_stats([1.0, 2.0])['sr'])


def test_dsr_rejects_the_best_of_many_skill_less_trials():
    '''100 skill-less series of 250 points. The best raw Sharpe usually looks significant (PSR vs 0);
    its DSR must be below 0.95 in about 95% of the draws.'''
    rng = np.random.default_rng(1)
    n_draws, n_trials, T = 200, 100, 250
    raw_hits = dsr_hits = 0
    for _ in range(n_draws):
        series = {f's{i}': rng.normal(size=T) for i in range(n_trials)}
        tab = deflated_sharpe_table(series, {}, step=1)
        best = max(tab.values(), key=lambda d: d['sr'])
        raw_hits += probabilistic_sharpe(best['sr'], best['n'], best['skew'], best['kurt'], 0.0) >= 0.95
        dsr_hits += best['dsr'] >= 0.95
    assert raw_hits / n_draws > 0.8            # without deflation the best chance result looks real
    assert dsr_hits / n_draws < 0.15           # with deflation it almost never does


def test_dsr_keeps_a_true_signal():
    rng = np.random.default_rng(2)
    series = {f's{i}': rng.normal(size=500) for i in range(30)}
    series['real'] = rng.normal(0.35, 1.0, size=500)          # true Sharpe 0.35 per observation
    tab = deflated_sharpe_table(series, {}, step=1)
    assert tab['real']['dsr'] > 0.95


def test_null_variance_mode_is_less_strict_than_trials_mode():
    rng = np.random.default_rng(4)
    series = {f's{i}': rng.normal(size=400) for i in range(30)}
    series['real'] = rng.normal(0.25, 1.0, size=400)
    null = deflated_sharpe_table(series, {}, step=1, var_mode='null')
    trials = deflated_sharpe_table(series, {}, step=1, var_mode='trials')
    assert null['real']['sr0'] < trials['real']['sr0']
    assert null['real']['sr0'] == pytest.approx(expected_max_sharpe(31, 1 / 399))
    with pytest.raises(ValueError):
        deflated_sharpe_table(series, {}, step=1, var_mode='x')


def test_wrong_sign_gives_low_dsr():
    rng = np.random.default_rng(3)
    series = {'a': rng.normal(0.25, 1.0, size=500), 'b': rng.normal(size=500)}
    good = deflated_sharpe_table(series, {'a': +1}, step=1)['a']['dsr']
    bad = deflated_sharpe_table(series, {'a': -1}, step=1)['a']['dsr']
    assert good > 0.9 and bad < 0.01


# ----------------------------------------------------------------------------- gates

def stat(signal='good', sign=-1, family='reversal', mean_ic=-0.03, p_rw=0.001, dsr=0.99, nw=-5.0):
    return dict(signal=signal, sign=sign, family=family, mean_ic=mean_ic, p_single=None if p_rw is None else p_rw / 2, p_rw=p_rw, dsr=dsr,
                nw_tstat=nw, nonoverlap_tstat=-3.0, qvalue=0.001)


def run(stats, **kw):
    return apply_gates(stats, SelectionConfig(**kw)).to_dicts()


def test_good_signal_is_admitted_with_negative_prior():
    row = run([stat()])[0]
    assert row['selected'] and row['failed_gate'] is None


@pytest.mark.parametrize('change, gate', [
    (dict(sign=None, family=None), 'prior'),
    (dict(mean_ic=0.03), 'sign'),                              # right size, wrong direction
    (dict(p_rw=0.06), 'fwer'),
])
def test_each_gate_stops_its_signal(change, gate):
    row = run([stat(**change)])[0]
    assert not row['selected'] and row['failed_gate'] == gate


def test_missing_statistic_fails():
    assert run([stat(p_rw=None)])[0]['failed_gate'] == 'fwer'


def test_cross_checks_decide_nothing():
    """A tiny Newey-West t or a low DSR does not stop a signal: they are printed cross-checks, not gates."""
    assert run([stat(nw=-0.1, dsr=0.1)])[0]['selected']


def test_earlier_trials_scale_the_p_value():
    """4 signals now, 12 tried before: 4 families of 4, so p_all = 4 x p_RW."""
    rows = run([stat(f's{i}', p_rw=0.02) for i in range(4)], n_trials_extra=12)
    assert rows[0]['p_all'] == pytest.approx(0.08) and not rows[0]['selected']
    assert run([stat(f's{i}', p_rw=0.02) for i in range(4)])[0]['selected']
    assert run([stat(p_rw=0.5)], n_trials_extra=100)[0]['p_all'] == 1.0


def test_exploration_mode_skips_prior_and_sign():
    s = stat(sign=None, family=None, mean_ic=0.03)
    assert not run([s])[0]['selected']
    assert run([s], require_prior=False)[0]['selected']


def test_thresholds_are_configurable():
    assert run([stat(p_rw=0.08)], fwer_alpha=0.10)[0]['selected']
    assert not run([stat(p_rw=0.04)], fwer_alpha=0.01)[0]['selected']


def test_gate_summary_counts_first_failure_only():
    tab = apply_gates([stat('a'), stat('b', p_rw=0.5), stat('c', mean_ic=0.03, p_rw=0.5)], SelectionConfig())
    s = gate_summary(tab)
    assert s['admitted'] == 1 and s['fwer'] == 1 and s['sign'] == 1 and s['tested'] == 3
    assert set(GATES) <= set(s)
