'''Tests for the comparison rules of src/evaluation/reference.py and the adjusted Rand index.'''

import pytest

from src.evaluation.reference import compare, RULES, adjusted_rand


def golden():
    return {'admitted_signals': ['a'], 'families': ['f'], 'holdout_ic': 0.05, 'avg_n_clusters': 8.0, 'cluster_ari': 0.9,
            'market_mode_removed_share': 1.0, 'beta_exposure_max_abs': 1e-15, 'realised_beta_full': 0.01,
            'sharpe_gross_full': 1.5, 'sharpe_net_full': 0.5, 'annual_cost_bps_full': 680.0, 'alpha_t_full': 1.3}


def test_identical_numbers_pass():
    assert all(r['ok'] for r in compare(golden(), golden()))


def test_each_rule_fails_when_it_should():
    m = golden()
    m['admitted_signals'] = ['b']              # exact
    m['holdout_ic'] = 0.07                     # abs 0.01
    m['beta_exposure_max_abs'] = 1e-3          # max 1e-6
    bad = {r['metric'] for r in compare(m, golden()) if not r['ok']}
    assert bad == {'admitted_signals', 'holdout_ic', 'beta_exposure_max_abs'}


def test_missing_metric_fails():
    m = golden()
    del m['alpha_t_full']
    assert [r['metric'] for r in compare(m, golden()) if not r['ok']] == ['alpha_t_full']


def test_max_rule_uses_absolute_value():
    m = golden()
    m['realised_beta_full'] = -0.04
    assert all(r['ok'] for r in compare(m, golden()))
    m['realised_beta_full'] = -0.06
    assert not all(r['ok'] for r in compare(m, golden()))


def test_every_rule_metric_is_checked():
    assert {r['metric'] for r in compare(golden(), golden())} == set(RULES)


def test_adjusted_rand():
    assert adjusted_rand([0, 0, 1, 1], [5, 5, 9, 9]) == pytest.approx(1.0)       # same partition, other names
    assert adjusted_rand([0, 0, 1, 1], [0, 1, 0, 1]) < 0.1
