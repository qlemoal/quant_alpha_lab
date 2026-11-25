'''
Tests for src/evaluation/portfolio/attribution.py: the alpha regression and the yearly table.

The HAC (Newey-West) standard errors are checked against a slow, plain double-loop implementation.
'''

import numpy as np
import pytest

from src.evaluation.portfolio.attribution import alpha_regression, yearly_table


def simulate(n=2000, alpha=0.0004, beta=0.3, seed=0, phi=0.0):
    rng = np.random.default_rng(seed)
    mkt = rng.normal(0.0003, 0.01, n)
    e = np.zeros(n)
    shocks = rng.normal(0, 0.008, n)
    for t in range(1, n):
        e[t] = phi * e[t - 1] + shocks[t]
    return alpha + beta * mkt + e, mkt


def slow_hac_alpha_t(y, x, L):
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    u = y - X @ b
    S = np.zeros((2, 2))
    for k in range(0, L + 1):
        w = 1.0 if k == 0 else 1.0 - k / (L + 1.0)
        for t in range(k, n):
            g = np.outer(X[t] * u[t], X[t - k] * u[t - k])
            S += w * g if k == 0 else w * (g + g.T)
    V = XtXi @ S @ XtXi
    return b[0] / np.sqrt(V[0, 0]), b[1] / np.sqrt(V[1, 1])


def test_recovers_beta_and_alpha():
    y, x = simulate()
    r = alpha_regression(y, x)
    assert r['beta'] == pytest.approx(0.3, abs=0.03)
    assert r['alpha_daily'] == pytest.approx(0.0004, abs=0.0004)
    assert r['alpha_ann'] == pytest.approx(r['alpha_daily'] * 252)


def test_tstats_match_the_slow_implementation():
    y, x = simulate(n=400, phi=0.3, seed=3)
    r = alpha_regression(y, x, max_lag=6)
    ta, tb = slow_hac_alpha_t(y, x, 6)
    assert r['alpha_t'] == pytest.approx(ta, rel=1e-9)
    assert r['beta_t'] == pytest.approx(tb, rel=1e-9)


def test_autocorrelation_widens_the_standard_error():
    '''Same data generator, positive autocorrelation of the errors: the HAC t must be smaller than the plain OLS t.'''
    y, x = simulate(n=3000, phi=0.5, seed=4)
    hac = alpha_regression(y, x, max_lag=20)['alpha_t']
    plain = alpha_regression(y, x, max_lag=0)['alpha_t']
    assert abs(hac) < abs(plain)


def test_pure_market_exposure_has_no_alpha():
    rng = np.random.default_rng(5)
    x = rng.normal(0.0004, 0.01, 3000)
    y = 0.8 * x + rng.normal(0, 0.002, 3000)
    r = alpha_regression(y, x)
    assert r['beta'] == pytest.approx(0.8, abs=0.02)
    assert abs(r['alpha_t']) < 3 and r['r2'] > 0.9


def test_nan_rows_are_dropped_and_short_input_is_reported():
    y, x = simulate(n=200)
    y[5], x[7] = np.nan, np.nan
    assert alpha_regression(y, x)['n'] == 198
    assert 'note' in alpha_regression(y[:10], x[:10])
    assert 'note' in alpha_regression(y, np.zeros(200))


def test_yearly_table():
    dates = np.arange(np.datetime64('2020-01-01'), np.datetime64('2022-01-01'))          # 731 days
    dates = dates[np.is_busday(dates)]
    rng = np.random.default_rng(6)
    g = rng.normal(0.0004, 0.005, len(dates))
    c = np.full(len(dates), 0.0001)
    m = rng.normal(0, 0.01, len(dates))
    tdates = np.array([np.datetime64('2020-03-02'), np.datetime64('2020-09-01'), np.datetime64('2021-05-03')])
    rows = yearly_table(dates, g, c, m, tdates, np.array([0.2, 0.3, 0.4]))
    assert [r['year'] for r in rows] == [2020, 2021]
    assert sum(r['n_days'] for r in rows) == len(dates)
    assert rows[0]['turnover'] == pytest.approx(0.5) and rows[1]['turnover'] == pytest.approx(0.4)
    r0 = rows[0]
    s = np.array([d.astype('datetime64[Y]').astype(int) + 1970 for d in dates]) == 2020
    assert r0['net_return'] == pytest.approx(np.prod(1 + g[s] - c[s]) - 1)
    assert r0['cost_bps'] == pytest.approx(c[s].sum() * 1e4)
    assert r0['sharpe_gross'] > r0['sharpe_net']
