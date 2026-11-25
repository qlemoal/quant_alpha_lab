import datetime as dt

import numpy as np

from src.evaluation.signals.yearly import yearly_ic_table


def _dates(n, start=dt.date(2018, 1, 1)):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def test_stable_signal_is_ok_every_year():
    d = _dates(1000)
    ic = np.full(1000, 0.03)
    t = yearly_ic_table({'a': (d, ic)}, {'a': 1})
    assert t['years_ok'][0] == t['n_years'][0] >= 3
    assert t['share_ok'][0] == 1.0


def test_sign_is_aligned_with_prior():
    d = _dates(1000)
    ic = np.full(1000, -0.03)
    assert yearly_ic_table({'a': (d, ic)}, {'a': -1})['share_ok'][0] == 1.0
    assert yearly_ic_table({'a': (d, ic)}, {'a': 1})['share_ok'][0] == 0.0


def test_fading_signal_shows_in_halves():
    d = _dates(1500)
    ic = np.where(np.arange(1500) < 750, 0.05, 0.0) + 1e-6
    t = yearly_ic_table({'a': (d, ic)}, {'a': 1})
    assert t['first_half'][0] > 0.04 > t['second_half'][0] + 0.03


def test_partial_year_skipped():
    d = _dates(300)                       # last year has few dates
    t = yearly_ic_table({'a': (d, np.full(300, 0.01))}, {'a': 1})
    assert all(c.startswith('ic_') or c in ('signal', 'years_ok', 'n_years', 'share_ok', 'first_half', 'second_half') for c in t.columns)
    assert t['n_years'][0] >= 1


def test_missing_values_are_ignored():
    d = _dates(600)
    ic = np.full(600, 0.02)
    ic[::7] = np.nan
    t = yearly_ic_table({'a': (d, ic)}, {'a': None})
    assert t['share_ok'][0] == 1.0
