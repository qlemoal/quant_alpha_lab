'''
Tests for the point-in-time universe, the dead-ticker handling and the survivorship audit.

  1. dead_tickers finds the tickers that stop early
  2. rule_universe: top N by dollar volume, no look-ahead, dead names leave, history and ADV floors
  3. membership: events -> intervals -> per-date mask, validation of bad tables
  4. terminal labels: a dying stock keeps its crash in the labels; alive stocks keep nulls; no cross-ticker leak
  5. apply_universe filters rows
  6. audit verdicts
'''

import datetime as dt

import numpy as np
import polars as pl
import pytest

from src.features.returns import add_fwdret_horizon
from src.signals.signal_pipeline import add_dead_flag, add_labels, apply_universe, check_labels
from src.universe.point_in_time import (
    audit_universe, dead_tickers, membership_from_events, membership_mask, rule_universe, validate_membership,
)

D0 = dt.date(2020, 1, 1)


def _prices(spec: dict, n_days=400):
    '''spec: {ticker: (n_rows, volume_per_day)}. close_raw = 10, so dollar volume = 10 x volume.'''
    dates = [D0 + dt.timedelta(days=i) for i in range(n_days)]
    rows = []
    for tk, (n, vol) in spec.items():
        for i in range(n):
            rows.append((tk, dates[i], 10.0, float(vol), 10.0, 10.0))
    return pl.DataFrame(rows, schema=['ticker', 'date', 'close_raw', 'volume', 'close', 'open_adj'], orient='row'), dates


# ---------------------------------------------------------------- 1. dead tickers
def test_dead_tickers():
    df, _ = _prices({'ALIVE': (400, 1e5), 'DEAD': (200, 1e5), 'LATE': (395, 1e5)})
    d = dead_tickers(df, tolerance_days=10).to_dicts()
    flags = {r['ticker']: r['is_dead'] for r in d}
    assert flags == {'ALIVE': False, 'DEAD': True, 'LATE': False}      # LATE ends 4 days early: inside the tolerance


# ---------------------------------------------------------------- 2. rule universe
def test_rule_universe_takes_the_most_liquid():
    spec = {f'T{k:02d}': (400, 1e5 * (k + 1)) for k in range(30)}
    df, dates = _prices(spec)
    u = rule_universe(df, top_n=10, adv_window=20, min_adv=1.0, min_history=30, reconstitute_every=10)
    members = u.filter((pl.col('date') == dates[200]) & pl.col('in_universe'))['ticker'].sort().to_list()
    assert members == [f'T{k:02d}' for k in range(20, 30)]


def test_rule_universe_has_no_look_ahead():
    '''A stock that becomes liquid on day 250 is not a member on any earlier day.'''
    dates = [D0 + dt.timedelta(days=i) for i in range(400)]
    rows = []
    for k in range(12):
        for i in range(400):
            vol = 1e5 * (k + 1)
            if k == 0 and i >= 250:
                vol = 1e9                                           # becomes the most liquid stock on day 250
            rows.append((f'T{k:02d}', dates[i], 10.0, vol))
    df = pl.DataFrame(rows, schema=['ticker', 'date', 'close_raw', 'volume'], orient='row')
    u = rule_universe(df, top_n=3, adv_window=20, min_adv=1.0, min_history=30, reconstitute_every=5)
    t0 = u.filter(pl.col('ticker') == 'T00').sort('date')
    before = t0.filter(pl.col('date') < dates[250])
    assert not before['in_universe'].any()
    # it enters at the first re-ranking that can SEE the jump (day 250 itself: the average includes day 250's close)
    first_in = t0.filter(pl.col('in_universe'))['date'].min()
    assert first_in == dates[250]
    assert t0.filter(pl.col('date') == dates[399])['in_universe'][0]


def test_rule_universe_dead_names_leave_and_floors_apply():
    spec = {'BIG_DEAD': (200, 1e7), 'BIG_ALIVE': (400, 5e6), 'SMALL': (400, 10.0), 'NEW': (60, 1e8)}
    df, dates = _prices(spec)
    u = rule_universe(df, top_n=3, adv_window=20, min_adv=1e4, min_history=126, reconstitute_every=10)
    day = u.filter(pl.col('date') == dates[150])
    assert set(day.filter(pl.col('in_universe'))['ticker']) == {'BIG_DEAD', 'BIG_ALIVE'}     # SMALL: ADV floor. NEW: no row yet
    late = u.filter(pl.col('date') == dates[300])
    assert set(late.filter(pl.col('in_universe'))['ticker']) == {'BIG_ALIVE'}                # BIG_DEAD has no row
    new_rows = u.filter(pl.col('ticker') == 'NEW')
    assert not new_rows['in_universe'].any()                                                 # history floor (60 < 126 rows)


def test_rule_universe_input_errors():
    df, _ = _prices({'A': (100, 1e5)})
    with pytest.raises(ValueError):
        rule_universe(df.drop('close_raw'))
    with pytest.raises(ValueError):
        rule_universe(df, top_n=0)


# ---------------------------------------------------------------- 3. membership
def _events():
    return pl.DataFrame({
        'date': [dt.date(2020, 3, 1), dt.date(2020, 6, 1), dt.date(2020, 9, 1), dt.date(2020, 2, 1)],
        'ticker': ['A', 'A', 'A', 'B'],
        'action': ['added', 'removed', 'added', 'removed'],
    })


def test_membership_from_events():
    m = membership_from_events(_events(), first_date=dt.date(2019, 1, 1)).sort(['ticker', 'start'])
    rows = m.to_dicts()
    assert rows == [
        {'ticker': 'A', 'start': dt.date(2020, 3, 1), 'end': dt.date(2020, 6, 1)},
        {'ticker': 'A', 'start': dt.date(2020, 9, 1), 'end': None},
        {'ticker': 'B', 'start': dt.date(2019, 1, 1), 'end': dt.date(2020, 2, 1)},     # removed with no add: member since the start
    ]


def test_membership_mask_is_closed_on_both_ends():
    m = membership_from_events(_events(), first_date=dt.date(2019, 1, 1))
    keys = pl.DataFrame({'date': [dt.date(2020, 3, 1), dt.date(2020, 6, 1), dt.date(2020, 6, 2), dt.date(2020, 10, 1)],
                         'ticker': ['A'] * 4})
    out = membership_mask(keys, m).sort('date')
    assert out['in_universe'].to_list() == [True, True, False, True]


def test_membership_validation():
    bad_order = pl.DataFrame({'ticker': ['A'], 'start': [dt.date(2020, 5, 1)], 'end': [dt.date(2020, 1, 1)]})
    with pytest.raises(ValueError):
        validate_membership(bad_order)
    overlap = pl.DataFrame({'ticker': ['A', 'A'], 'start': [dt.date(2020, 1, 1), dt.date(2020, 3, 1)],
                            'end': [dt.date(2020, 5, 1), None]})
    with pytest.raises(ValueError):
        validate_membership(overlap)
    with pytest.raises(ValueError):
        validate_membership(pl.DataFrame({'ticker': ['A']}))


# ---------------------------------------------------------------- 4. terminal labels
def _two(h=5):
    dates = [D0 + dt.timedelta(days=i) for i in range(60)]
    rows = [('ALIVE', dates[i], 100.0 + i, 100.0 + i) for i in range(60)]
    rows += [('DEAD', dates[i], 100.0, 100.0) for i in range(30)]
    rows[-1] = ('DEAD', dates[29], 100.0, 50.0)                    # the delisting return is in the last adjusted close
    return pl.DataFrame(rows, schema=['ticker', 'date', 'open_adj', 'close'], orient='row')


def test_terminal_labels_keep_the_crash():
    lf = add_dead_flag(_two().lazy())
    lf, _ = add_labels(lf, [5], terminal=True)
    dead = lf.collect().filter(pl.col('ticker') == 'DEAD').sort('date')
    assert dead['is_dead'].all()
    # t = day 24: entry = open day 25 (100), exit would be day 30 (missing) -> last close 50
    assert dead['fwdret5'][24] == pytest.approx(np.log(50 / 100))
    assert dead['fwdret5'][23] == pytest.approx(0.0)                # exit day 29 exists, open 100
    assert dead['fwdret5'][29] is None                              # no entry price after the last row


def test_terminal_off_keeps_the_old_behaviour_and_alive_names_keep_nulls():
    lf = add_dead_flag(_two().lazy())
    off = add_fwdret_horizon(lf, 5).collect().sort(['ticker', 'date'])
    assert off.filter(pl.col('ticker') == 'DEAD')['fwdret5'][24] is None
    on, _ = add_labels(lf, [5], terminal=True)
    alive = on.collect().filter(pl.col('ticker') == 'ALIVE').sort('date')
    assert alive['fwdret5'].tail(6).is_null().all()                 # alive: the last h+1 rows stay null
    assert not alive['is_dead'].any()


def test_check_labels_accepts_terminal_values_only_for_dead_tickers():
    lf, _ = add_labels(add_dead_flag(_two().lazy()), [5], terminal=True)
    df = lf.collect().select(['ticker', 'date', 'fwdret5'])
    with pytest.raises(ValueError):
        check_labels(df, [5])                                       # without the exemption it looks like a leak
    rep = check_labels(df, [5], dead_tickers={'DEAD'})
    assert rep['fwdret5']['n_abs_above_2'] == 0


# ---------------------------------------------------------------- 5. apply_universe
def test_apply_universe_filters_rows(tmp_path):
    lf = pl.DataFrame({'date': [D0, D0, D0 + dt.timedelta(days=1)], 'ticker': ['A', 'B', 'A'], 'x': [1.0, 2.0, 3.0]}).lazy()
    uni = pl.DataFrame({'date': [D0, D0], 'ticker': ['A', 'B'], 'in_universe': [True, False]})   # day 2: no entry
    f = tmp_path / 'universe.parquet'
    uni.write_parquet(f)
    out, info = apply_universe(lf, f)
    got = out.collect()
    assert got['x'].to_list() == [1.0]                              # B is out; (day 2, A) has no entry -> not kept
    assert 'in_universe' not in got.columns


# ---------------------------------------------------------------- 6. audit
def _audit_frame(n_dead, n_alive=40, years=12, crash=True):
    rows, per = [], 250 * years
    dates = [D0 + dt.timedelta(days=int(i * 365.25 / 250)) for i in range(per)]
    for k in range(n_alive):
        rows += [(f'A{k}', d, 10.0, 1e5, 10.0) for d in dates]
    rng = np.random.default_rng(0)
    for k in range(n_dead):
        end = int(rng.integers(300, per - 300))
        for i in range(end):
            rows.append((f'D{k}', dates[i], 10.0, 1e5, 10.0))
        if crash:
            rows[-1] = (f'D{k}', dates[end - 1], 10.0, 1e5, 5.0)
    return pl.DataFrame(rows, schema=['ticker', 'date', 'close_raw', 'volume', 'close'], orient='row')


def test_audit_fails_on_survivors_only():
    a = audit_universe(_audit_frame(n_dead=0))
    assert a['verdict'] == 'FAIL' and a['n_dead'] == 0


def test_audit_warns_on_very_few_dead_names():
    a = audit_universe(_audit_frame(n_dead=2, n_alive=60))
    assert a['verdict'] == 'WARN'


def test_audit_ok_on_plausible_file():
    a = audit_universe(_audit_frame(n_dead=40, n_alive=60))
    assert a['verdict'] == 'OK', a['notes']
    assert a['share_dead'] == pytest.approx(0.4)


def test_audit_warns_when_delisting_returns_are_missing():
    a = audit_universe(_audit_frame(n_dead=40, n_alive=60, crash=False))
    assert a['verdict'] == 'WARN' and any('delisting returns' in n for n in a['notes'])
