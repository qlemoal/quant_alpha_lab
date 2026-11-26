'''
Tests for the delistings of the synthetic market (src/data/synthetic.py, delist_rate > 0).

  1. the default market has no dead stock (so the reference run is unchanged)
  2. a dead stock has no data after its last day; the delisting return is in the adjusted close of the last row only
  3. delistings follow bad years: dead stocks had a worse past year than the average stock
  4. drop_dead=True writes only the survivors; the files of dead stocks stop at their last day
  5. the same seed gives the same dead stocks
'''

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.synthetic import SyntheticConfig, simulate_market, write_market, market_digest

BASE = SyntheticConfig(n_tickers=120, n_days=700, seed=5)
DEAD = replace(BASE, delist_rate=0.15, delist_sensitivity=2.0)


@pytest.fixture(scope='module')
def mkt():
    return simulate_market(DEAD)


def test_default_has_no_dead_stock():
    m = simulate_market(BASE)
    assert not m['dead'].any() and (m['last_idx'] == BASE.n_days - 1).all()
    assert np.isfinite(m['close']).all()


def test_nothing_after_the_last_day_and_delisting_return_in_adj_close_only(mkt):
    assert mkt['dead'].sum() >= 10
    for j in np.where(mkt['dead'])[0]:
        last = mkt['last_idx'][j]
        assert np.isfinite(mkt['close'][last, j]) and np.isnan(mkt['close'][last + 1:, j]).all()
        assert np.isnan(mkt['volume'][last + 1:, j]).all()
        adj_ret = mkt['adj_close'][last, j] / mkt['adj_close'][last - 1, j] - 1
        price_ret = mkt['close'][last, j] / mkt['close'][last - 1, j] - 1
        total = np.exp(mkt['total_ret'][last, j]) - 1
        assert adj_ret == pytest.approx((1 + total) * (1 + mkt['dlret'][j]) - 1, abs=1e-9)
        assert price_ret == pytest.approx(np.exp(mkt['price_ret'][last, j]) - 1, abs=1e-9)
        assert mkt['dlret'][j] > -1.0


def test_delisting_returns_are_crashes_or_takeovers(mkt):
    dl = mkt['dlret'][mkt['dead']]
    assert ((dl <= -0.3) | ((dl >= 0.0) & (dl <= 0.35))).all()
    assert (dl < 0).any() and (dl >= 0).any()


def test_dead_stocks_had_a_bad_year(mkt):
    '''Before dying, the past-year return is lower than the average stock's: delistings follow bad years.'''
    csum = np.vstack([np.zeros(mkt['total_ret'].shape[1]), np.cumsum(mkt['total_ret'], axis=0)])
    past = []
    for j in np.where(mkt['dead'])[0]:
        t = mkt['last_idx'][j]
        if t >= 252:
            col = csum[t] - csum[t - 252]                 # past-year return of every stock at that date
            past.append((col[j] - col.mean()) / col.std())
    assert len(past) >= 5 and np.mean(past) < -0.3


def test_drop_dead_writes_only_survivors_and_files_stop_early(mkt, tmp_path):
    n_dead = int(mkt['dead'].sum())
    info = write_market(mkt, DEAD, tmp_path / 'all')
    surv = write_market(mkt, DEAD, tmp_path / 'surv', drop_dead=True)
    assert info['n_tickers'] == DEAD.n_tickers and surv['n_tickers'] == DEAD.n_tickers - n_dead
    j = int(np.where(mkt['dead'])[0][0])
    f = pd.read_csv(tmp_path / 'all' / f'{mkt["tickers"][j]}.csv')
    assert len(f) == mkt['last_idx'][j] + 1
    assert not (tmp_path / 'surv' / f'{mkt["tickers"][j]}.csv').exists()


def test_same_seed_same_dead_stocks():
    a, b = simulate_market(DEAD), simulate_market(DEAD)
    assert (a['last_idx'] == b['last_idx']).all() and market_digest(a) == market_digest(b)
    c = simulate_market(replace(DEAD, seed=6))
    assert not (a['last_idx'] == c['last_idx']).all()
