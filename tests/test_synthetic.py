'''
Tests for src/data/synthetic.py (the synthetic market).

What is checked:
  1. shapes, positive prices, OHLC order, adjusted close <= close and equal on the last day
  2. same seed -> same market (same digest); other seed -> other market
  3. the planted effects are really in the data (tested with STRONG settings so a small sample can see them)
  4. the CSV files have the yfinance layout and can be read back
  5. industries: stocks of one industry move together more than stocks of different industries
'''

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.data.synthetic import SyntheticConfig, simulate_market, market_digest, write_market, YAHOO_ORDER

SMALL = SyntheticConfig(n_tickers=80, n_days=500, seed=3)


@pytest.fixture(scope='module')
def mkt():
    return simulate_market(SMALL)


def test_shapes_and_price_sanity(mkt):
    T, N = SMALL.n_days, SMALL.n_tickers
    for k in ('close', 'adj_close', 'open', 'high', 'low', 'volume'):
        assert mkt[k].shape == (T, N) and np.isfinite(mkt[k]).all()
    assert (mkt['close'] > 0).all() and (mkt['volume'] > 0).all()
    assert (mkt['high'] >= np.maximum(mkt['open'], mkt['close']) - 1e-12).all()
    assert (mkt['low'] <= np.minimum(mkt['open'], mkt['close']) + 1e-12).all()
    assert len(mkt['dates']) == T and mkt['dates'].is_monotonic_increasing


def test_adjusted_close_convention(mkt):
    '''Adj Close is anchored on the last day (equal to Close there) and is below Close in the past (dividends).'''
    assert np.allclose(mkt['adj_close'][-1], mkt['close'][-1])
    assert (mkt['adj_close'] <= mkt['close'] * (1 + 1e-9)).all()
    assert (mkt['adj_close'][0] < mkt['close'][0]).mean() > 0.9


def test_same_seed_same_market_other_seed_other_market(mkt):
    assert market_digest(simulate_market(SMALL)) == market_digest(mkt)
    assert market_digest(simulate_market(replace(SMALL, seed=4))) != market_digest(mkt)


def test_low_volatility_and_illiquidity_effects_are_planted():
    cfg = replace(SMALL, n_tickers=200, n_days=800, c_vol=0.002, c_liq=0.002, c_rev=0.0, c_mom=0.0)
    m = simulate_market(cfg)
    mean_ret = m['total_ret'].mean(axis=0)
    assert np.corrcoef(mean_ret, m['z_vol'])[0, 1] < -0.5
    assert np.corrcoef(mean_ret, m['z_liq'])[0, 1] < -0.5


def test_reversal_is_planted():
    cfg = replace(SMALL, n_tickers=200, n_days=800, c_vol=0.0, c_liq=0.0, c_rev=0.004, c_mom=0.0)
    r = simulate_market(cfg)['total_ret']
    past = np.array([r[t - 5:t].sum(axis=0) for t in range(6, 795)])
    nxt = r[6:795]
    c = np.mean([np.corrcoef(a, b)[0, 1] for a, b in zip(past, nxt)])
    assert c < -0.02


def test_momentum_is_planted():
    cfg = replace(SMALL, n_tickers=200, n_days=800, c_vol=0.0, c_liq=0.0, c_rev=0.0, c_mom=0.004)
    r = simulate_market(cfg)['total_ret']
    cs = np.vstack([np.zeros((1, r.shape[1])), r.cumsum(axis=0)])
    cors = []
    for t in range(260, 790):
        mom = cs[t - 21] - cs[t - 252]
        cors.append(np.corrcoef(mom, r[t])[0, 1])
    assert np.mean(cors) > 0.02


def test_stocks_of_one_industry_move_together(mkt):
    r = mkt['total_ret']
    r = r - r.mean(axis=1, keepdims=True)              # remove the market
    c = np.corrcoef(r.T)
    same = mkt['industry'][:, None] == mkt['industry'][None, :]
    off = ~np.eye(len(c), dtype=bool)
    assert c[same & off].mean() > c[~same].mean() + 0.05


def test_csv_files_have_the_yfinance_layout(tmp_path):
    small = replace(SMALL, n_tickers=5, n_days=60)
    m = simulate_market(small)
    info = write_market(m, small, tmp_path)
    files = sorted(tmp_path.glob('S*.csv'))
    assert len(files) == 5 and info['digest'] == market_digest(m)
    df = pd.read_csv(files[0])
    assert list(df.columns) == YAHOO_ORDER and len(df) == 60
    truth = json.loads((tmp_path / 'truth.json').read_text())
    assert set(truth['industry']) == {f'S{i:03d}' for i in range(5)}
    assert truth['planted_effects']['reversal'] < 0 < truth['planted_effects']['momentum']
