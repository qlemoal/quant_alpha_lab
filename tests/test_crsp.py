'''
Tests for src/data/crsp.py on a small fake table shaped like crsp.dsf_v2. (No WRDS connection is used.)

What is checked
  1. split: the price halves, the return does not -> Close and Adj Close are continuous, the volume is rescaled
  2. dividends: Adj Close is below Close in the past (total return > price return), equal at the anchor
  3. dollar volume (Close x Volume) is the raw dollar volume, on any split basis
  4. negative prices (bid/ask midpoints) are used as absolute values
  5. a DEAD stock: the delisting return is in the LAST Adj Close only, and Adj Close <= Close in the past
  6. a dead stock with NO delisting return gets the -30% assumption, and it is counted
  7. Nasdaq volume before 2004 is divided (Gao-Ritter)
  8. missing open/high/low are filled and the OHLC order holds
  9. a wrong column name stops with a readable message
  10. ids are unique when a ticker is reused
'''

import numpy as np
import pandas as pd
import pytest

from src.data.crsp import (COLUMNS, convert_daily, convert_one, nasdaq_divisor, safe_name, check_columns, TERMINAL_RETURN)

C = COLUMNS['daily']


def _daily(permno, n, price0=100.0, vol=1000.0, start='2010-01-04', retx=None, ret=None, split_at=None):
    '''A fake permno. Raw prices follow retx; at split_at the raw price is halved (a 2-for-1 split, no return).'''
    dates = pd.bdate_range(start, periods=n)
    retx = np.zeros(n) if retx is None else np.asarray(retx, float)
    ret = retx.copy() if ret is None else np.asarray(ret, float)
    raw = np.empty(n)
    raw[0] = price0
    for i in range(1, n):
        raw[i] = raw[i - 1] * (1 + retx[i])
        if split_at is not None and i == split_at:
            raw[i] /= 2.0
    volume = np.full(n, vol)
    if split_at is not None:
        volume[split_at:] *= 2.0                                  # twice the shares trade after the split
    return pd.DataFrame({'permno': permno, 'dlycaldt': dates, 'dlyprc': raw, 'dlyret': ret, 'dlyretx': retx,
                         'dlyvol': volume, 'dlyopen': raw * 0.999, 'dlyhigh': raw * 1.01, 'dlylow': raw * 0.99})


def _info(permno, ticker='AAA', exch='N', start='1990-01-01', end='2100-01-01'):
    return pd.DataFrame({'permno': [permno], 'ticker': [ticker], 'primaryexch': [exch],
                         'secinfostartdt': [start], 'secinfoenddt': [end]})


def test_split_is_continuous_and_volume_rescaled():
    n = 40
    retx = np.r_[0.0, np.full(n - 1, 0.001)]
    d = _daily(1, n, retx=retx, split_at=20)
    out, rep = convert_one(d, is_dead=False)
    r = out['Close'].pct_change().dropna()
    assert np.allclose(r, 0.001, atol=1e-9)                       # no jump at the split
    assert out['Close'].iloc[-1] == pytest.approx(d['dlyprc'].iloc[-1])     # anchored on the last day
    # volume on the split-adjusted basis is constant before and after (1000 shares before, 2000 after = 1000 old shares x ...)
    assert out['Volume'].iloc[5] == pytest.approx(out['Volume'].iloc[30], rel=1e-6)


def test_dividend_makes_adj_close_lower_in_the_past():
    n = 30
    retx = np.r_[0.0, np.full(n - 1, 0.001)]
    ret = retx.copy()
    ret[10] += 0.01                                               # a 1% dividend on day 10
    out, _ = convert_one(_daily(1, n, retx=retx, ret=ret), is_dead=False)
    assert out['Adj Close'].iloc[-1] == pytest.approx(out['Close'].iloc[-1])
    assert (out['Adj Close'].iloc[:10] < out['Close'].iloc[:10]).all()
    assert (out['Adj Close'] <= out['Close'] + 1e-9).all()
    got = out['Adj Close'].pct_change().iloc[10]
    assert got == pytest.approx(0.011, abs=1e-9)


def test_dollar_volume_is_the_raw_dollar_volume():
    n = 40
    retx = np.r_[0.0, np.full(n - 1, 0.002)]
    d = _daily(1, n, retx=retx, split_at=15)
    out, _ = convert_one(d, is_dead=False)
    raw_dollars = d['dlyprc'].to_numpy() * d['dlyvol'].to_numpy()
    adj_dollars = out['Close'].to_numpy() * out['Volume'].to_numpy()
    assert np.allclose(adj_dollars, raw_dollars, rtol=1e-3)       # Volume is rounded to a whole number


def test_negative_prices_are_midpoints():
    d = _daily(1, 20)
    d['dlyprc'] = -d['dlyprc']
    out, _ = convert_one(d, is_dead=False)
    assert (out['Close'] > 0).all() and out['Close'].iloc[-1] == pytest.approx(100.0)


def test_dead_stock_delisting_return_is_in_the_last_adj_close_only():
    n = 30
    retx = np.r_[0.0, np.full(n - 1, 0.001)]
    ret = retx.copy()
    ret[-1] = -0.60                                               # delisting return in the last total return
    retx[-1] = 0.001                                              # the last TRADE price moved normally
    d = _daily(1, n, retx=retx, ret=ret)
    out, rep = convert_one(d, is_dead=True)
    adj_ret = out['Adj Close'].pct_change().iloc[-1]
    assert adj_ret == pytest.approx(-0.60, abs=1e-9)
    assert out['Close'].pct_change().iloc[-1] == pytest.approx(0.001, abs=1e-9)
    assert (out['Adj Close'].iloc[:-1] <= out['Close'].iloc[:-1] + 1e-9).all()     # the ETL check stays quiet
    assert rep['terminal_assumed'] == 0


def test_dead_stock_without_delisting_return_gets_the_assumption():
    n = 30
    retx = np.r_[0.0, np.full(n - 1, 0.001)]
    d = _daily(1, n, retx=retx)
    d.loc[n - 1, ['dlyret', 'dlyretx']] = np.nan                   # the price exists, CRSP has no return for the last day
    out, rep = convert_one(d, is_dead=True)
    assert rep['terminal_assumed'] == 1
    assert out['Adj Close'].pct_change().iloc[-1] == pytest.approx(TERMINAL_RETURN, abs=1e-9)


def test_alive_stock_with_missing_last_return_is_not_punished():
    n = 30
    d = _daily(1, n, retx=np.r_[0.0, np.full(n - 1, 0.001)])
    d.loc[n - 1, ['dlyret', 'dlyretx']] = np.nan
    out, rep = convert_one(d, is_dead=False)
    assert rep['terminal_assumed'] == 0


def test_nasdaq_volume_adjustment():
    d = nasdaq_divisor(pd.Series(pd.to_datetime(['2000-06-01', '2001-06-01', '2003-01-01', '2005-01-01'])))
    assert d.tolist() == [2.0, 1.8, 1.6, 1.0]
    frame = _daily(1, 10, start='2000-06-01')
    nas, _ = convert_one(frame, False, exch=pd.Series(['Q'] * 10))
    nyse, _ = convert_one(frame, False, exch=pd.Series(['N'] * 10))
    assert nas['Volume'].iloc[0] * 2 == pytest.approx(nyse['Volume'].iloc[0], abs=2)


def test_missing_open_high_low_are_filled_and_ordered():
    d = _daily(1, 20)
    d.loc[5, ['dlyopen', 'dlyhigh', 'dlylow']] = np.nan
    out, rep = convert_one(d, False)
    assert rep['ohl_filled'] == 1
    assert (out['High'] >= out[['Open', 'Close']].max(axis=1) - 1e-9).all()
    assert (out['Low'] <= out[['Open', 'Close']].min(axis=1) + 1e-9).all()


def test_convert_daily_end_to_end_and_ids():
    alive = _daily(10, 300)
    dead = _daily(11, 120)
    daily = pd.concat([alive, dead], ignore_index=True)
    info = pd.concat([_info(10, 'AAA'), _info(11, 'AAA', exch='Q')], ignore_index=True)       # the ticker AAA is reused
    frames, master, rep = convert_daily(daily, info, end_date=alive['dlycaldt'].max())
    assert set(frames) == {'AAA_10', 'AAA_11'}
    assert rep['n_dead'] == 1 and master.set_index('id').loc['AAA_11', 'is_dead']
    assert not master.set_index('id').loc['AAA_10', 'is_dead']
    assert list(frames['AAA_10'].columns) == ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']


def test_wrong_column_names_stop_with_a_readable_message():
    d = _daily(1, 10).rename(columns={'dlyprc': 'prc'})
    with pytest.raises(ValueError, match='lacks the columns'):
        check_columns(d, COLUMNS['daily'], 'crsp.dsf_v2')


def test_safe_name():
    assert safe_name('BRK/B', 17) == 'BRKB_17'
    assert safe_name('AAPL', 14593) == 'AAPL_14593'
