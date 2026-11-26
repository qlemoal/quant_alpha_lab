'''
Tests for the backtest engine (src/portfolio/weights_backtest.py), on tiny hand-made price tables.

  1. a dead stock gets an exit mark: the day after its last row, its "open" is its last adjusted close
  2. the P&L of a long position in a dying stock books the crash (the delisting return)
  3. the same for a short position: the short earns the crash
  4. no trade at a price that is not real: after death, and on a missing day of an alive stock
  5. the market return ignores stale (forward-filled) prices
  6. a stock alive at the end of the data gets no exit mark
'''

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from src.portfolio.weights_backtest import build_price_panel, exit_marks, simulate

COSTLESS = {'aum': 1e8, 'spread_model': 'fixed', 'spread_kwargs': {'spread_bps': 0.0},
            'impact_kappa': 0.0, 'commission_bps': 0.0}


def _dates(n):
    d0 = date(2020, 1, 1)
    return [d0 + timedelta(days=i) for i in range(n)]


def _prices(n_days=40, dead_last=19, crash=0.5):
    '''
    Two stocks. ALIVE: open = close = 100 every day.
    DEAD: open = close = 100 until its last row (index dead_last); on that row the adjusted close is
    100 * (1 - crash) (the delisting return is in the close), the open is still 100.
    '''
    d = _dates(n_days)
    rows = [(x, 'ALIVE', 100.0, 100.0) for x in d]
    for i, x in enumerate(d[:dead_last + 1]):
        rows.append((x, 'DEAD', 100.0, 100.0 * (1 - crash) if i == dead_last else 100.0))
    prices = pl.DataFrame(rows, schema={'date': pl.Date, 'ticker': pl.Utf8, 'open_adj': pl.Float64, 'close': pl.Float64},
                          orient='row')
    feats = prices.select(['date', 'ticker']).with_columns(pl.lit(np.log(1e9)).alias('log_adv60'),
                                                           pl.lit(0.02).alias('vol60'))
    return prices, feats, d


def _weights(day, w):
    return pl.DataFrame({'date': [day] * len(w), 'ticker': list(w), 'weight': list(w.values())},
                        schema={'date': pl.Date, 'ticker': pl.Utf8, 'weight': pl.Float64})


def test_exit_mark_is_last_close_on_the_next_date():
    prices, _, d = _prices()
    m = exit_marks(prices)
    assert m.height == 1 and m['ticker'][0] == 'DEAD'
    assert m['date'][0] == d[20] and m['open_adj'][0] == pytest.approx(50.0)


def test_no_exit_mark_for_a_stock_alive_at_the_end():
    prices, _, _ = _prices()
    assert 'ALIVE' not in exit_marks(prices)['ticker'].to_list()


def test_long_position_books_the_delisting_crash():
    prices, feats, d = _prices(crash=0.5)
    panel = build_price_panel(prices, feats)
    rets, _ = simulate(_weights(d[10], {'DEAD': 0.5, 'ALIVE': -0.5}), panel, COSTLESS)
    nav = np.prod(1 + rets['gross_ret'].to_numpy())
    assert nav == pytest.approx(1 - 0.5 * 0.5)          # half the book in DEAD, DEAD loses 50%


def test_short_position_earns_the_delisting_crash():
    prices, feats, d = _prices(crash=0.5)
    panel = build_price_panel(prices, feats)
    rets, _ = simulate(_weights(d[10], {'DEAD': -0.5, 'ALIVE': 0.5}), panel, COSTLESS)
    assert np.prod(1 + rets['gross_ret'].to_numpy()) == pytest.approx(1 + 0.5 * 0.5)


def test_no_entry_after_death():
    prices, feats, d = _prices()
    panel = build_price_panel(prices, feats)
    _, trades = simulate(_weights(d[25], {'DEAD': 0.5, 'ALIVE': -0.5}), panel, COSTLESS)
    assert trades['n_untradable'][0] == 1


def test_no_entry_on_a_missing_day_of_an_alive_stock():
    prices, feats, d = _prices()
    prices = prices.filter(~((pl.col('ticker') == 'ALIVE') & (pl.col('date') == d[11])))
    panel = build_price_panel(prices, feats)
    _, trades = simulate(_weights(d[10], {'ALIVE': 0.5, 'DEAD': -0.5}), panel, COSTLESS)   # entry on d[11]
    assert trades['n_untradable'][0] == 1


def test_market_return_ignores_stale_prices():
    prices, feats, d = _prices(crash=0.5)
    panel = build_price_panel(prices, feats)
    rets, _ = simulate(_weights(d[5], {'DEAD': 0.5, 'ALIVE': -0.5}), panel, COSTLESS)
    mkt = dict(zip(rets['date'].to_list(), rets['market_ret'].to_list()))
    assert mkt[d[20]] == pytest.approx((-0.5 + 0.0) / 2)   # the exit day: both stocks count
    assert mkt[d[25]] == pytest.approx(0.0)               # later: only ALIVE counts (DEAD is stale)
