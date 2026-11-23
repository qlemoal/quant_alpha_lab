'''Tests for src/data/vendors.py on a tiny fake vendor file (no real vendor data, no network).'''

import pandas as pd
import pytest

from src.data.vendors import PRESETS, to_yahoo_format, check_price_meaning, write_ticker_csvs, YAHOO_COLUMNS


def fake(dup=False):
    rows = [
        ('AAA', '2020-01-03', 10, 11, 9, 10.5, 1000, 10.2),
        ('AAA', '2020-01-02', 10, 11, 9, 10.0, 900, 9.7),
        ('BBB', '2020-01-02', 50, 52, 49, 51.0, 500, 50.5),
    ]
    if dup:
        rows.append(rows[0])
    return pd.DataFrame(rows, columns=['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'closeadj'])


def test_sharadar_preset_gives_yahoo_columns_sorted():
    out = to_yahoo_format(fake(), PRESETS['sharadar'])
    assert list(out.columns) == ['ticker'] + YAHOO_COLUMNS
    assert out['Date'].tolist() == ['2020-01-02', '2020-01-03', '2020-01-02']
    assert out['Adj Close'].tolist()[0] == 9.7 and out['Close'].tolist()[0] == 10.0


def test_missing_column_and_duplicates_are_errors():
    with pytest.raises(ValueError, match='lacks'):
        to_yahoo_format(fake().drop(columns=['closeadj']), PRESETS['sharadar'])
    with pytest.raises(ValueError, match='duplicated'):
        to_yahoo_format(fake(dup=True), PRESETS['sharadar'])
    with pytest.raises(ValueError, match='mapping lacks'):
        to_yahoo_format(fake(), {'ticker': 'ticker'})


def test_price_meaning_flags_a_split_in_the_wrong_column():
    out = to_yahoo_format(fake(), PRESETS['sharadar'])
    assert check_price_meaning(out)['tickers_flagged'] == []
    out.loc[0, 'Adj Close'] = out.loc[0, 'Close'] / 10          # a 10:1 split sitting in Adj Close only
    assert check_price_meaning(out)['tickers_flagged'] == ['AAA']


def test_write_ticker_csvs(tmp_path):
    out = to_yahoo_format(fake(), PRESETS['sharadar'])
    assert write_ticker_csvs(out, tmp_path) == 2
    back = pd.read_csv(tmp_path / 'AAA.csv')
    assert list(back.columns) == YAHOO_COLUMNS and len(back) == 2


def test_unsafe_ticker_is_refused(tmp_path):
    out = to_yahoo_format(fake(), PRESETS['sharadar'])
    out.loc[0, 'ticker'] = '../evil'
    with pytest.raises(ValueError, match='unsafe'):
        write_ticker_csvs(out, tmp_path)