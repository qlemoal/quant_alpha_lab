'''
Tests for src/data/download.py with a fake downloader (no network).
'''

import pandas as pd
import pytest

from src.data import download as dl


def fake_prices(n=30):
    return pd.DataFrame({c: range(n) for c in ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']},
                        index=pd.date_range('2020-01-01', periods=n, name='Date'))


def test_to_yahoo_symbol():
    assert dl.to_yahoo_symbol('BRK.B') == 'BRK-B' and dl.to_yahoo_symbol(' AAPL ') == 'AAPL'


def test_fetch_sp500_symbols_from_html():
    html = '<table><tr><th>Symbol</th></tr><tr><td>BRK.B</td></tr><tr><td>AAPL</td></tr><tr><td>AAPL</td></tr></table>'
    assert dl.fetch_sp500_symbols(html) == ['BRK-B', 'AAPL']


def test_flatten_two_level_columns():
    df = fake_prices()
    df.columns = pd.MultiIndex.from_product([df.columns, ['AAPL']])
    assert list(dl.flatten_columns(df).columns) == ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']


def test_retry_succeeds_after_failures(monkeypatch):
    monkeypatch.setattr(dl.time, 'sleep', lambda s: None)
    calls = {'n': 0}

    def flaky(ticker):
        calls['n'] += 1
        if calls['n'] < 3:
            raise ConnectionError('rate limit')
        return fake_prices()
    assert len(dl.download_with_retry(flaky, 'X', retries=3)) == 30 and calls['n'] == 3


def test_retry_gives_up_and_reports(monkeypatch):
    monkeypatch.setattr(dl.time, 'sleep', lambda s: None)
    with pytest.raises(RuntimeError, match='only 0 rows'):
        dl.download_with_retry(lambda t: pd.DataFrame(), 'X', retries=2)


def test_download_all_never_stops_on_one_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(dl.time, 'sleep', lambda s: None)
    fetch = lambda t: fake_prices() if t != 'BAD' else pd.DataFrame()
    out = dl.download_all(fetch, ['AAA', 'BAD', 'BBB'], tmp_path, retries=1)
    assert out['ok'] == ['AAA', 'BBB'] and list(out['failed']) == ['BAD']
    assert sorted(p.name for p in tmp_path.glob('*.csv')) == ['AAA.csv', 'BBB.csv']
    assert pd.read_csv(tmp_path / 'AAA.csv').columns[0] == 'Date'      # stage 02 reads the column "Date"