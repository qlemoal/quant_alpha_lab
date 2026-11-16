'''
Plumbing for stage 01 (download). The script 01_download_data.py only calls these functions.

    Wikipedia table of S&P 500 members
          |  fetch_sp500_symbols      the list of tickers, as written on Wikipedia
          |  to_yahoo_symbol          BRK.B -> BRK-B (Yahoo uses a dash, Wikipedia a dot)
          v
    for each ticker: download_with_retry  ->  one CSV per ticker in data/raw/.../ohlcv/
          |
          v
    a report: which tickers worked, which failed and why

The prices are NOT adjusted by us: auto_adjust=False keeps both "Close" (as traded) and "Adj Close"
(splits and dividends). Stage 02 needs both.
'''

import time
from io import StringIO
from pathlib import Path

import pandas as pd

WIKI_URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
EXPECTED_COLUMNS = {'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'}
MIN_ROWS = 20                       # fewer rows than this is a failed download, not a short history


def to_yahoo_symbol(symbol: str) -> str:
    '''Wikipedia writes share classes with a dot (BRK.B, BF.B). Yahoo wants a dash (BRK-B, BF-B).'''
    return symbol.strip().replace('.', '-')


def fetch_sp500_symbols(html: str) -> list:
    '''Tickers of the first table of the Wikipedia page, in Yahoo format, no duplicate, order kept.'''
    table = pd.read_html(StringIO(html))[0]
    seen, symbols = set(), []
    for raw in table['Symbol'].astype(str):
        s = to_yahoo_symbol(raw)
        if s and s not in seen:
            seen.add(s)
            symbols.append(s)
    return symbols


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    '''yfinance may return two-level column names (field, ticker). Keep the field.'''
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def check_download(df: pd.DataFrame) -> None:
    '''Raise ValueError if the table is not a usable price history.'''
    if df is None or len(df) < MIN_ROWS:
        raise ValueError(f'only {0 if df is None else len(df)} rows')
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f'missing columns {sorted(missing)}')


def download_with_retry(fetch, ticker: str, retries: int = 3, wait_s: float = 2.0) -> pd.DataFrame:
    '''
    Call fetch(ticker) up to `retries` times, waiting wait_s * attempt seconds between tries.
    `fetch` returns a DataFrame (in the script it wraps yfinance). Raises the last error if all tries fail.
    '''
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            df = flatten_columns(fetch(ticker))
            check_download(df)
            return df
        except Exception as e:                                   # network, rate limit, empty answer
            last_error = e
            if attempt < retries:
                time.sleep(wait_s * attempt)
    raise RuntimeError(str(last_error))


def download_all(fetch, symbols: list, out_dir: Path, retries: int = 3, wait_s: float = 2.0, progress=None) -> dict:
    '''
    Download every symbol and write {out_dir}/{symbol}.csv. Never stops on one failure.
    Returns {'ok': [symbols], 'failed': {symbol: reason}}.
    '''
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ok, failed = [], {}
    for symbol in (progress(symbols) if progress else symbols):
        try:
            df = download_with_retry(fetch, symbol, retries, wait_s)
            df.to_csv(out_dir / f'{symbol}.csv')
            ok.append(symbol)
        except Exception as e:
            failed[symbol] = str(e)
    return {'ok': ok, 'failed': failed}