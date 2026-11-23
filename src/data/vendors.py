'''
GOAL: turn the daily-price file of a paid data vendor into the per-ticker CSV files that stage 02 reads.
    Stage 02 and everything after it stay unchanged. Only the SOURCE of the raw files changes.

    vendor file (one big CSV or parquet)
          |  [1] rename the columns to the Yahoo names      Date, Open, High, Low, Close, Adj Close, Volume
          |  [2] check the meaning of the price columns     (see below)
          |  [3] write one CSV per ticker                   same folder and format as 01_download_data.py
          v
    data/raw/{source}/.../ohlcv/AAPL.csv, ...   -->   02_etl_prices.py as usual

THE ONE THING TO GET RIGHT: WHAT DO THE PRICE COLUMNS MEAN?
    Stage 02 expects:   Close      = close adjusted for SPLITS only  (the price a trader saw, on today's share count)
                        Adj Close  = close adjusted for splits AND dividends
                        Open, High, Low = split-adjusted, NOT dividend-adjusted
    (02 builds open_adj = Open * Adj Close / Close.)
    A vendor column that is adjusted differently gives silently wrong returns. Check ONE split stock
    and ONE high-dividend stock by hand before you trust a new source.

PRESETS (column mappings written from the vendors' documentation, NOT tested on a real file. Check the first rows.)
    'sharadar'  Nasdaq Data Link, table SHARADAR/SEP. Columns: ticker, date, open, high, low, close, volume, closeadj,
                closeunadj. open/high/low/close are split-adjusted; closeadj is also dividend-adjusted.
                Includes delisted companies: this is what removes survivorship bias.
    'generic'   you give the mapping yourself (a dict {yahoo name: your column name}), plus 'Ticker'.

WHY A VENDOR AT ALL? (see docs/data_sources.md in the answer to your question)
    Yahoo gives only today's survivors. A strategy tested on survivors looks better than it was.
    A vendor with delisted names and point-in-time index membership fixes this.
'''

from pathlib import Path

import pandas as pd

YAHOO_COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

PRESETS = {
    'sharadar': {
        'ticker': 'ticker', 'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low',
        'Close': 'close', 'Adj Close': 'closeadj', 'Volume': 'volume',
    },
}


def read_vendor_file(path: Path) -> pd.DataFrame:
    '''Read a .csv or .parquet vendor file. Nothing else is done to it.'''
    path = Path(path)
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    if path.suffix in ('.csv', '.gz'):
        return pd.read_csv(path)
    raise ValueError(f'unsupported file type: {path.suffix} (use .csv or .parquet)')


def to_yahoo_format(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    '''
    Rename the vendor columns to ['ticker'] + YAHOO_COLUMNS and clean the types.
    mapping: {'ticker': col, 'Date': col, 'Open': col, ... 'Volume': col}. A missing key is an error.
    Rows with a missing date or ticker are dropped. Rows are sorted by ticker, date. Duplicates of
    (ticker, date) are an ERROR (the vendor file is not what we think it is), not silently removed.
    '''
    need = ['ticker'] + YAHOO_COLUMNS
    missing_keys = [k for k in need if k not in mapping]
    if missing_keys:
        raise ValueError(f'mapping lacks {missing_keys}')
    missing_cols = [mapping[k] for k in need if mapping[k] not in df.columns]
    if missing_cols:
        raise ValueError(f'the vendor file lacks the columns {missing_cols}. It has {list(df.columns)}')

    out = pd.DataFrame({k: df[mapping[k]] for k in need})
    out['Date'] = pd.to_datetime(out['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
    out = out.dropna(subset=['ticker', 'Date'])
    dup = out.duplicated(['ticker', 'Date'])
    if dup.any():
        ex = out.loc[dup, ['ticker', 'Date']].head(3).values.tolist()
        raise ValueError(f'{int(dup.sum())} duplicated (ticker, date) rows, e.g. {ex}')
    return out.sort_values(['ticker', 'Date']).reset_index(drop=True)


def check_price_meaning(df: pd.DataFrame, min_ratio: float = 0.2, max_ratio: float = 1.0 + 1e-6) -> dict:
    '''
    Cheap sanity check of the Close / Adj Close convention.
    For each ticker the ratio Adj Close / Close must be <= 1 on the LAST day's scale only when the
    adjustment is anchored on the last day, which is the usual vendor choice. We test two weak facts:
      - the ratio is positive and below max_ratio on at least 99% of the rows (dividend adjustment lowers the past);
      - the ratio is not below min_ratio (a 5x gap is a split that landed in the wrong column).
    Returns {'bad_share': share of rows outside [min_ratio, max_ratio], 'tickers_flagged': [...]}.
    It does NOT prove the convention is right. The manual split-stock check stays necessary.
    '''
    ratio = pd.to_numeric(df['Adj Close'], errors='coerce') / pd.to_numeric(df['Close'], errors='coerce')
    bad = ~ratio.between(min_ratio, max_ratio)
    flagged = sorted(df.loc[bad, 'ticker'].unique().tolist())
    return {'bad_share': float(bad.mean()) if len(df) else 0.0, 'tickers_flagged': flagged}


def write_ticker_csvs(df: pd.DataFrame, out_dir: Path) -> int:
    '''One CSV per ticker, with the Yahoo columns. Tickers with a path-unsafe character are refused. Returns the count.'''
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for ticker, g in df.groupby('ticker', sort=True):
        t = str(ticker)
        if not t or any(ch in t for ch in '/\\:*?"<>|') or t.startswith('.'):
            raise ValueError(f'unsafe ticker name for a file: {t!r}')
        g[YAHOO_COLUMNS].to_csv(out_dir / f'{t}.csv', index=False)
        n += 1
    return n