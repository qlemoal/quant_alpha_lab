#!/usr/bin/env python
'''
===============================================
Phase 01: Download the Prices
===============================================

QUESTION ANSWERED
    "Which stocks are in the S&P 500 today, and what are their daily prices since 2000?"

THE STAGE IN ONE PICTURE

    Wikipedia: List of S&P 500 companies
            |
            v
     [1] read the tickers   BRK.B -> BRK-B  (Yahoo writes the share class with a dash)
            |
            v           one ticker at a time, 3 tries each
     [2] Yahoo Finance via yfinance
            auto_adjust=False  ->  we keep "Close" (split-adjusted) AND "Adj Close" (splits and dividends)
            |
            v
     [3] data/raw/yfinance/.../ohlcv/AAPL.csv, MSFT.csv, ...      one CSV per ticker
            |
            v
     [4] report: how many worked, which failed and why  (config.json)

WHERE IS THE COMPLICATED PART?
    src/data/download.py   symbol format, retries, checks. This script only calls it, in order.

HONEST LIMITS OF THIS DATA
    - Survivorship bias: only TODAY's members are downloaded. Companies that left the index are missing.
    - Yahoo is a free, unofficial source. Re-downloading later can give slightly different adjusted prices.
    - A ticker that fails here is missing from the study. The script stops with an error if more than
      MAX_FAILED_SHARE of the tickers fail, so a network problem does not give a silent half-universe.

OUTPUTS   data/raw/{source}/{asset}/{region}/{universe}/{freq}/ohlcv/{TICKER}.csv
          data/raw/.../ohlcv/config.json   (date of download, tickers, failures)

USAGE
    python scripts/pipeline/01_download_data.py
'''

import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Make `import src...` and `import config...` work wherever the script is started from.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import requests
import yfinance as yf
from tqdm import tqdm

from config.paths import get_paths
from src.data.download import WIKI_URL, fetch_sp500_symbols, download_all
from src.utils.helpers import get_git_commit

START_DATE = '2000-01-01'
MAX_FAILED_SHARE = 0.05          # stop with an error if more than 5% of the tickers fail
RETRIES = 3


def fetch_yahoo(ticker: str):
    '''One ticker from Yahoo. auto_adjust=False keeps Close and Adj Close.'''
    return yf.download(ticker, start=START_DATE, auto_adjust=False, progress=False)


def main():
    start = time.time()
    paths = get_paths()
    out_dir = paths.raw_ohlcv_dir

    print('=' * 70)
    print('PHASE 01: DOWNLOAD THE PRICES')
    print('=' * 70)
    print(f'Output: {out_dir}')

    # ------------------------------------------------------------------
    # [1/3] The ticker list
    # ------------------------------------------------------------------
    print('\n[1/3] Reading the S&P 500 members from Wikipedia...')
    response = requests.get(WIKI_URL, headers={'User-Agent': 'Mozilla/5.0'}, timeout=30)   # Wikipedia refuses requests without an agent
    response.raise_for_status()
    symbols = fetch_sp500_symbols(response.text)
    print(f'  {len(symbols)} tickers')

    # ------------------------------------------------------------------
    # [2/3] Download
    # ------------------------------------------------------------------
    print(f'\n[2/3] Downloading from {START_DATE} ({RETRIES} tries per ticker)...')
    result = download_all(fetch_yahoo, symbols, out_dir, retries=RETRIES,
                          progress=lambda it: tqdm(it, desc='  Tickers', unit='ticker'))
    n_failed, share = len(result['failed']), len(result['failed']) / len(symbols)
    print(f'  ok: {len(result["ok"])}   failed: {n_failed} ({share:.1%})')
    for symbol, reason in list(result['failed'].items())[:15]:
        print(f'    {symbol}: {reason}')

    # ------------------------------------------------------------------
    # [3/3] Save the report, then stop if too much is missing
    # ------------------------------------------------------------------
    print('\n[3/3] Saving the report...')
    with open(out_dir / 'config.json', 'w') as f:
        json.dump({
            'run_timestamp': datetime.now().isoformat(),
            'source': paths.source,
            'start_date': START_DATE,
            'wikipedia_url': WIKI_URL,
            'git_commit': get_git_commit(ROOT),
            'n_symbols': len(symbols),
            'n_ok': len(result['ok']),
            'failed': result['failed'],
            'symbols_ok': result['ok'],
        }, f, indent=2)
    print(f'  report: {out_dir / "config.json"}')

    if share > MAX_FAILED_SHARE:
        raise SystemExit(f'\nERROR: {share:.1%} of the tickers failed (limit {MAX_FAILED_SHARE:.0%}). '
                         f'Check the network and rerun.')
    print(f'\nDone in {(time.time() - start) / 60:.1f} min.')
    print('Next: python scripts/pipeline/02_etl_prices.py')


if __name__ == '__main__':
    main()
