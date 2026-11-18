#!/usr/bin/env python
'''
===============================================
Phase 02: ETL of the Prices
===============================================

QUESTION ANSWERED
    "Can we turn the raw Yahoo files into ONE clean price table that every later stage can trust?"

THE STAGE IN ONE PICTURE

    data/raw/yfinance/.../ohlcv/AAPL.csv, MSFT.csv, ...      (from 01_download_data.py)
            |
            v
     [1] read all the CSV files into one table (every column as text)
            |
            v
     [2] src/data/prices_etl.py  build_clean_table()
            unreadable rows         removed, counted
            price or volume <= 0    removed, counted
            duplicate (ticker,date) one kept, counted
            open_adj = open * close / close_raw
            |
            v
     [3] quality report: impossible highs and lows, bad adjustment factors, short histories
            |
            v
     [4] ohlcv.parquet + config.json (with the counts)

WHERE IS THE COMPLICATED PART?
    src/data/prices_etl.py   the SQL and the checks. This script only calls it, in order.

THE COLUMNS OF ohlcv.parquet
    ticker, date
    open, high, low   as traded
    close_raw         close as traded
    close             ADJUSTED close (splits and dividends). Used for returns.
    volume            shares traded
    open_adj          open on the adjusted scale. Used for the labels and the backtest (we trade at the open).

KNOWN LIMITS OF THE DATA (they bias the results, read them before you quote a number)
    - Survivorship: the tickers are today's S&P 500 members. Companies that left the index (bankrupt,
      acquired, demoted) are missing. Backtests look better than reality, mostly before 2015.
    - Source: Yahoo Finance, adjusted by Yahoo. Good for research, not for production.
    - A day without volume is removed. It leaves a gap in the ticker's dates (04 reports the gaps).

INPUT     data/raw/{source}/{asset}/{region}/{universe}/{freq}/ohlcv/*.csv
OUTPUTS   {clean_base}/ohlcv/ohlcv.parquet, {clean_base}/ohlcv/config.json

USAGE
    python scripts/pipeline/02_etl_prices.py
'''

import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Make `import src...` and `import config...` work wherever the script is started from.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import duckdb

from config.paths import get_paths
from src.data.etl_prices import read_raw_prices, build_clean_table, quality_report, write_parquet
from src.utils.helpers import get_git_commit, file_hash

MIN_HISTORY = 252          # a ticker with fewer rows than this is reported as "short history"


def main():
    start = time.time()
    paths = get_paths()
    raw_dir, out_file = paths.raw_ohlcv_dir, paths.ohlcv_file

    print('=' * 70)
    print('PHASE 02: ETL OF THE PRICES')
    print('=' * 70)
    print(f'Raw files: {raw_dir}')
    print(f'Output:    {out_file}')

    con = duckdb.connect()

    # ------------------------------------------------------------------
    # [1/4] Read
    # ------------------------------------------------------------------
    print('\n[1/4] Reading the raw CSV files...')
    n_raw = read_raw_prices(con, raw_dir)
    print(f'  {n_raw:,} raw rows')

    # ------------------------------------------------------------------
    # [2/4] Clean
    # ------------------------------------------------------------------
    print('\n[2/4] Cleaning...')
    removed = build_clean_table(con)
    for reason, n in removed.items():
        print(f'  removed {n:>9,}  {reason}')

    # ------------------------------------------------------------------
    # [3/4] Quality report
    # ------------------------------------------------------------------
    print('\n[3/4] Quality report...')
    report = quality_report(con, MIN_HISTORY)
    print(f'  {report["n_rows"]:,} rows, {report["n_tickers"]} tickers, {report["n_dates"]} dates '
          f'({report["first_date"]} to {report["last_date"]})')
    for key, text in [
        ('high_below_low', 'rows with high < low'),
        ('open_outside_range', 'rows with the open outside [low, high]'),
        ('adj_factor_above_one', 'rows with close / close_raw > 1.001'),
        ('short_history', f'tickers with fewer than {MIN_HISTORY} rows'),
    ]:
        flag = '' if report[key] == 0 else '   <-- look at it'
        print(f'  {report[key]:>9,}  {text}{flag}')

    # ------------------------------------------------------------------
    # [4/4] Save
    # ------------------------------------------------------------------
    print('\n[4/4] Saving...')
    write_parquet(con, out_file)
    with open(out_file.parent / 'config.json', 'w') as f:
        json.dump({
            'run_timestamp': datetime.now().isoformat(),
            'source': paths.source,
            'raw_dir': str(raw_dir),
            'output_file': out_file.name,
            'output_hash': file_hash(out_file),
            'git_commit': get_git_commit(ROOT),
            'n_raw_rows': n_raw,
            'removed': removed,
            'quality': report,
        }, f, indent=2, default=str)
    print(f'  ohlcv: {out_file}')
    print(f'\nDone in {time.time() - start:.1f} s.')
    print('Next: python scripts/pipeline/03_build_features.py')


if __name__ == '__main__':
    main()