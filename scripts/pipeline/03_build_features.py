#!/usr/bin/env python
'''
===============================================
Phase 03: Build Features
===============================================

QUESTION ANSWERED
    "What do we know about each stock at the close of each day?"

THE STAGE IN ONE PICTURE

    ohlcv.parquet  (from 02_etl_prices.py)
    ticker, date, open_adj, close (adjusted), close_raw, volume
            |
            v
     [1] check the prices        duplicates, bad prices, suspicious daily moves
            |
            v
     [2] src/features/feature_pipeline.py  build_features()
            logret          log(close[t] / close[t-1])
            beta{w}         sensitivity to the equal-weight market, w days
            mom{w}          return over the last w days
            vol{w}          daily volatility over the last w days
            dollar_volume   close_raw * volume          (dollars traded)
            log_adv{w}      log of the average dollar volume over w days
            fwdret          LABEL: the next return. Uses future prices (see the warning below)
            |
            v
     [3] check the features      no NaN, no inf
            |
            v
     [4] features.parquet + config.json

WHERE IS THE COMPLICATED PART?
    src/features/*.py   one small file per feature family (the formulas are in their headers)
    This script only chooses the windows and calls them, in order.

THE RULE: A FEATURE AT DATE t USES DATA UP TO THE CLOSE OF t, NOTHING LATER.
    Proof: tests/test_features.py cuts the prices at a date and checks that the features of the
    earlier dates do not change.
    The only exception is the column `fwdret` (a LABEL). It stays in the file for the research scripts.
    Stage 04 builds its own labels and does not use it. Never build a signal from `fwdret`.

NULLS ARE KEPT
    The first w rows of a ticker have no w-day feature. We keep these rows (they may carry other features).

INPUT     {clean_base}/ohlcv/ohlcv.parquet
OUTPUTS   {clean_base}/features/features.parquet     one row per (ticker, date), sorted
          {clean_base}/features/config.json

USAGE
    python scripts/pipeline/03_build_features.py
'''

import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Make `import src...` and `import config...` work wherever the script is started from.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import polars as pl

from config.paths import get_paths
from src.features.feature_pipeline import build_features, check_ohlcv, check_features, LABEL_COLS
from src.utils.helpers import get_git_commit, file_hash


# =============================================================================
# CONFIGURATION
# =============================================================================

WINDOWS = {                                    # in trading days
    'beta':       [60, 252],
    'momentum':   [5, 10, 20, 60, 120, 252],
    'volatility': [5, 10, 20, 60, 120, 252],
    'adv':        [5, 10, 20, 60, 120, 252],
}

# A date needs this many stocks to have a market return, hence a beta. The data holds today's S&P 500
# members only, so the early years (around 2000) hold few stocks.
MIN_TICKERS_FOR_MARKET = 100


def main():
    start = time.time()
    paths = get_paths()
    if not paths.ohlcv_file.exists():
        raise FileNotFoundError(f'Missing input: {paths.ohlcv_file}. Run 02_etl_prices.py first.')
    out_file = paths.features_file
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print('PHASE 03: BUILD FEATURES')
    print('=' * 70)
    print(f'Input:   {paths.ohlcv_file}')
    print(f'Output:  {out_file}')
    print(f'Windows: {WINDOWS}')

    # ------------------------------------------------------------------
    # [1/4] Check the prices
    # ------------------------------------------------------------------
    print('\n[1/4] Checking the prices...')
    lf = pl.scan_parquet(paths.ohlcv_file)
    ohlcv_report = check_ohlcv(lf)                       # raises on duplicates and bad prices
    r = ohlcv_report
    print(f'  {r["n_rows"]:,} rows, {r["n_tickers"]} tickers, {r["n_dates"]} dates ({r["first_date"]} to {r["last_date"]})')
    print(f'  daily moves beyond +-40%: {r["n_extreme_moves"]}')
    for m in r['largest_moves']:
        print(f'    {m["ticker"]} {m["date"]}  log-return {m["logret"]:+.2f}')
    if r['n_extreme_moves']:
        print('  NOTE: a move of +-0.69 or +-1.10 can be an unadjusted split (log 2, log 3). Check the raw files.')

    # ------------------------------------------------------------------
    # [2/4] Build the features
    # ------------------------------------------------------------------
    print('\n[2/4] Building the features...')
    df = build_features(lf, WINDOWS, MIN_TICKERS_FOR_MARKET).collect()
    print(f'  {df.height:,} rows x {df.width} columns')

    # ------------------------------------------------------------------
    # [3/4] Check the features
    # ------------------------------------------------------------------
    print('\n[3/4] Checking the features...')
    feature_report = check_features(df, WINDOWS)         # raises on NaN or inf
    worst = sorted(feature_report['null_pct'].items(), key=lambda kv: -kv[1])[:3]
    print('  no NaN, no inf. Highest null share (warm-up of long windows is normal): '
          + ', '.join(f'{k} {v:.1f}%' for k, v in worst))

    # ------------------------------------------------------------------
    # [4/4] Save
    # ------------------------------------------------------------------
    print('\n[4/4] Saving...')
    df.write_parquet(out_file)
    with open(out_file.parent / 'config.json', 'w') as f:
        json.dump({
            'run_timestamp': datetime.now().isoformat(),
            'source_ohlcv': str(paths.ohlcv_file),
            'ohlcv_hash': file_hash(paths.ohlcv_file),
            'git_commit': get_git_commit(ROOT),
            'windows': WINDOWS,
            'min_tickers_for_market': MIN_TICKERS_FOR_MARKET,
            'label_columns': LABEL_COLS,
            'n_rows': df.height,
            'columns': df.columns,
            'ohlcv_report': ohlcv_report,
            'feature_report': feature_report,
        }, f, indent=2, default=str)
    print(f'  features: {out_file}')
    print(f'\nDone in {time.time() - start:.1f} s.')
    print('Next: python scripts/pipeline/04_generate_signals.py --date YYYY-MM-DD --tag default')


if __name__ == '__main__':
    main()