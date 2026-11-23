#!/usr/bin/env python
'''
===============================================
Phase 01 (alternative): Import a vendor price file
===============================================

QUESTION ANSWERED
    "I bought (or was given) daily prices from a vendor. How do I feed them to the pipeline?"

    vendor file --> src/data/vendors.py (rename, check, split by ticker) --> data/raw/{source}/.../ohlcv/*.csv
                                                                              --> 02_etl_prices.py (unchanged)

BEFORE YOU RUN IT
    1. Pick a source name and a universe name, for example --source sharadar --universe USALLCAP.
    2. Put the same two names in config/dataset_config.json, so that 02 to 09 read the same folder.
    3. Read src/data/vendors.py: the price columns must have the Yahoo meaning (Close = split-adjusted,
       Adj Close = split and dividend adjusted). Check one split stock by hand.

USAGE
    python scripts/pipeline/01_import_vendor_data.py --file sep.csv --preset sharadar --source sharadar --universe USALLCAP
    python scripts/pipeline/01_import_vendor_data.py --file bars.parquet --map map.json --source myvendor --universe X
        (map.json: {"ticker": "sym", "Date": "dt", "Open": "o", "High": "h", "Low": "l", "Close": "c", "Adj Close": "ac", "Volume": "v"})
'''

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.paths import get_paths
from src.data.vendors import PRESETS, read_vendor_file, to_yahoo_format, check_price_meaning, write_ticker_csvs


def main(file: str, preset: "str | None", map_file: "str | None", source: str, universe: str):
    if (preset is None) == (map_file is None):
        raise ValueError('give exactly one of --preset and --map')
    mapping = PRESETS[preset] if preset else json.loads(Path(map_file).read_text())

    paths = get_paths(source=source, universe=universe)
    out_dir = paths.raw_ohlcv_dir
    print(f'Input:  {file}\nOutput: {out_dir}')

    df = to_yahoo_format(read_vendor_file(Path(file)), mapping)
    print(f'  {len(df):,} rows, {df["ticker"].nunique():,} tickers, {df["Date"].min()} to {df["Date"].max()}')

    chk = check_price_meaning(df)
    print(f'  price-meaning check: {chk["bad_share"]:.2%} of rows outside the plausible Adj Close / Close range')
    if chk['tickers_flagged']:
        print(f'  WARNING: look at these tickers first: {chk["tickers_flagged"][:10]}')

    n = write_ticker_csvs(df, out_dir)
    (out_dir / 'config.json').write_text(json.dumps({
        'vendor_file': str(file), 'preset': preset, 'mapping': mapping, 'n_tickers': n,
        'rows': len(df), 'first_date': df['Date'].min(), 'last_date': df['Date'].max(),
    }, indent=2))
    print(f'  wrote {n} CSV files.\nNext: python scripts/pipeline/02_etl_prices.py (after setting source and universe in config/dataset_config.json)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Import a vendor daily-price file')
    ap.add_argument('--file', required=True)
    ap.add_argument('--preset', choices=sorted(PRESETS))
    ap.add_argument('--map', dest='map_file')
    ap.add_argument('--source', required=True, help='folder name under data/raw, e.g. sharadar')
    ap.add_argument('--universe', required=True, help='e.g. USALLCAP')
    a = ap.parse_args()
    main(a.file, a.preset, a.map_file, a.source, a.universe)