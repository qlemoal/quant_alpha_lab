#!/usr/bin/env python
'''
===============================================
Phase 01 (survivorship-free): download CRSP from WRDS
===============================================

QUESTION ANSWERED
    "How do I get US stock prices that include the companies that died, with their delisting returns?"

    WRDS (CRSP crsp.dsf_v2 + crsp.stksecurityinfohist)
        |  [1] choose the permnos: common stocks on NYSE/AMEX/Nasdaq that were EVER above a market-cap floor
        |  [2] download them in batches (a few hundred at a time)
        |  [3] src/data/crsp.py: rebuild Close / Adj Close from the returns, adjust the volume, keep the delisting return
        v
    data/raw/crsp/equities/US/{universe}/daily/ohlcv/{TICKER}_{PERMNO}.csv        (Yahoo layout)
    data/raw/crsp/equities/US/{universe}/daily/security_master.csv
    data/raw/crsp/equities/US/{universe}/daily/config.json
        |
        v   02_etl_prices.py --> 02b_build_universe.py --> 03 ... (see docs/survivorship.md)

BEFORE YOU RUN IT (once)
    1. pip install wrds
    2. An account on WRDS (your @epfl.ch address; see the EPFL library page). The first connection asks for the password
       and can save it in ~/.pgpass.
    3. config/dataset_config.json:  {"source": "crsp", "asset": "equities", "region": "US", "universe": "USALLCAP",
                                     "freq": "daily", "universe_mode": "rule", "universe_rule": {"top_n": 1000}}
    4. python scripts/pipeline/01_import_crsp.py --probe        prints the real column names. If they differ from
                                                                COLUMNS in src/data/crsp.py, edit that dict.

USAGE
    python scripts/pipeline/01_import_crsp.py --wrds_user YOURNAME --start 1999-01-01 --end 2026-11-20 --min_cap 200000
        --min_cap is in THOUSANDS of dollars (200000 = 200 million dollars). A stock is downloaded if it EVER had at
        least this market cap in the period. This keeps the dead stocks that were big once (the ones that matter) and
        skips the permanent micro-caps. 200000 gives roughly 3000 to 4000 stocks over 25 years. Lower it to include more.
    python scripts/pipeline/01_import_crsp.py --dry_run          prints the SQL, connects to nothing.

LIMITS (honest)
    - Not tested against a live WRDS account by the author. --probe and the column check are there for that reason.
    - CRSP is licensed: do not publish the data or the CSV files. Only code and aggregate results go in the repo.
    - Prices start in 1925 for NYSE. Volume is reliable from 1993 (Nasdaq double counting is adjusted before 2004).
'''

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from config.paths import get_paths
from src.data.crsp import (SQL_PERMNOS, SQL_DAILY, SQL_INFO, CAP_COLUMN, COLUMNS, convert_daily)

BATCH = 150


def _ids(permnos) -> str:
    return ','.join(str(int(p)) for p in permnos)


def main(wrds_user, start, end, min_cap, probe, dry_run, no_nasdaq_adjust):
    paths = get_paths()
    sql_perm = SQL_PERMNOS.format(start=start, end=end, cap=CAP_COLUMN, min_cap=int(min_cap))
    if dry_run:
        print('--- permnos ---', sql_perm, '--- daily ---', SQL_DAILY.format(permnos='...', start=start, end=end),
              '--- info ---', SQL_INFO.format(permnos='...'), sep='\n')
        return
    try:
        import wrds
    except ImportError:
        raise SystemExit('Install the WRDS package first: pip install wrds')
    db = wrds.Connection(wrds_username=wrds_user) if wrds_user else wrds.Connection()

    if probe:
        for name, sql in [('crsp.dsf_v2', 'SELECT * FROM crsp.dsf_v2 LIMIT 3'),
                          ('crsp.stksecurityinfohist', 'SELECT * FROM crsp.stksecurityinfohist LIMIT 3')]:
            df = db.raw_sql(sql)
            print(f'\n{name}: {len(df.columns)} columns\n  {list(df.columns)}\n{df.head(3).T.to_string()}')
        print(f'\nThe importer expects: {COLUMNS}\n and {CAP_COLUMN} for the market cap.')
        return

    out_dir = paths.raw_ohlcv_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output: {out_dir}')
    print('[1/3] Choosing the permnos...')
    permnos = db.raw_sql(sql_perm)['permno'].astype(int).tolist()
    print(f'  {len(permnos):,} stocks were above the market-cap floor at least once between {start} and {end}')

    print('[2/3] Downloading and converting...')
    master, total = [], {'n_permnos': 0, 'n_dead': 0, 'terminal_assumed': 0, 'ret_missing': 0, 'ohl_filled': 0, 'skipped': 0}
    for i in range(0, len(permnos), BATCH):
        chunk = permnos[i:i + BATCH]
        daily = db.raw_sql(SQL_DAILY.format(permnos=_ids(chunk), start=start, end=end))
        info = db.raw_sql(SQL_INFO.format(permnos=_ids(chunk)))
        frames, m, rep = convert_daily(daily, info, end_date=end, adjust_nasdaq=not no_nasdaq_adjust)
        for name, f in frames.items():
            f.to_csv(out_dir / f'{name}.csv', index=False)
        master.append(m)
        for k in total:
            total[k] += rep[k]
        print(f'  {min(i + BATCH, len(permnos)):>6,} / {len(permnos):,}   dead so far {total["n_dead"]:,}', flush=True)

    print('[3/3] Writing the security master...')
    sm = pd.concat(master, ignore_index=True)
    sm.to_csv(paths.raw_base / 'security_master.csv', index=False)
    (paths.raw_base / 'config.json').write_text(json.dumps({
        'vendor': 'CRSP (WRDS), crsp.dsf_v2', 'start': start, 'end': end, 'min_cap_thousands': min_cap,
        'nasdaq_volume_adjusted': not no_nasdaq_adjust, 'report': total, 'n_files': len(sm)}, indent=2))
    print(f'  {len(sm):,} stocks written, {total["n_dead"]:,} dead.')
    print(f'  delisting return assumed (-30%) for {total["terminal_assumed"]} stocks (CRSP had none)')
    print(f'  open/high/low missing and filled with the close: {total["ohl_filled"]:,} rows')
    print('Next: python scripts/pipeline/02_etl_prices.py ; python scripts/pipeline/02b_build_universe.py')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Download CRSP daily stock data from WRDS in the Yahoo layout')
    ap.add_argument('--wrds_user', default=None)
    ap.add_argument('--start', default='1999-01-01')
    ap.add_argument('--end', default='2026-11-20')
    ap.add_argument('--min_cap', type=float, default=200000, help='market cap floor in THOUSANDS of dollars')
    ap.add_argument('--probe', action='store_true', help='print the real column names and stop')
    ap.add_argument('--dry_run', action='store_true', help='print the SQL and stop')
    ap.add_argument('--no_nasdaq_adjust', action='store_true')
    a = ap.parse_args()
    main(a.wrds_user, a.start, a.end, a.min_cap, a.probe, a.dry_run, a.no_nasdaq_adjust)
