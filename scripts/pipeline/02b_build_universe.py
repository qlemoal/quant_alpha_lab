#!/usr/bin/env python
'''
===============================================
Phase 02b: Build the point-in-time universe (and audit for survivorship)
===============================================

QUESTION ANSWERED
    "On each date, which stocks could we really have traded, using only what was known that day?
     And does my price file still look like 'survivors only'?"

    ohlcv.parquet (02) --+--> rule_universe       (top N by dollar volume, re-ranked every k days)   mode "rule"
                         +--> membership.csv      (vendor index list)                                mode "membership"
                         +--> every ticker, always                                                   mode "all"
                                    |
                                    v
                       universe.parquet [date, ticker, in_universe]   --> read by 04
                       universe_audit.json                            --> the survivorship verdict

CHOOSE THE MODE in the dataset config (config/dataset_config.json), keys "universe_mode" and "universe_rule".
    "all"         the old behaviour. Fine for the synthetic market. For real data it is the survivorship trap.
    "rule"        recommended. Needs a price file that CONTAINS the dead companies (CRSP, Sharadar, Norgate ...).
    "membership"  your vendor's index list: data/raw/{source}/.../membership.csv with columns ticker,start,end
                  (end empty = still a member), or an event list date,ticker,action (see docs/survivorship.md).

THE VERDICT (read it before you trust any result of 04 to 09)
    FAIL  no dead ticker: the file is survivors only. Any result is biased upwards.
    WARN  few dead tickers or a low delisting rate: probably incomplete.
    OK    dead tickers at a plausible rate.
    The audit cannot PROVE that the file is complete. It catches the typical failures.

USAGE
    python scripts/pipeline/02b_build_universe.py
    python scripts/pipeline/02b_build_universe.py --mode rule --top_n 500      (override the config)
'''

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import polars as pl

from config.paths import get_paths, UNIVERSE_MODE, UNIVERSE_RULE
from src.universe.point_in_time import (
    DEFAULT_RULE, rule_universe, membership_mask, membership_from_events, audit_universe,
)


def load_membership(path: Path) -> pl.DataFrame:
    '''Read membership.csv. Two layouts: [ticker,start,end] or an event list [date,ticker,action].'''
    raw = pl.read_csv(path, try_parse_dates=True)
    cols = set(raw.columns)
    if {'ticker', 'start', 'end'} <= cols:
        return raw.with_columns(pl.col('start').cast(pl.Date), pl.col('end').cast(pl.Date))
    if {'date', 'ticker', 'action'} <= cols:
        return membership_from_events(raw.with_columns(pl.col('date').cast(pl.Date)))
    raise ValueError(f'{path}: need the columns ticker,start,end or date,ticker,action. Got {raw.columns}')


def main(mode: str, rule: dict):
    t0 = time.time()
    paths = get_paths()
    print('=' * 70)
    print('PHASE 02b: POINT-IN-TIME UNIVERSE')
    print('=' * 70)
    print(f'Dataset: {paths.asset}/{paths.region}/{paths.universe}   mode: {mode}')
    if not paths.ohlcv_file.exists():
        raise FileNotFoundError(f'Missing {paths.ohlcv_file}. Run 02_etl_prices.py first.')
    ohlcv = pl.read_parquet(paths.ohlcv_file, columns=['ticker', 'date', 'close_raw', 'close', 'volume'])
    print(f'  {ohlcv.height:,} rows, {ohlcv["ticker"].n_unique()} tickers, {ohlcv["date"].min()} to {ohlcv["date"].max()}')

    if mode == 'all':
        uni = ohlcv.select(['date', 'ticker']).with_columns(pl.lit(True).alias('in_universe'))
        print('  mode "all": every ticker is in the universe on every date it trades.')
    elif mode == 'rule':
        print(f'  rule: {rule}')
        uni = rule_universe(ohlcv, **rule).select(['date', 'ticker', 'in_universe'])
    elif mode == 'membership':
        f = paths.raw_membership_file
        if not f.exists():
            raise FileNotFoundError(f'Missing {f}. Put the vendor index list there (ticker,start,end).')
        m = load_membership(f)
        print(f'  membership: {m.height:,} intervals, {m["ticker"].n_unique()} tickers')
        uni = membership_mask(ohlcv.select(['date', 'ticker']), m)
        covered = set(m['ticker'].to_list())
        missing = sorted(covered - set(ohlcv['ticker'].unique().to_list()))
        print(f'  index members with NO price file: {len(missing)}' + (f'   e.g. {missing[:8]}' if missing else ''))
        if missing:
            print('  WARNING: members without prices are silently absent from every backtest. Check the ticker mapping.')
    else:
        raise ValueError(f'universe_mode must be all, rule or membership. Got {mode!r}')

    out = paths.universe_file
    out.parent.mkdir(parents=True, exist_ok=True)
    uni.write_parquet(out)

    audit = audit_universe(ohlcv, uni)
    print('\nAUDIT')
    print(f'  tickers {audit["n_tickers"]}, dead {audit["n_dead"]} ({audit["share_dead"]:.0%}), span {audit["years_span"]:.1f} years')
    print(f'  median yearly delisting rate: {audit["delist_rate_median"]:.1%}')
    if audit['last_return_median'] is not None:
        print(f'  last-day return of dead tickers: median {audit["last_return_median"]:+.1%}, '
              f'{audit["share_crash"]:.0%} below -20%')
    if 'members_median' in audit:
        print(f'  members per date: min {audit["members_min"]}, median {audit["members_median"]:.0f}, max {audit["members_max"]}')
    print(f'  VERDICT: {audit["verdict"]}')
    for n in audit['notes']:
        print(f'    - {n}')
    (out.parent / 'universe_audit.json').write_text(json.dumps({**audit, 'mode': mode, 'rule': rule if mode == 'rule' else None},
                                                              indent=2, default=str))
    print(f'\n  universe: {out}\nDone in {time.time() - t0:.1f} s.\nNext: python scripts/pipeline/03_build_features.py')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Build the point-in-time universe and the survivorship audit')
    ap.add_argument('--mode', choices=['all', 'rule', 'membership'], default=UNIVERSE_MODE)
    ap.add_argument('--top_n', type=int, default=None)
    ap.add_argument('--adv_window', type=int, default=None)
    ap.add_argument('--min_adv', type=float, default=None)
    ap.add_argument('--reconstitute_every', type=int, default=None)
    a = ap.parse_args()
    rule = {**DEFAULT_RULE, **UNIVERSE_RULE}
    for k in ('top_n', 'adv_window', 'min_adv', 'reconstitute_every'):
        if getattr(a, k) is not None:
            rule[k] = getattr(a, k)
    main(a.mode, rule)
