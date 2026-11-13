#!/usr/bin/env python
'''
===============================================
Phase 04a: Generate Signals
===============================================

QUESTION ANSWERED
    "Which candidate signals do we test, and what are the returns they will be tested against?"

THE STAGE IN ONE PICTURE

    features.parquet                       SIGNAL_SPECS (below)            HORIZONS (below)
    (mom20, vol60, log_adv60, open_adj)    (feature, transform, params)    (1, 5, 10, 20 days)
            |                                      |                              |
            v                                      v                              v
     [1] load the features, cut the dates (optional)
            |
            v
     [2] src/signals/signal_pipeline.py
            clean features    NaN and inf -> null
            build signals     one column per spec, normalised across the stocks of the SAME date
            add labels        fwdret{h} = log(open_adj[t+1+h]) - log(open_adj[t+1])
            |
            v
     [3] checks: labels aligned per ticker, no missing dates, signals not empty or constant
            |
            v
     [4] signals.parquet   +   config.json (specs, hashes, git commit, descriptive statistics)

WHERE IS THE COMPLICATED PART?
    src/signals/signal_pipeline.py   cleaning, signals, labels, checks, statistics
    src/signals/combine.py           the transforms (zscore_tanh, zscore_clip, rank, decile)
    This script only lists the signals you want and calls them, in order.

WHAT 04a NEVER DOES
    It never measures how well a signal predicts returns (no IC, no Sharpe, no drawdown).
    That is stage 04b, on the search period only. Looking at IC over all dates here would let the
    holdout influence which signals you decide to build. The statistics in config.json use no returns.

THE CLOCK
    The signal of row t uses data up to the close of day t.
    The label of row t is the return from the open of t+1 to the open of t+1+h. It uses only later prices.

HOW TO ADD A SIGNAL
    Add one line to SIGNAL_SPECS: (feature column, transform, parameters). Then run 04a, 04b, ...
    Every signal you ever try counts as one test for the false discovery rate in 04b.
    Keep the list of what you tried, also the ones you removed.

INPUTS    {clean_base}/features/features.parquet
OUTPUTS   {clean_base}/signals/{run_id}/
    signals.parquet   [date, ticker, <signals>, fwdret1, fwdret5, fwdret10, fwdret20]
    config.json       specs, skipped specs, hashes, git commit, statistics, label checks

USAGE
    python scripts/pipeline/04a_generate_signals.py --date 2026-11-07 --tag default
'''

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Make `import src...` and `import config...` work wherever the script is started from.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import polars as pl

from config.paths import get_paths, run_id_from_args
from src.signals.signal_pipeline import (
    validate_specs, clean_features, build_signals, add_labels, restrict_dates,
    check_labels, calendar_gaps, signal_descriptives, signal_warnings,
)
from src.utils.helpers import get_git_commit, get_git_diff, file_hash, params_hash


# =============================================================================
# CONFIGURATION
# =============================================================================

# One line per signal: (feature column, transform, parameters). The signal is named '{feature}_{transform}'.
#   'zscore_tanh'  z-score across the stocks of the date, then tanh.   params: scale (default 1.0)
#   'zscore_clip'  z-score, then clip.                                 params: low, high (default -3, 3)
#   'zscore'       z-score only.
#   'rank'         rank across the stocks, scaled to [-1, 1].          params: descending (default False)
#   'decile'       +1 top decile, -1 bottom decile, 0 otherwise.       params: n_buckets (default 10)
# The z-score transforms take `descending=True` to flip the sign (a feature where LOW is good).
SIGNAL_SPECS = [
    # Momentum: several horizons
    ('mom5',   'zscore_tanh', {}),
    ('mom10',  'zscore_tanh', {}),
    ('mom20',  'zscore_tanh', {}),
    ('mom60',  'zscore_tanh', {}),
    ('mom120', 'zscore_tanh', {}),
    ('mom252', 'zscore_tanh', {}),

    # Volatility (the sign is NOT flipped here: 04b and the combiner learn the sign from the data)
    ('vol5',   'zscore_clip', {'low': -3, 'high': 3}),
    ('vol10',  'zscore_clip', {'low': -3, 'high': 3}),
    ('vol20',  'zscore_clip', {'low': -3, 'high': 3}),
    ('vol60',  'zscore_clip', {'low': -3, 'high': 3}),
    ('vol120', 'zscore_clip', {'low': -3, 'high': 3}),
    ('vol252', 'zscore_clip', {'low': -3, 'high': 3}),

    # Liquidity (average dollar volume, log)
    ('log_adv10', 'zscore_tanh', {}),
    ('log_adv20', 'zscore_tanh', {}),
    ('log_adv60', 'zscore_tanh', {}),

    # Tried and switched off (keep this list: every signal you try counts for the false discovery rate)
    # ('log_adv5',   'zscore_tanh', {}),
    # ('log_adv120', 'zscore_tanh', {}),
    # ('log_adv252', 'zscore_tanh', {}),
    # ('beta60',  'rank', {'descending': True}),
    # ('beta120', 'rank', {'descending': True}),
    # ('beta252', 'rank', {'descending': True}),
]

# Forward-return horizons (trading days). 04b and 04c use ONE of them (default 20).
HORIZONS = [1, 5, 10, 20]

# Optional date cut, 'YYYY-MM-DD' or None. The labels are built after the cut, so the last rows have no label.
DATE_START = None
DATE_END = None

MAX_MISSING_SHARE = 0.01        # warn if more than 1% of the expected (ticker, date) rows are missing


def main(run_date: str, tag: str):
    start = time.time()
    run_id = run_id_from_args(run_date, tag)
    cfg = get_paths(run_id=run_id)
    out_dir = cfg.stage_dir('signals')

    if not cfg.features_file.exists():
        raise FileNotFoundError(f'Missing input: {cfg.features_file}. Run 03_build_features.py first.')

    print('=' * 70)
    print('PHASE 04a: GENERATE SIGNALS')
    print('=' * 70)
    print(f'Run ID:    {run_id}')
    print(f'Dataset:   {cfg.asset}/{cfg.region}/{cfg.universe}/{cfg.freq}')
    print(f'Features:  {cfg.features_file}')
    print(f'Output:    {out_dir}')
    print(f'Horizons:  {HORIZONS}')
    print(f'Dates:     {DATE_START or "first"} to {DATE_END or "last"}')

    # ------------------------------------------------------------------
    # [1/4] Load (lazy: nothing is read yet) and validate the specs
    # ------------------------------------------------------------------
    print('\n[1/4] Loading the features...')
    lf = restrict_dates(pl.scan_parquet(cfg.features_file), DATE_START, DATE_END)
    available = set(lf.collect_schema().names())
    missing_cols = {'date', 'ticker', 'open_adj'} - available
    if missing_cols:
        raise ValueError(f'features.parquet lacks the columns {sorted(missing_cols)}')

    specs, skipped = validate_specs(SIGNAL_SPECS, available)
    for sk in skipped:
        print(f'  WARNING: skipped {sk["spec"][0]} ({sk["reason"]})')
    if not specs:
        raise ValueError('No signal can be built: none of the SIGNAL_SPECS features is in features.parquet.')
    print(f'  {len(specs)} signals to build, {len(skipped)} skipped')

    # ------------------------------------------------------------------
    # [2/4] Build the signals and the labels (lazy), then compute once
    # ------------------------------------------------------------------
    print('\n[2/4] Building signals and labels...')
    lf = clean_features(lf, sorted({feat for feat, _, _ in specs}))
    lf, signal_cols = build_signals(lf, specs)
    lf, label_cols = add_labels(lf, HORIZONS)
    for s in signal_cols:
        print(f'  {s}')
    print(f'  labels: {label_cols}')

    output_cols = ['date', 'ticker'] + signal_cols + label_cols
    df = lf.select(output_cols).collect()
    print(f'  {df.height:,} rows x {len(output_cols)} columns, {df["ticker"].n_unique()} tickers, '
          f'{df["date"].n_unique()} dates ({df["date"].min()} to {df["date"].max()})')

    # ------------------------------------------------------------------
    # [3/4] Checks
    # ------------------------------------------------------------------
    print('\n[3/4] Checks...')
    label_report = check_labels(df, HORIZONS)                       # raises if labels cross tickers
    for col, r in label_report.items():
        print(f'  {col:<10} null {r["null_pct"]:5.2f}%   std {r["std"]:.4f}   max |value| {r["max_abs"]:.2f}'
              f'   rows with |value| > 2: {r["n_abs_above_2"]}')
        if r['n_abs_above_2']:
            print(f'    WARNING: {r["n_abs_above_2"]} extreme labels. Look for an unadjusted split in the prices.')

    gaps = calendar_gaps(lf)
    print(f'  missing (ticker, date) rows inside the life of the tickers: {gaps["missing_rows"]:,} '
          f'({gaps["missing_share"]:.2%}); worst ticker {gaps["worst_ticker"]} ({gaps["worst_ticker_missing"]} rows)')
    if gaps['missing_share'] > MAX_MISSING_SHARE:
        print('  WARNING: labels shift by rows, so missing dates make the real horizon longer than h days.')

    stats = signal_descriptives(df, signal_cols)
    warnings = signal_warnings(stats)
    for w in warnings:
        print(f'  WARNING: {w}')
    if not warnings:
        print('  signals: none empty, constant or mostly null')

    # ------------------------------------------------------------------
    # [4/4] Save
    # ------------------------------------------------------------------
    print('\n[4/4] Saving...')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'signals.parquet'
    df.write_parquet(out_file)

    config = {
        'run_id': run_id,
        'signal_tag': tag,
        'run_date': run_date,
        'run_timestamp': datetime.now().isoformat(),
        'signal_specs': SIGNAL_SPECS,
        'skipped_specs': skipped,
        'signal_cols': signal_cols,
        'horizons': HORIZONS,
        'date_start': DATE_START,
        'date_end': DATE_END,
        'n_signals': len(signal_cols),
        'n_rows': df.height,
        'output_cols': output_cols,
        'source': cfg.source, 'asset': cfg.asset, 'region': cfg.region, 'universe': cfg.universe, 'freq': cfg.freq,
        # provenance: which code and which data made this file
        'git_commit': get_git_commit(),
        'git_diff': get_git_diff() or None,
        'features_file': str(cfg.features_file),
        'features_hash': file_hash(cfg.features_file),
        'params_hash': params_hash([SIGNAL_SPECS, HORIZONS, DATE_START, DATE_END]),
        'script_path': str(Path(__file__).resolve()),
        # statistics that use NO returns
        'signal_stats': stats,
        'warnings': warnings,
        'label_checks': label_report,
        'calendar_gaps': gaps,
    }
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)

    print(f'  signals: {out_file}')
    print(f'  config:  {out_dir / "config.json"}')
    print(f'\nDone in {time.time() - start:.1f} s.')
    print(f'Next: python scripts/pipeline/04b_evaluate_signals.py --date {run_date} --tag {tag}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate the candidate signals and the forward returns')
    parser.add_argument('--date', type=str, required=True, help='Run date, YYYY-MM-DD (used in the run_id)')
    parser.add_argument('--tag', type=str, default='default', help='Run tag (default: default)')
    args = parser.parse_args()
    main(args.date, args.tag)