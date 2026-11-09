#!/usr/bin/env python
'''
===============================================
Phase 0a: Generate Signals (on-demand via make_signal)
===============================================

Generates signals from features using make_signal().
No persistent signals.parquet - signals are created fresh each run.

Outputs:
  data/clean/{asset}/{region}/{universe}/{freq}/signals/{run_id}/signals.parquet
  data/clean/{asset}/{region}/{universe}/{freq}/signals/{run_id}/config.json

Usage:
    -> Dataset defined in config/dataset_config.json
    python scripts/pipeline/04a_generate_signals.py --date 2026-11-07 --tag default
  
'''

import argparse
from src.signals.combine import make_signal
from src.features.returns import add_fwdret_horizon
from src.evaluation.signals.ic.core import compute_ic, summarize_ic
from src.evaluation.signals.performance import decile_longshort_returns, sharpe_ratio, max_drawdown, compute_stability
from src.utils.helpers import get_git_commit, get_git_diff, file_hash, params_hash
import polars as pl
import json
import os
import numpy as np
from datetime import datetime
from pathlib import Path

# Import centralized paths
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'config'))
from config.paths import get_paths, run_id_from_args

# =============================================================================
# ARGUMENT PARSING
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--date', type=str, required=True,
                    help='Date in YYYY-MM-DD format (e.g., 2026-10-29)')
parser.add_argument('--tag', type=str, default='default',
                    help='Signal tag (default: default)')
args = parser.parse_args()

RUN_DATE = args.date
SIGNAL_TAG = args.tag
run_id = run_id_from_args(RUN_DATE, SIGNAL_TAG)

# Create data config from dataset_config.json (no CLI args needed)
cfg = get_paths(run_id=run_id)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input (from config)
FEATURES_FILE = cfg.features_file

# Output (tagged so you can keep multiple signal sets)
OUTPUT_DIR = cfg.stage_dir('signals')
OUTPUT_FILE = OUTPUT_DIR / 'signals.parquet'

# Signal specifications: list of (feature_col, method, params)
# method: 'zscore_tanh', 'zscore_clip', 'rank', 'decile'
# kwargs per method:
#   zscore_tanh: scale (default 1.0) -- tanh naturally bounds to [-1, 1], no clipping needed
#   zscore_clip: low, high (default -3, 3) -- hard clipping after z-score
#   rank: descending (default False)
#   decile: n_buckets (default 10)
SIGNAL_SPECS = [
    # Momentum family - multiple horizons
    ('mom5', 'zscore_tanh', {}),
    ('mom10', 'zscore_tanh', {}),
    ('mom20', 'zscore_tanh', {}),
    ('mom60', 'zscore_tanh', {}),
    ('mom120', 'zscore_tanh', {}),
    ('mom252', 'zscore_tanh', {}),
    
    # Volatility family (inverse signal - low vol tends to outperform)
    ('vol5', 'zscore_clip', {'low': -3, 'high': 3}),
    ('vol10', 'zscore_clip', {'low': -3, 'high': 3}),
    ('vol20', 'zscore_clip', {'low': -3, 'high': 3}),
    ('vol60', 'zscore_clip', {'low': -3, 'high': 3}),
    ('vol120', 'zscore_clip', {'low': -3, 'high': 3}),
    ('vol252', 'zscore_clip', {'low': -3, 'high': 3}),
    
    # # ADV family - liquidity signal (higher ADV = more liquid = lower returns?)
    # ('log_adv5', 'zscore_tanh', {}),
    ('log_adv10', 'zscore_tanh', {}),
    ('log_adv20', 'zscore_tanh', {}),
    ('log_adv60', 'zscore_tanh', {}),
    # ('log_adv120', 'zscore_tanh', {}),
    # ('log_adv252', 'zscore_tanh', {}),
    
    # Beta family (rank - robust to outliers)
    # ('beta60', 'rank', {'descending': True}),
    # ('beta120', 'rank', {'descending': True}),
    # ('beta252', 'rank', {'descending': True}),
]

# Forward return horizons to compute
HORIZONS = [1, 5, 10, 20]

# Date range (optional, None = all)
DATE_START = None  # '2010-01-01'
DATE_END = None    # '2024-12-31'


# ---------------------------------------------------------------------
#  Printing the chosen parameters before generating the signals
print('=' * 70)
print('PHASE 4.a: GENERATE SIGNALS (on-demand)')
print('=' * 70)
print(f'Dataset:       {cfg.asset}/{cfg.region}/{cfg.universe}/{cfg.freq}')
print(f'Features:      {FEATURES_FILE}')
print(f'Output:        {OUTPUT_FILE}')
print(f'Signal tag:    {SIGNAL_TAG}')
print(f'Signal specs:  {len(SIGNAL_SPECS)} signals')
for feat, method, params in SIGNAL_SPECS:
    print(f'    {feat:15s} -> {method:12s} {params}')
print(f'Horizons:      {HORIZONS}')
print('=' * 70)


# ---------------------------------------------------------------------
# Load features


print('\n[1/3] Loading features...')
features_lf = pl.scan_parquet(FEATURES_FILE)

if DATE_START or DATE_END:
    features_lf = features_lf.filter(
        (pl.col('date') >= DATE_START) & (pl.col('date') <= DATE_END)
    )


# Get available columns from schema (no collect needed)
available_cols = set(features_lf.collect_schema().names())
print(f'  Loaded schema: {len(available_cols)} cols')


# ---------------------------------------------------------------------
# Generate each signal (lazy - compose expressions only)

print('\n[2/3] Generating signals...')
signal_cols = []
for feat_col, method, params in SIGNAL_SPECS:
    if feat_col not in available_cols:
        print(f'  ⚠ {feat_col} not in features, skipping')
        continue

    signal_name = f'{feat_col}_{method}'
    print(f'  {feat_col:15s} -> {signal_name}')

    features_lf = make_signal(
        features_lf,
        cols=feat_col,
        method=method,
        **params
    )
    signal_cols.append(signal_name)

print(f'\n  Generated {len(signal_cols)} signal columns (lazy)')


# ---------------------------------------------------------------------
# Add forward returns (features already have open/close, use add_fwdret_horizon)

print('\n[3/3] Adding forward returns...')
for h in HORIZONS:
    features_lf = add_fwdret_horizon(features_lf, horizon=h)
    print(f'  Added fwdret{h}')


# ---------------------------------------------------------------------
# Save (collect only at the end)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f'\nSaving to {OUTPUT_FILE}...')
output_cols = ['date', 'ticker'] + signal_cols + [f'fwdret{h}' for h in HORIZONS]
features_lf.select(output_cols).collect().write_parquet(OUTPUT_FILE)

# Collect for stats and row count
print('\n[4/4] Computing signal statistics...')
full_df = features_lf.select(output_cols).collect()
row_count = len(full_df)
print(f'  Saved {row_count:,} rows x {len(output_cols)} cols to {OUTPUT_FILE}')

# Compute per-signal stats
signal_stats = {}
for sig in signal_cols:
    s = full_df[sig]
    signal_stats[sig] = {
        'mean': float(s.mean()),
        'std': float(s.std()),
        'min': float(s.min()),
        'max': float(s.max()),
        'null_count': int(s.null_count()),
        'null_pct': float(s.null_count() / len(s) * 100),
    }

# Cross-sectional stats (per date)
cs_stats = {}
if len(signal_cols) > 0:
    cs_means = full_df.group_by('date').agg([
        pl.col(s).mean().alias(f'{s}_mean') for s in signal_cols
    ])
    cs_stds = full_df.group_by('date').agg([
        pl.col(s).std().alias(f'{s}_std') for s in signal_cols
    ])
    # Average cross-sectional mean/std across dates
    for sig in signal_cols:
        cs_stats[sig] = {
            'avg_cs_mean': float(cs_means[f'{sig}_mean'].mean()),
            'avg_cs_std': float(cs_stds[f'{sig}_std'].mean()),
        }

# IC & Performance stats (using evaluation modules)
ic_stats = {}
perf_stats = {}
if len(signal_cols) > 0 and 'fwdret20' in full_df.columns:
    fwd_col = 'fwdret20'  # main evaluation horizon
    lf = pl.scan_parquet(OUTPUT_FILE)
    
    for sig in signal_cols:
        # IC stats
        try:
            ic_lf = compute_ic(lf, sig, forward_return_col=fwd_col)
            ic_stats[sig] = summarize_ic(ic_lf)
            ic_stats[sig] = {k: (float(v) if v is not None else None) for k, v in ic_stats[sig].items()}
        except Exception as e:
            ic_stats[sig] = {'error': str(e)}
        
        # Performance (decile long-short)
        try:
            ls_lf = decile_longshort_returns(lf, sig, fwd_ret_col=fwd_col)
            ls_df = ls_lf.collect().to_pandas()
            if len(ls_df) > 0:
                perf_stats[sig] = {
                    'spread_mean': float(ls_df['spread_ret'].mean()),
                    'spread_std': float(ls_df['spread_ret'].std()),
                    'sharpe': sharpe_ratio(ls_df['spread_ret'].values),
                    'max_dd': max_drawdown(ls_df['spread_ret'].values),
                    'n_days': len(ls_df),
                }
            else:
                perf_stats[sig] = {'error': 'no data'}
        except Exception as e:
            perf_stats[sig] = {'error': str(e)}
        
        # Stability / Turnover
        try:
            turnover_lf = compute_stability(lf, sig)
            turnover_df = turnover_lf.collect().to_pandas()
            if len(turnover_df) > 0:
                perf_stats[sig]['turnover_mean'] = float(turnover_df['turnover'].mean())
                perf_stats[sig]['turnover_std'] = float(turnover_df['turnover'].std())
                perf_stats[sig]['turnover_median'] = float(turnover_df['turnover'].median())
            else:
                perf_stats[sig]['turnover_error'] = 'no data'
        except Exception as e:
            perf_stats[sig]['turnover_error'] = str(e)

# Save config for reproducibility
config = {
    'signal_tag': SIGNAL_TAG,
    'run_date': RUN_DATE,
    'run_timestamp': datetime.now().isoformat(),
    'signal_specs': SIGNAL_SPECS,
    'horizons': HORIZONS,
    'date_start': DATE_START,
    'date_end': DATE_END,
    'n_signals': len(signal_cols),
    'n_rows': row_count,
    'output_cols': output_cols,
    # Dataset configuration
    'source': cfg.source,
    'asset': cfg.asset,
    'region': cfg.region,
    'universe': cfg.universe,
    'freq': cfg.freq,
    # Provenance
    'git_commit': get_git_commit(),
    'git_diff': get_git_diff() or None,
    'features_file': str(FEATURES_FILE),
    'features_hash': file_hash(FEATURES_FILE),
    'params_hash': params_hash(SIGNAL_SPECS),
    'script_path': __file__,
    # Signal statistics
    'signal_stats': signal_stats,
    'cross_sectional_stats': cs_stats,
    'ic_stats': ic_stats,
    'performance_stats': perf_stats,
}
config_file = cfg.stage_dir('signals') / 'config.json'
with open(config_file, 'w') as f:
    json.dump(config, f, indent=2, default=str)
print(f'  Config saved to {config_file}')
print(f'  Signal stats computed for {len(signal_cols)} signals')

# Print quick summary
print('\n--- Quick Signal Summary ---')
for sig in signal_cols:
    ic = ic_stats.get(sig, {})
    perf = perf_stats.get(sig, {})
    print(f'  {sig}:')
    if 'error' not in ic:
        print(f'    IC: mean={ic.get("mean", 0):.4f}, IR={ic.get("ir", 0):.3f}, hit_rate={ic.get("hit_rate", 0):.2%}')
    if 'error' not in perf:
        print(f'    LS: Sharpe={perf.get("sharpe", 0):.3f}, MaxDD={perf.get("max_dd", 0):.3f}, Turnover={perf.get("turnover_mean", 0):.4f}')

print('\nDone!')
print(f'\nOutput directory: {OUTPUT_DIR}')
print(f'  signals.parquet  - the signal data')
print(f'  config.json      - full config for reproducibility')
print(f'\nNext: python scripts/pipeline/04b_evaluate_signals.py --date {RUN_DATE} --tag {SIGNAL_TAG}')






