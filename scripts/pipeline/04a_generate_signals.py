#!/usr/bin/env python
'''
===============================================
Phase 0a: Generate Signals (on-demand via make_signal)
===============================================

Generates signals from features using make_signal().
No persistent signals.parquet - signals are created fresh each run.

Outputs:
  data/processed/signals_<tag>.parquet   # [date, ticker, <signal_cols>, fwd_ret_<H>d]

Usage:
  python scripts/pipeline/04a_generate_signals.py
'''

from src.signals.combine import make_signal
from src.features.returns import add_forward_returns
import polars as pl

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input
FEATURES_FILE = 'data/processed/features.parquet'
PRICES_FILE = 'data/processed/prices.parquet'

# Output (tagged so you can keep multiple signal sets)
SIGNAL_TAG = 'default'  # e.g., 'mom_rev', 'all_zscore', 'rank_beta'
OUTPUT_FILE = f'data/processed/signals_{SIGNAL_TAG}.parquet'

# Signal specifications: list of (feature_col, method, params)
# method: 'zscore_tanh', 'zscore_clip', 'rank', 'decile'
SIGNAL_SPECS = [
    # Momentum family
    ('mom5', 'zscore_tanh', {'clip_val': 3}),
    ('mom20', 'zscore_tanh', {'clip_val': 3}),
    ('mom60', 'zscore_tanh', {'clip_val': 3}),
    ('mom126', 'zscore_tanh', {'clip_val': 3}),
    ('mom252', 'zscore_tanh', {'clip_val': 3}),
    # Volatility family
    ('std20', 'zscore_tanh', {'clip_val': 3}),
    ('std60', 'zscore_tanh', {'clip_val': 3}),
    ('std126', 'zscore_tanh', {'clip_val': 3}),
    ('std252', 'zscore_tanh', {'clip_val': 3}),
    # ADV family
    ('log_adv20', 'zscore_tanh', {'clip_val': 3}),
    ('log_adv60', 'zscore_tanh', {'clip_val': 3}),
    ('log_adv126', 'zscore_tanh', {'clip_val': 3}),
    # Beta family (use rank - not fully trusted yet)
    ('beta60', 'rank', {}),
    ('beta126', 'rank', {}),
    ('beta252', 'rank', {}),
]

# Forward return horizons to compute
HORIZONS = [1, 5, 10, 20]

# Date range (optional, None = all)
DATE_START = None  # '2010-01-01'
DATE_END = None    # '2024-12-31'


def main():
    print('=' * 70)
    print('PHASE 0a: GENERATE SIGNALS (on-demand)')
    print('=' * 70)
    print(f'Features:      {FEATURES_FILE}')
    print(f'Output:        {OUTPUT_FILE}')
    print(f'Signal tag:    {SIGNAL_TAG}')
    print(f'Signal specs:  {len(SIGNAL_SPECS)} signals')
    for feat, method, params in SIGNAL_SPECS:
        print(f'    {feat:15s} -> {method:12s} {params}')
    print(f'Horizons:      {HORIZONS}')
    print('=' * 70)

    # Load features
    print('\n[1/3] Loading features...')
    features_lf = pl.scan_parquet(FEATURES_FILE)

    if DATE_START or DATE_END:
        features_lf = features_lf.filter(
            (pl.col('date') >= DATE_START) & (pl.col('date') <= DATE_END)
        )

    features_df = features_lf.collect()
    print(f'  Loaded {len(features_df):,} rows, {len(features_df.columns)} cols')

    # Generate each signal
    print('\n[2/3] Generating signals...')
    signal_cols = []
    for feat_col, method, params in SIGNAL_SPECS:
        if feat_col not in features_df.columns:
            print(f'  ⚠ {feat_col} not in features, skipping')
            continue

        signal_name = f'{feat_col}_{method}'
        print(f'  {feat_col:15s} -> {signal_name}')

        features_df = make_signal(
            features_df,
            feature_col=feat_col,
            method=method,
            output_col=signal_name,
            **params
        )
        signal_cols.append(signal_name)

    print(f'\n  Generated {len(signal_cols)} signal columns')

    # Add forward returns (using prices for correct open-to-open)
    print('\n[3/3] Adding forward returns...')
    prices_lf = pl.scan_parquet(PRICES_FILE)
    if DATE_START or DATE_END:
        prices_lf = prices_lf.filter(
            (pl.col('date') >= DATE_START) & (pl.col('date') <= DATE_END)
        )
    prices_df = prices_lf.collect()

    for h in HORIZONS:
        features_df = add_forward_returns(features_df, prices_df, horizon=h)
        print(f'  Added fwd_ret_{h}d')

    # Save
    print(f'\nSaving to {OUTPUT_FILE}...')
    output_cols = ['date', 'ticker'] + signal_cols + [f'fwd_ret_{h}d' for h in HORIZONS]
    features_df.select(output_cols).write_parquet(OUTPUT_FILE)
    print(f'  Saved {len(features_df):,} rows x {len(output_cols)} cols')

    print('\nDone!')
    print(f'\nNext: python scripts/pipeline/04b_evaluate_signals.py --signal-tag {SIGNAL_TAG}')


if __name__ == '__main__':
    main()