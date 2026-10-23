#!/usr/bin/env python
'''
===============================================
Phase 0c: Fit Elastic Net Combiner
===============================================

Fits Elastic Net on selected signals from Phase 0b.
Produces combined_score for all dates.

Inputs:
  data/processed/signals_<tag>.parquet
  data/exports/signal_eval_<tag>/selected_signals.json

Outputs:
  data/processed/en_scores_<tag>.parquet   # [date, ticker, combined_score, <signal_cols>]
  data/exports/en_model_<tag>.json         # Model coefficients, CV results

Usage:
  python scripts/pipeline/04c_fit_combiner.py --signal-tag default
'''

import argparse
import json
from pathlib import Path
import numpy as np
import polars as pl

from src.models.elastic_net_combiner import fit_elastic_net_combiner

# =============================================================================
# CONFIGURATION
# =============================================================================

SIGNAL_TAG = 'default'

# EN Configuration
EN_CONFIG = {
    'alpha': 'auto',           # 'auto' or float
    'l1_ratio': 'auto',        # 'auto' or float
    'cv_method': 'walkforward', # 'walkforward' or 'cpcv'
    'q_value_weighting': True,  # Apply (1-q) column weighting
    'score_col': 'combined_score',
}

# Walk-forward CV config
WF_CONFIG = {
    'n_folds': 5,
    'embargo': 37,
}

# CPCV config (if cv_method='cpcv')
CPCV_CONFIG = {
    'n_blocks': 12,
    'n_test_blocks': 2,
    'embargo': 37,
}

# Training date range
TRAIN_START = '2010-01-01'
TRAIN_END = '2024-12-31'


def main(signal_tag: str):
    print('=' * 70)
    print('PHASE 0c: FIT ELASTIC NET COMBINER')
    print('=' * 70)
    print(f'Signal tag:    {signal_tag}')
    print(f'EN config:     {EN_CONFIG}')
    print(f'CV config:     {WF_CONFIG if EN_CONFIG["cv_method"]=="walkforward" else CPCV_CONFIG}')
    print(f'Train period:  {TRAIN_START} to {TRAIN_END}')
    print('=' * 70)

    # Load selected signals from evaluation
    eval_dir = f'data/exports/signal_eval_{signal_tag}'
    selected_file = f'{eval_dir}/selected_signals.json'
    if not Path(selected_file).exists():
        raise FileNotFoundError(f'Selected signals not found: {selected_file}. Run 04b_evaluate_signals.py first.')

    with open(selected_file) as f:
        selected_data = json.load(f)

    signal_cols = selected_data['signals']
    fwd_ret_col = selected_data['fwd_ret_col']
    print(f'\nUsing {len(signal_cols)} selected signals: {signal_cols}')
    print(f'Target: {fwd_ret_col}')

    # Load q-values for weighting
    q_values = {}
    if EN_CONFIG['q_value_weighting']:
        fdr_file = f'{eval_dir}/fdr_report.parquet'
        if Path(fdr_file).exists():
            fdr_df = pl.read_parquet(fdr_file)
            for row in fdr_df.iter_rows(named=True):
                if row['signal'] in signal_cols:
                    q_values[row['signal']] = row['q_value']
            print(f'  Loaded q-values for {len(q_values)} signals')

    # Load signal data
    signal_file = f'data/processed/signals_{signal_tag}.parquet'
    print(f'\nLoading {signal_file}...')
    df = pl.read_parquet(signal_file)

    # Filter to train period
    train_df = df.filter(
        (pl.col('date') >= TRAIN_START) & (pl.col('date') <= TRAIN_END)
    ).select(['date', 'ticker'] + signal_cols + [fwd_ret_col])

    print(f'  Train samples: {len(train_df):,}')

    # Fit Elastic Net
    print('\nFitting Elastic Net...')
    result = fit_elastic_net_combiner(
        signals_df=train_df,
        signal_cols=signal_cols,
        fwd_ret_col=fwd_ret_col,
        alpha=EN_CONFIG['alpha'] if EN_CONFIG['alpha'] != 'auto' else None,
        l1_ratio=EN_CONFIG['l1_ratio'] if EN_CONFIG['l1_ratio'] != 'auto' else None,
        cv_method=EN_CONFIG['cv_method'],
        n_folds=WF_CONFIG['n_folds'],
        embargo=WF_CONFIG['embargo'],
        q_values=q_values if EN_CONFIG['q_value_weighting'] else None,
    )

    coef = result['coefficients']
    intercept = result['intercept']
    alpha_used = result['alpha']
    l1_ratio_used = result['l1_ratio']

    print(f'\nFitted: alpha={alpha_used:.6f}, l1_ratio={l1_ratio_used:.3f}')
    print(f'Non-zero: {(np.abs(coef) > 1e-6).sum()}/{len(coef)}')
    for name, c in zip(signal_cols, coef):
        if abs(c) > 1e-6:
            print(f'  {name:25s}: {c:10.6f}')

    # Score FULL dataset (not just train)
    print('\nScoring full dataset...')
    X_full = df.select(signal_cols).to_numpy()
    X_full = np.nan_to_num(X_full, nan=0.0)  # Impute missing with 0 (neutral)
    scores = X_full @ coef + intercept

    df = df.with_columns(pl.Series(name=EN_CONFIG['score_col'], values=scores))

    # Save scores
    output_file = f'data/processed/en_scores_{signal_tag}.parquet'
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.select(['date', 'ticker', EN_CONFIG['score_col']] + signal_cols).write_parquet(output_file)
    print(f'\n✓ Saved scores to {output_file} ({len(df):,} rows)')

    # Save model info
    model_info = {
        'signal_tag': signal_tag,
        'alpha': float(alpha_used),
        'l1_ratio': float(l1_ratio_used),
        'signals': signal_cols,
        'coefficients': {k: float(v) for k, v in zip(signal_cols, coef)},
        'intercept': float(intercept),
        'cv_method': EN_CONFIG['cv_method'],
        'q_weighting': EN_CONFIG['q_value_weighting'],
        'train_start': TRAIN_START,
        'train_end': TRAIN_END,
        'fwd_ret_col': fwd_ret_col,
    }
    model_file = f'data/exports/en_model_{signal_tag}.json'
    Path(model_file).parent.mkdir(parents=True, exist_ok=True)
    with open(model_file, 'w') as f:
        json.dump(model_info, f, indent=2, default=str)
    print(f'  Model info: {model_file}')

    print('\nDone!')
    print(f'\nNext: python scripts/pipeline/04d_estimate_risk_model.py --signal-tag {signal_tag}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--signal-tag', type=str, default=SIGNAL_TAG)
    args = parser.parse_args()
    main(args.signal_tag)