#!/usr/bin/env python
'''
===============================================
Phase 0c: Fit Elastic Net Combiner
===============================================

Fits Elastic Net on selected signals from Phase 0b using CPCV search + holdout.
Produces combined_score for all dates.

Inputs:
  data/processed/signals/{date}_{tag}/signals.parquet
  data/processed/signal_eval/{date}_{tag}/selected_signals.json
  data/processed/signal_eval/{date}_{tag}/fdr_report.parquet (for q-values)

Outputs:
  data/processed/en_scores/{date}_{tag}/en_scores.parquet   # [date, ticker, combined_score, <signal_cols>]
  data/processed/en_scores/{date}_{tag}/config.json
  data/processed/en_model/{date}_{tag}/model.json           # Model coefficients, CV results, holdout report
  data/processed/en_model/{date}_{tag}/stability.parquet    # Per-fold coefficient stability

Usage:
  python scripts/pipeline/04c_fit_combiner.py --signal-dir data/processed/signals/2026-10-29_default
'''

import argparse
import json
from pathlib import Path
import numpy as np
import polars as pl

from src.models.elastic_net_combiner import run_en_combiner, coefficient_stability_by_fold
from config.constants import EMBARGO_WINDOW


# =============================================================================
# CONFIGURATION
# =============================================================================

EN_CONFIG = {
    'horizon': 20,
    'n_blocks': 10,
    'n_test_blocks': 2,
    'holdout_dates': 200,
    'l1_ratio_grid': [.1, .5, .7, .9, .95, .99, 1],
    'q_value_weighting': True,  # Keep Q-value weighting as requested
}

# Training: uses CPCV search region + embargo gap + holdout
# (no hardcoded TRAIN_START/TRAIN_END dates - derived from data)


def main(run_id: str):
    print('=' * 70)
    print('PHASE 0c: FIT ELASTIC NET COMBINER (CPCV + Holdout)')
    print('=' * 70)
    print(f'Run ID:    {run_id}')
    print(f'EN config:     {EN_CONFIG}')
    print('=' * 70)

    # Load selected signals from evaluation
    eval_dir = f'data/processed/signal_eval/{run_id}'
    selected_file = f'{eval_dir}/selected_signals.json'
    if not Path(selected_file).exists():
        raise FileNotFoundError(f'Selected signals not found: {selected_file}. Run 04b_evaluate_signals.py first.')

    with open(selected_file) as f:
        selected_data = json.load(f)

    signal_cols = selected_data['signals']
    fwd_ret_col = selected_data['fwd_ret_col']
    horizon = selected_data.get('horizon', EN_CONFIG['horizon'])
    print(f'\nUsing {len(signal_cols)} selected signals: {signal_cols}')
    print(f'Target: {fwd_ret_col}, horizon: {horizon}')

    # Load q-values for weighting
    q_values = {}
    if EN_CONFIG['q_value_weighting']:
        fdr_file = f'{eval_dir}/fdr_report.parquet'
        if Path(fdr_file).exists():
            fdr_df = pl.read_parquet(fdr_file)
            for row in fdr_df.iter_rows(named=True):
                if row['signal'] in signal_cols:
                    q_values[row['signal']] = row['qvalue']
            print(f'  Loaded q-values for {len(q_values)} signals: {q_values}')

    # Load signal data (versioned path)
    signal_file = Path(f'data/processed/signals/{run_id}/signals.parquet')
    print(f'\nLoading {signal_file}...')
    lf = pl.scan_parquet(signal_file)

    # Verify forward return column exists
    schema = lf.collect_schema()
    if fwd_ret_col not in schema.names():
        available = [c for c in schema.names() if c.startswith('fwdret')]
        raise ValueError(f'Forward return column {fwd_ret_col} not found. Available: {available}')

    # Run EN combiner with CPCV + holdout (uses run_en_combiner from module)
    print('\nRunning Elastic Net combiner (CPCV search + holdout)...')
    result = run_en_combiner(
        lf=lf,
        signal_cols=signal_cols,
        horizon=horizon,
        embargo_window=EMBARGO_WINDOW,
        n_blocks=EN_CONFIG['n_blocks'],
        n_test_blocks=EN_CONFIG['n_test_blocks'],
        holdout_dates=EN_CONFIG['holdout_dates'],
        q_values=q_values if EN_CONFIG['q_value_weighting'] else None,
        l1_ratio_grid=EN_CONFIG['l1_ratio_grid'],
    )

    model = result['model']
    coef = model.coef_
    intercept = model.intercept_
    alpha_used = model.alpha_
    l1_ratio_used = model.l1_ratio_

    print(f'\nFitted: alpha={alpha_used:.6f}, l1_ratio={l1_ratio_used:.3f}')
    print(f'Non-zero: {(np.abs(coef) > 1e-6).sum()}/{len(coef)}')
    for name, c in zip(signal_cols, coef):
        if abs(c) > 1e-6:
            print(f'  {name:25s}: {c:10.6f}')

    # --- Per-fold coefficient stability (diagnostic) ---
    print('\nPer-fold coefficient stability:')
    stability_df = coefficient_stability_by_fold(result, result['search_panel'])
    print(stability_df)
    
    # Save stability
    stability_dir = f'data/processed/en_model/{run_id}'
    Path(stability_dir).mkdir(parents=True, exist_ok=True)
    stability_file = f'{stability_dir}/stability.parquet'
    stability_df.write_parquet(stability_file)
    print(f'  Saved stability to {stability_file}')

    # --- Score FULL dataset ---
    print('\nScoring full dataset...')
    # Use combined_score_column to apply same q-value weighting at inference
    from src.models.elastic_net_combiner import combined_score_column
    full_lf = combined_score_column(lf, result, out_col='combined_score')
    full_df = full_lf.collect()
    
    print(f'  Full dataset: {len(full_df):,} rows')
    print(f'  Score stats: mean={full_df["combined_score"].mean():.6f}, std={full_df["combined_score"].std():.6f}')

    # Save scores
    en_scores_dir = f'data/processed/en_scores/{run_id}'
    signal_tag = run_id.split('_', 1)[1] if '_' in run_id else 'default'

    Path(en_scores_dir).mkdir(parents=True, exist_ok=True)
    output_file = f'{en_scores_dir}/en_scores.parquet'
    full_df.select(['date', 'ticker', 'combined_score'] + signal_cols).write_parquet(output_file)
    print(f'\n✓ Saved scores to {output_file}')

    # Save config for scores
    score_config = {
        'run_id': run_id,
        'signal_tag': signal_tag,
        'signal_dir': signal_dir,
        'fwd_ret_col': fwd_ret_col,
        'horizon': horizon,
        'signal_cols': signal_cols,
        'q_values': q_values,
    }
    with open(f'{en_scores_dir}/config.json', 'w') as f:
        json.dump(score_config, f, indent=2, default=str)

    # Save model info (includes holdout report)
    holdout_report = result.get('holdout_report', {})
    model_info = {
        'signal_tag': signal_tag,
        'run_id': run_id,
        'signal_dir': signal_dir,
        'alpha': float(alpha_used),
        'l1_ratio': float(l1_ratio_used),
        'signals': signal_cols,
        'coefficients': {k: float(v) for k, v in zip(signal_cols, coef)},
        'intercept': float(intercept),
        'horizon': horizon,
        'cv_method': 'cpcv',
        'n_blocks': EN_CONFIG['n_blocks'],
        'n_test_blocks': EN_CONFIG['n_test_blocks'],
        'holdout_dates': EN_CONFIG['holdout_dates'],
        'embargo_window': EMBARGO_WINDOW,
        'purge_w': result.get('purge_w'),
        'embargo_w': result.get('embargo_w'),
        'q_weighting': EN_CONFIG['q_value_weighting'],
        'q_values': q_values,
        'fwd_ret_col': fwd_ret_col,
        'holdout_report': holdout_report,
    }
    model_dir = f'data/processed/en_model/{run_id}'
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model_file = f'{model_dir}/model.json'
    with open(model_file, 'w') as f:
        json.dump(model_info, f, indent=2, default=str)
    print(f'  Model info: {model_file}')

    # Print holdout summary
    if holdout_report:
        print(f'\nHoldout performance:')
        print(f'  IC mean: {holdout_report.get("ic_mean", "N/A"):.6f}')
        print(f'  IC IR:   {holdout_report.get("ic_ir", "N/A"):.4f}')
        print(f'  Hit rate: {holdout_report.get("hit_rate", "N/A"):.2%}')

    print('\nDone!')
    print(f'\nNext: python scripts/pipeline/04d_estimate_risk_model.py --signal-tag {signal_tag}')





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', type=str, required=True,
                        help='Run ID (e.g., 2026-10-29_default)')
    args = parser.parse_args()
    main(args.run_id)