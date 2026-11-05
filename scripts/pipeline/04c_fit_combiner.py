#!/usr/bin/env python
'''
===============================================
Phase 04c: Fit Elastic Net Combiner
===============================================

Fits Elastic Net on selected signals from Phase 04b using CPCV search + holdout.
Produces combined_score for all dates.

Inputs:
  data/clean/{asset}/{region}/{universe}/{freq}/signals/{run_id}/signals.parquet
  data/clean/{asset}/{region}/{universe}/{freq}/signal_eval/{run_id}/selected_signals.json
  data/clean/{asset}/{region}/{universe}/{freq}/signal_eval/{run_id}/fdr_report.parquet (for q-values)

Outputs:
  data/clean/{asset}/{region}/{universe}/{freq}/en_scores/{run_id}/en_scores.parquet
  data/clean/{asset}/{region}/{universe}/{freq}/en_model/{run_id}/model.json
  data/clean/{asset}/{region}/{universe}/{freq}/en_model/{run_id}/stability.parquet

Usage:
    python scripts/pipeline/04c_fit_combiner.py --date 2026-10-29 --tag default
'''

import argparse
import json
from pathlib import Path
import numpy as np
import polars as pl

from config.paths import get_data_config, run_id_from_args
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
    'q_value_weighting': True,
}


def main(date: str, tag: str = 'default'):
    run_id = run_id_from_args(date, tag)
    
    # Use centralized config
    cfg = get_data_config(run_id=run_id)
    
    print('=' * 70)
    print('PHASE 04c: FIT ELASTIC NET COMBINER (CPCV + Holdout)')
    print('=' * 70)
    print(f'Run ID:    {run_id}')
    print(f'EN config:     {EN_CONFIG}')
    print('=' * 70)

    signal_tag = tag

    # Load selected signals from evaluation
    selected_file = cfg.stage_dir('signal_eval') / 'selected_signals.json'
    if not selected_file.exists():
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
        fdr_file = cfg.stage_dir('signal_eval') / 'fdr_report.parquet'
        if fdr_file.exists():
            fdr_df = pl.read_parquet(fdr_file)
            for row in fdr_df.iter_rows(named=True):
                if row['signal'] in signal_cols:
                    q_values[row['signal']] = row['qvalue']
            print(f'  Loaded q-values for {len(q_values)} signals: {q_values}')

    # Load signal data
    signals_file = cfg.stage_dir('signals') / 'signals.parquet'
    print(f'\nLoading {signals_file}...')
    lf = pl.scan_parquet(signals_file)

    # Verify forward return column exists
    schema = lf.collect_schema()
    if fwd_ret_col not in schema.names():
        available = [c for c in schema.names() if c.startswith('fwdret')]
        raise ValueError(f'Forward return column {fwd_ret_col} not found. Available: {available}')

    # Run EN combiner with CPCV + holdout
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
    cfg.stage_dir('en_model').mkdir(parents=True, exist_ok=True)
    stability_df.write_parquet(cfg.stage_dir('en_model') / 'stability.parquet')
    print(f'  Saved stability to {cfg.stage_dir("en_model") / "stability.parquet"}')

    # --- Score FULL dataset ---
    print('\nScoring full dataset...')
    from src.models.elastic_net_combiner import combined_score_column
    full_lf = combined_score_column(lf, result, out_col='combined_score')
    full_df = full_lf.collect()
    
    print(f'  Full dataset: {len(full_df):,} rows')
    print(f'  Score stats: mean={full_df["combined_score"].mean():.6f}, std={full_df["combined_score"].std():.6f}')

    # Save scores
    cfg.stage_dir('en_scores').mkdir(parents=True, exist_ok=True)
    full_df.select(['date', 'ticker', 'combined_score'] + signal_cols).write_parquet(cfg.stage_dir('en_scores') / 'en_scores.parquet')
    print(f'\n✓ Saved scores to {cfg.stage_dir("en_scores") / "en_scores.parquet"}')

    # Save config for scores
    score_config = {
        'run_id': run_id,
        'signal_tag': signal_tag,
        'signal_dir': str(cfg.stage_dir('signals')),
        'fwd_ret_col': fwd_ret_col,
        'horizon': horizon,
        'signal_cols': signal_cols,
        'q_values': q_values,
    }
    with open(cfg.stage_dir('en_scores') / 'config.json', 'w') as f:
        json.dump(score_config, f, indent=2, default=str)

    # Save model info (includes holdout report)
    holdout_report = result.get('holdout_report', {})
    model_info = {
        'signal_tag': signal_tag,
        'run_id': run_id,
        'signal_dir': str(cfg.signals_dir),
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
    cfg.en_model_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.en_model_file, 'w') as f:
        json.dump(model_info, f, indent=2, default=str)
    print(f'  Model info: {cfg.en_model_file}')

    # Print holdout summary
    if holdout_report:
        print(f'\nHoldout performance:')
        print(f'  IC mean: {holdout_report.get("ic_mean", "N/A"):.6f}')
        print(f'  IC IR:   {holdout_report.get("ic_ir", "N/A"):.4f}')
        print(f'  Hit rate: {holdout_report.get("hit_rate", "N/A"):.2%}')

    print('\nDone!')
    print(f'\nNext: python scripts/pipeline/04d_estimate_risk_model.py --date {date} --tag {tag}')





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, required=True,
                        help='Date in YYYY-MM-DD format (e.g., 2026-10-29)')
    parser.add_argument('--tag', type=str, default='default',
                        help='Signal tag (default: default)')
    args = parser.parse_args()
    
    main(args.date, args.tag)
