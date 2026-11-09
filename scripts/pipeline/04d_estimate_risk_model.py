#!/usr/bin/env python
'''
===============================================
Phase 04d: Estimate Risk Model (Covariance + Clustering)
===============================================

Thin wrapper calling src.risk.risk_model functions.
Estimates correlation, cleans with RMT, builds covariance, runs Louvain.

Inputs:
  data/clean/{asset}/{region}/{universe}/{freq}/en_scores/{run_id}/en_scores.parquet
  data/clean/{asset}/{region}/{universe}/{freq}/ohlcv/ohlcv.parquet
  data/clean/{asset}/{region}/{universe}/{freq}/features/features.parquet

Outputs:
  data/clean/{asset}/{region}/{universe}/{freq}/risk_model/{run_id}/
    corr_raw.parquet
    corr_clean.parquet
    cov.parquet
    clusters.parquet
    eigenvalues.parquet
    summary.json
    config.json

Usage:
  python scripts/pipeline/04d_estimate_risk_model.py --date 2026-11-07 --tag default
'''

import argparse
import json
import os
from pathlib import Path
from datetime import date
import numpy as np
import polars as pl
from joblib import Parallel, delayed
from concurrent.futures import as_completed
from tqdm import tqdm

from config.paths import get_paths, run_id_from_args
from src.risk.risk_model import estimate_risk_model_for_date, build_returns_panel
from src.utils.risk_io import (
    save_long_matrix,
    save_clusters,
    combine_parquet_files,
    save_eigenvalues,
)
from src.utils.helpers import get_rebalance_dates
from src.portfolio.backtest import prepare_backtest_data


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_COL = 'combined_score'

RISK_CONFIG = {
    'rebalance_freq': 'weekly',
    'lookback_window': 252,
    'min_obs': 60,
    'cov_method': 'ledoit_wolf',    # 'ledoit_wolf', 'oas', 'factor', 'sample'
    'cluster_resolution': 1.0,
    'n_factors': 5,                 # For factor model
}


def main(run_date: str, tag: str):
    run_id = run_id_from_args(run_date, tag)
    paths = get_paths('yfinance', 'equities', 'US', 'SP500', 'daily', run_id=run_id)

    # --- Inputs ---
    en_scores_dir = paths.stage_dir('en_scores')
    en_scores_file = en_scores_dir / 'en_scores.parquet'
    ohlcv_file = paths.ohlcv_file

    print(f'--> Estimating risk model for {run_id}')
    print(f'    EN scores: {en_scores_file}')
    print(f'    OHLCV: {ohlcv_file}')

    # --- Load data lazily ---
    en_scores_lf = pl.scan_parquet(en_scores_file)
    ohlcv_lf = pl.scan_parquet(ohlcv_file)

    # Get tickers and date range from en_scores (only collect small columns)
    tickers = en_scores_lf.select('ticker').unique().sort(by='ticker').collect().to_series().to_list()
    dates = en_scores_lf.select('date').unique().sort(by='date').collect().to_series().to_list()
    start_date = dates[0]
    end_date = dates[-1]

    print(f'    Tickers: {len(tickers)}')
    print(f'    Date range: {start_date} to {end_date}')

    # --- Prepare returns for risk model ---
    # Use prepare_backtest_data with LazyFrames directly
    signals_lf = en_scores_lf.select(['date', 'ticker', SCORE_COL])
    features_lf = pl.scan_parquet(paths.features_file).rename({'log_adv60': 'adv60', 'vol60': 'std60'})
    # Drop duplicate columns from features (only raw OHLCV columns, keep computed features)
    ohlcv_cols = {'date', 'ticker', 'open', 'high', 'low', 'close_raw', 'close', 'volume', 'open_adj'}
    feature_cols = [c for c in features_lf.collect_schema().names() if c not in ohlcv_cols]
    features_lf = features_lf.select(feature_cols)
    price_lf = pl.concat([ohlcv_lf, features_lf], how='horizontal')

    bt_data = prepare_backtest_data(
        signals_lf=signals_lf,
        score_cols=[SCORE_COL],
        price_lf=price_lf,
        start_date=start_date,
        end_date=end_date,
        rebalance_freq=RISK_CONFIG['rebalance_freq'],
        lookback_window=RISK_CONFIG['lookback_window'],
        min_obs=RISK_CONFIG['min_obs'],
    )

    rebalance_dates = bt_data['rebalance_dates']
    # Test mode: limit rebalance dates for faster testing
    import os
    test_mode = os.environ.get('QUANTALPHA_TEST_MODE', '').lower() in ('1', 'true', 'yes')
    if test_mode:
        rebalance_dates = rebalance_dates[-10:]  # Last 10 weeks for quick test
        print(f'TEST MODE: limiting to last {len(rebalance_dates)} rebalance dates')

    # Build returns panel using utility function
    from src.risk.risk_model import build_returns_panel
    returns_panel = build_returns_panel(
        bt_data['returns_panel'],
        rebalance_dates,
        RISK_CONFIG['lookback_window']
    )
    valid_tickers = bt_data['returns_panel'].columns[1:]  # exclude date column

    # --- Output directory ---
    out_dir = paths.stage_dir('risk_model')
    out_dir.mkdir(parents=True, exist_ok=True)

    corr_raw_file = out_dir / 'corr_raw.parquet'
    corr_clean_file = out_dir / 'corr_clean.parquet'
    cov_file = out_dir / 'cov.parquet'
    clusters_file = out_dir / 'clusters.parquet'
    eigen_file = out_dir / 'eigenvalues.parquet'
    summary_file = out_dir / 'summary.json'

    date_suffixes = [d.isoformat() for d in rebalance_dates]

    # --- Pre-filter valid rebalances (serial, fast) ---
    print(f'\n[1/4] Pre-filtering {len(rebalance_dates)} rebalances...')
    valid_items = []  # list of (idx, date_str, ret_matrix)
    
    for idx, reb_date in enumerate(rebalance_dates):
        date_str = reb_date.isoformat()
        ret_matrix = returns_panel[:, :, idx]
        n_obs = ret_matrix.shape[1]

        if n_obs < RISK_CONFIG['min_obs']:
            print(f'  Skipping {date_str}: insufficient observations ({n_obs})')
            continue

        valid_per_ticker = np.sum(np.isfinite(ret_matrix), axis=1)
        n_valid_tickers = np.sum(valid_per_ticker >= RISK_CONFIG['min_obs'])
        if n_valid_tickers < 50:
            print(f'  Skipping {date_str}: only {n_valid_tickers} tickers with enough data')
            continue

        valid_items.append((idx, date_str, ret_matrix))

    print(f'  {len(valid_items)}/{len(rebalance_dates)} rebalances passed pre-filter')

    # --- Parallel risk model computation ---
    print(f'\n[2/4] Computing risk model in parallel ({os.cpu_count()} cores)...')
    n_jobs = min(os.cpu_count() or 4, len(valid_items))
    print(f'  Using n_jobs={n_jobs}')
    
    def compute_one(item):
        idx, date_str, ret_matrix = item
        return estimate_risk_model_for_date(ret_matrix, valid_tickers, date_str, RISK_CONFIG)
    
    # Use Parallel with backend but no verbose, track progress with tqdm
    print(f'  Processing {len(valid_items)} rebalances...')
    with tqdm(total=len(valid_items), desc='  Risk model', unit='date', 
              bar_format='  {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        results = []
        # Submit all jobs
        with Parallel(n_jobs=n_jobs, backend='loky', verbose=0) as parallel:
            futures = [delayed(compute_one)(item) for item in valid_items]
            for future in parallel(futures):
                result = future
                # Update progress with meaningful info
                pbar.set_postfix_str(f"date={result['date']}, clusters={result['n_clusters']}, λ_max={result['lambda_max']:.2f}")
                pbar.update(1)
                results.append(result)

    # --- Save outputs and build summaries (serial, IO-safe) ---
    print('\n[3/4] Saving outputs...')
    summaries = []
    
    for (idx, date_str, ret_matrix), result in zip(valid_items, results):
        # Save per-date outputs
        save_long_matrix(result['corr_raw'], valid_tickers, date_str, corr_raw_file, 'corr')
        save_long_matrix(result['corr_clean'], valid_tickers, date_str, corr_clean_file, 'corr')
        save_long_matrix(result['cov'], valid_tickers, date_str, cov_file, 'cov')
        save_clusters(result['cluster_map'], valid_tickers, date_str, clusters_file)
        save_eigenvalues(result['eigenvalues'], date_str, eigen_file)

        # Build summary
        cluster_sizes = result['cluster_sizes']
        summaries.append({
            'date': date_str,
            'n_tickers': result['n_tickers'],
            'n_obs': result['n_obs'],
            'q_ratio': round(result['q_ratio'], 4),
            'lambda_max': round(result['lambda_max'], 4),
            'n_clusters': result['n_clusters'],
            'cluster_size_stats': {
                'min': min(cluster_sizes.values()),
                'max': max(cluster_sizes.values()),
                'median': sorted(cluster_sizes.values())[len(cluster_sizes) // 2],
                'mean': round(sum(cluster_sizes.values()) / len(cluster_sizes), 2),
            },
            'shrinkage': result['shrinkage'],
            'cov_method': result['cov_method'],
        })

    # --- Combine parquet files ---
    print('\n[4/4] Combining parquet files...')
    combine_parquet_files(out_dir, 'corr_raw', date_suffixes)
    combine_parquet_files(out_dir, 'corr_clean', date_suffixes)
    combine_parquet_files(out_dir, 'cov', date_suffixes)
    combine_parquet_files(out_dir, 'clusters', date_suffixes)
    combine_parquet_files(out_dir, 'eigenvalues', date_suffixes)

    # Save summary with aggregate stats
    print('\n[3/3] Saving summary...')
    agg = {
        'n_rebalances': len(summaries),
        'date_range': {'start': summaries[0]['date'], 'end': summaries[-1]['date']},
        'avg_n_clusters': round(sum(s['n_clusters'] for s in summaries) / len(summaries), 1),
        'avg_lambda_max': round(sum(s['lambda_max'] for s in summaries) / len(summaries), 4),
        'avg_shrinkage': round(sum(s['shrinkage'] for s in summaries if s['shrinkage'] is not None) / len([s for s in summaries if s['shrinkage'] is not None]), 4) if any(s['shrinkage'] is not None for s in summaries) else None,
        'per_date': summaries,
    }
    with open(summary_file, 'w') as f:
        json.dump(agg, f, indent=2, default=str)

    # Save config
    config = {
        'run_id': run_id,
        'tag': tag,
        'risk_config': RISK_CONFIG,
        'start_date': str(start_date),
        'end_date': str(end_date),
    }
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)

    print(f'\nRisk model estimated for {len(summaries)} rebalances')
    print(f'  Output: {out_dir}/')

    print(f'\nNext: python scripts/pipeline/04e_construct_weights.py --date {run_date} --tag {tag}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, required=True,
                        help='Date in YYYY-MM-DD format (e.g., 2026-10-29)')
    parser.add_argument('--tag', type=str, default='default',
                        help='Signal tag (default: default)')
    args = parser.parse_args()
    main(args.date, args.tag)
