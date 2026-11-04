#!/usr/bin/env python
'''
===============================================
Phase 1: Estimate Risk Model (Covariance + Clustering)
===============================================

Thin wrapper calling src/risk functions.
Estimates correlation, cleans with RMT, runs Louvain, builds covariance.

Inputs:
  data/processed/en_scores/{date}_{tag}/en_scores.parquet (for ticker universe + dates)
  data/processed/ohlcv/{date}_{tag}/ohlcv.parquet
  data/processed/features/{date}_{tag}/features.parquet (for vol, ADV)

Outputs:
  data/processed/risk_model/{date}_{tag}/
    ├── corr_raw.parquet       # [date, ticker_i, ticker_j, corr]
    ├── corr_clean.parquet     # RMT-cleaned
    ├── cov.parquet            # [date, ticker_i, ticker_j, cov]
    ├── clusters.parquet       # [date, ticker, cluster_id]
    ├── eigenvalues.parquet    # [date, eigenvalue, component]
    ├── summary.json           # Per-rebalance stats
    └── config.json            # Run configuration

Usage:
  python scripts/pipeline/04d_estimate_risk_model.py --signal-tag default --run-id 2026-10-29_default
'''

import argparse
import json
from pathlib import Path
from datetime import date
import numpy as np
import polars as pl

from src.risk.correlation_cleaning import clean_correlation_matrix as clean_correlation_rmt, marchenko_pastur_bounds
from src.risk.clustering import louvain_clustering
from src.risk.covariance import (
    build_covariance_matrix,
    ledoit_wolf_shrinkage,
    oracle_approximating_shrinkage,
    factor_model_covariance,
)
from src.portfolio.backtest import prepare_backtest_data

# =============================================================================
# CONFIGURATION
# =============================================================================

SIGNAL_TAG = 'default'

# Risk model config
RISK_CONFIG = {
    'rebalance_freq': 'weekly',
    'lookback_window': 252,
    'min_obs': 60,
    'cov_method': 'ledoit_wolf',    # 'ledoit_wolf', 'oas', 'factor', 'sample'
    'cluster_resolution': 1.0,
    'n_factors': 5,                 # For factor model
}

# Score column (from en_scores)
SCORE_COL = 'combined_score'


def save_long_matrix(matrix, tickers, date_str, file_path, value_name, append=True):
    '''Save symmetric matrix in long format - write per rebalance, combine at end'''
    n = len(tickers)
    rows = []
    for i in range(n):
        for j in range(i, n):
            rows.append({
                'date': date_str,
                'ticker_i': tickers[i],
                'ticker_j': tickers[j],
                value_name: float(matrix[i, j])
            })
    df = pl.DataFrame(rows)
    # Write to a temporary per-date file
    temp_path = Path(str(file_path).replace('.parquet', f'_{date_str}.parquet'))
    df.write_parquet(temp_path)


def save_clusters(cluster_map, valid_tickers, date_str, file_path, append=True):
    '''cluster_map has integer indices, valid_tickers has ticker symbols'''
    rows = [{'date': date_str, 'ticker': valid_tickers[t], 'cluster_id': int(c)} for t, c in cluster_map.items()]
    df = pl.DataFrame(rows)
    temp_path = Path(str(file_path).replace('.parquet', f'_{date_str}.parquet'))
    df.write_parquet(temp_path)


def combine_parquet_files(output_dir, base_name, date_suffixes):
    '''Combine per-date parquet files into one'''
    files = [output_dir / f'{base_name}_{ds}.parquet' for ds in date_suffixes if (output_dir / f'{base_name}_{ds}.parquet').exists()]
    if not files:
        return
    dfs = [pl.read_parquet(f) for f in files]
    combined = pl.concat(dfs)
    combined.write_parquet(output_dir / f'{base_name}.parquet')
    # Clean up temp files
    for f in files:
        f.unlink()


def save_eigenvalues(eigenvals, date_str, file_path):
    rows = [{'date': date_str, 'eigenvalue': float(v), 'component': i} for i, v in enumerate(eigenvals)]
    df = pl.DataFrame(rows)
    # Write to a temporary per-date file
    temp_path = Path(str(file_path).replace('.parquet', f'_{date_str}.parquet'))
    df.write_parquet(temp_path)


def main(signal_tag: str, run_id: str = None):
    print('=' * 70)
    print('PHASE 1: ESTIMATE RISK MODEL')
    print('=' * 70)
    print(f'Signal tag:    {signal_tag}')
    print(f'Run ID:        {run_id}')
    print(f'Config:        {RISK_CONFIG}')
    print('=' * 70)

    # Output dir - new structure: data/processed/risk_model/{run_id}/
    out_dir = Path(f'data/processed/risk_model/{run_id}')
    out_dir.mkdir(parents=True, exist_ok=True)

    corr_raw_file = out_dir / 'corr_raw.parquet'
    corr_clean_file = out_dir / 'corr_clean.parquet'
    cov_file = out_dir / 'cov.parquet'
    clusters_file = out_dir / 'clusters.parquet'
    eigen_file = out_dir / 'eigenvalues.parquet'
    summary_file = out_dir / 'summary.json'

    # Load data
    print('\n[1/4] Loading data...')
    # Find the ohlcv file for this run
    ohlcv_files = list(Path('data/processed/ohlcv').glob(f'*_{signal_tag}/ohlcv.parquet'))
    if not ohlcv_files:
        raise FileNotFoundError(f'No ohlcv file found for tag {signal_tag}')
    ohlcv_file = sorted(ohlcv_files)[-1]
    print(f'  Using OHLCV: {ohlcv_file}')
    price_lf = pl.scan_parquet(ohlcv_file)

    # Find features file for this run
    features_files = list(Path('data/processed/features').glob(f'*_{signal_tag}/features.parquet'))
    if not features_files:
        raise FileNotFoundError(f'No features file found for tag {signal_tag}')
    features_file = sorted(features_files)[-1]
    print(f'  Using features: {features_file}')
    features_lf = pl.scan_parquet(features_file)

    # Find en_scores file for this run
    en_scores_file = f'data/processed/en_scores/{run_id}/en_scores.parquet'
    if not Path(en_scores_file).exists():
        raise FileNotFoundError(f'No en_scores file found: {en_scores_file}')
    print(f'  Using en_scores: {en_scores_file}')
    signal_lf = pl.scan_parquet(en_scores_file).select(['date', 'ticker', SCORE_COL])

    # Prepare features with required columns (adv60, std60/logret from features)
    # features has log_adv60 and vol60, we need to derive adv60 and std60
    # adv60 = exp(log_adv60), std60 = vol60
    features_lf = features_lf.with_columns([
        pl.col('log_adv60').exp().alias('adv60'),
        pl.col('vol60').alias('std60'),
        pl.col('logret'),
    ]).select(['date', 'ticker', 'logret', 'adv60', 'std60'])

    # Join prices with features for complete price data
    complete_price_lf = price_lf.join(features_lf, on=['date', 'ticker'], how='left')

    # Get date range from data
    date_range = complete_price_lf.select(
        pl.col('date').min().alias('min_date'),
        pl.col('date').max().alias('max_date')
    ).collect()
    start_date = date_range['min_date'][0]
    end_date = date_range['max_date'][0]
    print(f'  Date range: {start_date} to {end_date}')

    prepared = prepare_backtest_data(
        signals_lf=signal_lf,
        score_cols=[SCORE_COL],
        price_lf=complete_price_lf,
        start_date=start_date,
        end_date=end_date,
        rebalance_freq=RISK_CONFIG['rebalance_freq'],
        lookback_window=RISK_CONFIG['lookback_window'],
        min_obs=RISK_CONFIG['min_obs'],
        score_col=SCORE_COL,
        adv_col='adv60',
        vol_col='std60',
    )

    rebalance_dates = prepared['rebalance_dates']
    # Use long-format returns from features for risk model estimation
    returns_long = features_lf.select(['date', 'ticker', 'logret']).collect()
    print(f'  {len(rebalance_dates)} rebalance dates')

    # Process each rebalance
    print('\n[2/4] Estimating risk model per rebalance...')
    summaries = []
    date_suffixes = []

    for i, reb_date in enumerate(rebalance_dates):
        if i % 50 == 0:
            print(f'  [{i+1}/{len(rebalance_dates)}] {reb_date}')

        date_str = str(reb_date)
        date_suffixes.append(date_str)

        # Get lookback returns
        lookback_start = np.datetime64(reb_date) - np.timedelta64(RISK_CONFIG['lookback_window'], 'D')
        window_rets = returns_long.filter(
            (pl.col('date') > lookback_start) & (pl.col('date') <= reb_date)
        )

        # Pivot to [ticker, time]
        ret_wide = window_rets.pivot(values='logret', index='ticker', on='date', aggregate_function='first')

        # Filter valid tickers
        valid_tickers = []
        ret_matrix = []
        for row in ret_wide.iter_rows(named=True):
            ticker = row['ticker']
            vals = np.array([v for k, v in row.items() if k != 'ticker' and v is not None])
            if len(vals) >= RISK_CONFIG['min_obs']:
                valid_tickers.append(ticker)
                ret_matrix.append(vals)

        if len(valid_tickers) < 10:
            continue

        # Align all tickers to the same length (use minimum observations)
        min_len = min(len(v) for v in ret_matrix)
        ret_matrix = [v[-min_len:] for v in ret_matrix]
        ret_matrix = np.array(ret_matrix)  # [n, T]
        n_tickers, n_obs = ret_matrix.shape

        # Correlation
        corr = np.corrcoef(ret_matrix)
        np.fill_diagonal(corr, 1.0)
        save_long_matrix(corr, valid_tickers, date_str, corr_raw_file, 'corr')

        # RMT Cleaning
        q = n_tickers / n_obs
        lambda_min, lambda_max = marchenko_pastur_bounds(n_tickers, n_obs, variance=1.0)
        corr_clean = clean_correlation_rmt(corr, n_obs)
        # Compute eigenvalues for saving
        eigenvalues, eigenvectors = np.linalg.eigh(corr)
        eigenvals = eigenvalues  # Ascending order
        save_long_matrix(corr_clean, valid_tickers, date_str, corr_clean_file, 'corr')
        save_eigenvalues(eigenvals, date_str, eigen_file)

        # Volatilities (annualized)
        vols = np.std(ret_matrix, axis=1, ddof=1) * np.sqrt(252)

        # Covariance
        method = RISK_CONFIG['cov_method']
        if method == 'ledoit_wolf':
            cov, shrinkage = ledoit_wolf_shrinkage(corr_clean, returns=ret_matrix, target='identity')
        elif method == 'oas':
            cov, shrinkage = oracle_approximating_shrinkage(corr_clean, returns=ret_matrix)
        elif method == 'factor':
            n_factors = min(RISK_CONFIG['n_factors'], n_tickers - 1)
            cov = factor_model_covariance(ret_matrix.T, n_factors)
            shrinkage = None
        else:
            cov = build_covariance_matrix(corr_clean, vols)
            shrinkage = None

        save_long_matrix(cov, valid_tickers, date_str, cov_file, 'cov')

        # Louvain Clustering (on cleaned correlation)
        cluster_map = louvain_clustering(corr_clean, resolution=RISK_CONFIG['cluster_resolution'])
        save_clusters(cluster_map, valid_tickers, date_str, clusters_file)

        # Cluster summary
        clusters = {}
        for t, c in cluster_map.items():
            clusters.setdefault(c, []).append(t)

        summaries.append({
            'date': date_str,
            'n_tickers': n_tickers,
            'n_obs': n_obs,
            'q_ratio': q,
            'lambda_max': float(lambda_max),
            'n_clusters': len(clusters),
            'cluster_sizes': {str(k): len(v) for k, v in clusters.items()},
            'shrinkage': float(shrinkage) if shrinkage is not None else None,
            'cov_method': method,
        })

    # Combine per-date parquet files into single files
    print('\n[3/4] Combining parquet files...')
    combine_parquet_files(out_dir, 'corr_raw', date_suffixes)
    combine_parquet_files(out_dir, 'corr_clean', date_suffixes)
    combine_parquet_files(out_dir, 'cov', date_suffixes)
    combine_parquet_files(out_dir, 'clusters', date_suffixes)
    combine_parquet_files(out_dir, 'eigenvalues', date_suffixes)

    # Save summary
    print('\n[4/4] Saving summary...')
    with open(summary_file, 'w') as f:
        json.dump(summaries, f, indent=2, default=str)

    # Save config
    config = {
        'run_id': run_id,
        'signal_tag': signal_tag,
        'risk_config': RISK_CONFIG,
        'start_date': str(start_date),
        'end_date': str(end_date),
    }
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)

    print(f'\n✓ Risk model estimated for {len(summaries)} rebalances')
    print(f'  Output: {out_dir}/')
    print(f'    corr_raw.parquet, corr_clean.parquet, cov.parquet')
    print(f'    clusters.parquet, eigenvalues.parquet, summary.json, config.json')

    print('\nDone!')
    print(f'\nNext: python scripts/pipeline/04e_construct_weights.py --run-id {run_id}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--signal-tag', type=str, required=True)
    parser.add_argument('--run-id', type=str, required=True,
                        help='Run ID in format YYYY-MM-DD_tag (e.g., 2026-10-29_default)')
    args = parser.parse_args()
    main(args.signal_tag, args.run_id)