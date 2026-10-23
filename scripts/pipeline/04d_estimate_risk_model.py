#!/usr/bin/env python
'''
===============================================
Phase 1: Estimate Risk Model (Covariance + Clustering)
===============================================

Thin wrapper calling src/risk functions.
Estimates correlation, cleans with RMT, runs Louvain, builds covariance.

Inputs:
  data/processed/en_scores_<tag>.parquet (for ticker universe + dates)
  data/processed/prices.parquet
  data/processed/features.parquet (for vol, ADV)

Outputs:
  data/exports/risk_model_<tag>/
    ├── corr_raw.parquet       # [date, ticker_i, ticker_j, corr]
    ├── corr_clean.parquet     # RMT-cleaned
    ├── cov.parquet            # [date, ticker_i, ticker_j, cov]
    ├── clusters.parquet       # [date, ticker, cluster_id]
    ├── eigenvalues.parquet    # [date, eigenvalue, component]
    └── summary.json           # Per-rebalance stats

Usage:
  python scripts/pipeline/04d_estimate_risk_model.py --signal-tag default
'''

import argparse
import json
from pathlib import Path
from datetime import date
import numpy as np
import polars as pl

from src.risk.correlation_cleaning import clean_correlation_rmt
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

# Date range
START_DATE = '2010-01-01'
END_DATE = '2024-12-31'

# Score column (from en_scores)
SCORE_COL = 'combined_score'


def save_long_matrix(matrix, tickers, date_str, file_path, value_name, append=True):
    '''Save symmetric matrix in long format'''
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
    if append and Path(file_path).exists():
        df = pl.concat([pl.read_parquet(file_path), df])
    df.write_parquet(file_path)


def save_clusters(cluster_map, date_str, file_path, append=True):
    rows = [{'date': date_str, 'ticker': t, 'cluster_id': int(c)} for t, c in cluster_map.items()]
    df = pl.DataFrame(rows)
    if append and Path(file_path).exists():
        df = pl.concat([pl.read_parquet(file_path), df])
    df.write_parquet(file_path)


def save_eigenvalues(eigenvals, date_str, file_path, append=True):
    rows = [{'date': date_str, 'eigenvalue': float(v), 'component': i} for i, v in enumerate(eigenvals)]
    df = pl.DataFrame(rows)
    if append and Path(file_path).exists():
        df = pl.concat([pl.read_parquet(file_path), df])
    df.write_parquet(file_path)


def main(signal_tag: str):
    print('=' * 70)
    print('PHASE 1: ESTIMATE RISK MODEL')
    print('=' * 70)
    print(f'Signal tag:    {signal_tag}')
    print(f'Config:        {RISK_CONFIG}')
    print(f'Date range:    {START_DATE} to {END_DATE}')
    print('=' * 70)

    # Output dir
    out_dir = Path(f'data/exports/risk_model_{signal_tag}')
    out_dir.mkdir(parents=True, exist_ok=True)

    corr_raw_file = out_dir / 'corr_raw.parquet'
    corr_clean_file = out_dir / 'corr_clean.parquet'
    cov_file = out_dir / 'cov.parquet'
    clusters_file = out_dir / 'clusters.parquet'
    eigen_file = out_dir / 'eigenvalues.parquet'
    summary_file = out_dir / 'summary.json'

    # Load data
    print('\n[1/4] Loading data...')
    price_lf = pl.scan_parquet('data/processed/prices.parquet')
    signal_lf = pl.scan_parquet(f'data/processed/en_scores_{signal_tag}.parquet').select(['date', 'ticker', SCORE_COL])

    prepared = prepare_backtest_data(
        signals_lf=signal_lf,
        score_cols=[SCORE_COL],
        price_lf=price_lf,
        start_date=date.fromisoformat(START_DATE),
        end_date=date.fromisoformat(END_DATE),
        rebalance_freq=RISK_CONFIG['rebalance_freq'],
        lookback_window=RISK_CONFIG['lookback_window'],
        min_obs=RISK_CONFIG['min_obs'],
        score_col=SCORE_COL,
        adv_col='adv60',
        vol_col='std60',
    )

    rebalance_dates = prepared['rebalance_dates']
    returns_panel = prepared['returns_panel'].collect()
    print(f'  {len(rebalance_dates)} rebalance dates')

    # Process each rebalance
    print('\n[2/4] Estimating risk model per rebalance...')
    summaries = []

    for i, reb_date in enumerate(rebalance_dates):
        if i % 50 == 0:
            print(f'  [{i+1}/{len(rebalance_dates)}] {reb_date}')

        # Get lookback returns
        lookback_start = np.datetime64(reb_date) - np.timedelta64(RISK_CONFIG['lookback_window'], 'D')
        window_rets = returns_panel.filter(
            (pl.col('date') > lookback_start) & (pl.col('date') <= reb_date)
        )

        # Pivot to [ticker, time]
        ret_wide = window_rets.pivot(values='logret', index='ticker', columns='date', aggregate_function='first')

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

        ret_matrix = np.array(ret_matrix)  # [n, T]
        n_tickers, n_obs = ret_matrix.shape

        # Correlation
        corr = np.corrcoef(ret_matrix)
        np.fill_diagonal(corr, 1.0)
        save_long_matrix(corr, valid_tickers, str(reb_date), corr_raw_file, 'corr', append=(i>0))

        # RMT Cleaning
        q = n_tickers / n_obs
        lambda_max = (1 + np.sqrt(q)) ** 2
        corr_clean, eigenvals = clean_correlation_rmt(corr, return_eigenvals=True)
        save_long_matrix(corr_clean, valid_tickers, str(reb_date), corr_clean_file, 'corr', append=(i>0))
        save_eigenvalues(eigenvals, str(reb_date), eigen_file, append=(i>0))

        # Volatilities (annualized)
        vols = np.std(ret_matrix, axis=1, ddof=1) * np.sqrt(252)

        # Covariance
        method = RISK_CONFIG['cov_method']
        if method == 'ledoit_wolf':
            cov, shrinkage = ledoit_wolf_shrinkage(corr_clean, ret_matrix.T)
        elif method == 'oas':
            cov = oracle_approximating_shrinkage(corr_clean, ret_matrix.T)
            shrinkage = None
        elif method == 'factor':
            n_factors = min(RISK_CONFIG['n_factors'], n_tickers - 1)
            cov = factor_model_covariance(ret_matrix.T, n_factors)
            shrinkage = None
        else:
            cov = build_covariance_matrix(corr_clean, vols)
            shrinkage = None

        save_long_matrix(cov, valid_tickers, str(reb_date), cov_file, 'cov', append=(i>0))

        # Louvain Clustering (on cleaned correlation)
        cluster_map = louvain_clustering(corr_clean, valid_tickers, resolution=RISK_CONFIG['cluster_resolution'])
        save_clusters(cluster_map, str(reb_date), clusters_file, append=(i>0))

        # Cluster summary
        clusters = {}
        for t, c in cluster_map.items():
            clusters.setdefault(c, []).append(t)

        summaries.append({
            'date': str(reb_date),
            'n_tickers': n_tickers,
            'n_obs': n_obs,
            'q_ratio': q,
            'lambda_max': float(lambda_max),
            'n_clusters': len(clusters),
            'cluster_sizes': {str(k): len(v) for k, v in clusters.items()},
            'shrinkage': float(shrinkage) if shrinkage is not None else None,
            'cov_method': method,
        })

    # Save summary
    print('\n[3/4] Saving summary...')
    with open(summary_file, 'w') as f:
        json.dump(summaries, f, indent=2, default=str)

    print(f'\n✓ Risk model estimated for {len(summaries)} rebalances')
    print(f'  Output: {out_dir}/')
    print(f'    corr_raw.parquet, corr_clean.parquet, cov.parquet')
    print(f'    clusters.parquet, eigenvalues.parquet, summary.json')

    print('\nDone!')
    print(f'\nNext: python scripts/pipeline/04e_construct_weights.py --signal-tag {signal_tag}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--signal-tag', type=str, default=SIGNAL_TAG)
    args = parser.parse_args()
    main(args.signal_tag)