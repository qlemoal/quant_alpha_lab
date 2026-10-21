#!/usr/bin/env python
'''
Phase 1: Estimate Risk Model (Covariance + Clustering)
========================================================

Estimates the correlation/covariance matrix and Louvain clusters at each rebalance date.
Saves intermediate results for inspection.

Run AFTER: 04a_generate_en_scores.py (or have signals.parquet with combined_score)

Outputs:
  data/exports/risk_model/
    ├── correlation_raw.parquet      # [date, ticker_i, ticker_j, corr] per rebalance
    ├── correlation_clean.parquet    # RMT-cleaned
    ├── covariance.parquet           # [date, ticker_i, ticker_j, cov]
    ├── clusters.parquet             # [date, ticker, cluster_id]
    ├── cluster_summary.json         # Per-rebalance cluster stats
    └── eigenvalues.parquet          # [date, eigenvalue, component] for diagnostics

Usage:
  python scripts/pipeline/04b_estimate_risk_model.py

Env vars:
  REBALANCE_FREQ        = 'weekly'
  LOOKBACK_WINDOW       = 252
  MIN_OBS               = 60
  COV_METHOD            = 'ledoit_wolf'  # 'ledoit_wolf', 'oas', 'factor', 'sample'
  CLUSTER_RESOLUTION    = 1.0
  START_DATE            = '2010-01-01'
  END_DATE              = '2024-12-31'
'''

import os
import json
from pathlib import Path
from datetime import date
import polars as pl
import numpy as np

from src.risk.correlation_cleaning import clean_correlation_rmt
from src.risk.clustering import louvain_clustering
from src.risk.covariance import (
    build_covariance_matrix,
    ledoit_wolf_shrinkage,
    oracle_approximating_shrinkage,
    factor_model_covariance,
)
from src.risk.portfolio_construction import (
    raw_weights_from_score,
    apply_cluster_caps,
    apply_vol_target,
)
from src.portfolio.backtest import prepare_backtest_data

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
EXPORTS_DIR = DATA_DIR / 'exports'
RISK_DIR = EXPORTS_DIR / 'risk_model'

# Input
SIGNALS_FILE = PROCESSED_DIR / 'signals.parquet'  # or en_scores.parquet
PRICES_FILE = PROCESSED_DIR / 'prices.parquet'
FEATURES_FILE = PROCESSED_DIR / 'features.parquet'

# Output
RISK_DIR.mkdir(parents=True, exist_ok=True)
CORR_RAW_FILE = RISK_DIR / 'correlation_raw.parquet'
CORR_CLEAN_FILE = RISK_DIR / 'correlation_clean.parquet'
COV_FILE = RISK_DIR / 'covariance.parquet'
CLUSTERS_FILE = RISK_DIR / 'clusters.parquet'
CLUSTER_SUMMARY_FILE = RISK_DIR / 'cluster_summary.json'
EIGEN_FILE = RISK_DIR / 'eigenvalues.parquet'

# Config
REBALANCE_FREQ = os.environ.get('REBALANCE_FREQ', 'weekly')
LOOKBACK_WINDOW = int(os.environ.get('LOOKBACK_WINDOW', '252'))
MIN_OBS = int(os.environ.get('MIN_OBS', '60'))
COV_METHOD = os.environ.get('COV_METHOD', 'ledoit_wolf')
CLUSTER_RESOLUTION = float(os.environ.get('CLUSTER_RESOLUTION', '1.0'))
START_DATE = os.environ.get('START_DATE', '2010-01-01')
END_DATE = os.environ.get('END_DATE', '2024-12-31')

SCORE_COL = os.environ.get('SCORE_COL', 'combined_score')
ADV_COL = 'adv60'
VOL_COL = 'std60'


def estimate_correlation_matrix(returns_panel, tickers, date_str):
    '''Estimate correlation matrix from returns panel for a given date'''
    # returns_panel: [date, ticker, logret]
    # Filter to lookback window ending at date_str
    pass  # Will use prepare_backtest_data which handles this


def save_correlation_long(corr_matrix, tickers, date_str, file_path, append=True):
    '''Save correlation matrix in long format: [date, ticker_i, ticker_j, corr]'''
    n = len(tickers)
    rows = []
    for i in range(n):
        for j in range(i, n):  # Upper triangle including diagonal
            rows.append({
                'date': date_str,
                'ticker_i': tickers[i],
                'ticker_j': tickers[j],
                'corr': float(corr_matrix[i, j])
            })
    df = pl.DataFrame(rows)
    if append and file_path.exists():
        existing = pl.read_parquet(file_path)
        df = pl.concat([existing, df])
    df.write_parquet(file_path)


def save_covariance_long(cov_matrix, tickers, date_str, file_path, append=True):
    '''Save covariance matrix in long format'''
    n = len(tickers)
    rows = []
    for i in range(n):
        for j in range(i, n):
            rows.append({
                'date': date_str,
                'ticker_i': tickers[i],
                'ticker_j': tickers[j],
                'cov': float(cov_matrix[i, j])
            })
    df = pl.DataFrame(rows)
    if append and file_path.exists():
        existing = pl.read_parquet(file_path)
        df = pl.concat([existing, df])
    df.write_parquet(file_path)


def save_clusters(cluster_map, date_str, file_path, append=True):
    '''Save cluster assignments: [date, ticker, cluster_id]'''
    rows = [{'date': date_str, 'ticker': t, 'cluster_id': int(c)} for t, c in cluster_map.items()]
    df = pl.DataFrame(rows)
    if append and file_path.exists():
        existing = pl.read_parquet(file_path)
        df = pl.concat([existing, df])
    df.write_parquet(file_path)


def save_eigenvalues(eigenvals, date_str, file_path, append=True):
    '''Save eigenvalues for diagnostics'''
    rows = [{'date': date_str, 'eigenvalue': float(v), 'component': i} for i, v in enumerate(eigenvals)]
    df = pl.DataFrame(rows)
    if append and file_path.exists():
        existing = pl.read_parquet(file_path)
        df = pl.concat([existing, df])
    df.write_parquet(file_path)


def main():
    print('=' * 70)
    print('PHASE 1: ESTIMATE RISK MODEL (Covariance + Clustering)')
    print('=' * 70)
    print(f'Period:         {START_DATE} to {END_DATE}')
    print(f'Rebalance freq: {REBALANCE_FREQ}')
    print(f'Lookback:       {LOOKBACK_WINDOW} days')
    print(f'Min obs:        {MIN_OBS}')
    print(f'Cov method:     {COV_METHOD}')
    print(f'Cluster res:    {CLUSTER_RESOLUTION}')
    print(f'Output dir:     {RISK_DIR}')
    print('=' * 70)

    # Load data
    print('\n[1/4] Loading data...')
    price_lf = pl.scan_parquet(PRICES_FILE)

    # Prepare backtest data (gets rebalance dates, aligned returns, signals, ADV, vol)
    print('[2/4] Preparing backtest data structure...')
    prepared = prepare_backtest_data(
        signals_lf=pl.scan_parquet(SIGNALS_FILE).select(['date', 'ticker', SCORE_COL]),
        score_cols=[SCORE_COL],
        price_lf=price_lf,
        start_date=date.fromisoformat(START_DATE),
        end_date=date.fromisoformat(END_DATE),
        rebalance_freq=REBALANCE_FREQ,
        lookback_window=LOOKBACK_WINDOW,
        min_obs=MIN_OBS,
        score_col=SCORE_COL,
        adv_col=ADV_COL,
        vol_col=VOL_COL,
    )

    rebalance_dates = prepared['rebalance_dates']
    returns_panel = prepared['returns_panel']  # [date, ticker, logret]
    print(f'    {len(rebalance_dates)} rebalance dates')
    print(f'    Date range: {rebalance_dates[0]} to {rebalance_dates[-1]}')

    # Collect returns panel for easier slicing
    returns_df = returns_panel.collect()

    # Process each rebalance date
    print('\n[3/4] Estimating risk model at each rebalance...')

    cluster_summaries = []

    for i, reb_date in enumerate(rebalance_dates):
        if i % 50 == 0:
            print(f'  [{i+1}/{len(rebalance_dates)}] {reb_date}')

        # Get lookback window returns
        lookback_start = reb_date - np.timedelta64(LOOKBACK_WINDOW, 'D')
        window_returns = returns_df.filter(
            (pl.col('date') > lookback_start) & (pl.col('date') <= reb_date)
        )

        # Pivot to wide matrix [ticker, time]
        ret_wide = window_returns.pivot(
            values='logret', index='ticker', columns='date', aggregate_function='first'
        )

        # Drop tickers with insufficient obs
        valid_tickers = []
        ret_matrix = []
        for row in ret_wide.iter_rows(named=True):
            ticker = row['ticker']
            vals = np.array([v for k, v in row.items() if k != 'ticker' and v is not None])
            if len(vals) >= MIN_OBS:
                valid_tickers.append(ticker)
                ret_matrix.append(vals)

        if len(valid_tickers) < 10:
            print(f'    WARNING: Only {len(valid_tickers)} valid tickers, skipping')
            continue

        ret_matrix = np.array(ret_matrix)  # [n_tickers, n_obs]
        n_tickers, n_obs = ret_matrix.shape

        # --- Correlation matrix ---
        corr_matrix = np.corrcoef(ret_matrix)
        np.fill_diagonal(corr_matrix, 1.0)

        # Save raw correlation
        save_correlation_long(corr_matrix, valid_tickers, str(reb_date), CORR_RAW_FILE, append=(i>0))

        # --- RMT Cleaning ---
        q = n_tickers / n_obs
        lambda_max = (1 + np.sqrt(q)) ** 2
        corr_clean, eigenvals = clean_correlation_rmt(corr_matrix, return_eigenvals=True)

        # Save cleaned correlation
        save_correlation_long(corr_clean, valid_tickers, str(reb_date), CORR_CLEAN_FILE, append=(i>0))

        # Save eigenvalues
        save_eigenvalues(eigenvals, str(reb_date), EIGEN_FILE, append=(i>0))

        # --- Volatilities ---
        vols = np.std(ret_matrix, axis=1, ddof=1) * np.sqrt(252)  # Annualized

        # --- Covariance estimation ---
        if COV_METHOD == 'ledoit_wolf':
            cov_matrix, shrinkage = ledoit_wolf_shrinkage(corr_clean, ret_matrix.T)
        elif COV_METHOD == 'oas':
            cov_matrix = oracle_approximating_shrinkage(corr_clean, ret_matrix.T)
        elif COV_METHOD == 'factor':
            n_factors = min(5, n_tickers - 1)
            cov_matrix = factor_model_covariance(ret_matrix.T, n_factors)
        else:  # sample
            cov_matrix = build_covariance_matrix(corr_clean, vols)

        # Save covariance
        save_covariance_long(cov_matrix, valid_tickers, str(reb_date), COV_FILE, append=(i>0))

        # --- Louvain Clustering ---
        # Use cleaned correlation for clustering
        cluster_map = louvain_clustering(corr_clean, valid_tickers, resolution=CLUSTER_RESOLUTION)
        save_clusters(cluster_map, str(reb_date), CLUSTERS_FILE, append=(i>0))

        # Cluster summary
        clusters = {}
        for t, c in cluster_map.items():
            clusters.setdefault(c, []).append(t)

        cluster_summary = {
            'date': str(reb_date),
            'n_tickers': n_tickers,
            'n_obs': n_obs,
            'q_ratio': q,
            'lambda_max': lambda_max,
            'n_clusters': len(clusters),
            'cluster_sizes': {str(k): len(v) for k, v in clusters.items()},
            'shrinkage': float(shrinkage) if COV_METHOD == 'ledoit_wolf' else None,
            'cov_method': COV_METHOD,
        }
        cluster_summaries.append(cluster_summary)

    # Save cluster summary
    print('\n[4/4] Saving cluster summary...')
    with open(CLUSTER_SUMMARY_FILE, 'w') as f:
        json.dump(cluster_summaries, f, indent=2, default=str)

    print(f'\n✓ Risk model estimated for {len(cluster_summaries)} rebalance dates')
    print(f'  Files saved to {RISK_DIR}/')
    print(f'    - correlation_raw.parquet')
    print(f'    - correlation_clean.parquet')
    print(f'    - covariance.parquet')
    print(f'    - clusters.parquet')
    print(f'    - eigenvalues.parquet')
    print(f'    - cluster_summary.json')

    print('\nDone!')
    print(f'\nNext: python scripts/pipeline/04c_construct_portfolio.py')


if __name__ == '__main__':
    main()