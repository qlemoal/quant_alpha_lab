#!/usr/bin/env python
'''
Phase 2: Construct Portfolio Weights
=====================================

Loads risk model (covariance, clusters) and signals, constructs weights at each rebalance.
Saves weights for inspection before backtest.

Run AFTER: 04b_estimate_risk_model.py

Outputs:
  data/exports/weights_pre_backtest.parquet  # [date, ticker, weight, cluster_id, signal, raw_weight, capped_weight, vol_scaled_weight]

Usage:
  python scripts/pipeline/04c_construct_portfolio.py

Env vars:
  REBALANCE_FREQ         = 'weekly'
  TARGET_VOL             = 0.10
  MAX_GROSS              = 1.0
  MAX_CLUSTER_MULTIPLE   = 2.0
  SIGNAL_METHOD          = 'proportional'  # 'proportional', 'inv_vol', 'rank'
  CAP_METHOD             = 'cap'           # 'cap', 'rescale'
  VOL_METHOD             = 'scale'         # 'scale', 'risk_parity'
  SCORE_COL              = 'combined_score'
  START_DATE             = '2010-01-01'
  END_DATE               = '2024-12-31'
'''

import os
import json
from pathlib import Path
from datetime import date
import polars as pl
import numpy as np

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
SIGNALS_FILE = PROCESSED_DIR / 'en_scores.parquet'  # From Phase 0
PRICES_FILE = PROCESSED_DIR / 'prices.parquet'
COV_FILE = RISK_DIR / 'covariance.parquet'
CLUSTERS_FILE = RISK_DIR / 'clusters.parquet'
CLUSTER_SUMMARY_FILE = RISK_DIR / 'cluster_summary.json'

# Output
WEIGHTS_FILE = EXPORTS_DIR / 'weights_pre_backtest.parquet'
WEIGHTS_DIAG_FILE = EXPORTS_DIR / 'weights_diagnostics.parquet'

# Config
REBALANCE_FREQ = os.environ.get('REBALANCE_FREQ', 'weekly')
TARGET_VOL = float(os.environ.get('TARGET_VOL', '0.10'))
MAX_GROSS = float(os.environ.get('MAX_GROSS', '1.0'))
MAX_CLUSTER_MULTIPLE = float(os.environ.get('MAX_CLUSTER_MULTIPLE', '2.0'))
SIGNAL_METHOD = os.environ.get('SIGNAL_METHOD', 'proportional')
CAP_METHOD = os.environ.get('CAP_METHOD', 'cap')
VOL_METHOD = os.environ.get('VOL_METHOD', 'scale')
SCORE_COL = os.environ.get('SCORE_COL', 'combined_score')
START_DATE = os.environ.get('START_DATE', '2010-01-01')
END_DATE = os.environ.get('END_DATE', '2024-12-31')
LOOKBACK_WINDOW = int(os.environ.get('LOOKBACK_WINDOW', '252'))
MIN_OBS = int(os.environ.get('MIN_OBS', '60'))


def load_covariance_matrix(cov_file, reb_date, tickers):
    '''Load covariance matrix for a specific date from long format'''
    df = pl.read_parquet(cov_file).filter(pl.col('date') == str(reb_date))
    if len(df) == 0:
        return None

    n = len(tickers)
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    cov = np.zeros((n, n))

    for row in df.iter_rows(named=True):
        i = ticker_to_idx[row['ticker_i']]
        j = ticker_to_idx[row['ticker_j']]
        cov[i, j] = row['cov']
        cov[j, i] = row['cov']  # Symmetric

    return cov


def load_clusters(clusters_file, reb_date, tickers):
    '''Load cluster assignments for a specific date'''
    df = pl.read_parquet(clusters_file).filter(pl.col('date') == str(reb_date))
    if len(df) == 0:
        return None
    return {row['ticker']: row['cluster_id'] for row in df.iter_rows(named=True)}


def main():
    print('=' * 70)
    print('PHASE 2: CONSTRUCT PORTFOLIO WEIGHTS')
    print('=' * 70)
    print(f'Period:           {START_DATE} to {END_DATE}')
    print(f'Rebalance freq:   {REBALANCE_FREQ}')
    print(f'Target vol:       {TARGET_VOL:.1%}')
    print(f'Max gross:        {MAX_GROSS:.1f}x')
    print(f'Cluster cap:      {MAX_CLUSTER_MULTIPLE:.1f}x equal share')
    print(f'Signal method:    {SIGNAL_METHOD}')
    print(f'Cap method:       {CAP_METHOD}')
    print(f'Vol method:       {VOL_METHOD}')
    print(f'Score column:     {SCORE_COL}')
    print(f'Output:           {WEIGHTS_FILE}')
    print('=' * 70)

    # Check inputs
    for f in [SIGNALS_FILE, COV_FILE, CLUSTERS_FILE, CLUSTER_SUMMARY_FILE]:
        if not f.exists():
            raise FileNotFoundError(f'Required file not found: {f}. Run previous phases first.')

    # Load cluster summary to get rebalance dates
    with open(CLUSTER_SUMMARY_FILE) as f:
        cluster_summaries = json.load(f)

    rebalance_dates = [c['date'] for c in cluster_summaries]
    print(f'\nLoaded {len(rebalance_dates)} rebalance dates from risk model')

    # Load signals
    print('\n[1/3] Loading signals...')
    signals_lf = pl.scan_parquet(SIGNALS_FILE)
    # We'll collect per date to avoid memory issues

    # Load price data for preparation
    print('[2/3] Preparing backtest data...')
    price_lf = pl.scan_parquet(PRICES_FILE)

    prepared = prepare_backtest_data(
        signals_lf=signals_lf.select(['date', 'ticker', SCORE_COL]),
        score_cols=[SCORE_COL],
        price_lf=price_lf,
        start_date=date.fromisoformat(START_DATE),
        end_date=date.fromisoformat(END_DATE),
        rebalance_freq=REBALANCE_FREQ,
        lookback_window=LOOKBACK_WINDOW,
        min_obs=MIN_OBS,
        score_col=SCORE_COL,
        adv_col='adv60',
        vol_col='std60',
    )

    # Verify dates match
    prep_dates = [str(d) for d in prepared['rebalance_dates']]
    if prep_dates != rebalance_dates:
        print(f'  WARNING: Date mismatch! Prepared: {len(prep_dates)}, Risk model: {len(rebalance_dates)}')
        # Use intersection
        common_dates = sorted(set(prep_dates) & set(rebalance_dates))
        print(f'  Using {len(common_dates)} common dates')
        rebalance_dates = common_dates

    # Get returns panel for vol targeting
    returns_panel = prepared['returns_panel'].collect()

    # Construct weights at each rebalance
    print('\n[3/3] Constructing weights...')

    all_weights = []
    all_diagnostics = []

    for i, reb_date in enumerate(rebalance_dates):
        if i % 50 == 0:
            print(f'  [{i+1}/{len(rebalance_dates)}] {reb_date}')

        # Get signals for this date
        score_df = signals_lf.filter(pl.col('date') == reb_date).collect()
        if len(score_df) == 0:
            continue

        tickers = score_df['ticker'].to_list()
        scores = score_df[SCORE_COL].to_numpy()

        # Load covariance and clusters for this date
        cov_matrix = load_covariance_matrix(COV_FILE, reb_date, tickers)
        cluster_map = load_clusters(CLUSTERS_FILE, reb_date, tickers)

        if cov_matrix is None or cluster_map is None:
            print(f'    WARNING: Missing risk model for {reb_date}, skipping')
            continue

        # --- Step 1: Raw weights from score ---
        raw_weights = raw_weights_from_score(
            pl.Series('score', scores),
            max_gross=MAX_GROSS,
            method=SIGNAL_METHOD,
        )

        # --- Step 2: Cluster caps ---
        capped_weights = apply_cluster_caps(
            raw_weights,
            cluster_map,
            max_multiple=MAX_CLUSTER_MULTIPLE,
            method=CAP_METHOD,
        )

        # --- Step 3: Vol targeting ---
        final_weights = apply_vol_target(
            capped_weights,
            cov_matrix,
            TARGET_VOL,
            method=VOL_METHOD,
        )

        # Record weights with diagnostics
        for t, w_raw, w_cap, w_final in zip(tickers, raw_weights, capped_weights, final_weights):
            all_weights.append({
                'date': reb_date,
                'ticker': t,
                'signal': float(scores[tickers.index(t)]) if t in tickers else 0.0,
                'raw_weight': float(w_raw),
                'capped_weight': float(w_cap),
                'weight': float(w_final),
                'cluster_id': cluster_map.get(t, -1),
            })

        # Diagnostics
        gross_raw = np.sum(np.abs(raw_weights))
        gross_cap = np.sum(np.abs(capped_weights))
        gross_final = np.sum(np.abs(final_weights))
        n_long = np.sum(final_weights > 1e-6)
        n_short = np.sum(final_weights < -1e-6)

        # Cluster exposure
        cluster_exposure = {}
        for t, w in zip(tickers, final_weights):
            c = cluster_map.get(t, -1)
            cluster_exposure[c] = cluster_exposure.get(c, 0) + abs(w)

        max_cluster_exp = max(cluster_exposure.values()) if cluster_exposure else 0

        all_diagnostics.append({
            'date': reb_date,
            'n_tickers': len(tickers),
            'gross_raw': float(gross_raw),
            'gross_capped': float(gross_cap),
            'gross_final': float(gross_final),
            'n_long': int(n_long),
            'n_short': int(n_short),
            'max_cluster_exposure': float(max_cluster_exp),
            'target_vol': TARGET_VOL,
            'signal_method': SIGNAL_METHOD,
            'cap_method': CAP_METHOD,
            'vol_method': VOL_METHOD,
        })

    # Save weights
    print('\nSaving weights...')
    weights_df = pl.DataFrame(all_weights)
    weights_df.write_parquet(WEIGHTS_FILE)
    print(f'  Weights: {WEIGHTS_FILE} ({len(weights_df):,} rows)')

    # Save diagnostics
    diag_df = pl.DataFrame(all_diagnostics)
    diag_df.write_parquet(WEIGHTS_DIAG_FILE)
    print(f'  Diagnostics: {WEIGHTS_DIAG_FILE} ({len(diag_df)} rows)')

    # Summary stats
    print(f'\n✓ Portfolio weights constructed for {len(all_diagnostics)} rebalances')
    print(f'  Avg gross (raw):    {diag_df["gross_raw"].mean():.4f}')
    print(f'  Avg gross (capped): {diag_df["gross_capped"].mean():.4f}')
    print(f'  Avg gross (final):  {diag_df["gross_final"].mean():.4f}')
    print(f'  Avg names/period:   {diag_df["n_tickers"].mean():.0f}')
    print(f'  Avg long/short:     {diag_df["n_long"].mean():.0f} / {diag_df["n_short"].mean():.0f}')
    print(f'  Max cluster exp:    {diag_df["max_cluster_exposure"].max():.2%}')

    print('\nDone!')
    print(f'\nNext: python scripts/pipeline/04d_run_backtest.py')
    print(f'      (or inspect {WEIGHTS_FILE} first)')


if __name__ == '__main__':
    main()