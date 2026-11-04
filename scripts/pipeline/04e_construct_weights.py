#!/usr/bin/env python
'''
===============================================
Phase 2: Construct Portfolio Weights
===============================================

Thin wrapper calling src/risk/portfolio_construction.py functions.
Loads risk model + scores, constructs weights at each rebalance.

Inputs:
  data/clean/equities/US/SP500/daily/en_scores/{run_id}/en_scores.parquet
  data/clean/equities/US/SP500/daily/risk_model/{run_id}/

Outputs:
  data/clean/equities/US/SP500/daily/weights/{run_id}/weights.parquet    # [date, ticker, weight, cluster_id, signal, raw_w, capped_w]
  data/clean/equities/US/SP500/daily/weights/{run_id}/weights_diag.parquet
  data/clean/equities/US/SP500/daily/weights/{run_id}/config.json

Usage:
  python scripts/pipeline/04e_construct_weights.py --date 2026-10-29 --tag default
'''

import argparse
import json
from pathlib import Path
from datetime import date
import numpy as np
import polars as pl

from src.risk.portfolio_construction import (
    raw_weights_from_score,
    apply_cluster_caps,
    apply_vol_target,
)
from src.portfolio.backtest import prepare_backtest_data

# =============================================================================
# CONFIGURATION
# =============================================================================

# Portfolio construction config
PORTFOLIO_CONFIG = {
    'target_vol': 0.10,
    'max_gross': 1.0,
    'max_cluster_multiple': 2.0,
    'signal_method': 'proportional',   # 'proportional', 'inv_vol', 'rank'
    'cap_method': 'cap',               # 'cap', 'rescale'
    'vol_method': 'scale',             # 'scale', 'risk_parity'
}

SCORE_COL = 'combined_score'
LOOKBACK_WINDOW = 252
MIN_OBS = 60
REBALANCE_FREQ = 'weekly'


def load_covariance(cov_file, reb_date, tickers):
    df = pl.read_parquet(cov_file).filter(pl.col('date') == str(reb_date))
    if len(df) == 0:
        return None, None
    
    # Get tickers that are in both the scores and the covariance matrix
    cov_tickers = set(df['ticker_i'].unique()).union(set(df['ticker_j'].unique()))
    common_tickers = [t for t in tickers if t in cov_tickers]
    
    if len(common_tickers) == 0:
        return None, None
    
    n = len(common_tickers)
    idx = {t: i for i, t in enumerate(common_tickers)}
    cov = np.zeros((n, n))
    for row in df.iter_rows(named=True):
        ti, tj = row['ticker_i'], row['ticker_j']
        if ti in idx and tj in idx:
            i, j = idx[ti], idx[tj]
            cov[i, j] = cov[j, i] = row['cov']
    return cov, common_tickers


def load_clusters(clusters_file, reb_date, tickers):
    df = pl.read_parquet(clusters_file).filter(pl.col('date') == str(reb_date))
    if len(df) == 0:
        return None
    return {row['ticker']: row['cluster_id'] for row in df.iter_rows(named=True) if row['ticker'] in tickers}


def main(date: str, tag: str = 'default'):
    run_id = f'{date}_{tag}'
    print('=' * 70)
    print('PHASE 2: CONSTRUCT PORTFOLIO WEIGHTS')
    print('=' * 70)
    print(f'Run ID:        {run_id}')
    print(f'Date:          {date}')
    print(f'Tag:           {tag}')
    print(f'Config:        {PORTFOLIO_CONFIG}')
    print('=' * 70)

    # Input files
    en_scores_file = f'data/clean/equities/US/SP500/daily/en_scores/{run_id}/en_scores.parquet'
    risk_dir = Path(f'data/clean/equities/US/SP500/daily/risk_model/{run_id}')
    cov_file = risk_dir / 'cov.parquet'
    clusters_file = risk_dir / 'clusters.parquet'
    summary_file = risk_dir / 'summary.json'

    if not Path(en_scores_file).exists():
        raise FileNotFoundError(f'No en_scores file: {en_scores_file}')
    if not cov_file.exists():
        raise FileNotFoundError(f'No cov file: {cov_file}')
    if not clusters_file.exists():
        raise FileNotFoundError(f'No clusters file: {clusters_file}')
    if not summary_file.exists():
        raise FileNotFoundError(f'No summary file: {summary_file}')

    # Find OHLCV for this tag (not run_id since ohlcv is per tag)
    ohlcv_files = list(Path('data/clean/equities/US/SP500/daily/ohlcv').glob('ohlcv.parquet'))
    if not ohlcv_files:
        raise FileNotFoundError(f'No ohlcv file found for tag {tag}')
    ohlcv_file = sorted(ohlcv_files)[-1]
    print(f'  Using OHLCV: {ohlcv_file}')

    # Use single features file (no date/tag)
    features_file = Path('data/clean/equities/US/SP500/daily/features/features.parquet')
    if not features_file.exists():
        raise FileNotFoundError(f'No features file found: {features_file}')
    print(f'  Using features: {features_file}')

    out_dir = Path(f'data/clean/equities/US/SP500/daily/weights/{run_id}')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load rebalance dates from risk model
    with open(summary_file) as f:
        summaries = json.load(f)
    rebalance_dates = [s['date'] for s in summaries]

    # Prepare backtest data (aligns dates, gets returns panel)
    print('\n[1/3] Preparing data...')
    price_lf = pl.scan_parquet(ohlcv_file)
    features_lf = pl.scan_parquet(features_file)
    
    # Add adv60 and std60 from features
    features_lf = features_lf.with_columns([
        pl.col('log_adv60').exp().alias('adv60'),
        pl.col('vol60').alias('std60'),
        pl.col('logret'),
    ]).select(['date', 'ticker', 'logret', 'adv60', 'std60'])
    
    # Join prices with features
    complete_price_lf = price_lf.join(features_lf, on=['date', 'ticker'], how='left')
    
    signal_lf = pl.scan_parquet(en_scores_file).select(['date', 'ticker', SCORE_COL])

    # Get date range from data
    date_range = complete_price_lf.select(
        pl.col('date').min().alias('min_date'),
        pl.col('date').max().alias('max_date')
    ).collect()
    start_date = date_range['min_date'][0]
    
    # Add adv60 and std60 from features
    features_lf = features_lf.with_columns([
        pl.col('log_adv60').exp().alias('adv60'),
        pl.col('vol60').alias('std60'),
        pl.col('logret'),
    ]).select(['date', 'ticker', 'logret', 'adv60', 'std60'])
    
    # Join prices with features
    complete_price_lf = price_lf.join(features_lf, on=['date', 'ticker'], how='left')
    
    signal_lf = pl.scan_parquet(en_scores_file).select(['date', 'ticker', SCORE_COL])

    # Get date range from data
    date_range = complete_price_lf.select(
        pl.col('date').min().alias('min_date'),
        pl.col('date').max().alias('max_date')
    ).collect()
    start_date = date_range['min_date'][0]
    end_date = date_range['max_date'][0]

    prepared = prepare_backtest_data(
        signals_lf=signal_lf,
        score_cols=[SCORE_COL],
        price_lf=complete_price_lf,
        start_date=start_date,
        end_date=end_date,
        rebalance_freq=REBALANCE_FREQ,
        lookback_window=LOOKBACK_WINDOW,
        min_obs=MIN_OBS,
        score_col=SCORE_COL,
        adv_col='adv60',
        vol_col='std60',
    )

    prep_dates = [str(d) for d in prepared['rebalance_dates']]
    
    # Find dates with non-null combined_score
    from datetime import date as dt_date
    signal_lf_for_dates = pl.scan_parquet(en_scores_file)
    score_dates = set(str(d) for d in signal_lf_for_dates.filter(
        pl.col(SCORE_COL).is_not_null()
    ).select(pl.col('date').unique()).collect()['date'])
    
    # Risk model dates
    risk_dates = set(rebalance_dates)
    
    # Intersection: dates with risk model, backtest prep, AND non-null combined_score
    common_dates = sorted(set(prep_dates) & risk_dates & score_dates)
    print(f'  {len(common_dates)} common rebalance dates (with non-null {SCORE_COL})')

    returns_panel = prepared['returns_panel']  # Already a DataFrame

    # Pre-load signals for all common dates at once (much faster than filtering per date)
    print('\n[2/3] Pre-loading signals...')
    signals_lf = pl.scan_parquet(en_scores_file)
    # Only keep signals with non-null combined_score
    signals_df = signals_lf.filter(
        pl.col('date').dt.strftime('%Y-%m-%d').is_in(common_dates) & 
        pl.col(SCORE_COL).is_not_null()
    ).collect()
    # Group by date for fast lookup - use string dates as keys for consistency with cov/clusters
    signals_df = signals_df.with_columns(pl.col('date').dt.strftime('%Y-%m-%d').alias('date_str'))
    # Use to_dict to get string keys
    signals_by_date = {}
    for row in signals_df.iter_rows(named=True):
        d = row['date_str']
        if d not in signals_by_date:
            signals_by_date[d] = []
        signals_by_date[d].append(row)
    # Convert to DataFrames
    signals_by_date = {d: pl.DataFrame(g) for d, g in signals_by_date.items()}
    print(f'  Loaded signals for {len(signals_by_date)} dates (with non-null {SCORE_COL})')

    # Pre-load cov and clusters for all common dates
    print('\n[2/3b] Pre-loading covariance and clusters...')
    cov_lf = pl.scan_parquet(cov_file)
    clusters_lf = pl.scan_parquet(clusters_file)
    cov_df = cov_lf.filter(pl.col('date').is_in(common_dates)).collect()
    clusters_df = clusters_lf.filter(pl.col('date').is_in(common_dates)).collect()
    
    # Ensure date column is string type for consistent key access
    if cov_df['date'].dtype != pl.Utf8:
        cov_df = cov_df.with_columns(pl.col('date').cast(pl.Utf8).alias('date_str'))
    else:
        cov_df = cov_df.with_columns(pl.col('date').alias('date_str'))
    if clusters_df['date'].dtype != pl.Utf8:
        clusters_df = clusters_df.with_columns(pl.col('date').cast(pl.Utf8).alias('date_str'))
    else:
        clusters_df = clusters_df.with_columns(pl.col('date').alias('date_str'))
    
    # Build dicts using string keys
    cov_by_date = {}
    for row in cov_df.iter_rows(named=True):
        d = row['date_str']
        cov_by_date.setdefault(d, []).append(row)
    cov_by_date = {d: pl.DataFrame(g) for d, g in cov_by_date.items()}
    
    clusters_by_date = {}
    for row in clusters_df.iter_rows(named=True):
        d = row['date_str']
        clusters_by_date.setdefault(d, []).append(row)
    clusters_by_date = {d: pl.DataFrame(g) for d, g in clusters_by_date.items()}
    print(f'  Loaded cov for {len(cov_by_date)} dates, clusters for {len(clusters_by_date)} dates')

    # Construct weights
    print('\n[2/3] Constructing weights...')
    all_weights = []
    all_diag = []

    for i, reb_date in enumerate(common_dates):
        if i % 50 == 0:
            print(f'  [{i+1}/{len(common_dates)}] {reb_date}')

        # Get signals for this date (from pre-loaded dict)
        score_df = signals_by_date.get(reb_date)
        if score_df is None or len(score_df) == 0:
            continue

        tickers = score_df['ticker'].to_list()
        scores = score_df[SCORE_COL].to_numpy()

        # Get cov from pre-loaded dict
        cov_df = cov_by_date.get(reb_date)
        if cov_df is None or len(cov_df) == 0:
            continue
        
        cov_tickers = set(cov_df['ticker_i'].unique()).union(set(cov_df['ticker_j'].unique()))
        common_tickers = [t for t in tickers if t in cov_tickers]
        
        if len(common_tickers) == 0:
            continue
        
        n = len(common_tickers)
        idx = {t: i for i, t in enumerate(common_tickers)}
        cov = np.zeros((n, n))
        for row in cov_df.iter_rows(named=True):
            ti, tj = row['ticker_i'], row['ticker_j']
            if ti in idx and tj in idx:
                i, j = idx[ti], idx[tj]
                cov[i, j] = cov[j, i] = row['cov']
        
        # Filter tickers and scores to only those in covariance matrix
        ticker_mask = [t in common_tickers for t in tickers]
        tickers = [t for t in tickers if t in common_tickers]
        scores = scores[ticker_mask]
        
        # Get clusters from pre-loaded dict
        cluster_df = clusters_by_date.get(reb_date)
        if cluster_df is None or len(cluster_df) == 0:
            continue
        cluster_map = {row['ticker']: row['cluster_id'] for row in cluster_df.iter_rows(named=True) if row['ticker'] in tickers}

        # Step 1: Raw weights
        raw_w = raw_weights_from_score(
            pl.Series('score', scores),
            max_gross=PORTFOLIO_CONFIG['max_gross'],
            method=PORTFOLIO_CONFIG['signal_method'],
        )

        # Step 2: Cluster caps
        capped_w = apply_cluster_caps(
            raw_w, cluster_map,
            max_multiple=PORTFOLIO_CONFIG['max_cluster_multiple'],
            method=PORTFOLIO_CONFIG['cap_method'],
        )

        # Step 3: Vol targeting
        final_w = apply_vol_target(
            capped_w, cov, PORTFOLIO_CONFIG['target_vol'],
            method=PORTFOLIO_CONFIG['vol_method'],
        )

        # Record
        for t, w_r, w_c, w_f in zip(tickers, raw_w, capped_w, final_w):
            all_weights.append({
                'date': reb_date,
                'ticker': t,
                'signal': float(scores[tickers.index(t)]),
                'raw_weight': float(w_r),
                'capped_weight': float(w_c),
                'weight': float(w_f),
                'cluster_id': cluster_map.get(t, -1),
            })

        # Diagnostics - convert to numpy for numpy operations
        raw_w_np = raw_w.to_numpy()
        capped_w_np = capped_w.to_numpy()
        final_w_np = final_w.to_numpy()
        all_diag.append({
            'date': reb_date,
            'n_tickers': len(tickers),
            'gross_raw': float(np.sum(np.abs(raw_w_np))),
            'gross_capped': float(np.sum(np.abs(capped_w_np))),
            'gross_final': float(np.sum(np.abs(final_w_np))),
            'n_long': int(np.sum(final_w_np > 1e-6)),
            'n_short': int(np.sum(final_w_np < -1e-6)),
            'max_cluster_exp': float(max(
                sum(abs(w_f) for t, w_f in zip(tickers, final_w_np) if cluster_map.get(t) == c)
                for c in set(cluster_map.values())
            ) if cluster_map else 0),
        })

    # Save
    print('\n[3/3] Saving...')
    out_dir = Path(f'data/clean/equities/US/SP500/daily/weights/{run_id}')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    weights_df = pl.DataFrame(all_weights)
    weights_file = out_dir / 'weights.parquet'
    weights_df.write_parquet(weights_file)

    diag_df = pl.DataFrame(all_diag)
    diag_file = out_dir / 'weights_diag.parquet'
    diag_df.write_parquet(diag_file)

    # Save config
    config = {
        'run_id': run_id,
        'tag': tag,
        'portfolio_config': PORTFOLIO_CONFIG,
        'score_col': SCORE_COL,
        'lookback_window': LOOKBACK_WINDOW,
        'min_obs': MIN_OBS,
        'rebalance_freq': REBALANCE_FREQ,
        'start_date': str(start_date),
        'end_date': str(end_date),
    }
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)

    print(f'  Weights: {len(weights_df):,} rows -> {weights_file}')
    print(f'  Diag:    {len(diag_df)} rows -> {diag_file}')
    print(f'  Config:  {out_dir}/config.json')
    print(f'  Avg gross: {diag_df["gross_final"].mean():.4f}')
    print(f'  Avg names: {diag_df["n_tickers"].mean():.0f}')

    print('\nDone!')
    print(f'\nNext: python scripts/pipeline/04f_run_backtest.py --date {date} --tag {tag}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, required=True,
                        help='Date in YYYY-MM-DD format (e.g., 2026-10-29)')
    parser.add_argument('--tag', type=str, default='default',
                        help='Signal tag (default: default)')
    args = parser.parse_args()
    main(args.date, args.tag)