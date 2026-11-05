#!/usr/bin/env python
'''
===============================================
Phase 04e: Construct Portfolio Weights
===============================================

Thin wrapper calling src/risk/portfolio_construction.py functions.
Loads risk model + scores, constructs weights at each rebalance.

Inputs:
  data/clean/{asset}/{region}/{universe}/{freq}/en_scores/{run_id}/en_scores.parquet
  data/clean/{asset}/{region}/{universe}/{freq}/risk_model/{run_id}/

Outputs:
  data/clean/{asset}/{region}/{universe}/{freq}/weights/{run_id}/weights.parquet
  data/clean/{asset}/{region}/{universe}/{freq}/weights/{run_id}/weights_diag.parquet
  data/clean/{asset}/{region}/{universe}/{freq}/weights/{run_id}/config.json

Usage:
  python scripts/pipeline/04e_construct_weights.py --date 2026-10-29 --tag default
'''

import argparse
import json
from pathlib import Path
from datetime import date
import numpy as np
import polars as pl

from config.paths import get_data_config, run_id_from_args
from src.risk.portfolio_construction import (
    raw_weights_from_score,
    apply_cluster_caps,
    apply_vol_target,
)
from src.portfolio.backtest import prepare_backtest_data

# =============================================================================
# CONFIGURATION
# =============================================================================

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
    run_id = run_id_from_args(date, tag)
    
    # Use centralized config
    cfg = get_data_config(run_id=run_id)
    
    print('=' * 70)
    print('PHASE 04e: CONSTRUCT PORTFOLIO WEIGHTS')
    print('=' * 70)
    print(f'Run ID:        {run_id}')
    print(f'Config:        {PORTFOLIO_CONFIG}')
    print('=' * 70)

    # Input files
    en_scores_file = cfg.en_scores_file
    risk_dir = cfg.risk_model_dir
    cov_file = risk_dir / 'cov.parquet'
    clusters_file = risk_dir / 'clusters.parquet'
    summary_file = risk_dir / 'summary.json'

    if not en_scores_file.exists():
        raise FileNotFoundError(f'No en_scores file: {en_scores_file}')
    if not cov_file.exists():
        raise FileNotFoundError(f'No cov file: {cov_file}')
    if not clusters_file.exists():
        raise FileNotFoundError(f'No clusters file: {clusters_file}')
    if not summary_file.exists():
        raise FileNotFoundError(f'No summary file: {summary_file}')

    # OHLCV and features (shared base tables)
    ohlcv_file = cfg.clean_ohlcv_file
    features_file = cfg.clean_features_file
    if not ohlcv_file.exists():
        raise FileNotFoundError(f'No ohlcv file: {ohlcv_file}')
    if not features_file.exists():
        raise FileNotFoundError(f'No features file: {features_file}')

    out_dir = cfg.weights_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load rebalance dates from risk model
    with open(summary_file) as f:
        summaries = json.load(f)
    rebalance_dates = [s['date'] for s in summaries]

    # Prepare backtest data
    print('\n[1/3] Preparing data...')
    price_lf = pl.scan_parquet(ohlcv_file)
    features_lf = pl.scan_parquet(features_file)
    
    features_lf = features_lf.with_columns([
        pl.col('log_adv60').exp().alias('adv60'),
        pl.col('vol60').alias('std60'),
        pl.col('logret'),
    ]).select(['date', 'ticker', 'logret', 'adv60', 'std60'])
    
    complete_price_lf = price_lf.join(features_lf, on=['date', 'ticker'], how='left')
    
    signal_lf = pl.scan_parquet(en_scores_file).select(['date', 'ticker', SCORE_COL])

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
    signal_lf_for_dates = pl.scan_parquet(en_scores_file)
    score_dates = set(str(d) for d in signal_lf_for_dates.filter(
        pl.col(SCORE_COL).is_not_null()
    ).select(pl.col('date').unique()).collect()['date'])
    
    risk_dates = set(rebalance_dates)
    common_dates = sorted(set(prep_dates) & risk_dates & score_dates)
    print(f'  {len(common_dates)} common rebalance dates (with non-null {SCORE_COL})')

    # Pre-load signals for all common dates
    print('\n[2/3] Pre-loading signals...')
    signals_lf = pl.scan_parquet(en_scores_file)
    signals_df = signals_lf.filter(
        pl.col('date').dt.strftime('%Y-%m-%d').is_in(common_dates) & 
        pl.col(SCORE_COL).is_not_null()
    ).collect()
    signals_df = signals_df.with_columns(pl.col('date').dt.strftime('%Y-%m-%d').alias('date_str'))
    signals_by_date = {}
    for row in signals_df.iter_rows(named=True):
        d = row['date_str']
        if d not in signals_by_date:
            signals_by_date[d] = []
        signals_by_date[d].append(row)
    signals_by_date = {d: pl.DataFrame(g) for d, g in signals_by_date.items()}
    print(f'  Loaded signals for {len(signals_by_date)} dates')

    # Pre-load cov and clusters
    print('\n[2/3b] Pre-loading covariance and clusters...')
    cov_lf = pl.scan_parquet(cov_file)
    clusters_lf = pl.scan_parquet(clusters_file)
    cov_df = cov_lf.filter(pl.col('date').is_in(common_dates)).collect()
    clusters_df = clusters_lf.filter(pl.col('date').is_in(common_dates)).collect()
    
    if cov_df['date'].dtype != pl.Utf8:
        cov_df = cov_df.with_columns(pl.col('date').cast(pl.Utf8).alias('date_str'))
    else:
        cov_df = cov_df.with_columns(pl.col('date').alias('date_str'))
    if clusters_df['date'].dtype != pl.Utf8:
        clusters_df = clusters_df.with_columns(pl.col('date').cast(pl.Utf8).alias('date_str'))
    else:
        clusters_df = clusters_df.with_columns(pl.col('date').alias('date_str'))
    
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

    # Construct weights per date
    print('\n[3/3] Constructing weights...')
    all_weights = []
    all_diag = []

    for i, reb_date in enumerate(common_dates):
        if i % 50 == 0:
            print(f'  [{i+1}/{len(common_dates)}] {reb_date}')

        # Get signals for this date
        if reb_date not in signals_by_date:
            continue
        sig_df = signals_by_date[reb_date]
        tickers = sig_df['ticker'].to_list()
        scores = sig_df[SCORE_COL].to_numpy()
        n = len(tickers)

        # Get covariance and clusters for this date
        cov_df_date = cov_by_date.get(reb_date)
        cluster_df_date = clusters_by_date.get(reb_date)
        
        if cov_df_date is None:
            continue

        # Build covariance matrix
        cov_tickers = set(cov_df_date['ticker_i'].unique()).union(set(cov_df_date['ticker_j'].unique()))
        common = [t for t in tickers if t in cov_tickers]
        if len(common) < 10:
            continue

        idx = {t: i for i, t in enumerate(common)}
        cov = np.zeros((len(common), len(common)))
        for row in cov_df_date.iter_rows(named=True):
            ti, tj = row['ticker_i'], row['ticker_j']
            if ti in idx and tj in idx:
                i, j = idx[ti], idx[tj]
                cov[i, j] = cov[j, i] = row['cov']

        # Clusters
        cluster_map = {}
        if cluster_df_date is not None:
            cluster_map = {row['ticker']: row['cluster_id'] for row in cluster_df_date.iter_rows(named=True) if row['ticker'] in common}

        # Align scores to common tickers
        scores_aligned = np.array([scores[tickers.index(t)] for t in common])

        # Raw weights
        raw_w = raw_weights_from_score(
            scores=scores_aligned,
            method=PORTFOLIO_CONFIG['signal_method'],
        )

        # Cluster caps
        capped_w = apply_cluster_caps(
            weights=raw_w,
            tickers=common,
            cluster_map=cluster_map,
            max_cluster_multiple=PORTFOLIO_CONFIG['max_cluster_multiple'],
            method=PORTFOLIO_CONFIG['cap_method'],
        )

        # Vol target
        final_w = apply_vol_target(
            weights=capped_w,
            cov=cov,
            target_vol=PORTFOLIO_CONFIG['target_vol'],
            method=PORTFOLIO_CONFIG['vol_method'],
        )

        # Gross cap
        gross = np.sum(np.abs(final_w))
        if gross > PORTFOLIO_CONFIG['max_gross']:
            final_w = final_w * PORTFOLIO_CONFIG['max_gross'] / gross

        # Save weights
        for j, t in enumerate(common):
            all_weights.append({
                'date': reb_date,
                'ticker': t,
                'weight': float(final_w[j]),
                'cluster_id': int(cluster_map.get(t, -1)) if cluster_map else -1,
                'raw_weight': float(raw_w[j]),
                'capped_weight': float(capped_w[j]),
            })

        # Diagnostics
        all_diag.append({
            'date': reb_date,
            'n_tickers': len(common),
            'gross': float(np.sum(np.abs(final_w))),
            'long_exposure': float(np.sum(final_w[final_w > 0])),
            'short_exposure': float(np.abs(np.sum(final_w[final_w < 0]))),
            'max_weight': float(np.max(final_w)) if len(final_w) > 0 else 0.0,
            'min_weight': float(np.min(final_w)) if len(final_w) > 0 else 0.0,
            'turnover': float(np.sum(np.abs(final_w))) if i == 0 else None,  # Will fill below
        })

    # Fill turnover
    for i in range(1, len(all_diag)):
        prev = [w for w in all_weights if w['date'] == all_diag[i-1]['date']]
        curr = [w for w in all_weights if w['date'] == all_diag[i]['date']]
        if prev and curr:
            prev_w = {w['ticker']: w['weight'] for w in prev}
            curr_w = {w['ticker']: w['weight'] for w in curr}
            all_t = set(prev_w.keys()) | set(curr_w.keys())
            turnover = sum(abs(curr_w.get(t, 0) - prev_w.get(t, 0)) for t in all_t) / 2
            all_diag[i]['turnover'] = turnover

    # Write outputs
    weights_df = pl.DataFrame(all_weights)
    diag_df = pl.DataFrame(all_diag)
    weights_df.write_parquet(cfg.weights_file)
    diag_df.write_parquet(cfg.weights_diag_file)
    print(f'\n✓ Saved weights to {cfg.weights_file}')
    print(f'  {len(all_weights):,} weight entries across {len(all_diag)} dates')

    # Save config
    config = {
        'run_id': run_id,
        'tag': tag,
        'portfolio_config': PORTFOLIO_CONFIG,
        'score_col': SCORE_COL,
    }
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)

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
