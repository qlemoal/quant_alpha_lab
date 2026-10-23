#!/usr/bin/env python
'''
===============================================
Phase 2: Construct Portfolio Weights
===============================================

Thin wrapper calling src/risk/portfolio_construction.py functions.
Loads risk model + scores, constructs weights at each rebalance.

Inputs:
  data/processed/en_scores_<tag>.parquet
  data/exports/risk_model_<tag>/

Outputs:
  data/exports/weights_<tag>.parquet    # [date, ticker, weight, cluster_id, signal, raw_w, capped_w]
  data/exports/weights_diag_<tag>.parquet

Usage:
  python scripts/pipeline/04e_construct_weights.py --signal-tag default
'''

import argparse
import json
from pathlib import Path
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

SIGNAL_TAG = 'default'

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
START_DATE = '2010-01-01'
END_DATE = '2024-12-31'
LOOKBACK_WINDOW = 252
MIN_OBS = 60
REBALANCE_FREQ = 'weekly'


def load_covariance(cov_file, reb_date, tickers):
    df = pl.read_parquet(cov_file).filter(pl.col('date') == str(reb_date))
    if len(df) == 0:
        return None
    n = len(tickers)
    idx = {t: i for i, t in enumerate(tickers)}
    cov = np.zeros((n, n))
    for row in df.iter_rows(named=True):
        i, j = idx[row['ticker_i']], idx[row['ticker_j']]
        cov[i, j] = cov[j, i] = row['cov']
    return cov


def load_clusters(clusters_file, reb_date, tickers):
    df = pl.read_parquet(clusters_file).filter(pl.col('date') == str(reb_date))
    if len(df) == 0:
        return None
    return {row['ticker']: row['cluster_id'] for row in df.iter_rows(named=True)}


def main(signal_tag: str):
    print('=' * 70)
    print('PHASE 2: CONSTRUCT PORTFOLIO WEIGHTS')
    print('=' * 70)
    print(f'Signal tag:    {signal_tag}')
    print(f'Config:        {PORTFOLIO_CONFIG}')
    print('=' * 70)

    # Paths
    risk_dir = Path(f'data/exports/risk_model_{signal_tag}')
    cov_file = risk_dir / 'cov.parquet'
    clusters_file = risk_dir / 'clusters.parquet'
    summary_file = risk_dir / 'summary.json'
    signal_file = f'data/processed/en_scores_{signal_tag}.parquet'

    for f in [cov_file, clusters_file, summary_file, signal_file]:
        if not Path(f).exists():
            raise FileNotFoundError(f'Missing: {f}. Run previous phases.')

    # Load rebalance dates from risk model
    with open(summary_file) as f:
        summaries = json.load(f)
    rebalance_dates = [s['date'] for s in summaries]

    # Prepare backtest data (aligns dates, gets returns panel)
    print('\n[1/3] Preparing data...')
    price_lf = pl.scan_parquet('data/processed/prices.parquet')
    signal_lf = pl.scan_parquet(signal_file).select(['date', 'ticker', SCORE_COL])

    prepared = prepare_backtest_data(
        signals_lf=signal_lf,
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

    prep_dates = [str(d) for d in prepared['rebalance_dates']]
    common_dates = sorted(set(prep_dates) & set(rebalance_dates))
    print(f'  {len(common_dates)} common rebalance dates')

    returns_panel = prepared['returns_panel'].collect()
    signals_lf = pl.scan_parquet(signal_file)

    # Construct weights
    print('\n[2/3] Constructing weights...')
    all_weights = []
    all_diag = []

    for i, reb_date in enumerate(common_dates):
        if i % 50 == 0:
            print(f'  [{i+1}/{len(common_dates)}] {reb_date}')

        # Get signals for this date
        score_df = signals_lf.filter(pl.col('date') == reb_date).collect()
        if len(score_df) == 0:
            continue

        tickers = score_df['ticker'].to_list()
        scores = score_df[SCORE_COL].to_numpy()

        cov = load_covariance(cov_file, reb_date, tickers)
        cluster_map = load_clusters(clusters_file, reb_date, tickers)
        if cov is None or cluster_map is None:
            continue

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

        # Diagnostics
        all_diag.append({
            'date': reb_date,
            'n_tickers': len(tickers),
            'gross_raw': float(np.sum(np.abs(raw_w))),
            'gross_capped': float(np.sum(np.abs(capped_w))),
            'gross_final': float(np.sum(np.abs(final_w))),
            'n_long': int(np.sum(final_w > 1e-6)),
            'n_short': int(np.sum(final_w < -1e-6)),
            'max_cluster_exp': float(max(
                sum(abs(w_f) for t, w_f in zip(tickers, final_w) if cluster_map.get(t) == c)
                for c in set(cluster_map.values())
            ) if cluster_map else 0),
        })

    # Save
    print('\n[3/3] Saving...')
    out_dir = Path('data/exports')
    weights_df = pl.DataFrame(all_weights)
    weights_df.write_parquet(out_dir / f'weights_{signal_tag}.parquet')

    diag_df = pl.DataFrame(all_diag)
    diag_df.write_parquet(out_dir / f'weights_diag_{signal_tag}.parquet')

    print(f'  Weights: {len(weights_df):,} rows -> data/exports/weights_{signal_tag}.parquet')
    print(f'  Diag:    {len(diag_df)} rows')
    print(f'  Avg gross: {diag_df["gross_final"].mean():.4f}')
    print(f'  Avg names: {diag_df["n_tickers"].mean():.0f}')

    print('\nDone!')
    print(f'\nNext: python scripts/pipeline/04f_run_backtest.py --signal-tag {signal_tag}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--signal-tag', type=str, default=SIGNAL_TAG)
    args = parser.parse_args()
    main(args.signal_tag)