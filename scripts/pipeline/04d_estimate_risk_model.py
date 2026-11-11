#!/usr/bin/env python
'''
===============================================
Phase 04d: Estimate Risk Model (Covariance + Clusters)
===============================================

QUESTION ANSWERED
    "At each rebalance date, how do the stocks move together, and how risky is each one?"

THE STAGE IN ONE PICTURE

    features.parquet                 en_scores.parquet
    (date, ticker, logret)           (dates that have a score)
            |                                |
            v                                v
     [1] returns panel              [2] rebalance dates
     (dates x tickers)              (last trading day of each week)
            \\                              /
             \\                            /
              v                          v
             [3] plan: for each date, the 252-day window + the eligible tickers
                              |
                              v        one task per date, run in parallel
             [4] src/risk/risk_model.py (the maths, one date at a time):
                    raw correlation
                      -> Marchenko-Pastur cleaning
                      -> shrinkage (Ledoit-Wolf)
                      -> covariance = D x C x D   (annualised)
                      -> Louvain clusters
                              |
                              v
             [5] save files + summary.json

WHERE IS THE COMPLICATED PART?
    src/risk/risk_model.py     the statistics (one date)
    src/risk/risk_pipeline.py  windows, eligibility, parallel run, summary
    src/utils/risk_io.py       reading and writing the matrix files
    This script only calls them, in order.

NO LOOK-AHEAD
    The window for date t holds returns up to the CLOSE of t. The signal of t is also known at the close of t.
    The portfolio trades at the open of t+1.

INPUTS
    {clean_base}/features/features.parquet                    (columns: date, ticker, logret)
    {clean_base}/en_scores/{run_id}/en_scores.parquet         (columns: date, ticker, combined_score)

OUTPUTS   {clean_base}/risk_model/{run_id}/
    corr_clean.parquet    cleaned correlation, long format, upper triangle  [date, ticker_i, ticker_j, corr]
    cov.parquet           ANNUALISED covariance, same format                [date, ticker_i, ticker_j, cov]
    clusters.parquet      [date, ticker, cluster_id]
    eigenvalues.parquet   [date, eigenvalue, component]
    corr_raw.parquet      only if RISK_CONFIG['save_corr_raw'] is True
    summary.json          a DICT; the per-date list is under the key 'per_date'
    config.json
    Read a matrix back with src.utils.risk_io.load_matrix_for_date().

USAGE
    python scripts/pipeline/04d_estimate_risk_model.py --date 2026-11-07 --tag default
    QUANTALPHA_TEST_MODE=1 python scripts/pipeline/04d_estimate_risk_model.py --date 2026-11-07 --tag test
        (test mode: only the last 10 rebalance dates)
'''

import argparse
import os
import sys
from pathlib import Path

# Make `import src...` and `import config...` work wherever the script is started from.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import polars as pl

from config.paths import get_paths, run_id_from_args
from src.risk.risk_pipeline import (
    make_returns_panel,
    pick_rebalance_dates,
    plan_rebalances,
    run_risk_model,
    build_summary,
    write_json,
)
from src.utils.risk_io import combine_parquet_files


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_COL = 'combined_score'

RISK_CONFIG = {
    'rebalance_freq': 'weekly',       # 'weekly' or 'monthly': last trading day of the period
    'lookback_window': 252,           # trading days of returns per date
    'min_obs': 60,                    # a ticker needs this many returns in the window to be eligible
    'min_assets': 50,                 # a date needs this many eligible tickers, else it is skipped
    'cov_method': 'ledoit_wolf',      # 'ledoit_wolf', 'oas', 'factor', 'sample'
    'cluster_resolution': 1.0,        # Louvain resolution (higher = more, smaller clusters)
    'n_factors': 5,                   # only for cov_method='factor'
    'save_corr_raw': False,           # the raw correlation is big and not used downstream
}


def main(run_date: str, tag: str):
    run_id = run_id_from_args(run_date, tag)
    paths = get_paths(run_id=run_id)

    en_scores_file = paths.stage_dir('en_scores') / 'en_scores.parquet'
    features_file = paths.features_file
    out_dir = paths.stage_dir('risk_model')

    for f in (en_scores_file, features_file):
        if not f.exists():
            raise FileNotFoundError(f'Missing input: {f}')
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print('PHASE 04d: ESTIMATE RISK MODEL')
    print('=' * 70)
    print(f'Run ID:   {run_id}')
    print(f'Config:   {RISK_CONFIG}')
    print(f'Output:   {out_dir}')

    # ------------------------------------------------------------------
    # [1/5] Returns panel: rows = dates, columns = tickers
    # ------------------------------------------------------------------
    print('\n[1/5] Loading returns...')
    returns_long = pl.read_parquet(features_file, columns=['date', 'ticker', 'logret'])
    panel = make_returns_panel(returns_long)
    print(f'  {len(panel.dates)} dates x {len(panel.tickers)} tickers '
          f'({panel.dates[0]} to {panel.dates[-1]})')

    # ------------------------------------------------------------------
    # [2/5] Rebalance dates: last trading day of each week, among the dates that have a score
    # ------------------------------------------------------------------
    print('\n[2/5] Choosing rebalance dates...')
    score_dates = (
        pl.scan_parquet(en_scores_file)
        .filter(pl.col(SCORE_COL).is_not_null())
        .select(pl.col('date').unique())
        .collect()['date'].to_list()
    )
    panel_dates = set(panel.dates.astype(object))          # datetime.date objects
    usable_dates = [d for d in score_dates if d in panel_dates]

    first_date = panel.dates[RISK_CONFIG['min_obs']].astype(object)   # need min_obs rows of history
    rebalance_dates = pick_rebalance_dates(usable_dates, RISK_CONFIG['rebalance_freq'], first_date)
    if not rebalance_dates:
        raise ValueError('No rebalance date found. Check that en_scores.parquet and features.parquet overlap.')

    if os.environ.get('QUANTALPHA_TEST_MODE', '').lower() in ('1', 'true', 'yes'):
        rebalance_dates = rebalance_dates[-10:]
        print(f'  TEST MODE: keeping the last {len(rebalance_dates)} rebalance dates')
    print(f'  {len(rebalance_dates)} rebalance dates ({rebalance_dates[0]} to {rebalance_dates[-1]})')

    # ------------------------------------------------------------------
    # [3/5] Plan: window + eligible tickers for each date
    # ------------------------------------------------------------------
    print('\n[3/5] Planning windows and eligible tickers...')
    plans, skipped = plan_rebalances(
        panel, rebalance_dates,
        lookback_window=RISK_CONFIG['lookback_window'],
        min_obs=RISK_CONFIG['min_obs'],
        min_assets=RISK_CONFIG['min_assets'],
    )
    for s in skipped:
        print(f'  Skipped {s["date"]}: {s["reason"]}')
    print(f'  {len(plans)} dates to process, {len(skipped)} skipped')

    # ------------------------------------------------------------------
    # [4/5] Risk model for every date (parallel). Per-date files are written by the workers.
    # ------------------------------------------------------------------
    print(f'\n[4/5] Estimating the risk model ({os.cpu_count()} cores available)...')
    summaries = run_risk_model(panel, plans, RISK_CONFIG, out_dir)

    # ------------------------------------------------------------------
    # [5/5] Combine per-date files, write summary and config
    # ------------------------------------------------------------------
    print('\n[5/5] Combining files and writing the summary...')
    date_suffixes = [s['date'] for s in summaries]
    for base_name in ('corr_raw', 'corr_clean', 'cov', 'clusters', 'eigenvalues'):
        combine_parquet_files(out_dir, base_name, date_suffixes)

    summary = build_summary(summaries, skipped)
    write_json(out_dir / 'summary.json', summary)
    write_json(out_dir / 'config.json', {
        'run_id': run_id,
        'tag': tag,
        'risk_config': RISK_CONFIG,
        'score_col': SCORE_COL,
        'start_date': summary['date_range']['start'],
        'end_date': summary['date_range']['end'],
    })

    print(f'\nRisk model estimated for {summary["n_rebalances"]} rebalances')
    print(f'  average shrinkage:  {summary["avg_shrinkage"]}   (should vary by date, not sit at 1.0)')
    print(f'  average median vol: {summary["avg_median_vol"]}   (annualised, equities: about 0.2 to 0.5)')
    print(f'  average clusters:   {summary["avg_n_clusters"]}')
    print(f'  output: {out_dir}/')
    print(f'\nNext: python scripts/pipeline/04e_construct_weights.py --date {run_date} --tag {tag}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate the risk model (correlation, covariance, clusters)')
    parser.add_argument('--date', type=str, required=True, help='Run date, YYYY-MM-DD (used in the run_id)')
    parser.add_argument('--tag', type=str, default='default', help='Run tag (default: default)')
    args = parser.parse_args()
    main(args.date, args.tag)