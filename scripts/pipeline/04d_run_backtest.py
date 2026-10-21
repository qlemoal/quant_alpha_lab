#!/usr/bin/env python
'''
Phase 3: Run Backtest
=====================

Executes walk-forward backtest using pre-constructed weights.
Applies transaction costs, computes performance metrics and attribution.

Run AFTER: 04c_construct_portfolio.py

Outputs:
  data/exports/
    ├── weights.parquet          # Final weights [date, ticker, weight, cluster_id]
    ├── returns.parquet          # Daily returns [date, gross_return, net_return, cost, turnover]
    ├── diagnostics.parquet      # Per-rebalance diagnostics
    ├── metrics.json             # Summary performance metrics
    └── attribution.parquet      # Performance attribution (if factors available)

Usage:
  python scripts/pipeline/04d_run_backtest.py

Env vars:
  NOTIONAL               = 1_000_000
  COST_MODEL             = 'sqrt'        # 'sqrt', 'linear', 'almgren_chriss_full'
  COMMISSION_BPS         = 1.0
  EXECUTION_LAG          = 1             # Days from signal to trade
  START_DATE             = '2010-01-01'
  END_DATE               = '2024-12-31'
'''

import os
import json
from pathlib import Path
from datetime import date
import polars as pl
import numpy as np

from src.portfolio.backtest import (
    run_backtest_from_weights,
    compute_metrics,
    performance_attribution,
)
from src.portfolio.costs import estimate_transaction_costs

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
EXPORTS_DIR = DATA_DIR / 'exports'

# Input (from previous phases)
WEIGHTS_FILE = EXPORTS_DIR / 'weights_pre_backtest.parquet'
PRICES_FILE = PROCESSED_DIR / 'prices.parquet'
FEATURES_FILE = PROCESSED_DIR / 'features.parquet'

# Output
FINAL_WEIGHTS_FILE = EXPORTS_DIR / 'weights.parquet'
RETURNS_FILE = EXPORTS_DIR / 'returns.parquet'
DIAGNOSTICS_FILE = EXPORTS_DIR / 'diagnostics.parquet'
METRICS_FILE = EXPORTS_DIR / 'metrics.json'
ATTRIBUTION_FILE = EXPORTS_DIR / 'attribution.parquet'

# Config
NOTIONAL = float(os.environ.get('NOTIONAL', '1_000_000'))
COST_MODEL = os.environ.get('COST_MODEL', 'sqrt')
COMMISSION_BPS = float(os.environ.get('COMMISSION_BPS', '1.0'))
EXECUTION_LAG = int(os.environ.get('EXECUTION_LAG', '1'))
START_DATE = os.environ.get('START_DATE', '2010-01-01')
END_DATE = os.environ.get('END_DATE', '2024-12-31')


def main():
    print('=' * 70)
    print('PHASE 3: RUN BACKTEST')
    print('=' * 70)
    print(f'Period:        {START_DATE} to {END_DATE}')
    print(f'Notional:      ${NOTIONAL:,.0f}')
    print(f'Cost model:    {COST_MODEL}')
    print(f'Commission:    {COMMISSION_BPS} bps')
    print(f'Execution lag: {EXECUTION_LAG} day(s)')
    print(f'Input weights: {WEIGHTS_FILE}')
    print('=' * 70)

    # Check inputs
    for f in [WEIGHTS_FILE, PRICES_FILE, FEATURES_FILE]:
        if not f.exists():
            raise FileNotFoundError(f'Required file not found: {f}. Run previous phases first.')

    # Load pre-constructed weights
    print('\n[1/4] Loading pre-constructed weights...')
    weights_df = pl.read_parquet(WEIGHTS_FILE)
    print(f'  Loaded {len(weights_df):,} weight records')
    print(f'  Date range: {weights_df["date"].min()} to {weights_df["date"].max()}')
    print(f'  Unique dates: {weights_df["date"].n_unique()}')

    # Load prices and features for cost model
    print('\n[2/4] Loading price and feature data...')
    prices_lf = pl.scan_parquet(PRICES_FILE)
    features_lf = pl.scan_parquet(FEATURES_FILE)

    # Get ADV and vol for cost model at rebalance dates
    rebalance_dates = sorted(weights_df['date'].unique().to_list())

    # For each rebalance, get ADV and vol of traded tickers
    # We'll compute costs in the backtest loop

    # Run backtest
    print('\n[3/4] Running walk-forward backtest with costs...')

    result = run_backtest_from_weights(
        weights_df=weights_df,
        prices_lf=prices_lf,
        features_lf=features_lf,
        start_date=date.fromisoformat(START_DATE),
        end_date=date.fromisoformat(END_DATE),
        notional=NOTIONAL,
        cost_model=COST_MODEL,
        commission_bps=COMMISSION_BPS,
        execution_lag=EXECUTION_LAG,
        adv_col='adv60',
        vol_col='std60',
    )

    weights_history = result['weights_history']
    returns_history = result['returns_history']
    diagnostics_history = result['diagnostics_history']
    summary = result['summary_metrics']

    print(f'    Weight records: {len(weights_history)}')
    print(f'    Return records: {len(returns_history)}')
    print(f'    Rebalances:     {len(diagnostics_history)}')

    # Save outputs
    print('\n[4/4] Saving results...')

    # Weights (final, with any adjustments from backtest)
    if len(weights_history) > 0:
        weights_history.write_parquet(FINAL_WEIGHTS_FILE)
        print(f'  Weights:     {FINAL_WEIGHTS_FILE} ({len(weights_history):,} rows)')

    # Returns
    if len(returns_history) > 0:
        returns_history.write_parquet(RETURNS_FILE)
        print(f'  Returns:     {RETURNS_FILE} ({len(returns_history):,} rows)')

    # Diagnostics
    if diagnostics_history:
        diag_df = pl.DataFrame(diagnostics_history)
        diag_df.write_parquet(DIAGNOSTICS_FILE)
        print(f'  Diagnostics: {DIAGNOSTICS_FILE} ({len(diag_df)} rows)')

    # Metrics
    with open(METRICS_FILE, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'  Metrics:     {METRICS_FILE}')

    # Attribution (if factor data available)
    print('\nComputing attribution...')
    try:
        factor_loadings_file = PROCESSED_DIR / 'factor_loadings.parquet'
        factor_returns_file = PROCESSED_DIR / 'factor_returns.parquet'

        if factor_loadings_file.exists() and factor_returns_file.exists():
            factor_loadings = pl.read_parquet(factor_loadings_file)
            factor_returns = pl.read_parquet(factor_returns_file)

            attribution = performance_attribution(
                weights_history=weights_history,
                returns_panel=pl.scan_parquet(PRICES_FILE),  # Need to compute returns panel
                factor_loadings=factor_loadings,
                factor_returns=factor_returns,
            )
            attribution.write_parquet(ATTRIBUTION_FILE)
            print(f'  Attribution: {ATTRIBUTION_FILE}')
        else:
            print('  Factor data not found, skipping factor attribution.')
    except Exception as e:
        print(f'  Attribution skipped: {e}')

    # Print summary
    print('\n' + '=' * 70)
    print('BACKTEST SUMMARY')
    print('=' * 70)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f'  {k}: {v:.4f}')
        else:
            print(f'  {k}: {v}')
    print('=' * 70)

    if summary:
        print(f'\nKey Metrics:')
        print(f'  Net Sharpe:     {summary.get("sharpe_net", 0):.3f}')
        print(f'  Gross Sharpe:   {summary.get("sharpe_gross", 0):.3f}')
        print(f'  Net Ann Return: {summary.get("annual_return_net", 0):.2%}')
        print(f'  Net Ann Vol:    {summary.get("annual_vol_net", 0):.2%}')
        print(f'  Max Drawdown:   {summary.get("max_drawdown", 0):.2%}')
        print(f'  Hit Rate:       {summary.get("hit_rate", 0):.2%}')
        print(f'  Cost Drag:      {summary.get("cost_drag_annual_bps", 0):.1f} bps/yr')
        print(f'  Avg Turnover:   {summary.get("avg_turnover", 0):.2%}')

    print('\nDone!')
    print(f'\nResults in: {EXPORTS_DIR}/')
    print(f'  weights.parquet, returns.parquet, diagnostics.parquet, metrics.json')


if __name__ == '__main__':
    main()