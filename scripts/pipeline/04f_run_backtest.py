#!/usr/bin/env python
'''
===============================================
Phase 04f: Run Backtest
===============================================

Runs full backtest with transaction costs, computes performance metrics.

Inputs:
  data/clean/{asset}/{region}/{universe}/{freq}/weights/{run_id}/weights.parquet
  data/clean/{asset}/{region}/{universe}/{freq}/ohlcv/ohlcv.parquet
  data/clean/{asset}/{region}/{universe}/{freq}/features/features.parquet

Outputs:
  data/clean/{asset}/{region}/{universe}/{freq}/backtest/{run_id}/
    ├── returns.parquet      # [date, strategy_ret, long_ret, short_ret, tcost, turnover]
    ├── metrics.json         # Sharpe, IR, max_dd, etc.
    ├── equity_curve.png
    └── config.json

Usage:
  python scripts/pipeline/04f_run_backtest.py --date 2026-10-29 --tag default
'''

import argparse
import json
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config.paths import get_data_config, run_id_from_args
from src.portfolio.costs import estimate_costs
from src.portfolio.backtest import backtest_weights

# =============================================================================
# CONFIGURATION
# =============================================================================

BACKTEST_CONFIG = {
    'cost_bps': 5.0,          # 5 bps per side (10 bps round trip)
    'spread_pct': 0.0005,     # 5 bps half-spread
    'impact_coeff': 0.0,      # No market impact for now
    'adv_window': 60,
    'max_participation': 0.10,
}


def main(date: str, tag: str = 'default'):
    run_id = run_id_from_args(date, tag)
    
    # Use centralized config
    cfg = get_data_config(run_id=run_id)
    
    print('=' * 70)
    print('PHASE 04f: RUN BACKTEST')
    print('=' * 70)
    print(f'Run ID:        {run_id}')
    print(f'Config:        {BACKTEST_CONFIG}')
    print('=' * 70)

    # Input files
    weights_file = cfg.weights_file
    ohlcv_file = cfg.clean_ohlcv_file
    features_file = cfg.clean_features_file

    if not weights_file.exists():
        raise FileNotFoundError(f'No weights file: {weights_file}')
    if not ohlcv_file.exists():
        raise FileNotFoundError(f'No ohlcv file: {ohlcv_file}')
    if not features_file.exists():
        raise FileNotFoundError(f'No features file: {features_file}')

    out_dir = cfg.backtest_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print('\n[1/4] Loading data...')
    weights_df = pl.read_parquet(weights_file)
    print(f'  Weights: {len(weights_df):,} rows, {weights_df["date"].n_unique()} dates')

    price_lf = pl.scan_parquet(ohlcv_file)
    features_lf = pl.scan_parquet(features_file)

    # Prepare features for returns
    features_lf = features_lf.with_columns([
        pl.col('log_adv60').exp().alias('adv60'),
        pl.col('vol60').alias('std60'),
        pl.col('logret'),
    ]).select(['date', 'ticker', 'logret', 'adv60', 'std60'])

    complete_price_lf = price_lf.join(features_lf, on=['date', 'ticker'], how='left')

    # Get date range
    dates = sorted(weights_df['date'].unique().to_list())
    start_date = dates[0]
    end_date = dates[-1]
    print(f'  Backtest period: {start_date} to {end_date}')

    # Run backtest
    print('\n[2/4] Running backtest...')
    result = backtest_weights(
        weights_lf=weights_df.lazy(),
        price_lf=complete_price_lf,
        cost_bps=BACKTEST_CONFIG['cost_bps'],
        adv_col='adv60',
    )

    returns_df = result['returns']
    metrics = result['metrics']
    positions_df = result['positions']

    print(f'  Backtest complete: {len(returns_df)} periods')
    print(f'  Annualized Sharpe: {metrics.get("sharpe", "N/A"):.4f}')
    print(f'  Annualized IR:     {metrics.get("information_ratio", "N/A"):.4f}')
    print(f'  Max Drawdown:      {metrics.get("max_drawdown", "N/A"):.2%}')
    print(f'  Hit Rate:          {metrics.get("hit_rate", "N/A"):.2%}')
    print(f'  Avg Turnover:      {metrics.get("avg_turnover", "N/A"):.4f}')
    print(f'  Avg T-Cost:        {metrics.get("avg_tcost", "N/A"):.6f}')

    # Save returns
    print('\n[3/4] Saving results...')
    returns_file = out_dir / 'returns.parquet'
    returns_df.write_parquet(returns_file)
    print(f'  Returns: {returns_file}')

    # Save metrics
    metrics_file = out_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f'  Metrics: {metrics_file}')

    # Plot equity curve
    print('\n[4/4] Plotting equity curve...')
    equity_curve = (1 + returns_df['strategy_ret'].fill_null(0)).cumprod()
    plt.figure(figsize=(10, 5))
    plt.plot(equity_curve)
    plt.title(f'Equity Curve — {run_id}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    equity_file = out_dir / 'equity_curve.png'
    plt.savefig(equity_file, dpi=150)
    plt.close()
    print(f'  Equity curve: {equity_file}')

    # Save config
    config = {
        'run_id': run_id,
        'tag': tag,
        'backtest_config': BACKTEST_CONFIG,
        'start_date': str(start_date),
        'end_date': str(end_date),
        'n_periods': len(returns_df),
    }
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)

    print('\n✓ Backtest complete!')
    print(f'  Output: {out_dir}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, required=True,
                        help='Date in YYYY-MM-DD format (e.g., 2026-10-29)')
    parser.add_argument('--tag', type=str, default='default',
                        help='Signal tag (default: default)')
    args = parser.parse_args()
    main(args.date, args.tag)
