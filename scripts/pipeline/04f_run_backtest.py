#!/usr/bin/env python
'''
===============================================
Phase 3: Run Backtest with Costs
===============================================

Thin wrapper calling src/portfolio/backtest.py and src/portfolio/costs.py.
Executes walk-forward backtest on pre-constructed weights.

Inputs:
  data/exports/weights_<tag>.parquet
  data/processed/prices.parquet
  data/processed/features.parquet

Outputs:
  data/exports/backtest_<tag>/
    ├── weights.parquet        # Final weights with execution lag applied
    ├── returns.parquet        # [date, gross_ret, net_ret, cost, turnover]
    ├── diagnostics.parquet    # Per-rebalance details
    └── metrics.json           # Summary metrics

Usage:
  python scripts/pipeline/04f_run_backtest.py --signal-tag default
'''

import argparse
import json
from pathlib import Path
from datetime import date
import polars as pl

from src.portfolio.backtest import run_backtest_from_weights

# =============================================================================
# CONFIGURATION
# =============================================================================

SIGNAL_TAG = 'default'

BACKTEST_CONFIG = {
    'notional': 1_000_000,
    'cost_model': 'sqrt',           # 'sqrt', 'linear', 'almgren_chriss_full'
    'commission_bps': 1.0,
    'execution_lag': 1,             # Days from signal date to trade date
    'adv_col': 'adv60',
    'vol_col': 'std60',
}

START_DATE = '2010-01-01'
END_DATE = '2024-12-31'


def main(signal_tag: str):
    print('=' * 70)
    print('PHASE 3: RUN BACKTEST')
    print('=' * 70)
    print(f'Signal tag:    {signal_tag}')
    print(f'Config:        {BACKTEST_CONFIG}')
    print(f'Date range:    {START_DATE} to {END_DATE}')
    print('=' * 70)

    # Paths
    weights_file = Path(f'data/exports/weights_{signal_tag}.parquet')
    prices_file = Path('data/processed/prices.parquet')
    features_file = Path('data/processed/features.parquet')

    for f in [weights_file, prices_file, features_file]:
        if not f.exists():
            raise FileNotFoundError(f'Missing: {f}')

    # Output dir
    out_dir = Path(f'data/exports/backtest_{signal_tag}')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load weights
    print('\n[1/3] Loading weights...')
    weights_df = pl.read_parquet(weights_file)
    print(f'  {len(weights_df):,} weight records, {weights_df["date"].n_unique()} dates')

    # Run backtest
    print('\n[2/3] Running backtest...')
    prices_lf = pl.scan_parquet(prices_file)
    features_lf = pl.scan_parquet(features_file)

    result = run_backtest_from_weights(
        weights_df=weights_df,
        prices_lf=prices_lf,
        features_lf=features_lf,
        start_date=date.fromisoformat(START_DATE),
        end_date=date.fromisoformat(END_DATE),
        notional=BACKTEST_CONFIG['notional'],
        cost_model=BACKTEST_CONFIG['cost_model'],
        commission_bps=BACKTEST_CONFIG['commission_bps'],
        execution_lag=BACKTEST_CONFIG['execution_lag'],
        adv_col=BACKTEST_CONFIG['adv_col'],
        vol_col=BACKTEST_CONFIG['vol_col'],
    )

    weights_hist = result['weights_history']
    returns_hist = result['returns_history']
    diag_hist = result['diagnostics_history']
    metrics = result['summary_metrics']

    print(f'  Weight records: {len(weights_hist)}')
    print(f'  Return records: {len(returns_hist)}')
    print(f'  Rebalances:     {len(diag_hist)}')

    # Save
    print('\n[3/3] Saving...')
    weights_hist.write_parquet(out_dir / 'weights.parquet')
    returns_hist.write_parquet(out_dir / 'returns.parquet')
    pl.DataFrame(diag_hist).write_parquet(out_dir / 'diagnostics.parquet')

    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    # Print summary
    print('\n' + '=' * 70)
    print('BACKTEST SUMMARY')
    print('=' * 70)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f'  {k}: {v:.4f}')
        else:
            print(f'  {k}: {v}')
    print('=' * 70)

    if metrics:
        print(f'\nKey Metrics:')
        print(f'  Net Sharpe:     {metrics.get("sharpe_net", 0):.3f}')
        print(f'  Gross Sharpe:   {metrics.get("sharpe_gross", 0):.3f}')
        print(f'  Net Ann Return: {metrics.get("annual_return_net", 0):.2%}')
        print(f'  Net Ann Vol:    {metrics.get("annual_vol_net", 0):.2%}')
        print(f'  Max Drawdown:   {metrics.get("max_drawdown", 0):.2%}')
        print(f'  Hit Rate:       {metrics.get("hit_rate", 0):.2%}')
        print(f'  Cost Drag:      {metrics.get("cost_drag_annual_bps", 0):.1f} bps/yr')
        print(f'  Avg Turnover:   {metrics.get("avg_turnover", 0):.2%}')

    print(f'\n✓ Backtest complete. Results in {out_dir}/')
    print('\nDone!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--signal-tag', type=str, default=SIGNAL_TAG)
    args = parser.parse_args()
    main(args.signal_tag)