#!/usr/bin/env python
'''
===============================================
Phase 3: Run Backtest with Costs
===============================================

Thin wrapper calling src/portfolio/backtest.py and src/portfolio/costs.py.
Executes walk-forward backtest on pre-constructed weights.

Inputs:
  data/processed/weights/{run_id}/weights.parquet
  data/processed/en_scores/{run_id}/en_scores.parquet
  data/processed/ohlcv/{run_id}/ohlcv.parquet (has open, close, logret, dollar_volume)
  data/processed/features/{run_id}/features.parquet (has log_adv60, vol60)

Outputs:
  data/processed/backtest/{run_id}/
    ├── weights.parquet        # Final weights with execution lag applied
    ├── returns.parquet        # [date, gross_ret, net_ret, cost, turnover]
    ├── diagnostics.parquet    # Per-rebalance details
    ├── metrics.json           # Summary metrics
    └── config.json            # Run configuration

Usage:
  python scripts/pipeline/04f_run_backtest.py --run-id 2026-10-29_default
'''

import argparse
import json
from pathlib import Path
from datetime import date, timedelta
import polars as pl
import numpy as np

from src.portfolio.backtest import run_backtest, prepare_backtest_data, rebalance_step
from src.portfolio.costs import total_transaction_cost

# =============================================================================
# CONFIGURATION
# =============================================================================

BACKTEST_CONFIG = {
    'notional': 1_000_000,
    'cost_model': 'sqrt',           # 'sqrt', 'linear', 'almgren_chriss_full'
    'commission_bps': 1.0,
    'execution_lag': 1,             # Days from signal date to trade date
    'adv_col': 'adv60',
    'vol_col': 'std60',
}

REBALANCE_FREQ = 'weekly'
LOOKBACK_WINDOW = 252
MIN_OBS = 60
SCORE_COL = 'combined_score'


def main(run_id: str):
    print('=' * 70)
    print('PHASE 3: RUN BACKTEST')
    print('=' * 70)
    signal_tag = run_id.split('_', 1)[1] if '_' in run_id else 'default'
    print(f'Signal tag:    {signal_tag}')
    print(f'Run ID:        {run_id}')
    print(f'Config:        {BACKTEST_CONFIG}')
    print(f'Rebalance:     {REBALANCE_FREQ}')
    print('=' * 70)

    # Paths - new structure
    en_scores_file = f'data/processed/en_scores/{run_id}/en_scores.parquet'
    weights_file = f'data/processed/weights/{run_id}/weights.parquet'
    ohlcv_files = list(Path('data/processed/ohlcv').glob(f'*_{signal_tag}/ohlcv.parquet'))
    features_files = list(Path('data/processed/features').glob(f'*_{signal_tag}/features.parquet'))
    
    for f, name in [(en_scores_file, 'en_scores'), (weights_file, 'weights')]:
        if not Path(f).exists():
            raise FileNotFoundError(f'Missing {name}: {f}')
    
    if not ohlcv_files:
        raise FileNotFoundError(f'No ohlcv file found for tag {signal_tag}')
    ohlcv_file = sorted(ohlcv_files)[-1]
    
    if not features_files:
        raise FileNotFoundError(f'No features file found for tag {signal_tag}')
    features_file = sorted(features_files)[-1]
    
    print(f'  Using OHLCV: {ohlcv_file}')
    print(f'  Using features: {features_file}')

    # Output dir - new structure
    out_dir = Path(f'data/processed/backtest/{run_id}')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load EN scores (signals with combined_score)
    print('\n[1/4] Loading EN scores...')
    en_scores_df = pl.read_parquet(en_scores_file)
    print(f'  {len(en_scores_df):,} score records, {en_scores_df["date"].n_unique()} dates')

    # Load features (has logret, adv60, std60, open, close)
    print('\n[2/4] Loading features...')
    features_df = pl.read_parquet(features_file)
    print(f'  {len(features_df):,} feature records, {features_df["date"].n_unique()} dates')

    # Prepare data for backtest
    print('\n[3/4] Preparing backtest data...')
    
    # Use the ohlcv file for price data (has open, close, logret, dollar_volume)
    # Use features for adv60 (from log_adv60) and vol60
    ohlcv_lf = pl.scan_parquet(ohlcv_file)
    features_lf = pl.scan_parquet(features_file)
    
    # Features need: logret, adv60 (dollar volume), std60 (vol), open, close
    # log_adv60 is log of adv60, so adv60 = exp(log_adv60)
    # vol60 is the 60-day volatility (std of returns)
    features_for_backtest_lf = features_lf.select([
        'date', 'ticker', 
        pl.col('logret').alias('logret'),          # logret from features
        pl.col('log_adv60').exp().alias('adv60'),  # Convert log_adv60 to adv60
        pl.col('vol60').alias('std60'),            # Use vol60 as std60
    ])
    
    # OHLCV has: open, close, volume
    # Compute dollar_volume = close * volume
    ohlcv_for_backtest_lf = ohlcv_lf.select([
        'date', 'ticker', 'open', 'close', 
        (pl.col('close') * pl.col('volume')).alias('dollar_volume'),
    ])
    
    # Join OHLCV with features to get logret, adv60, and std60
    price_lf = ohlcv_for_backtest_lf.join(
        features_for_backtest_lf, 
        on=['date', 'ticker'], 
        how='left'
    )
    
    signals_lf = pl.scan_parquet(en_scores_file).select(['date', 'ticker', SCORE_COL])
    
    # Get date range from data
    date_range = price_lf.select(
        pl.col('date').min().alias('min_date'),
        pl.col('date').max().alias('max_date')
    ).collect()
    start_date = date_range['min_date'][0]
    end_date = date_range['max_date'][0]
    print(f'  Date range: {start_date} to {end_date}')
    
    # Prepare backtest data
    prepared = prepare_backtest_data(
        signals_lf=signals_lf,
        score_cols=[SCORE_COL],
        price_lf=price_lf,
        start_date=start_date,
        end_date=end_date,
        rebalance_freq=REBALANCE_FREQ,
        lookback_window=LOOKBACK_WINDOW,
        min_obs=MIN_OBS,
        date_col='date',
        ticker_col='ticker',
        score_col=SCORE_COL,
        adv_col='adv60',
        vol_col='std60',
        open_col='open',
        close_col='close',
        return_col='logret',
    )
    
    rebalance_dates = prepared['rebalance_dates']
    print(f'  Rebalance dates: {len(rebalance_dates)}')
    
    # Run the backtest
    print('\n[4/4] Running backtest...')
    
    # Path to pre-computed risk model from 04d
    risk_model_files = list(Path('data/processed/risk_model').glob(f'*_{signal_tag}/'))
    risk_model_dir = str(sorted(risk_model_files)[-1]) if risk_model_files else None
    if risk_model_dir:
        print(f'  Using pre-computed risk model: {risk_model_dir}')
    else:
        print('  No pre-computed risk model found, will re-estimate at each rebalance')
    
    result = run_backtest(
        prepared_data=prepared,
        target_vol=0.10,
        max_gross=1.0,
        max_cluster_multiple=2.0,
        signal_method='proportional',
        cap_method='cap',
        vol_method='scale',
        cov_method='ledoit_wolf',
        cluster_resolution=1.0,
        lookback_window=LOOKBACK_WINDOW,
        min_obs=MIN_OBS,
        notional=BACKTEST_CONFIG['notional'],
        cost_model=BACKTEST_CONFIG['cost_model'],
        commission_bps=BACKTEST_CONFIG['commission_bps'],
        execution_lag=BACKTEST_CONFIG['execution_lag'],
        score_col=SCORE_COL,
        risk_model_dir=risk_model_dir,
    )

    weights_hist = result['weights_history']
    returns_hist = result['returns_history']
    diag_hist = result['diagnostics_history']
    metrics = result['summary_metrics']

    print(f'  Weight records: {len(weights_hist)}')
    print(f'  Return records: {len(returns_hist)}')
    print(f'  Rebalances:     {len(diag_hist)}')

    # Save
    print('\n[5/5] Saving...')
    weights_hist.write_parquet(out_dir / 'weights.parquet')
    returns_hist.write_parquet(out_dir / 'returns.parquet')
    # Convert diagnostics to flat DataFrame (handle struct columns)
    if diag_hist:
        diag_df = pl.DataFrame(diag_hist)
        # Explode any struct columns to flat columns
        for col in diag_df.columns:
            if diag_df[col].dtype == pl.Struct:
                # Explode struct into separate columns
                struct_fields = diag_df[col].struct.fields
                for field in struct_fields:
                    diag_df = diag_df.with_columns(pl.col(col).struct.field(field).alias(f"{col}_{field}"))
                diag_df = diag_df.drop(col)
        diag_df.write_parquet(out_dir / 'diagnostics.parquet')

    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    # Save config
    config = {
        'run_id': run_id,
        'signal_tag': signal_tag,
        'backtest_config': BACKTEST_CONFIG,
        'rebalance_freq': REBALANCE_FREQ,
        'lookback_window': LOOKBACK_WINDOW,
        'min_obs': MIN_OBS,
        'score_col': SCORE_COL,
        'start_date': str(start_date),
        'end_date': str(end_date),
    }
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)

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
    parser.add_argument('--run-id', type=str, required=True,
                        help='Run ID in format YYYY-MM-DD_tag (e.g., 2026-10-29_default)')
    args = parser.parse_args()
    main(args.run_id)