'''
Full portfolio construction & backtest pipeline.


================================================================================
POSSIBLE CONSTRUCTIONS
================================================================================

1. Inverse Volatility / Risk Parity Weighting (Asness, Frazzini, Pedersen 2012)
   - Weight ∝ 1/σᵢ, risk-equalized positions
   - More robust than mean-variance; no expected return estimation needed
2. Mean-Variance with Shrinkage (Ledoit-Wolf 2004, 2012; Oracle Approximating Shrinkage)
   - $\hat{Sigma}=\rho \dot F + (1-\rho) \dot S$ or Σ̂ = ρ·F + (1-ρ)·S where F is structured estimator (single-factor, constant correlation)
   - Analytical optimal shrinkage intensity ρ
   - Used at AQR, BlackRock, etc.
3. Hierarchical Risk Parity (HRP) (López de Prado 2016)
   - Uses correlation clustering (like our Louvain) to build hierarchical structure
   - Recursive bisection allocates risk budget top-down
   - Solves instability of Markowitz without ad-hoc constraints
4. Clustered Risk Parity (Raffinot 2018; extends HRP)
   - Directly uses our Louvain clusters as risk allocation units
   - Equal risk contribution within and across clusters
5. Factor-Aware Optimization (Meucci 2007; Black-Litterman for signals)
   - Explicit factor model: r = B·f + ε
   - Optimize in factor space (low-dim) rather than asset space
   - Signal as "views" in Black-Litterman framework
6. Transaction Cost-Aware Optimization (Almgren-Chriss 2001; square-root impact)
   - Cost = spread/2 + commission + σ·√(participation_rate)
   - Integrated into objective: max w'μ - λ·w'Σw - κ·√(w'·ADV⁻¹·w)
   - Solved via convex optimization (CVXPY) or custom first-order methods
7. Robust / Distributionally Robust Optimization (Ben-Tal, El Ghaoui, Nemirovski; Bertsimas)
   - Max-min over uncertainty set for μ, Σ
   - Wasserstein ambiguity sets for modern DRO

================================================================================
WHAT THIS DOES
================================================================================

From combined signals to backtest results:

  1. Loads combined signals (now from Elastic Net, implemented also GBM combiner)
  2. Estimates correlation matrix C (RMT-cleaned + shrunk)
  3. Louvain clustering on C
  4. Loads ADV (average dollar volume) and volatility for cost model
  5. Runs walk-forward portfolio construction + backtest
  6. Computes performance metrics and attribution
  7. Saves results to data/exports/

This is the production entry point -- run this after:
  - 01_download_data.py
  - 02_etl_prices.py
  - 03_build_features.py
  - TO MODIFIY: elastic_net_combiner.py (or gbm_combiner.py) to generate combined_score 

================================================================================
USAGE
================================================================================

    python scripts/pipeline/04_build_portfolio.py

Optional arguments (via environment variables or edit below):
    BACKTEST_START_DATE    = '2010-01-01'
    BACKTEST_END_DATE      = '2024-12-31'
    REBALANCE_FREQ         = 'weekly'  # 'daily', 'weekly', 'monthly'
    TARGET_VOL             = 0.10       # 10% annualized
    MAX_GROSS              = 1.0        # 100% long + 100% short = 200% gross
    MAX_CLUSTER_MULTIPLE   = 2.0
    COV_METHOD             = 'ledoit_wolf'
    SIGNAL_METHOD          = 'proportional'
    NOTIONAL               = 1_000_000  # $1M portfolio

================================================================================
OUTPUTS
================================================================================

  data/exports/
    ├── weights.parquet          # Daily weights [date, ticker, weight, cluster_id]
    ├── returns.parquet          # Daily returns [date, gross_return, net_return, cost]
    ├── diagnostics.parquet      # Per-rebalance diagnostics
    ├── metrics.json             # Summary performance metrics
    └── attribution.parquet      # Performance attribution (if factors available)

================================================================================
'''

import os
import json
from pathlib import Path
from datetime import date, datetime
import polars as pl
import numpy as np

from src.portfolio.backtest import (
    prepare_backtest_data,
    run_backtest,
    compute_metrics,
    performance_attribution,
    regime_analysis,
)

# =============================================================================
# CONFIGURATION (edit these or set via environment variables)
# =============================================================================

# Backtest date range
START_DATE = os.environ.get('BACKTEST_START_DATE', '2010-01-01')
END_DATE = os.environ.get('BACKTEST_END_DATE', '2024-12-31')

# Rebalancing
REBALANCE_FREQ = os.environ.get('REBALANCE_FREQ', 'weekly')  # 'daily', 'weekly', 'monthly'

# Portfolio construction
TARGET_VOL = float(os.environ.get('TARGET_VOL', '0.10'))
MAX_GROSS = float(os.environ.get('MAX_GROSS', '1.0'))
MAX_CLUSTER_MULTIPLE = float(os.environ.get('MAX_CLUSTER_MULTIPLE', '2.0'))

# Signal -> weight method
SIGNAL_METHOD = os.environ.get('SIGNAL_METHOD', 'proportional')  # 'proportional', 'inv_vol', 'rank'
CAP_METHOD = os.environ.get('CAP_METHOD', 'cap')                  # 'cap', 'rescale'
VOL_METHOD = os.environ.get('VOL_METHOD', 'scale')               # 'scale', 'risk_parity'

# Covariance estimation
COV_METHOD = os.environ.get('COV_METHOD', 'ledoit_wolf')         # 'ledoit_wolf', 'oas', 'factor', 'sample'
CLUSTER_RESOLUTION = float(os.environ.get('CLUSTER_RESOLUTION', '1.0'))
LOOKBACK_WINDOW = int(os.environ.get('LOOKBACK_WINDOW', '252'))
MIN_OBS = int(os.environ.get('MIN_OBS', '60'))

# Cost model
NOTIONAL = float(os.environ.get('NOTIONAL', '1_000_000'))
COST_MODEL = os.environ.get('COST_MODEL', 'sqrt')                # 'sqrt', 'linear', 'almgren_chriss_full'
COMMISSION_BPS = float(os.environ.get('COMMISSION_BPS', '1.0'))

# Signal column (from elastic_net_combiner or gbm_combiner)
SCORE_COL = os.environ.get('SCORE_COL', 'combined_score')

# Execution
EXECUTION_LAG = int(os.environ.get('EXECUTION_LAG', '1'))  # days from signal to trade

# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
EXPORTS_DIR = DATA_DIR / 'exports'

# Input files
FEATURES_FILE = PROCESSED_DIR / 'features.parquet'
SIGNALS_FILE = PROCESSED_DIR / 'signals.parquet'  # Output from combiner
PRICES_FILE = PROCESSED_DIR / 'prices.parquet'

# Output files
WEIGHTS_FILE = EXPORTS_DIR / 'weights.parquet'
RETURNS_FILE = EXPORTS_DIR / 'returns.parquet'
DIAGNOSTICS_FILE = EXPORTS_DIR / 'diagnostics.parquet'
METRICS_FILE = EXPORTS_DIR / 'metrics.json'
ATTRIBUTION_FILE = EXPORTS_DIR / 'attribution.parquet'

EXPORTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print('=' * 70)
    print('PORTFOLIO CONSTRUCTION & BACKTEST PIPELINE')
    print('=' * 70)
    print(f'Start date:     {START_DATE}')
    print(f'End date:       {END_DATE}')
    print(f'Rebalance freq: {REBALANCE_FREQ}')
    print(f'Target vol:     {TARGET_VOL:.1%}')
    print(f'Max gross:      {MAX_GROSS:.1f}x')
    print(f'Cluster cap:    {MAX_CLUSTER_MULTIPLE:.1f}x equal share')
    print(f'Signal method:  {SIGNAL_METHOD}')
    print(f'Cov method:     {COV_METHOD}')
    print(f'Notional:       ${NOTIONAL:,.0f}')
    print(f'Cost model:     {COST_MODEL}')
    print(f'Commission:     {COMMISSION_BPS} bps')
    print(f'Execution lag:  {EXECUTION_LAG} day(s)')
    print('=' * 70)

    # -----------------------------------------------------------------
    # 1. Load signals (combined_score from elastic_net_combiner)
    # -----------------------------------------------------------------
    print('\n[1/6] Loading combined signals...')
    if not SIGNALS_FILE.exists():
        # Fallback: try to find signal file from combiner output
        possible_files = list(PROCESSED_DIR.glob('*signals*.parquet')) + \
                         list(PROCESSED_DIR.glob('*combined*.parquet')) + \
                         list(PROCESSED_DIR.glob('*en_combiner*.parquet'))
        if possible_files:
            SIGNALS_FILE = possible_files[0]
            print(f'    Using found signal file: {SIGNALS_FILE.name}')
        else:
            raise FileNotFoundError(
                f'Signals file not found at {SIGNALS_FILE}. '
                'Run elastic_net_combiner.py first to generate combined_score.'
            )

    signals_lf = pl.scan_parquet(SIGNALS_FILE)
    print(f'    Loaded: {SIGNALS_FILE.name}')

    # Check required columns
    required_cols = ['date', 'ticker', SCORE_COL]
    available_cols = signals_lf.collect_schema().names()
    missing = [c for c in required_cols if c not in available_cols]
    if missing:
        raise ValueError(f'Signals file missing required columns: {missing}')

    # -----------------------------------------------------------------
    # 2. Load price data (for returns, ADV, vol)
    # -----------------------------------------------------------------
    print('\n[2/6] Loading price data...')
    if not PRICES_FILE.exists():
        raise FileNotFoundError(f'Prices file not found at {PRICES_FILE}. Run 02_etl_prices.py first.')

    price_lf = pl.scan_parquet(PRICES_FILE)
    print(f'    Loaded: {PRICES_FILE.name}')

    # -----------------------------------------------------------------
    # 3. Prepare backtest data
    # -----------------------------------------------------------------
    print('\n[3/6] Preparing backtest data...')
    prepared = prepare_backtest_data(
        signals_lf=signals_lf,
        score_cols=[SCORE_COL],
        price_lf=price_lf,
        start_date=date.fromisoformat(START_DATE),
        end_date=date.fromisoformat(END_DATE),
        rebalance_freq=REBALANCE_FREQ,
        lookback_window=LOOKBACK_WINDOW,
        min_obs=MIN_OBS,
        score_col=SCORE_COL,
        adv_col='adv60',  # from features.py
        vol_col='std60',  # from features.py
    )
    print(f'    Rebalance dates: {len(prepared["rebalance_dates"])}')
    print(f'    Date range: {prepared["rebalance_dates"][0]} to {prepared["rebalance_dates"][-1]}')

    # -----------------------------------------------------------------
    # 4. Run backtest
    # -----------------------------------------------------------------
    print('\n[4/6] Running walk-forward backtest...')
    result = run_backtest(
        prepared_data=prepared,
        target_vol=TARGET_VOL,
        max_gross=MAX_GROSS,
        max_cluster_multiple=MAX_CLUSTER_MULTIPLE,
        signal_method=SIGNAL_METHOD,
        cap_method=CAP_METHOD,
        vol_method=VOL_METHOD,
        cov_method=COV_METHOD,
        cluster_resolution=CLUSTER_RESOLUTION,
        lookback_window=LOOKBACK_WINDOW,
        min_obs=MIN_OBS,
        notional=NOTIONAL,
        cost_model=COST_MODEL,
        commission_bps=COMMISSION_BPS,
        score_col=SCORE_COL,
        execution_lag=EXECUTION_LAG,
    )

    weights_history = result['weights_history']
    returns_history = result['returns_history']
    diagnostics_history = result['diagnostics_history']
    summary = result['summary_metrics']

    print(f'    Weight records: {len(weights_history)}')
    print(f'    Return records: {len(returns_history)}')
    print(f'    Rebalances:     {len(diagnostics_history)}')

    # -----------------------------------------------------------------
    # 5. Save outputs
    # -----------------------------------------------------------------
    print('\n[5/6] Saving results...')

    # Weights
    if len(weights_history) > 0:
        weights_history.write_parquet(WEIGHTS_FILE)
        print(f'    Weights:     {WEIGHTS_FILE} ({len(weights_history):,} rows)')

    # Returns
    if len(returns_history) > 0:
        returns_history.write_parquet(RETURNS_FILE)
        print(f'    Returns:     {RETURNS_FILE} ({len(returns_history):,} rows)')

    # Diagnostics
    if diagnostics_history:
        diag_df = pl.DataFrame(diagnostics_history)
        diag_df.write_parquet(DIAGNOSTICS_FILE)
        print(f'    Diagnostics: {DIAGNOSTICS_FILE} ({len(diag_df)} rows)')

    # Metrics
    with open(METRICS_FILE, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'    Metrics:     {METRICS_FILE}')

    # -----------------------------------------------------------------
    # 6. Performance attribution (if factor data available)
    # -----------------------------------------------------------------
    print('\n[6/6] Computing attribution...')
    try:
        # Try to load factor loadings/returns if they exist
        factor_loadings_file = PROCESSED_DIR / 'factor_loadings.parquet'
        factor_returns_file = PROCESSED_DIR / 'factor_returns.parquet'

        if factor_loadings_file.exists() and factor_returns_file.exists():
            factor_loadings = pl.read_parquet(factor_loadings_file)
            factor_returns = pl.read_parquet(factor_returns_file)

            attribution = performance_attribution(
                weights_history=weights_history,
                returns_panel=prepared['returns_panel'],
                factor_loadings=factor_loadings,
                factor_returns=factor_returns,
            )
            attribution.write_parquet(ATTRIBUTION_FILE)
            print(f'    Attribution: {ATTRIBUTION_FILE}')
        else:
            # Basic attribution without factors
            print('    Factor data not found, skipping factor attribution.')
            # Could compute sector attribution if sector data available
    except Exception as e:
        print(f'    Attribution skipped: {e}')

    # -----------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------
    print('\n' + '=' * 70)
    print('BACKTEST SUMMARY')
    print('=' * 70)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f'  {k}: {v:.4f}')
        else:
            print(f'  {k}: {v}')
    print('=' * 70)

    # Also print key metrics nicely
    if summary:
        print(f'\nKey Metrics:')
        print(f'  Net Sharpe:     {summary.get("sharpe_net", 0):.3f}')
        print(f'  Gross Sharpe:   {summary.get("sharpe_gross", 0):.3f}')
        print(f'  Net Ann Return: {summary.get("annual_return_net", 0):.2%}')
        print(f'  Net Ann Vol:    {summary.get("annual_vol_net", 0):.2%}')
        print(f'  Max Drawdown:   {summary.get("max_drawdown", 0):.2%}')
        print(f'  Hit Rate:       {summary.get("hit_rate", 0):.2%}')
        print(f'  Cost Drag:      {summary.get("cost_drag_annual_bps", 0):.1f} bps/yr')

    print('\nDone!')


if __name__ == '__main__':
    main()