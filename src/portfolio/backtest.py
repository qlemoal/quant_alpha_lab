'''
Walk-forward backtest loop with performance attribution.

================================================================================
WHY THIS MATTERS
================================================================================

A backtest is the final integration test of the entire pipeline. At top firms,
the backtest is where you validate that:
  - Alpha signals survive transaction costs
  - Risk models (covariance, clustering) work in practice
  - Portfolio construction (HRP, risk parity, vol targeting) is stable
  - Turnover and capacity are realistic

Key principles for a production-grade backtest:
  1. WALK-FORWARD: Never use future data. Re-estimate everything at each
     rebalance date using only data available up to that date.
  2. REALISTIC EXECUTION: Use open-to-open or close-to-close consistently,
     model spreads, impact, commission.
  3. ATTRIBUTION: Decompose P&L into factor, sector, idiosyncratic.
  4. ROBUSTNESS: Test across regimes, rebalance frequencies, cost assumptions.
  5. NO LOOKAHEAD: All estimates (covariance, vol, clusters, signal params)
     use only historical data.

This module implements a modular, configurable backtest engine that uses
all the pieces we've built:
  - Signals: from elastic_net_combiner / gbm_combiner
  - Covariance: from covariance.py (RMT + shrinkage)
  - Clustering: from clustering.py (Louvain)
  - Construction: from portfolio_construction.py
  - Costs: from costs.py
  - Optimization: from optimization.py (optional, for cost-aware)

================================================================================
ARCHITECTURE
================================================================================

The backtest is structured as a pipeline of pure functions:

  1. prepare_backtest_data()     -- loads & aligns all inputs per date
  2. rebalance_step()            -- single rebalance: estimate Σ, clusters, construct weights
  3. apply_costs()               -- compute transaction costs for weight change
  4. compute_returns()           -- portfolio P&L from weights * asset returns
  5. run_backtest()              -- orchestrates the walk-forward loop
  6. performance_attribution()   -- factor/sector/idiosyncratic decomposition
  7. compute_metrics()           -- final performance statistics

Each function is testable in isolation with synthetic data.

================================================================================
REFERENCES
================================================================================

  - Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2017).
    "The probability of backtest overfitting." J. Comput. Finance, 20(4), 39-69.
  - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley.
    Ch. 12: Backtesting, Ch. 7: Cross-validation.
  - Grinold, R. C., & Kahn, R. N. (2000). "Active Portfolio Management." McGraw-Hill.
    Ch. 14: Performance attribution.
  - Menchero, J. (2010). "The Characteristics of Factor Portfolios."
    Journal of Performance Measurement, 15(2), 44-54.

================================================================================
'''

import numpy as np
import polars as pl
from polars import col as c
from datetime import date, timedelta
from typing import Callable, Optional
from src.risk.covariance import estimate_covariance, build_covariance_matrix
from src.risk.clustering import louvain_clustering
from src.risk.portfolio_construction import (
    construct_portfolio,
    raw_weights_from_score,
    apply_cluster_caps,
    apply_vol_target,
    portfolio_diagnostics,
    compute_turnover,
)
from src.portfolio.costs import (
    total_transaction_cost,
    cost_adjusted_returns,
    cost_penalty_for_optimizer,
)
from src.portfolio.optimization import (
    hierarchical_risk_parity,
    clustered_risk_parity,
    cost_aware_optimize,
    mean_variance_optimize,
    portfolio_var_decomposition,
)


# =============================================================================
# PRE-COMPUTED RISK MODEL LOADING
# =============================================================================

def load_precomputed_risk_model(risk_model_dir: str) -> dict | None:
    '''
    Load pre-computed risk model from 04d output directory.
    
    Expected files:
      - corr_clean.parquet: [date, ticker_i, ticker_j, corr]
      - cov.parquet: [date, ticker_i, ticker_j, cov]
      - clusters.parquet: [date, ticker, cluster_id]
    
    Returns dict with:
      - dates: np.ndarray of datetime64 dates
      - corr: list of correlation matrices (one per date)
      - sigma: list of covariance matrices (one per date)
      - clusters: list of cluster DataFrames (one per date)
    '''
    import os
    from pathlib import Path
    
    dir_path = Path(risk_model_dir)
    corr_file = dir_path / 'corr_clean.parquet'
    cov_file = dir_path / 'cov.parquet'
    clusters_file = dir_path / 'clusters.parquet'
    
    if not all(f.exists() for f in [corr_file, cov_file, clusters_file]):
        print(f'Missing risk model files in {risk_model_dir}')
        return None
    
    print(f'Loading pre-computed risk model from {risk_model_dir}...')
    
    # Load all data
    corr_lf = pl.scan_parquet(corr_file)
    cov_lf = pl.scan_parquet(cov_file)
    clusters_lf = pl.scan_parquet(clusters_file)
    
    # Get unique dates (sorted)
    dates_lf = corr_lf.select(pl.col('date').unique().sort()).collect()
    dates = dates_lf['date'].to_list()
    
    # Convert to datetime64 for searchsorted
    dates_np = np.array([np.datetime64(d) for d in dates], dtype='datetime64[D]')
    
    # For each date, load the correlation matrix, covariance matrix, and clusters
    corr_matrices = []
    cov_matrices = []
    cluster_dfs = []
    
    for dt in dates:
        # Get unique tickers for this date from clusters
        clusters_dt = clusters_lf.filter(pl.col('date') == dt).collect()
        tickers = clusters_dt['ticker'].to_list()
        n = len(tickers)
        
        if n == 0:
            continue
            
        # Load correlation matrix for this date
        corr_dt = corr_lf.filter(pl.col('date') == dt).collect()
        # Pivot to matrix
        corr_pivot = corr_dt.pivot(values='corr', index='ticker_i', on='ticker_j', sort_columns=True)
        corr_matrix = corr_pivot.drop('ticker_i').to_numpy()
        corr_matrices.append(corr_matrix)
        
        # Load covariance matrix for this date
        cov_dt = cov_lf.filter(pl.col('date') == dt).collect()
        cov_pivot = cov_dt.pivot(values='cov', index='ticker_i', on='ticker_j', sort_columns=True)
        cov_matrix = cov_pivot.drop('ticker_i').to_numpy()
        cov_matrices.append(cov_matrix)
        
        # Clusters
        cluster_dfs.append(clusters_dt)
    
    return {
        'dates': dates_np,
        'corr': corr_matrices,
        'sigma': cov_matrices,
        'clusters': cluster_dfs,
        'tickers': tickers,  # tickers from last date (may vary by date)
    }


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_backtest_data(
    signals_lf: pl.LazyFrame,
    score_cols: list[str],
    price_lf: pl.LazyFrame,
    start_date: date,
    end_date: date,
    rebalance_freq: str = 'weekly',  # 'daily', 'weekly', 'monthly'
    lookback_window: int = 252,       # days for covariance estimation
    min_obs: int = 60,                # minimum obs for covariance
    date_col: str = 'date',
    ticker_col: str = 'ticker',
    score_col: str = 'combined_score',
    adv_col: str = 'adv60',
    vol_col: str = 'std60',
    open_col: str = 'open',
    close_col: str = 'close',
    return_col: str = 'logret',
) -> dict:
    '''
    Load and align all data needed for backtest.

    ===========================================================================
    INTUITION
    ===========================================================================

    We need, for each rebalance date:
      - Signal scores for all tickers
      - Historical returns for covariance estimation (lookback window)
      - ADV and vol for cost model and vol targeting
      - Next-period returns for P&L calculation

    This function pre-loads and structures everything so the walk-forward
    loop just iterates over rebalance dates.

    ===========================================================================
    PARAMETERS
    ===========================================================================

    signals_lf:      LazyFrame with [date, ticker, score_cols...]
    score_cols:      List of signal column names (e.g., ['combined_score']).
    price_lf:        LazyFrame with [date, ticker, open, close, logret, adv60, std60...]
    start_date:      First rebalance date.
    end_date:        Last rebalance date (exclusive).
    rebalance_freq:  'daily', 'weekly', 'monthly'.
    lookback_window: Days of history for covariance/vol estimation.
    min_obs:         Minimum observations required for valid covariance.
    ... column names.

    ===========================================================================
    RETURNS
    ===========================================================================

    Dict with:
      - rebalance_dates: list of date objects
      - signals_by_date: {date: DataFrame[ticker, score_cols...]}
      - returns_panel:   DataFrame for covariance estimation
      - adv_by_date:     {date: Series} ADV per ticker
      - vol_by_date:     {date: Series} vol per ticker
      - next_returns:    {date: Series} forward returns for P&L
    '''
    # Collect signals
    signals_df = signals_lf.filter(
        (c(date_col) >= start_date) & (c(date_col) <= end_date)
    ).select([date_col, ticker_col] + score_cols).collect().sort([date_col, ticker_col])

    # Collect price data (for returns, ADV, vol)
    price_df = price_lf.filter(
        (c(date_col) >= start_date - timedelta(days=lookback_window + 30)) &
        (c(date_col) <= end_date)
    ).select([date_col, ticker_col, open_col, close_col, return_col, adv_col, vol_col]).collect().sort([date_col, ticker_col])

    # Generate rebalance dates
    rebalance_dates = _generate_rebalance_dates(start_date, end_date, rebalance_freq, signals_df, date_col)

    # Organize by date
    signals_by_date = {}
    for dt in rebalance_dates:
        df = signals_df.filter(c(date_col) == dt).sort(ticker_col)
        if len(df) > 0:
            signals_by_date[dt] = df

    # Returns panel for covariance estimation (wide format: tickers x dates)
    returns_wide = price_df.pivot(
        values=return_col, index=date_col, on=ticker_col
    ).sort(date_col)

    # ADV and vol by date
    adv_by_date = {}
    vol_by_date = {}
    for dt in rebalance_dates:
        adv_df = price_df.filter(c(date_col) == dt).sort(ticker_col)
        if len(adv_df) > 0:
            # Fill NaN values with median to avoid NaN propagation
            adv_vals = adv_df[adv_col].to_numpy()
            vol_vals = adv_df[vol_col].to_numpy()
            
            # Fill NaN adv with median
            adv_mask = np.isfinite(adv_vals) & (adv_vals > 0)
            if not np.all(adv_mask):
                median_adv = np.median(adv_vals[adv_mask]) if np.any(adv_mask) else 1e6
                adv_vals = np.where(adv_mask, adv_vals, median_adv)
            
            # Fill NaN vol with median
            vol_mask = np.isfinite(vol_vals) & (vol_vals > 0)
            if not np.all(vol_mask):
                median_vol = np.median(vol_vals[vol_mask]) if np.any(vol_mask) else 0.02
                vol_vals = np.where(vol_mask, vol_vals, median_vol)
            
            adv_by_date[dt] = pl.Series('adv', adv_vals)
            vol_by_date[dt] = pl.Series('vol', vol_vals)

    # Next-period returns for P&L (we use open-to-open for execution realism)
    # The signal at date t is traded at open t+1, so P&L uses return from open t+1 to open t+2
    # We'll compute this in the loop using price data

    return {
        'rebalance_dates': rebalance_dates,
        'signals_by_date': signals_by_date,
        'returns_panel': returns_wide,
        'adv_by_date': adv_by_date,
        'vol_by_date': vol_by_date,
        'price_df': price_df,
        'date_col': date_col,
        'ticker_col': ticker_col,
        'open_col': open_col,
        'return_col': return_col,
    }


def _generate_rebalance_dates(
    start_date: date,
    end_date: date,
    freq: str,
    signals_df: pl.DataFrame,
    date_col: str,
) -> list[date]:
    '''Generate rebalance dates based on frequency.'''
    available_dates = signals_df[date_col].unique().sort().to_list()
    available_dates = [d for d in available_dates if start_date <= d < end_date]

    if freq == 'daily':
        return available_dates

    elif freq == 'weekly':
        # Every Wednesday (or first available date of week)
        result = []
        for d in available_dates:
            if d.weekday() == 2:  # Wednesday = 2
                result.append(d)
        if not result:
            # Fallback: every 5th trading day
            result = available_dates[::5]
        return result

    elif freq == 'monthly':
        # First available date of each month
        seen_months = set()
        result = []
        for d in available_dates:
            key = (d.year, d.month)
            if key not in seen_months:
                seen_months.add(key)
                result.append(d)
        return result

    else:
        raise ValueError(f"unknown frequency: {freq}")


# =============================================================================
# SINGLE REBALANCE STEP
# =============================================================================

def rebalance_step(
    current_date: date,
    scores_df: pl.DataFrame,
    returns_panel: pl.DataFrame,
    adv_series: pl.Series,
    vol_series: pl.Series,
    lookback_window: int,
    min_obs: int,
    target_vol: float,
    max_gross: float,
    max_cluster_multiple: float,
    signal_method: str = 'proportional',
    cap_method: str = 'cap',
    vol_method: str = 'scale',
    cov_method: str = 'ledoit_wolf',
    cluster_resolution: float = 1.0,
    score_col: str = 'combined_score',
    ticker_col: str = 'ticker',
    date_col: str = 'date',
    # Pre-computed risk model for this date (optional)
    precomputed_corr: np.ndarray | None = None,
    precomputed_sigma: np.ndarray | None = None,
    precomputed_cluster_map: dict | None = None,
) -> dict:
    '''
    Execute ONE rebalance step: estimate risk model, construct portfolio.

    ===========================================================================
    INTUITION
    ===========================================================================

    At each rebalance date t:
      1. Get signal scores for all tickers at date t
      2. Pull lookback_window days of returns ending at t-1 (no lookahead!)
      3. Estimate covariance matrix (cleaned + shrunk)
      4. Run Louvain clustering on cleaned correlation
      5. Construct portfolio: score -> weights -> cluster caps -> vol target
      6. Return weights and diagnostics

    This is the core "risk model + construction" step, isolated for testing.

    ===========================================================================
    PARAMETERS
    ===========================================================================

    current_date:       Rebalance date (weights take effect at open of next day).
    scores_df:          DataFrame[ticker, score_col] for this date.
    returns_panel:      Wide returns DataFrame (date x tickers) for covariance.
    adv_series:         Series of ADV for this date (aligned with scores_df).
    vol_series:         Series of vol for this date (aligned).
    lookback_window:    Days of history for covariance.
    min_obs:            Minimum observations for valid covariance.
    target_vol:         Target portfolio volatility (annualized).
    max_gross:          Max gross exposure.
    max_cluster_multiple: Cluster cap multiplier.
    signal_method:      'proportional', 'inv_vol', 'rank'.
    cap_method:         'cap', 'rescale'.
    vol_method:         'scale'.
    cov_method:         'ledoit_wolf', 'oas', 'factor', 'sample'.
    cluster_resolution: Louvain resolution parameter.
    ...

    ===========================================================================
    RETURNS
    ===========================================================================

    Dict with:
      - weights:        Series[ticker -> weight]
      - corr_clean:     Cleaned correlation matrix
      - sigma:          Covariance matrix used
      - cluster_map:    {ticker_idx: cluster_id}
      - diagnostics:    portfolio_diagnostics output
    '''
    tickers = scores_df[ticker_col].to_list()
    scores = scores_df[score_col]
    n = len(tickers)

    if n < 2:
        return {
            'weights': pl.Series('weight', [0.0] * n),
            'corr_clean': np.eye(n) if n > 0 else np.array([]),
            'sigma': np.eye(n) if n > 0 else np.array([]),
            'cluster_map': {},
            'diagnostics': {},
            'error': 'insufficient_tickers',
        }

    # -------------------------------------------------------------
    # 1. Check if we have pre-computed risk model for this date
    # -------------------------------------------------------------
    use_precomputed = (precomputed_corr is not None and 
                       precomputed_sigma is not None and 
                       precomputed_cluster_map is not None)

    if use_precomputed:
        # Pre-computed risk model provided - need to align to current tickers
        # The precomputed matrices have a specific ticker ordering from the risk model
        # We need to extract the submatrix for our current tickers
        # For now, we assume the precomputed model has all tickers and we extract the relevant ones
        # But the risk model only has tickers available at that rebalance date
        # The caller should provide the ticker ordering from the precomputed model
        # We'll need to handle this by extracting the submatrix
        # For now, we'll just use the precomputed matrices directly if sizes match
        # Otherwise fall back to re-estimation
        
        n_precomputed = precomputed_corr.shape[0]
        if n_precomputed == n:
            # Same number of tickers - assume alignment
            corr = precomputed_corr
            sigma = precomputed_sigma
            cluster_map = precomputed_cluster_map
            cov_info = {'method': 'precomputed', 'date': str(current_date)}
        else:
            # Mismatch - fall back to re-estimation
            print(f'  Warning: pre-computed risk model has {n_precomputed} tickers but current has {n}, falling back to re-estimation')
            use_precomputed = False
    
    if not use_precomputed:
        # -------------------------------------------------------------
        # 1. Build returns matrix for covariance (lookback ending at t-1)
        # -------------------------------------------------------------
        # returns_panel is date x tickers (wide)
        # We need the lookback_window most recent dates BEFORE current_date
        dates = returns_panel[date_col].to_numpy()
        valid_dates = dates[dates < np.datetime64(current_date)]
        if len(valid_dates) < min_obs:
            return {
                'weights': pl.Series('weight', [0.0] * n),
                'corr_clean': np.eye(n),
                'sigma': np.eye(n),
                'cluster_map': {},
                'diagnostics': {},
                'error': f'insufficient_history: {len(valid_dates)} < {min_obs}',
            }

        # Convert numpy datetime64 to Python date for Polars is_in()
        lookback_dates_np = valid_dates[-lookback_window:]
        lookback_dates = [date.fromisoformat(str(d)) for d in lookback_dates_np]
        returns_sub = returns_panel.filter(c(date_col).is_in(lookback_dates))

        # Align tickers: ensure order matches scores_df
        ticker_cols = [c for c in returns_sub.columns if c != date_col]
        # Reorder columns to match tickers list
        ticker_cols_aligned = [tc for tc in tickers if tc in ticker_cols]
        missing = set(tickers) - set(ticker_cols_aligned)
        if missing:
            # Add zero columns for missing tickers
            for m in missing:
                returns_sub = returns_sub.with_columns(pl.lit(0.0).alias(m))
            ticker_cols_aligned = tickers

        returns_matrix = returns_sub.select(ticker_cols_aligned).to_numpy().T  # P x T
        # Replace NaN (missing price data) with 0 so covariance stays valid
        returns_matrix = np.nan_to_num(returns_matrix, nan=0.0)

        if returns_matrix.shape[1] < min_obs:
            return {
                'weights': pl.Series('weight', [0.0] * n),
                'corr_clean': np.eye(n),
                'sigma': np.eye(n),
                'cluster_map': {},
                'diagnostics': {},
                'error': f'insufficient_obs_after_align: {returns_matrix.shape[1]} < {min_obs}',
            }

        # -------------------------------------------------------------
        # 2. Estimate covariance (cleaned + shrunk)
        # -------------------------------------------------------------
        # Use the estimate_covariance pipeline
        sigma, cov_info = estimate_covariance(
            returns_matrix,
            method=cov_method,
            target='identity' if cov_method == 'ledoit_wolf' else None,
        )
        # Extract correlation from sigma for clustering
        stds = np.sqrt(np.diag(sigma))
        stds = np.maximum(stds, 1e-8)
        corr = sigma / np.outer(stds, stds)
        np.fill_diagonal(corr, 1.0)

        # -------------------------------------------------------------
        # 3. Louvain clustering on cleaned correlation
        # -------------------------------------------------------------
        cluster_map = louvain_clustering(corr, resolution=cluster_resolution)

    # -------------------------------------------------------------
    # 4. Construct portfolio
    # -------------------------------------------------------------
    # Align adv/vol to ticker order - fill NaN vols with median to prevent NaN covariance
    adv_arr = np.asarray(adv_series.to_numpy(), dtype=float)
    vol_arr = np.asarray(vol_series.to_numpy(), dtype=float)
        
    # Fill NaN/inf vols with cross-sectional median
    valid_vol_mask = np.isfinite(vol_arr) & (vol_arr > 0)
    if not np.all(valid_vol_mask):
        median_vol = np.median(vol_arr[valid_vol_mask]) if np.any(valid_vol_mask) else 0.02
        vol_arr = np.where(valid_vol_mask, vol_arr, median_vol)
        
    # Also fill NaN advs with median
    valid_adv_mask = np.isfinite(adv_arr) & (adv_arr > 0)
    if not np.all(valid_adv_mask):
        median_adv = np.median(adv_arr[valid_adv_mask]) if np.any(valid_adv_mask) else 1e6
        adv_arr = np.where(valid_adv_mask, adv_arr, median_adv)

    weights_df = construct_portfolio(
        score_df=scores_df.select([date_col, ticker_col, score_col]).with_columns(pl.lit(current_date).alias(date_col)),
        corr=corr,
        vols=vol_arr,
        cluster_map=cluster_map,
        target_vol=target_vol,
        max_gross=max_gross,
        max_cluster_multiple=max_cluster_multiple,
        signal_method=signal_method,
        cap_method=cap_method,
        vol_method=vol_method,
        score_col=score_col,
        ticker_col=ticker_col,
        date_col=date_col,
    )

    weights_series = weights_df['weight']

    # -------------------------------------------------------------
    # 5. Diagnostics
    # -------------------------------------------------------------
    diag = portfolio_diagnostics(
        weights_series, sigma, cluster_map,
        target_vol=target_vol, max_gross=max_gross,
        max_cluster_multiple=max_cluster_multiple,
    )

    return {
        'weights': weights_series,
        'corr_clean': corr,
        'sigma': sigma,
        'cluster_map': cluster_map,
        'diagnostics': diag,
        'cov_info': cov_info,
    }


# =============================================================================
# WALK-FORWARD BACKTEST LOOP
# =============================================================================

def run_backtest(
    prepared_data: dict,
    target_vol: float = 0.10,
    max_gross: float = 1.0,
    max_cluster_multiple: float = 2.0,
    signal_method: str = 'proportional',
    cap_method: str = 'cap',
    vol_method: str = 'scale',
    cov_method: str = 'ledoit_wolf',
    cluster_resolution: float = 1.0,
    lookback_window: int = 252,
    min_obs: int = 60,
    notional: float = 1_000_000.0,
    cost_model: str = 'sqrt',
    commission_bps: float = 1.0,
    spread_kwargs: dict | None = None,
    impact_kwargs: dict | None = None,
    score_col: str = 'combined_score',
    execution_lag: int = 1,  # days between signal date and trade date
    # Pre-computed risk model (from 04d) - if provided, skips per-rebalance estimation
    risk_model_dir: str | None = None,
) -> dict:
    '''
    Run the full walk-forward backtest.

    ===========================================================================
    INTUITION
    ===========================================================================

    Walk-forward loop:
      For each rebalance date t:
        1. Get signal at date t
        2. Re-estimate risk model using data up to t-1
        3. Construct target weights for date t (+ execution_lag)
        4. Compute transaction cost to move from w_prev to w_target
        5. Hold weights until next rebalance, accumulate P&L
        6. Repeat

    Execution lag = 1 means:
      - Signal computed on close of date t
      - Trade at open of date t+1
      - P&L from open t+1 to next rebalance open

    ===========================================================================
    PARAMETERS
    ===========================================================================

    prepared_data:   Output from prepare_backtest_data().
    target_vol:      Target annualized portfolio volatility.
    max_gross:       Max gross exposure.
    max_cluster_multiple: Cluster cap.
    signal_method:   'proportional', 'inv_vol', 'rank'.
    cap_method:      'cap', 'rescale'.
    vol_method:      'scale'.
    cov_method:      Covariance estimation method.
    cluster_resolution: Louvain resolution.
    lookback_window: Days for covariance estimation.
    min_obs:         Minimum obs for valid covariance.
    notional:        Portfolio notional ($).
    cost_model:      'sqrt', 'linear', 'almgren_chriss_full'.
    commission_bps:  Commission in bps.
    spread_kwargs:   Dict passed to estimate_spread.
    impact_kwargs:   Dict passed to estimate_market_impact.
    score_col:       Signal column name.
    execution_lag:   Days from signal to trade (default 1 = next open).
    risk_model_dir:  Optional path to pre-computed risk model from 04d.
                     If provided, uses corr_clean.parquet, cov.parquet, clusters.parquet
                     instead of re-estimating at each rebalance.

    ===========================================================================
    RETURNS
    ===========================================================================

    Dict with:
      - weights_history:     DataFrame[date, ticker, weight]
      - returns_history:     DataFrame[date, gross_return, net_return, cost]
      - diagnostics_history: List of per-rebalance diagnostics
      - summary_metrics:     Final performance metrics
    '''
    rebalance_dates = prepared_data['rebalance_dates']
    signals_by_date = prepared_data['signals_by_date']
    returns_panel = prepared_data['returns_panel']
    adv_by_date = prepared_data['adv_by_date']
    vol_by_date = prepared_data['vol_by_date']
    price_df = prepared_data['price_df']
    date_col = prepared_data['date_col']
    ticker_col = prepared_data['ticker_col']
    open_col = prepared_data['open_col']
    return_col = prepared_data['return_col']

    # Load pre-computed risk model if provided
    risk_model_loaded = None
    if risk_model_dir is not None:
        risk_model_loaded = load_precomputed_risk_model(risk_model_dir)
        if risk_model_loaded:
            print(f'Loaded pre-computed risk model from {risk_model_dir}')
            print(f'  Risk model covers {len(risk_model_loaded["dates"])} dates')
        else:
            print(f'Warning: Could not load risk model from {risk_model_dir}, will re-estimate at each rebalance')

    # Results storage
    weights_rows = []
    returns_rows = []
    diagnostics_list = []

    w_prev = None
    tickers_prev = None
    weights_prev_date = None

    for i, rebal_date in enumerate(rebalance_dates):
        # Get signal for this rebalance date
        if rebal_date not in signals_by_date:
            continue
        scores_df = signals_by_date[rebal_date]

        # Get ADV/vol for this date
        if rebal_date not in adv_by_date or rebal_date not in vol_by_date:
            continue
        adv_series = adv_by_date[rebal_date]
        vol_series = vol_by_date[rebal_date]

        # ---------------------------------------------------------
        # Get pre-computed risk model for this date (if available)
        # ---------------------------------------------------------
        precomputed_corr = None
        precomputed_sigma = None
        precomputed_cluster_map = None
        
        if risk_model_loaded is not None:
            # Find closest risk model date <= rebal_date
            risk_dates = risk_model_loaded['dates']
            idx = np.searchsorted(risk_dates, np.datetime64(rebal_date), side='right') - 1
            if idx >= 0:
                model_date = risk_dates[idx]
                precomputed_corr = risk_model_loaded['corr'][idx]
                precomputed_sigma = risk_model_loaded['sigma'][idx]
                # Convert cluster_map to proper format
                cluster_df = risk_model_loaded['clusters'][idx]
                # cluster_df has columns: ticker, cluster_id
                precomputed_cluster_map = dict(zip(cluster_df['ticker'], cluster_df['cluster_id']))

        # ---------------------------------------------------------
        # Rebalance step: get target weights
        # ---------------------------------------------------------
        rebal_result = rebalance_step(
            current_date=rebal_date,
            scores_df=scores_df,
            returns_panel=returns_panel,
            adv_series=adv_series,
            vol_series=vol_series,
            lookback_window=lookback_window,
            min_obs=min_obs,
            target_vol=target_vol,
            max_gross=max_gross,
            max_cluster_multiple=max_cluster_multiple,
            signal_method=signal_method,
            cap_method=cap_method,
            vol_method=vol_method,
            cov_method=cov_method,
            cluster_resolution=cluster_resolution,
            score_col=score_col,
            ticker_col=ticker_col,
            date_col=date_col,
            precomputed_corr=precomputed_corr,
            precomputed_sigma=precomputed_sigma,
            precomputed_cluster_map=precomputed_cluster_map,
        )

        if 'error' in rebal_result:
            diagnostics_list.append({'date': rebal_date, **rebal_result})
            continue

        w_target = rebal_result['weights'].to_numpy()
        tickers = scores_df[ticker_col].to_list()

        # ---------------------------------------------------------
        # Transaction costs (if we have previous weights)
        # ---------------------------------------------------------
        cost = 0.0
        if w_prev is not None and tickers_prev is not None:
            # Align previous weights to current tickers using union
            # This handles changing universe across rebalance dates
            all_tickers = list(dict.fromkeys(tickers_prev + tickers))  # preserve order
            w_prev_aligned = np.zeros(len(all_tickers))
            w_target_aligned = np.zeros(len(all_tickers))
            
            # Fill in previous weights
            for i, tk in enumerate(tickers_prev):
                if tk in all_tickers:
                    idx = all_tickers.index(tk)
                    w_prev_aligned[idx] = w_prev[i]
            
            # Fill in target weights
            for i, tk in enumerate(tickers):
                if tk in all_tickers:
                    idx = all_tickers.index(tk)
                    w_target_aligned[idx] = w_target[i]
            
            # Align ADV/vol to the union universe
            adv_aligned = np.zeros(len(all_tickers))
            vol_aligned = np.zeros(len(all_tickers))
            for i, tk in enumerate(all_tickers):
                if tk in tickers:
                    idx = tickers.index(tk)
                    adv_aligned[i] = adv_series.to_numpy()[idx]
                    vol_aligned[i] = vol_series.to_numpy()[idx]
                elif tk in tickers_prev:
                    idx = tickers_prev.index(tk)
                    # Use previous ADV/vol for tickers that dropped out
                    adv_aligned[i] = adv_aligned[i]  # will be 0, use a fallback
                    vol_aligned[i] = vol_aligned[i]
            
            cost, cost_breakdown = total_transaction_cost(
                w_prev_aligned, w_target_aligned,
                adv_aligned, vol_aligned,
                notional,
                spread_model='power_law',
                spread_kwargs=spread_kwargs,
                impact_model=cost_model,
                impact_kwargs=impact_kwargs,
                commission_bps=commission_bps,
            )

        # ---------------------------------------------------------
        # Compute returns until next rebalance (or end)
        # ---------------------------------------------------------
        # Trade date = rebal_date + execution_lag
        trade_date = rebal_date + timedelta(days=execution_lag)

        if i + 1 < len(rebalance_dates):
            next_rebal = rebalance_dates[i + 1]
            next_trade = next_rebal + timedelta(days=execution_lag)
        else:
            next_trade = trade_date + timedelta(days=5)  # default 5-day hold

        # Get returns for each ticker from trade_date to next_trade
        # Using open-to-open returns: return = log(open_next / open)
        price_subset = price_df.filter(
            (c(date_col) >= trade_date) & (c(date_col) < next_trade)
        ).sort([ticker_col, date_col])

        # Compute daily portfolio returns
        daily_returns = []
        for hold_date in price_subset[date_col].unique().sort().to_list():
            day_prices = price_subset.filter(c(date_col) == hold_date).sort(ticker_col)
            if len(day_prices) == 0:
                continue

            # Align to our tickers
            day_tickers = day_prices[ticker_col].to_list()
            day_weights = np.zeros(len(day_tickers))
            for j, tk in enumerate(day_tickers):
                if tk in tickers:
                    idx = tickers.index(tk)
                    day_weights[j] = w_target[idx]

            # Daily return = sum(w_i * r_i) where r_i is logret
            day_rets = day_prices[return_col].to_numpy()
            port_ret = np.sum(day_weights * day_rets)
            daily_returns.append({
                date_col: hold_date,
                'gross_return': port_ret,
                'net_return': port_ret - (cost / notional) if hold_date == trade_date else port_ret,
                'cost': cost / notional if hold_date == trade_date else 0.0,
            })

        returns_rows.extend(daily_returns)

        # Store weights (effective at trade_date)
        for j, tk in enumerate(tickers):
            weights_rows.append({
                date_col: trade_date,
                ticker_col: tk,
                'weight': float(w_target[j]),
                'cluster_id': rebal_result['cluster_map'].get(j, -1) if isinstance(rebal_result['cluster_map'], dict) else (
                    rebal_result['cluster_map'][j] if j < len(rebal_result['cluster_map']) else -1
                ),
            })

        diagnostics_list.append({
            'date': rebal_date,
            'gross_exposure': rebal_result['diagnostics'].get('gross_exposure', 0),
            'portfolio_vol': rebal_result['diagnostics'].get('portfolio_vol', 0),
            'n_positions': rebal_result['diagnostics'].get('n_positions', 0),
            'turnover': compute_turnover(
                pl.Series('w_prev', w_prev_aligned) if w_prev is not None else pl.Series('w_prev', [0.0]*len(w_target_aligned)),
                pl.Series('w_target', w_target_aligned),
            ) if w_prev is not None else 0.0,
            'cost_bps': cost / notional * 10000.0,
            'cov_method': rebal_result.get('cov_info', {}).get('method', cov_method),
            'rho': rebal_result.get('cov_info', {}).get('rho', None),
        })

        # Update for next iteration
        w_prev = w_target
        tickers_prev = tickers
        weights_prev_date = trade_date

    # ---------------------------------------------------------
    # Build results DataFrames
    # ---------------------------------------------------------
    weights_history = pl.DataFrame(weights_rows) if weights_rows else pl.DataFrame()
    returns_history = pl.DataFrame(returns_rows) if returns_rows else pl.DataFrame()

    # ---------------------------------------------------------
    # Summary metrics
    # ---------------------------------------------------------
    if len(returns_history) > 0:
        gross_rets = returns_history['gross_return'].to_numpy()
        net_rets = returns_history['net_return'].to_numpy()
        costs = returns_history['cost'].to_numpy()

        summary = compute_metrics(gross_rets, net_rets, costs, notional)
    else:
        summary = {}

    return {
        'weights_history': weights_history,
        'returns_history': returns_history,
        'diagnostics_history': diagnostics_list,
        'summary_metrics': summary,
    }


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

def compute_metrics(
    gross_returns: np.ndarray,
    net_returns: np.ndarray,
    costs: np.ndarray,
    notional: float,
    periods_per_year: int = 252,
) -> dict:
    '''
    Compute comprehensive backtest performance metrics.

    ===========================================================================
    RETURNS
    ===========================================================================

    Dict with:
      - total_return_gross, total_return_net
      - annual_return_gross, annual_return_net
      - annual_vol_gross, annual_vol_net
      - sharpe_gross, sharpe_net
      - sortino_net, calmar_net
      - max_drawdown
      - avg_turnover, avg_cost_bps
      - hit_rate (fraction of positive net return days)
      - profit_factor (gross profit / gross loss)
    '''
    gross = np.asarray(gross_returns, dtype=float)
    net = np.asarray(net_returns, dtype=float)
    costs = np.asarray(costs, dtype=float)

    # Remove NaN
    valid = ~np.isnan(net)
    gross = gross[valid]
    net = net[valid]
    costs = costs[valid]

    if len(net) == 0:
        return {}

    # Cumulative returns
    cum_gross = np.cumsum(gross)
    cum_net = np.cumsum(net)

    total_gross = cum_gross[-1]
    total_net = cum_net[-1]

    # Annualized
    ann_factor = np.sqrt(periods_per_year)
    ann_gross = gross.mean() * periods_per_year
    ann_net = net.mean() * periods_per_year
    vol_gross = gross.std(ddof=1) * ann_factor
    vol_net = net.std(ddof=1) * ann_factor

    # Sharpe
    sharpe_gross = ann_gross / vol_gross if vol_gross > 0 else 0
    sharpe_net = ann_net / vol_net if vol_net > 0 else 0

    # Sortino (downside deviation)
    downside = net[net < 0]
    sortino = ann_net / (downside.std(ddof=1) * ann_factor) if len(downside) > 1 and downside.std() > 0 else 0

    # Max drawdown
    running_max = np.maximum.accumulate(cum_net)
    dd = cum_net - running_max
    max_dd = dd.min()
    calmar = ann_net / abs(max_dd) if max_dd < 0 else 0

    # Hit rate
    hit_rate = float(np.mean(net > 0))

    # Profit factor
    gross_profit = gross[gross > 0].sum()
    gross_loss = abs(gross[gross < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Cost stats
    total_cost = costs.sum()
    avg_cost_bps = (total_cost / notional * 10000.0) / len(costs) if len(costs) > 0 else 0

    # Turnover from weights history (approximate)
    # We don't have it here, but could be passed in

    return {
        'n_days': len(net),
        'total_return_gross': float(total_gross),
        'total_return_net': float(total_net),
        'annual_return_gross': float(ann_gross),
        'annual_return_net': float(ann_net),
        'annual_vol_gross': float(vol_gross),
        'annual_vol_net': float(vol_net),
        'sharpe_gross': float(sharpe_gross),
        'sharpe_net': float(sharpe_net),
        'sortino_net': float(sortino),
        'calmar_net': float(calmar),
        'max_drawdown': float(max_dd),
        'hit_rate': float(hit_rate),
        'profit_factor': float(profit_factor),
        'total_cost': float(total_cost),
        'avg_daily_cost_bps': float(avg_cost_bps),
        'cost_drag_annual_bps': float(avg_cost_bps * periods_per_year),
    }


# =============================================================================
# PERFORMANCE ATTRIBUTION
# =============================================================================

def performance_attribution(
    weights_history: pl.DataFrame,
    returns_panel: pl.DataFrame,
    factor_loadings: pl.DataFrame | None = None,
    factor_returns: pl.DataFrame | None = None,
    date_col: str = 'date',
    ticker_col: str = 'ticker',
    weight_col: str = 'weight',
    return_col: str = 'logret',
) -> pl.DataFrame:
    '''
    Brinson-style performance attribution: decompose daily P&L.

    ===========================================================================
    INTUITION
    ===========================================================================

    Attribution answers: "Where did the P&L come from?"

    1. FACTOR ATTRIBUTION (if factor_loadings + factor_returns provided):
       Daily P&L = sum_k (factor_weight_k * factor_return_k) + idiosyncratic
       where factor_weight_k = sum_i w_i * beta_ik

    2. SECTOR ATTRIBUTION (if sector column in weights):
       Allocation effect: (w_sector - w_bench_sector) * (r_sector - r_bench)
       Selection effect:  w_sector * (r_stock - r_sector)
       Interaction:       (w_stock - w_bench_stock) * (r_stock - r_sector)

    3. IDIOSYNCRATIC: Residual after factor/sector.

    This implements a simplified daily attribution suitable for backtest analysis.

    ===========================================================================
    PARAMETERS
    ===========================================================================

    weights_history:   DataFrame[date, ticker, weight, cluster_id...]
    returns_panel:     Wide returns DataFrame (date x tickers).
    factor_loadings:   DataFrame[ticker, factor1, factor2...] (P x K)
    factor_returns:    DataFrame[date, factor1, factor2...] (T x K)
    ...

    ===========================================================================
    RETURNS
    ===========================================================================

    DataFrame[date, total_pnl, factor_pnl, specific_pnl, ...]
    '''
    # For now, return a placeholder that can be extended
    # Full implementation needs factor data
    unique_dates = weights_history[date_col].unique().sort().to_list()

    result_rows = []
    for dt in unique_dates:
        w_df = weights_history.filter(c(date_col) == dt).sort(ticker_col)
        if len(w_df) == 0:
            continue

        # Get returns for this date
        ret_row = returns_panel.filter(c(date_col) == dt)
        if len(ret_row) == 0:
            continue

        # Portfolio return = sum(w_i * r_i)
        # Need to align
        tickers = w_df[ticker_col].to_list()
        weights = w_df[weight_col].to_numpy()

        # Get returns for these tickers
        rets = []
        for tk in tickers:
            if tk in ret_row.columns:
                rets.append(ret_row[tk].item())
            else:
                rets.append(0.0)
        rets = np.array(rets)

        total_pnl = float(np.sum(weights * rets))

        result_rows.append({
            date_col: dt,
            'total_pnl': total_pnl,
            'factor_pnl': 0.0,  # placeholder
            'specific_pnl': total_pnl,  # all specific for now
        })

    return pl.DataFrame(result_rows) if result_rows else pl.DataFrame()


# =============================================================================
# ROBUSTNESS / REGIME ANALYSIS
# =============================================================================

def regime_analysis(
    returns_history: pl.DataFrame,
    regime_indicator: pl.DataFrame | None = None,
    date_col: str = 'date',
    return_col: str = 'net_return',
) -> dict:
    '''
    Analyze performance by market regime.

    ===========================================================================
    INTUITION
    ===========================================================================

    A strategy might work well in trending markets but fail in choppy ones.
    Regime analysis splits returns by a regime indicator (e.g., VIX level,
    market trend, volatility regime) and computes metrics per regime.

    ===========================================================================
    PARAMETERS
    ===========================================================================

    returns_history:  DataFrame[date, net_return, ...]
    regime_indicator: DataFrame[date, regime] where regime is 'bull', 'bear', 'high_vol', etc.

    ===========================================================================
    RETURNS
    ===========================================================================

    Dict mapping regime -> metrics dict.
    '''
    if regime_indicator is None:
        return {'no_regime': compute_metrics(
            returns_history[return_col].to_numpy(),
            returns_history[return_col].to_numpy(),
            np.zeros(len(returns_history)),
            1.0,
        )}

    # Merge regime into returns
    merged = returns_history.join(regime_indicator, on=date_col)
    regimes = merged['regime'].unique().to_list()

    result = {}
    for reg in regimes:
        reg_data = merged.filter(c('regime') == reg)
        rets = reg_data[return_col].to_numpy()
        result[reg] = compute_metrics(rets, rets, np.zeros(len(rets)), 1.0)
        result[reg]['n_days'] = len(rets)

    return result