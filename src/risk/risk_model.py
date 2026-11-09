import warnings
# Suppress RuntimeWarnings from np.nanmean/nanstd on empty slices (common in worker processes)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Degrees of freedom <= 0 for slice')

import numpy as np
import polars as pl
from typing import Optional, Tuple, Dict, Any, List

from src.risk.correlation_cleaning import clean_correlation_matrix, marchenko_pastur_bounds
from src.risk.clustering import louvain_clustering
from src.risk.covariance import (
    build_covariance_matrix,
    ledoit_wolf_shrinkage,
    oracle_approximating_shrinkage,
    factor_model_covariance,
)


def compute_raw_correlation(returns_matrix: np.ndarray, min_obs: int = 30) -> np.ndarray:
    '''Compute sample correlation matrix from returns (assets x observations).
    Handles NaN values by using pairwise complete observations.
    Optimized version using vectorized operations.
    '''
    n_assets = returns_matrix.shape[0]
    
    # Standardize returns (subtract mean, divide by std) ignoring NaN
    # Suppress warnings for assets with insufficient data - must be inside to affect workers
    with np.errstate(invalid='ignore', divide='ignore'):
        means = np.nanmean(returns_matrix, axis=1, keepdims=True)
        stds = np.nanstd(returns_matrix, axis=1, keepdims=True, ddof=1)
    
    # Assets with < 2 valid obs get NaN mean/std (will be handled below)
    # Assets with 0 valid obs get NaN mean/std
    
    # Avoid division by zero - only for assets with valid std
    stds = np.where((stds == 0) | ~np.isfinite(stds), 1.0, stds)
    
    # Z-score normalized returns (NaN stays NaN)
    z_returns = np.full_like(returns_matrix, np.nan)
    valid_std = np.isfinite(stds) & (stds > 0)
    if np.any(valid_std):
        z_returns[valid_std[:, 0]] = (returns_matrix[valid_std[:, 0]] - means[valid_std[:, 0]]) / stds[valid_std[:, 0]]
    
    # Correlation = mean of pairwise z-products
    # corr[i,j] = mean(z[i] * z[j]) where both are finite
    corr = np.zeros((n_assets, n_assets))
    
    for i in range(n_assets):
        zi = z_returns[i]
        for j in range(i, n_assets):
            zj = z_returns[j]
            # Pairwise complete: both finite
            mask = np.isfinite(zi) & np.isfinite(zj)
            n_valid = mask.sum()
            if n_valid > min_obs:
                with np.errstate(invalid='ignore'):
                    corr[i, j] = corr[j, i] = np.nanmean(zi[mask] * zj[mask])
            else:
                corr[i, j] = corr[j, i] = np.nan  # Will be handled by RMT
    
    np.fill_diagonal(corr, 1.0)
    
    # Fill remaining NaN with 0 (uncorrelated prior)
    corr = np.nan_to_num(corr, nan=0.0)
    return corr


def clean_correlation_rmt(corr: np.ndarray, n_obs: int) -> Tuple[np.ndarray, np.ndarray, float]:
    '''
    RMT cleaning of correlation matrix.
    Returns: (corr_clean, eigenvalues, lambda_max)
    '''
    n_assets = corr.shape[0]
    _, lambda_max = marchenko_pastur_bounds(n_assets, n_obs, variance=1.0)
    corr_clean = clean_correlation_matrix(corr, n_obs)
    eigenvalues, _ = np.linalg.eigh(corr_clean)
    return corr_clean, eigenvalues, lambda_max


def estimate_risk_model_for_date(
    returns_matrix: np.ndarray,
    tickers: List[str],
    date_str: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    '''
    Full risk model estimation for a single rebalance date.
    Returns dict with all outputs for this date.
    '''
    n_tickers = len(tickers)
    n_obs = returns_matrix.shape[1]

    # 1. Raw correlation
    min_obs = config.get('min_obs', 30)
    corr = compute_raw_correlation(returns_matrix, min_obs=min_obs)


def estimate_volatilities(returns_matrix: np.ndarray) -> np.ndarray:
    '''Annualized volatilities from returns matrix (assets x observations).'''
    return np.std(returns_matrix, axis=1, ddof=1) * np.sqrt(252)


def build_covariance(
    corr_clean: np.ndarray,
    vols: np.ndarray,
    method: str,
    returns_matrix: Optional[np.ndarray] = None,
    n_factors: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[float]]:
    '''
    Build covariance matrix from cleaned correlation and volatilities.
    Returns: (cov_matrix, shrinkage_value_or_None)
    '''
    if method == 'ledoit_wolf':
        cov, shrinkage = ledoit_wolf_shrinkage(corr_clean, returns=returns_matrix, target='identity')
    elif method == 'oas':
        cov, shrinkage = oracle_approximating_shrinkage(corr_clean, returns=returns_matrix)
    elif method == 'factor':
        n_factors = n_factors or 5
        cov = factor_model_covariance(returns_matrix.T, n_factors)
        shrinkage = None
    else:  # 'sample'
        cov = build_covariance_matrix(corr_clean, vols)
        shrinkage = None
    return cov, shrinkage


def cluster_correlation(corr_clean: np.ndarray, resolution: float) -> Dict[int, int]:
    '''Louvain clustering on cleaned correlation matrix. Returns {ticker_idx: cluster_id}.'''
    return louvain_clustering(corr_clean, resolution=resolution)


def estimate_risk_model_for_date(
    returns_matrix: np.ndarray,
    tickers: List[str],
    date_str: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    '''
    Full risk model estimation for a single rebalance date.
    Returns dict with all outputs for this date.
    '''
    n_tickers = len(tickers)
    n_obs = returns_matrix.shape[1]

    # 1. Raw correlation
    corr = compute_raw_correlation(returns_matrix)

    # 2. RMT cleaning
    corr_clean, eigenvalues, lambda_max = clean_correlation_rmt(corr, n_obs)
    q = n_tickers / n_obs

    # 3. Volatilities
    vols = estimate_volatilities(returns_matrix)

    # 4. Covariance
    method = config.get('cov_method', 'ledoit_wolf')
    n_factors = config.get('n_factors')
    cov, shrinkage = build_covariance(corr_clean, vols, method, returns_matrix, n_factors)

    # 5. Clustering
    cluster_resolution = config.get('cluster_resolution', 1.0)
    cluster_map = cluster_correlation(corr_clean, cluster_resolution)

    # Cluster summary
    clusters = {}
    for t, c in cluster_map.items():
        clusters.setdefault(c, []).append(t)

    return {
        'date': date_str,
        'corr_raw': corr,
        'corr_clean': corr_clean,
        'cov': cov,
        'eigenvalues': eigenvalues,
        'lambda_max': float(lambda_max),
        'q_ratio': q,
        'cluster_map': cluster_map,
        'cluster_sizes': {str(k): len(v) for k, v in clusters.items()},
        'n_clusters': len(clusters),
        'shrinkage': float(shrinkage) if shrinkage is not None else None,
        'cov_method': method,
        'n_tickers': n_tickers,
        'n_obs': n_obs,
    }


def build_returns_panel(returns_wide: pl.DataFrame, rebalance_dates: list, lookback_window: int) -> np.ndarray:
    '''
    Build 3D returns panel (n_assets x lookback_window x n_rebalances) from wide returns DataFrame.
    
    Args:
        returns_wide: DataFrame with date index and ticker columns
        rebalance_dates: list of rebalance dates (datetime.date)
        lookback_window: number of days to look back
    
    Returns:
        3D array (n_assets, lookback_window, n_rebalances)
    '''
    valid_tickers = returns_wide.columns[1:]  # exclude date column
    n_assets = len(valid_tickers)
    
    date_col = returns_wide.columns[0]
    returns_np = returns_wide.select(valid_tickers).to_numpy().T  # (n_assets, n_dates)
    dates_np = returns_wide[date_col].to_numpy()  # numpy.datetime64[D]
    
    # Map dates to indices - convert to datetime.date for consistent lookup
    date_to_idx = {np.datetime64(d).astype('datetime64[D]'): i for i, d in enumerate(dates_np)}
    # Also store as datetime.date for rebalance_dates lookup
    rebalance_dates_np = np.array([np.datetime64(d) for d in rebalance_dates], dtype='datetime64[D]')
    
    returns_panel = np.zeros((n_assets, lookback_window, len(rebalance_dates)))
    
    for i, reb_date_np in enumerate(rebalance_dates_np):
        if reb_date_np in date_to_idx:
            idx = date_to_idx[reb_date_np]
            start_idx = max(0, idx - lookback_window)
            end_idx = idx
            window_data = returns_np[:, start_idx:end_idx]
            if window_data.shape[1] > 0:
                # Pad if needed
                if window_data.shape[1] < lookback_window:
                    pad_width = lookback_window - window_data.shape[1]
                    window_data = np.pad(window_data, ((0, 0), (pad_width, 0)), mode='constant', constant_values=np.nan)
                returns_panel[:, :window_data.shape[1], i] = window_data[:, -lookback_window:]
    
    return returns_panel