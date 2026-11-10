'''
Risk model for one rebalance date: correlation -> RMT cleaning -> shrinkage -> covariance -> clusters.

Units
-----
- returns_matrix: shape (n_assets, n_obs), DAILY log returns, NaN allowed.
- vols:           ANNUALISED volatilities (daily std * sqrt(252)).
- cov:            ANNUALISED covariance. Its diagonal is vols**2.
                  src/risk/portfolio_construction.py does not annualise again,
                  so target_vol (e.g. 0.10) is compared with sqrt(w' cov w) directly.

Pipeline
--------
  returns -> pairwise correlation -> Marchenko-Pastur cleaning -> shrinkage
          -> cov = D * C_shrunk * D        (D = diag(vols))
  cleaned correlation -> Louvain clusters
'''
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Degrees of freedom <= 0 for slice')

import numpy as np
import polars as pl
from typing import Optional, Tuple, Dict, Any, List

from src.risk.correlation_cleaning import clean_correlation_matrix, marchenko_pastur_bounds
from src.risk.clustering import louvain_clustering
from src.risk.covariance import (
    build_covariance_matrix,
    oracle_approximating_shrinkage,
    factor_model_covariance,
)

TRADING_DAYS = 252


# =============================================================================
# 1. Correlation
# =============================================================================

def compute_raw_correlation(returns_matrix: np.ndarray, min_obs: int = 30) -> np.ndarray:
    '''
    Pairwise-complete Pearson correlation. Input: (assets x observations), NaN allowed.

    For each pair (i, j) only the dates where both are finite are used, and the
    means and variances are computed on those dates only.
    Pairs with n_valid <= min_obs get correlation 0 (uncorrelated prior).
    Fully vectorised: five matrix products, no Python loop over pairs.
    '''
    R = np.asarray(returns_matrix, dtype=float)
    M = np.isfinite(R).astype(float)          # 1 where observed
    R0 = np.where(M > 0, R, 0.0)              # NaN -> 0 (masked out by M)

    n = M @ M.T                               # n_ij: common observations
    sx = R0 @ M.T                             # sum of x_i over dates where j is observed (and i, as R0=0 elsewhere)
    sy = sx.T                                 # sum of x_j over dates where i is observed
    sxy = R0 @ R0.T
    sxx = (R0 ** 2) @ M.T
    syy = sxx.T

    with np.errstate(invalid='ignore', divide='ignore'):
        num = sxy - sx * sy / n
        var_x = sxx - sx ** 2 / n
        var_y = syy - sy ** 2 / n
        corr = num / np.sqrt(var_x * var_y)

    corr = np.where((n > min_obs) & np.isfinite(corr), corr, 0.0)
    corr = np.clip(corr, -1.0, 1.0)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    return corr


def clean_correlation_rmt(corr: np.ndarray, n_obs: int) -> Tuple[np.ndarray, np.ndarray, float]:
    '''
    RMT cleaning of a correlation matrix.
    Returns: (corr_clean, eigenvalues_of_corr_clean, lambda_max).

    If every eigenvalue is below the Marchenko-Pastur edge (no detectable signal),
    the raw correlation is returned unchanged and a warning is issued.
    '''
    n_assets = corr.shape[0]
    _, lambda_max = marchenko_pastur_bounds(n_assets, n_obs, variance=1.0)
    try:
        corr_clean = clean_correlation_matrix(corr, n_obs)
    except ValueError as exc:
        warnings.warn(f'RMT cleaning skipped: {exc}')
        corr_clean = corr
    eigenvalues, _ = np.linalg.eigh(corr_clean)
    return corr_clean, eigenvalues, lambda_max


# =============================================================================
# 2. Volatilities
# =============================================================================

def estimate_volatilities(returns_matrix: np.ndarray) -> np.ndarray:
    '''
    Annualised volatilities from daily returns (assets x observations).
    NaN-safe. Assets with fewer than 2 observations, or zero variance,
    get the cross-sectional median volatility (never 0, never NaN).
    '''
    R = np.asarray(returns_matrix, dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        vols = np.nanstd(R, axis=1, ddof=1) * np.sqrt(TRADING_DAYS)
    bad = ~np.isfinite(vols) | (vols <= 0)
    if bad.all():
        return np.full(R.shape[0], 0.30)
    if bad.any():
        vols = np.where(bad, np.median(vols[~bad]), vols)
    return vols


# =============================================================================
# 3. Shrinkage intensity (Ledoit-Wolf 2004, identity target, correlation space)
# =============================================================================

def _standardize(returns_matrix: np.ndarray) -> np.ndarray:
    '''Per-asset z-score, NaN-safe. Missing values and constant assets become 0.'''
    R = np.asarray(returns_matrix, dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        mu = np.nanmean(R, axis=1, keepdims=True)
        sd = np.nanstd(R, axis=1, ddof=0, keepdims=True)
        Z = (R - mu) / sd
    return np.where(np.isfinite(Z), Z, 0.0)


def ledoit_wolf_intensity(returns_matrix: np.ndarray) -> float:
    '''
    Optimal linear shrinkage intensity toward the identity, in correlation space.

        rho = (gamma / T) / delta
        S      : sample correlation of standardised returns
        delta  : ||S - I||_F^2                      (distance to target)
        gamma  : sum_ij Var_hat(sqrt(T) * S_ij)     (estimation noise of S)
        T      : number of observations

    The division by T matters. Without it, rho is almost always clipped to 1.
    Result in [0, 1]. Returns 0.0 if delta is 0 or not finite.
    '''
    Z = _standardize(returns_matrix)
    n_obs = Z.shape[1]
    if n_obs < 2:
        return 0.0
    S = (Z @ Z.T) / n_obs
    Z2 = Z ** 2
    pi_mat = np.maximum((Z2 @ Z2.T) / n_obs - S ** 2, 0.0)   # Var_hat of each entry, times T
    gamma = pi_mat.sum() / n_obs
    delta = np.sum((S - np.eye(S.shape[0])) ** 2)
    if not np.isfinite(delta) or delta <= 0 or not np.isfinite(gamma):
        return 0.0
    return float(np.clip(gamma / delta, 0.0, 1.0))


# =============================================================================
# 4. Covariance
# =============================================================================

def build_covariance(
    corr_clean: np.ndarray,
    vols: np.ndarray,
    method: str,
    returns_matrix: Optional[np.ndarray] = None,
    n_factors: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[float]]:
    '''
    Annualised covariance from the cleaned correlation and annualised vols.
    Returns: (cov, shrinkage_or_None). cov has diagonal vols**2 for every method
    except 'factor' (which uses its own variances).

    method:
      'ledoit_wolf'  C_s = rho*I + (1-rho)*C_clean, rho from ledoit_wolf_intensity
      'oas'          library OAS intensity, same convention
      'factor'       PCA factor model on the returns (needs returns_matrix)
      'sample'       C_clean as it is (no shrinkage)
    Then cov = D * C_s * D.
    '''
    corr_clean = 0.5 * (np.asarray(corr_clean, dtype=float) + np.asarray(corr_clean, dtype=float).T)
    n = corr_clean.shape[0]

    if method in ('ledoit_wolf', 'oas'):
        if returns_matrix is None:
            raise ValueError(f"method {method!r} needs returns_matrix to estimate the shrinkage intensity")
        if method == 'ledoit_wolf':
            rho = ledoit_wolf_intensity(returns_matrix)
            corr_s = rho * np.eye(n) + (1.0 - rho) * corr_clean
        else:
            corr_s, rho = oracle_approximating_shrinkage(corr_clean, returns=_standardize(returns_matrix))
        np.fill_diagonal(corr_s, 1.0)
        return build_covariance_matrix(corr_s, vols), float(rho)

    if method == 'factor':
        if returns_matrix is None:
            raise ValueError("method 'factor' needs returns_matrix")
        X = np.where(np.isfinite(returns_matrix), returns_matrix, 0.0)   # (assets x obs)
        sigma_daily, _, _ = factor_model_covariance(X, n_factors or 5)
        return sigma_daily * TRADING_DAYS, None

    if method == 'sample':
        return build_covariance_matrix(corr_clean, vols), None

    raise ValueError(f"unknown cov method: {method!r}. Use 'ledoit_wolf', 'oas', 'factor' or 'sample'.")


# =============================================================================
# 5. Clustering
# =============================================================================

def cluster_correlation(corr_clean: np.ndarray, resolution: float) -> Dict[int, int]:
    '''Louvain clustering on the cleaned correlation. Returns {ticker_idx: cluster_id}.'''
    return louvain_clustering(corr_clean, resolution=resolution)


# =============================================================================
# 6. One rebalance date
# =============================================================================

def estimate_risk_model_for_date(
    returns_matrix: np.ndarray,
    tickers: List[str],
    date_str: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    '''
    Full risk model for one rebalance date.
    returns_matrix: (n_assets x n_obs) daily log returns known BEFORE the rebalance date.
    '''
    n_tickers = len(tickers)
    n_obs = returns_matrix.shape[1]

    # 1. Raw correlation (pairwise complete)
    min_obs = config.get('min_obs', 30)
    corr = compute_raw_correlation(returns_matrix, min_obs=min_obs)

    # 2. RMT cleaning
    corr_clean, eigenvalues, lambda_max = clean_correlation_rmt(corr, n_obs)
    q = n_tickers / n_obs

    # 3. Annualised volatilities
    vols = estimate_volatilities(returns_matrix)

    # 4. Annualised covariance
    method = config.get('cov_method', 'ledoit_wolf')
    n_factors = config.get('n_factors')
    cov, shrinkage = build_covariance(corr_clean, vols, method, returns_matrix, n_factors)

    # 5. Clustering
    cluster_resolution = config.get('cluster_resolution', 1.0)
    cluster_map = cluster_correlation(corr_clean, cluster_resolution)

    clusters: Dict[int, list] = {}
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


# =============================================================================
# 7. Rolling windows
# =============================================================================

def build_returns_panel(returns_wide: pl.DataFrame, rebalance_dates: list, lookback_window: int) -> np.ndarray:
    '''
    3D returns panel (n_assets, lookback_window, n_rebalances) from a wide returns frame.

    For each rebalance date t the window holds the `lookback_window` returns
    strictly BEFORE t (rows idx-lookback .. idx-1). Missing history is NaN
    (not 0, so that it is not mistaken for a real zero return).

    Args:
        returns_wide: DataFrame, first column = date, other columns = tickers.
        rebalance_dates: list of datetime.date.
        lookback_window: number of days.
    '''
    valid_tickers = returns_wide.columns[1:]
    n_assets = len(valid_tickers)

    date_col = returns_wide.columns[0]
    returns_np = returns_wide.select(valid_tickers).to_numpy().T      # (n_assets, n_dates)
    dates_np = returns_wide[date_col].to_numpy()

    date_to_idx = {np.datetime64(d).astype('datetime64[D]'): i for i, d in enumerate(dates_np)}
    rebalance_dates_np = np.array([np.datetime64(d) for d in rebalance_dates], dtype='datetime64[D]')

    returns_panel = np.full((n_assets, lookback_window, len(rebalance_dates)), np.nan)

    for i, reb_date_np in enumerate(rebalance_dates_np):
        idx = date_to_idx.get(reb_date_np)
        if idx is None:
            continue
        start_idx = max(0, idx - lookback_window)
        window_data = returns_np[:, start_idx:idx]
        w = window_data.shape[1]
        if w > 0:
            returns_panel[:, lookback_window - w:, i] = window_data   # right-aligned, NaN pad on the left

    return returns_panel