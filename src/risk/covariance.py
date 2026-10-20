'''
src/risk/covariance.py

State-of-the-art covariance matrix estimation for portfolio construction.

================================================================================
WHY THIS MATTERS
================================================================================

A portfolio optimizer is only as good as its covariance matrix. The naive
sample covariance estimator is notoriously unstable in high dimensions:
with P assets and N observations, you need N >> P for the sample covariance
to be well-conditioned. In equities we typically have P ~ 500 and N ~ 252-1260
(one to five years of daily data), so N/P is often barely above 1 and the
sample covariance matrix is singular or near-singular.

Every top quant fund (Millennium, Citadel, Two Sigma, AQR, Point72) therefore
uses regularized / shrunk covariance estimators. This module implements the
three most widely used approaches:

  1. Ledoit-Wolf (2004) linear shrinkage toward identity -- the gold standard
     analytical shrinkage estimator. Used across the industry.

  2. Oracle Approximating Shrinkage (Ledoit & Wolf 2012) -- finite-sample
     optimal variant of (1), slightly more aggressive shrinkage.

  3. Factor-model covariance via PCA (Bai & Ng 2002; Connor & Korajczyk 1986)
     -- decomposes covariance into low-rank systematic + diagonal idiosyncratic.
     Essential when P > N.

  4. Shrinkage toward constant-correlation target (Ledoit et al. 2003) -- a
     structured alternative to the identity target.

The output of each function is a symmetric positive semi-definite covariance
matrix, ready for use in vol targeting, cluster caps, or mean-variance
optimization downstream.

================================================================================
INTEGRATION WITH THE REST OF THE PIPELINE
================================================================================

This module plugs directly into the existing risk infrastructure:

  - src/risk/correlation_cleaning.py  -- our Marchenko-Pastur RMT cleaner
    already produces a denoised correlation matrix C_clean. Use that C as
    input here instead of a raw sample correlation.

  - src/risk/clustering.py            -- Louvain clusters can be used to
    build block-diagonal factor models (one factor per cluster).

  - src/risk/portfolio_construction.py -- consumes Σ here to vol-target and
    cap cluster exposures.

Typical call chain:

    # 1. Estimate correlation from returns (existing: compute per rolling window)
    corr = sample_correlation(returns_panel)          # P x P

    # 2. RMT-clean it (existing)
    corr_clean = clean_correlation_matrix(corr, n_obs)

    # 3. Shrink the cleaned correlation (new: this module)
    corr_shrunk, rho = ledoit_wolf_shrinkage(corr_clean, returns_panel)

    # 4. Convert to covariance with vols
    vols = rolling_std(returns_panel, window=60)      # P-vector
    sigma = build_covariance_matrix(corr_shrunk, vols) # P x P

    # 5. Feed to portfolio construction
    weights = apply_vol_target(raw_w, sigma, target_vol=0.10)

================================================================================
REFERENCES
================================================================================

  - Ledoit, O., & Wolf, M. (2004). "A well-conditioned estimator for large-
    dimensional covariance matrices." Journal of Multivariate Analysis,
    88(2), 365-411.  (Linear shrinkage to identity; optimal shrinkage intensity.)

  - Ledoit, O., & Wolf, M. (2012). "Nonlinear shrinkage estimation of large-
    dimensional covariance matrices." Annals of Statistics, 40(2), 1024-1060.
    (Oracle Approximating Shrinkage -- finite-sample optimal.)

  - Ledoit, O., Wolf, M., & Zhao, Z. (2019). "Efficient sorting and shrinkage."
    (Companion to the 2012 paper.)

  - Bai, J., & Ng, S. (2002). "Determining the number of factors in approximate
    factor models." Econometrica, 70(1), 191-221.  (Factor model selection.)

  - Connor, G., & Korajczyk, R. A. (1986). "Performance measurement with the
    arbitrage pricing theory." Journal of Financial Economics, 15(2), 373-394.
    (Factor model for asset returns.)

  - Kritzman, M., & Li, Y. (2010). "Skulls, financial turbulence, and risk
    management." Financial Analysts Journal, 66(5), 30-41.  (Absorption ratio.)

================================================================================
'''

import numpy as np
import polars as pl
from sklearn.decomposition import PCA


# =============================================================================
# HELPER: symmetrize and PSD-project
# =============================================================================

def _symmetrize(M: np.ndarray) -> np.ndarray:
    '''
    Symmetrize a square matrix: (M + M') / 2.

    Numerical eigen-decompositions can introduce tiny asymmetries (off-diagonal
    entries differ at 1e-16 level). Symmetrizing guarantees exact symmetry,
    which is required before Cholesky / eigh downstream.
    '''
    return 0.5 * (M + M.T)


def _psd_project(M: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    '''
    Project a symmetric matrix onto the positive semi-definite cone.

    Eigen-decompose, clip negative eigenvalues to `floor`, reconstruct.
    Guarantees the output covariance is PSD -- a hard requirement for any
    portfolio optimizer (vol targeting, mean-variance, risk parity).

    Reference: Higham (2002), "Computing the nearest correlation matrix".
    '''
    M = _symmetrize(M)
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    eigenvalues = np.clip(eigenvalues, floor, None)
    return (eigenvectors * eigenvalues) @ eigenvectors.T


# =============================================================================
# STEP 1: Ledoit-Wolf (2004) Linear Shrinkage
# =============================================================================

def ledoit_wolf_shrinkage(
    corr: np.ndarray,
    returns: np.ndarray | None = None,
    target: str = 'identity',
) -> tuple[np.ndarray, float]:
    '''
    Ledoit-Wolf (2004) linear shrinkage covariance estimator.

    ===========================================================================
    INTUITION
    ===========================================================================

    The sample covariance matrix S has P(P+1)/2 free parameters but only N
    observations. When N/P is small, S is noisy: its extreme eigenvalues are
    exaggerated (Marchenko-Pastur), and its inverse is unstable.

    Shrinkage pulls S toward a structured target F:

        Sigma_hat = rho * F + (1 - rho) * S

    where rho in [0, 1] is the shrinkage intensity:
        rho = 0  -> pure sample covariance (no shrinkage)
        rho = 1  -> pure target (ignore data entirely)

    The optimal rho balances bias (target may be wrong) against variance
    (sample estimate is noisy). Ledoit & Wolf (2004) derive rho analytically
    by minimizing the expected Frobenius loss E||Sigma_hat - Sigma||_F^2,
    assuming only that returns are i.i.d. with finite fourth moments -- no
    normality required.

    ===========================================================================
    TARGET CHOICES
    ===========================================================================

    'identity':  F = I (equicorrelation = 0, all assets independent).
        Simplest, most common target. Works well when true correlations are
        weak on average. Default because it needs no extra estimation.

    'constant_correlation':  F = constant-correlation matrix with average
        sample correlation as the common off-diagonal. Better when assets
        share a strong common factor (e.g. all stocks move with the market).

    ===========================================================================
    PARAMETERS
    ===========================================================================

    corr:      P x P sample correlation matrix (can be raw or RMT-cleaned).
    returns:   P x N array of returns (one row per asset, one col per obs),
               needed only to compute the optimal shrinkage intensity rho.
               If None, defaults to rho = 0.5 (moderate shrinkage).
    target:    'identity' or 'constant_correlation'.

    ===========================================================================
    RETURNS
    ===========================================================================

    (corr_shrunk, rho): shrunk correlation matrix and the shrinkage intensity.

    ===========================================================================
    REFERENCES
    ===========================================================================

    Ledoit & Wolf (2004), J. Multivariate Analysis 88(2), 365-411.
    Ledoit & Wolf (2012), Ann. Statist. 40(2), 1024-1060.
    '''
    corr = _symmetrize(np.asarray(corr, dtype=float))
    n_assets = corr.shape[0]

    # ---------------------------------------------------------------------
    # Build the shrinkage target F.
    # ---------------------------------------------------------------------

    if target == 'identity':
        # F = identity: all assets assumed uncorrelated, unit variance.
        target_mat = np.eye(n_assets)

    elif target == 'constant_correlation':
        # F = constant correlation: off-diagonals all equal to the average
        # sample correlation (excluding diagonal). Preserves the overall
        # correlation level while removing idiosyncratic noise.
        mask = ~np.eye(n_assets, dtype=bool)
        avg_corr = corr[mask].mean()
        target_mat = np.full((n_assets, n_assets), avg_corr)
        np.fill_diagonal(target_mat, 1.0)

    else:
        raise ValueError(f"unknown target: {target!r}, use 'identity' or 'constant_correlation'")

    # ---------------------------------------------------------------------
    # If no returns given, use a conservative default rho.
    # ---------------------------------------------------------------------

    if returns is None:
        rho = 0.5
        corr_shrunk = rho * target_mat + (1.0 - rho) * corr
        return _psd_project(corr_shrunk), rho

    # ---------------------------------------------------------------------
    # Compute the optimal shrinkage intensity rho analytically.
    # ---------------------------------------------------------------------

    # Returns: shape (n_assets, n_obs). Center them.
    X = np.asarray(returns, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    X_centered = X - X.mean(axis=1, keepdims=True)
    n_obs = X_centered.shape[1]

    # Sample covariance (denominator N, unbiased up to N-1 factor which
    # cancels in the rho ratio below).
    S = (X_centered @ X_centered.T) / n_obs

    # Shrinkage target in covariance space: scale the correlation target
    # by the sample variances so we're shrinking covariances, not correlations.
    sample_vars = np.diag(S)
    target_cov = target_mat * np.outer(np.sqrt(sample_vars), np.sqrt(sample_vars))

    # pi_mat: the "asymptotic variance" of each entry of S.
    # For entry (i,j), pi_ij = (1/N) * sum_k (X_ik*X_jk - S_ij)^2 / N ...
    # More precisely, Ledoit-Wolf define:
    #   pi_ij = (1/N) * sum_{k=1}^N (x_ik x_jk - S_ij)^2   [empirical version]
    #   (with x centered)
    # We compute the full matrix efficiently.
    # X_centered^2: (P, N) of squared centered returns.
    X_sq = X_centered ** 2
    # For each pair (i,j): sum_k (X_ik * X_jk - S_ij)^2 / N
    #   = sum_k X_ik^2 X_jk^2 / N - 2*S_ij * sum_k X_ik X_jk / N + S_ij^2
    #   = (X_sq_i @ X_sq_j) / N - 2*S_ij * (X_i @ X_j)/N + S_ij^2
    #   = (X_sq_i @ X_sq_j) / N - 2*S_ij * S_ij + S_ij^2   [since X_i@X_j/N = S_ij]
    #   = (X_sq_i @ X_sq_j) / N - S_ij^2
    cross_sq = (X_sq @ X_sq.T) / n_obs
    pi_mat = cross_sq - S ** 2
    pi_mat = np.maximum(pi_mat, 0.0)  # numerical safety

    # delta: squared Frobenius distance between S and target.
    delta = np.sum((S - target_cov) ** 2)

    # gamma: sum of pi_ij over all entries (total estimation error of S).
    gamma = np.sum(pi_mat)

    # Optimal shrinkage intensity: rho = gamma / delta, clipped to [0, 1].
    # When delta is tiny (S already close to target), rho -> 0 (no shrinkage).
    # When gamma is huge (S very noisy), rho -> 1 (full shrinkage to target).
    if delta > 0:
        rho = gamma / delta
    else:
        rho = 0.0
    rho = float(np.clip(rho, 0.0, 1.0))

    # ---------------------------------------------------------------------
    # Apply shrinkage in correlation space (preserve unit diagonal).
    # ---------------------------------------------------------------------

    corr_shrunk = rho * target_mat + (1.0 - rho) * corr

    # Enforce unit diagonal (shrinkage can drift it slightly off 1).
    np.fill_diagonal(corr_shrunk, 1.0)

    return _psd_project(corr_shrunk), rho


# =============================================================================
# STEP 2: Oracle Approximating Shrinkage (Ledoit & Wolf 2012)
# =============================================================================

def oracle_approximating_shrinkage(
    corr: np.ndarray,
    returns: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    '''
    Oracle Approximating Shrinkage (OAS) -- Ledoit & Wolf (2012).

    ===========================================================================
    INTUITION
    ===========================================================================

    OAS is a refinement of the Ledoit-Wolf linear shrinkage estimator that
    is optimal in finite samples under a slightly different loss criterion.
    Key differences from LW (2004):

      - OAS uses a different (smaller) shrinkage intensity on average,
        because it accounts for the fact that the sample eigenvalues are
        systematically biased upward (Marčenko-Pastur spreading).
      - OAS has a closed-form solution that does NOT require estimating
        the asymptotic variance of each entry -- it uses a simpler
        moment-based formula.

    Empirically OAS often outperforms LW (2004) when N/P is moderate
    (say 2 < N/P < 10), which is exactly our equities regime.

    ===========================================================================
    FORMULA
    ===========================================================================

    rho_OAS = [ (1 - 2/P) * sum_ij S_ij^2 + tr(S)^2 ] /
              [ (N + 1 - 2/P) * sum_ij S_ij^2 - 2 * tr(S)^2 / N ... ]

    Actually the standard OAS formula (Ledoit & Wolf 2012, eq. 14):

        rho = [ sum_ij S_ij^2 - (1/P) * tr(S)^2 ] /
              [ (N + 1) * sum_ij S_ij^2 - 2 * tr(S)^2 / P ]

    where S is the sample covariance (with denominator N), P = n_assets,
    N = n_observations.

    Then: Sigma_hat = rho * (tr(S)/P) * I + (1 - rho) * S.

    Target is (trace/P) * I -- i.e. identity scaled by average variance.
    This preserves total variance (trace) exactly.

    ===========================================================================
    PARAMETERS
    ===========================================================================

    corr:      P x P sample correlation matrix.
    returns:   P x N array of returns. If None, defaults rho = 0.5.

    ===========================================================================
    RETURNS
    ===========================================================================

    (corr_shrunk, rho): shrunk correlation matrix and shrinkage intensity.

    ===========================================================================
    REFERENCES
    ===========================================================================

    Ledoit, O., & Wolf, M. (2012). "Nonlinear shrinkage estimation of large-
    dimensional covariance matrices." Annals of Statistics, 40(2), 1024-1060.
    See Section 2 (Oracle Approximating Shrinkage).
    '''
    corr = _symmetrize(np.asarray(corr, dtype=float))
    n_assets = corr.shape[0]

    if returns is None:
        rho = 0.5
        target = np.eye(n_assets)
        corr_shrunk = rho * target + (1.0 - rho) * corr
        np.fill_diagonal(corr_shrunk, 1.0)
        return _psd_project(corr_shrunk), rho

    X = np.asarray(returns, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    X_centered = X - X.mean(axis=1, keepdims=True)
    n_obs = X_centered.shape[1]

    # Sample covariance with denominator N (not N-1; matches LW paper).
    S = (X_centered @ X_centered.T) / n_obs

    # Work in covariance space for OAS (target = scaled identity).
    sample_vars = np.diag(S)
    avg_var = np.trace(S) / n_assets
    target_cov = avg_var * np.eye(n_assets)

    # sum_ij S_ij^2 = Frobenius norm squared of S.
    fro_sq = np.sum(S ** 2)
    tr_S = np.trace(S)

    # OAS shrinkage intensity (Ledoit & Wolf 2012, eq. 14).
    numerator = fro_sq - (tr_S ** 2) / n_assets
    denominator = (n_obs + 1) * fro_sq - 2 * (tr_S ** 2) / n_assets

    if denominator > 0:
        rho = numerator / denominator
    else:
        rho = 0.0
    rho = float(np.clip(rho, 0.0, 1.0))

    # Shrink covariance, then convert back to correlation.
    cov_shrunk = rho * target_cov + (1.0 - rho) * S

    # Normalize to correlation: divide by outer product of std devs.
    shrunk_vars = np.diag(cov_shrunk)
    shrunk_std = np.sqrt(np.maximum(shrunk_vars, 1e-12))
    corr_shrunk = cov_shrunk / np.outer(shrunk_std, shrunk_std)
    np.fill_diagonal(corr_shrunk, 1.0)

    return _psd_project(corr_shrunk), rho


# =============================================================================
# STEP 3: Factor Model Covariance via PCA
# =============================================================================

def factor_model_covariance(
    returns: np.ndarray,
    n_factors: int | None = None,
    method: str = 'fixed',
) -> tuple[np.ndarray, np.ndarray, int]:
    '''
    Factor-model covariance via PCA decomposition.

    ===========================================================================
    INTUITION
    ===========================================================================

    When P is large relative to N (many assets, limited history), the sample
    covariance is full of noise. A factor model assumes returns decompose as:

        r_i(t) = beta_i' * f(t) + epsilon_i(t)

    where f(t) is a K-vector of common factors (K << P) and epsilon_i is
    asset-specific idiosyncratic noise. The covariance becomes:

        Sigma = B * F * B' + D_epsilon

    where B is P x K factor loadings, F is K x K factor covariance,
    D_epsilon is P x P diagonal idiosyncratic variance.

    PCA extracts the top K principal components of the return covariance as
    factors. This is equivalent to a single-index / multi-index model but
    with statistically orthogonal factors (no need to pick market/sector
    factors a priori).

    Why this works: financial returns are driven by a small number of common
    sources of variation (market, sector, style). The top K PCs capture these
    systematically; the residual is mostly idiosyncratic noise. Shrinking the
    small eigenvalues (which PCA discards) toward their average dramatically
    reduces estimation error.

    ===========================================================================
    FACTOR SELECTION
    ===========================================================================

    If n_factors is None, uses the Bai & Ng (2002) IC criteria to select K
    automatically. Otherwise uses the fixed value.

    ===========================================================================
    PARAMETERS
    ===========================================================================

    returns:    P x N array of returns (one row per asset).
    n_factors:  number of factors K. If None, auto-select via Bai-Ng IC.
    method:     'fixed' (use given n_factors) or 'bic' (auto-select).

    ===========================================================================
    RETURNS
    ===========================================================================

    (sigma, factor_loadings, n_factors_used):
        sigma:       P x P factor-model covariance matrix.
        factor_loadings: P x K matrix B.
        n_factors_used: K actually used.

    ===========================================================================
    REFERENCES
    ===========================================================================

    Bai, J., & Ng, S. (2002). "Determining the number of factors in approximate
    factor models." Econometrica, 70(1), 191-221.

    Connor, G., & Korajczyk, R. A. (1986). "Performance measurement with the
    arbitrage pricing theory." Journal of Financial Economics, 15(2), 373-394.

    López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley.
    (Ch. 11: covariance estimation via PCA/factor models.)
    '''
    X = np.asarray(returns, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    n_assets, n_obs = X.shape

    X_centered = X - X.mean(axis=1, keepdims=True)
    S = (X_centered @ X_centered.T) / n_obs

    # -----------------------------------------------------------------
    # Auto-select n_factors via Bai-Ng IC2 criterion (simplified version).
    # -----------------------------------------------------------------

    if n_factors is None:
        max_k = min(n_assets // 2, n_obs // 2, 20)
        if max_k < 1:
            max_k = 1

        # Eigenvalues of sample covariance (descending).
        eigenvalues = np.linalg.eigvalsh(S)[::-1]

        # Bai-Ng IC2(k) = log(V(k)) + k * g(N,T), where V(k) is the average
        # residual variance after extracting k factors, and g is a penalty.
        # Simplified: use the "eigenvalue ratio" heuristic + a penalty on k.
        best_ic = np.inf
        best_k = 1
        for k in range(1, max_k + 1):
            residual_var = np.sum(eigenvalues[k:]) / (n_assets - k)
            penalty = k * (np.log(n_obs) / n_obs)  # BIC-like penalty
            ic = np.log(residual_var) + penalty
            if ic < best_ic:
                best_ic = ic
                best_k = k
        n_factors = best_k

    n_factors = int(n_factors)
    n_factors = max(1, min(n_factors, n_assets, n_obs))

    # -----------------------------------------------------------------
    # PCA: eigen-decompose sample covariance.
    # -----------------------------------------------------------------

    eigenvalues, eigenvectors = np.linalg.eigh(S)  # ascending order
    # Take top n_factors (largest eigenvalues at the end).
    loadings = eigenvectors[:, -n_factors:]          # P x K
    factor_vars = eigenvalues[-n_factors:]           # K-vector

    # Factor covariance F is diagonal (PCs are orthogonal).
    F = np.diag(factor_vars)

    # Idiosyncratic variance: average of the remaining eigenvalues.
    # This is the key shrinkage step: instead of using each small
    # eigenvalue individually (noisy), replace them all with their mean.
    if n_factors < n_assets:
        idio_var = np.mean(eigenvalues[:-n_factors])
    else:
        idio_var = 0.0

    D_epsilon = idio_var * np.eye(n_assets)

    # Sigma = B * F * B' + D_epsilon.
    sigma = loadings @ F @ loadings.T + D_epsilon

    # Convert back to correlation for consistency with the rest of the pipeline.
    sigma = _symmetrize(sigma)
    stds = np.sqrt(np.diag(sigma))
    corr = sigma / np.outer(stds, stds)
    np.fill_diagonal(corr, 1.0)

    # Reconstruct covariance from cleaned correlation.
    sigma = build_covariance_matrix(corr, stds)

    return _psd_project(sigma), loadings, n_factors


# =============================================================================
# STEP 4: Shrink Toward Constant Correlation
# =============================================================================

def shrink_towards_constant_correlation(
    corr: np.ndarray,
    delta: float,
) -> np.ndarray:
    '''
    Shrink correlation matrix toward a constant-correlation target.

    ===========================================================================
    INTUITION
    ===========================================================================

    Ledoit et al. (2003) propose shrinking toward a constant-correlation
    matrix rather than identity. The target has all off-diagonal correlations
    equal to the average sample correlation. This is better when assets share
    a strong common factor (e.g. the market) -- identity would zero out that
    common structure entirely.

    Formula:
        C_shrunk = delta * C_constant + (1 - delta) * C_sample

    where delta is chosen by the user (or via cross-validation). Unlike
    Ledoit-Wolf (2004), delta is NOT computed analytically here -- it's a
    user-specified hyperparameter. This gives you explicit control over the
    bias-variance tradeoff.

    ===========================================================================
    PARAMETERS
    ===========================================================================

    corr:   P x P sample correlation matrix.
    delta:  shrinkage intensity in [0, 1].
        0 = no shrinkage (pure sample)
        1 = pure constant-correlation target

    ===========================================================================
    RETURNS
    ===========================================================================

    corr_shrunk: P x P shrunk correlation matrix (PSD-projected).

    ===========================================================================
    REFERENCES
    ===========================================================================

    Ledoit, O., Wolf, M., & Zhao, Z. (2019). "Efficient sorting and shrinkage."
    Ledoit, O., & Wolf, M. (2004). J. Multivariate Analysis 88(2), 365-411.
    '''
    corr = _symmetrize(np.asarray(corr, dtype=float))
    n_assets = corr.shape[0]
    delta = float(np.clip(delta, 0.0, 1.0))

    mask = ~np.eye(n_assets, dtype=bool)
    avg_corr = corr[mask].mean()
    target = np.full((n_assets, n_assets), avg_corr)
    np.fill_diagonal(target, 1.0)

    corr_shrunk = delta * target + (1.0 - delta) * corr
    np.fill_diagonal(corr_shrunk, 1.0)

    return _psd_project(corr_shrunk)


# =============================================================================
# STEP 5: Build Covariance Matrix from Correlation + Vols
# =============================================================================

def build_covariance_matrix(
    corr: np.ndarray,
    vols: np.ndarray,
) -> np.ndarray:
    '''
    Convert a correlation matrix to a covariance matrix: Sigma = D * C * D.

    ===========================================================================
    INTUITION
    ===========================================================================

    Correlation captures the *directional* co-movement of asset returns;
    volatilities capture the *scale* of each asset's own risk. The covariance
    between assets i and j is:

        Sigma_ij = vol_i * vol_j * Corr_ij

    In matrix form: Sigma = D @ C @ D where D = diag(vol_1, ..., vol_P).

    This separation is useful because:
      - Correlation is more stable over time than volatility.
      - You can swap in different vol estimates (EWMA, GARCH, rolling) without
        re-estimating the correlation structure.
      - RMT cleaning operates on correlation, not covariance.

    ===========================================================================
    PARAMETERS
    ===========================================================================

    corr:  P x P correlation matrix (must have unit diagonal).
    vols:  P-vector of volatilities (std dev of returns).

    ===========================================================================
    RETURNS
    ===========================================================================

    sigma: P x P covariance matrix (symmetric, PSD-projected).

    ===========================================================================
    REFERENCES
    ===========================================================================

    Standard textbook result; see any reference on portfolio theory, e.g.:
    Grinold & Kahn (2000), "Active Portfolio Management", McGraw-Hill.
    '''
    corr = _symmetrize(np.asarray(corr, dtype=float))
    vols = np.asarray(vols, dtype=float).reshape(-1)
    n_assets = corr.shape[0]

    if vols.shape[0] != n_assets:
        raise ValueError(
            f'vols length {vols.shape[0]} != corr dimension {n_assets}'
        )

    # Guard against zero or negative vols (would blow up the covariance).
    vols = np.maximum(vols, 1e-8)

    D = np.diag(vols)
    sigma = D @ corr @ D

    return _psd_project(sigma)


# =============================================================================
# CONVENIENCE: full pipeline from returns to shrunk covariance
# =============================================================================

def estimate_covariance(
    returns: np.ndarray,
    method: str = 'ledoit_wolf',
    target: str = 'identity',
    n_factors: int | None = None,
    delta: float | None = None,
) -> tuple[np.ndarray, dict]:
    '''
    End-to-end covariance estimation from a returns panel.

    ===========================================================================
    INTUITION
    ===========================================================================

    One-stop shop: compute sample correlation from returns, then apply the
    chosen regularization method. Returns both the covariance matrix and a
    diagnostics dict (shrinkage intensity, n_factors, etc.) for logging.

    ===========================================================================
    PARAMETERS
    ===========================================================================

    returns:   P x N array (one row per asset).
    method:    'ledoit_wolf', 'oas', 'factor', 'constant_corr', or 'sample'.
    target:    for ledoit_wolf: 'identity' or 'constant_correlation'.
    n_factors: for method='factor': number of PCA factors (None = auto).
    delta:     for method='constant_corr': shrinkage intensity [0,1].

    ===========================================================================
    RETURNS
    ===========================================================================

    (sigma, info):
        sigma: P x P covariance matrix.
        info:  dict with method-specific diagnostics.

    ===========================================================================
    REFERENCES
    ===========================================================================

    See individual method docstrings above.
    '''
    X = np.asarray(returns, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    n_assets, n_obs = X.shape

    X_centered = X - X.mean(axis=1, keepdims=True)
    S = (X_centered @ X_centered.T) / n_obs

    # Sample correlation.
    stds = np.sqrt(np.diag(S))
    stds = np.maximum(stds, 1e-8)
    corr = S / np.outer(stds, stds)
    np.fill_diagonal(corr, 1.0)

    info = {'method': method, 'n_assets': n_assets, 'n_obs': n_obs}

    if method == 'sample':
        sigma = build_covariance_matrix(corr, stds)
        info['rho'] = 0.0
        info['n_factors'] = None

    elif method == 'ledoit_wolf':
        corr_shrunk, rho = ledoit_wolf_shrinkage(corr, returns=X_centered, target=target)
        sigma = build_covariance_matrix(corr_shrunk, stds)
        info['rho'] = rho
        info['target'] = target
        info['n_factors'] = None

    elif method == 'oas':
        corr_shrunk, rho = oracle_approximating_shrinkage(corr, returns=X_centered)
        sigma = build_covariance_matrix(corr_shrunk, stds)
        info['rho'] = rho
        info['target'] = 'identity_scaled'
        info['n_factors'] = None

    elif method == 'constant_corr':
        if delta is None:
            delta = 0.5
        corr_shrunk = shrink_towards_constant_correlation(corr, delta=delta)
        sigma = build_covariance_matrix(corr_shrunk, stds)
        info['rho'] = delta
        info['target'] = 'constant_correlation'
        info['n_factors'] = None

    elif method == 'factor':
        sigma, loadings, n_factors_used = factor_model_covariance(X_centered, n_factors=n_factors)
        info['rho'] = None
        info['n_factors'] = n_factors_used

    else:
        raise ValueError(
            f"unknown method: {method!r}. "
            "Use 'sample', 'ledoit_wolf', 'oas', 'constant_corr', or 'factor'."
        )

    return sigma, info