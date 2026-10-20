'''
src/portfolio/optimization.py

Optimization solvers for portfolio construction.

================================================================================
WHY THIS MATTERS
================================================================================

Portfolio construction ultimately requires solving an optimization problem.
At top firms (Millennium, Citadel, Two Sigma, AQR), the optimization is
where risk models, alpha signals, and cost models meet. Different firms
use different approaches:

  1. MEAN-VARIANCE (Markowitz 1952):  max w'μ - λ w'Σw  s.t. constraints
     - Standard, but sensitive to μ estimation error
     - Requires robust Σ (shrinkage, factor models) and regularization

  2. RISK PARITY / INVERSE VOL: w_i ∝ 1/σ_i (no μ needed)
     - More robust, no return forecasting needed
     - Used for volatility targeting and stable allocation

  3. HIERARCHICAL RISK PARITY (López de Prado 2016):
     - Uses correlation hierarchy (our Louvain clusters!) to allocate
     - No matrix inversion, numerically stable, no μ needed
     - Solves the "Markowitz instability" problem

  4. CLUSTERED RISK PARITY (Raffinot 2018):
     - Extends HRP: equal risk contribution WITHIN and BETWEEN clusters
     - Directly uses our Louvain output

  5. BLACK-LITTERMAN (1992):
     - Combines market equilibrium (CAPM) with alpha views
     - Our signal = "views", market prior = equilibrium
     - Used at BlackRock, Goldman, etc.

  6. COST-AWARE OPTIMIZATION (Gârleanu & Pedersen 2013):
     - max w'μ - λ w'Σw - 0.5 (w - w_prev)' Λ (w - w_prev)
     - Λ encodes transaction costs (from costs.py)
     - Balances alpha vs turnover

This module implements all of the above in a modular way. The "production"
choice depends on the strategy, but HRP/Clustered Risk Parity are the
modern gold standard for equity long-short.

================================================================================
INTEGRATION
================================================================================

Inputs:
  - μ (expected returns): from combined signal (Elastic Net / GBM)
  - Σ (covariance): from src/risk/covariance.py (shrunk + RMT-cleaned)
  - w_prev: previous weights (for turnover cost)
  - ADV, vol: from features (for cost model)
  - Cluster map: from src/risk/clustering.py (Louvain)

Outputs:
  - Optimal weights (P-vector)
  - Diagnostics (risk contrib, factor contrib, etc.)

================================================================================
REFERENCES
================================================================================

  - Markowitz, H. (1952). "Portfolio Selection." Journal of Finance, 7(1), 77-91.
  - Black, F., & Litterman, R. (1992). "Global Portfolio Optimization."
    Financial Analysts Journal, 48(5), 28-43.
  - López de Prado, M. (2016). "Building Diversified Portfolios that Outperform
    Out-of-Sample." Journal of Portfolio Management, 42(4), 59-69. (HRP)
  - Raffinot, T. (2018). "Hierarchical Clustering-Based Asset Allocation."
    (Clustered Risk Parity)
  - Gârleanu, N., & Pedersen, L. H. (2013). "Dynamic Trading with Predictable
    Returns and Transaction Costs." Journal of Finance, 68(6), 2309-2340.
  - Asness, C. S., Frazzini, A., & Pedersen, L. H. (2012). "Leverage Aversion
    and Risk Parity." Financial Analysts Journal, 68(1), 47-59.
  - Roncalli, T. (2013). "Introduction to Risk Parity and Budgeting." CRC Press.

================================================================================
'''

import numpy as np
import polars as pl
from polars import col as c
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree
from sklearn.covariance import shrunk_covariance


# =============================================================================
# HELPER: Inverse-Volatility Weights (simplest risk parity)
# =============================================================================

def inverse_volatility_weights(vols: np.ndarray) -> np.ndarray:
    '''
    Inverse-volatility weights: w_i = (1/σ_i) / Σ_j (1/σ_j).

    ===========================================================================
    INTUITION
    ===========================================================================

    The simplest form of risk parity: give each asset the same risk budget
    by sizing inversely to its volatility. No correlation needed, no
    optimization solver needed, extremely robust.

    This is the "naive risk parity" baseline. It ignores correlations
    (assumes diagonal Σ), so it can over-allocate to correlated assets.

    ===========================================================================
    PARAMETERS
    ===========================================================================

    vols: P-vector of volatilities (daily or annualized, scale cancels out).

    ===========================================================================
    RETURNS
    ===========================================================================

    P-vector of weights summing to 1.0 (long-only) or dollar-neutral if
    you center them. Here returns long-only summing to 1.
    '''
    vols = np.asarray(vols, dtype=float)
    vols = np.maximum(vols, 1e-8)  # floor
    inv_vol = 1.0 / vols
    weights = inv_vol / np.sum(inv_vol)
    return weights


# =============================================================================
# EQUAL RISK CONTRIBUTION (True Risk Parity)
# =============================================================================

def equal_risk_contribution_weights(
    sigma: np.ndarray,
    w_init: np.ndarray | None = None,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> np.ndarray:
    '''
    Solve for Equal Risk Contribution (ERC) weights.

    ===========================================================================
    INTUITION
    ===========================================================================

    ERC finds weights such that each asset contributes equally to portfolio
    variance:

      RC_i = w_i * (Σ w)_i / (w' Σ w) = 1 / P   for all i

    This is the true risk parity portfolio. It accounts for correlations
    (unlike inverse vol) but requires no expected return μ.

    Solution: fixed-point iteration (Roncalli 2013) or convex optimization.
    We use the fixed-point iteration which is fast and guaranteed to converge
    for positive definite Σ.

    Iteration: w_i^{new} = w_i * sqrt( target_rc / RC_i )
    where target_rc = 1/P and RC_i = w_i * (Σ w)_i / (w' Σ w)

    ===========================================================================
    PARAMETERS
    ===========================================================================

    sigma:   P x P covariance matrix (must be PSD).
    w_init:  Optional initial guess (default: inverse vol).
    max_iter: Maximum iterations.
    tol:     Convergence tolerance on max |RC_i - 1/P|.

    ===========================================================================
    RETURNS
    ===========================================================================

    P-vector of ERC weights (long-only, sum to 1).
    '''
    sigma = np.asarray(sigma, dtype=float)
    P = sigma.shape[0]

    # Initialize with inverse vol
    if w_init is None:
        vols = np.sqrt(np.diag(sigma))
        w = inverse_volatility_weights(vols)
    else:
        w = np.asarray(w_init, dtype=float)

    w = np.maximum(w, 1e-12)  # strictly positive for ERC
    w = w / np.sum(w)
    target_rc = 1.0 / P

    for iteration in range(max_iter):
        # Marginal risk contributions: MRC_i = (Σ w)_i
        mrc = sigma @ w
        # Total portfolio variance
        port_var = w @ mrc
        if port_var <= 0:
            break
        # Risk contributions: RC_i = w_i * MRC_i / port_var
        rc = w * mrc / port_var

        # Fixed-point update
        w = w * np.sqrt(target_rc / np.maximum(rc, 1e-12))
        w = w / np.sum(w)

        # Check convergence
        if np.max(np.abs(rc - target_rc)) < tol:
            break

    return w


# =============================================================================
# HIERARCHICAL RISK PARITY (HRP) - López de Prado 2016
# =============================================================================

def _get_quasi_diagonal(linkage_matrix: np.ndarray) -> list[int]:
    '''
    Get the quasi-diagonal ordering from hierarchical clustering.

    The dendrogram defines a recursive bisection order. Assets in the same
    cluster end up adjacent in the quasi-diagonal covariance matrix.
    '''
    n = linkage_matrix.shape[0] + 1  # number of original assets
    # Start with the root cluster (last row of linkage)
    clusters = [int(linkage_matrix[-1, 0]), int(linkage_matrix[-1, 1])]
    # Recursively expand clusters until we reach individual assets (indices < n)
    while any(c >= n for c in clusters):
        new_clusters = []
        for c in clusters:
            if c >= n:
                # This is a merged cluster, expand it
                row_idx = c - n
                new_clusters.extend([int(linkage_matrix[row_idx, 0]), int(linkage_matrix[row_idx, 1])])
            else:
                new_clusters.append(c)
        clusters = new_clusters
    return clusters


def _get_cluster_var(sigma: np.ndarray, cluster_indices: list[int]) -> float:
    '''
    Variance of a cluster using its sub-covariance matrix and inverse-vol weights.
    '''
    if len(cluster_indices) == 1:
        return sigma[cluster_indices[0], cluster_indices[0]]
    sub_sigma = sigma[np.ix_(cluster_indices, cluster_indices)]
    vols = np.sqrt(np.diag(sub_sigma))
    w = inverse_volatility_weights(vols)
    return float(w @ sub_sigma @ w)


def _recursive_bisection(
    sigma: np.ndarray,
    sorted_indices: list[int],
    weights: np.ndarray,
) -> np.ndarray:
    '''
    Recursive bisection allocation (HRP core).

    At each step:
      1. Split the cluster into two halves
      2. Compute each half's variance (using inverse-vol weights internally)
      3. Allocate risk budget inversely proportional to cluster variance
      4. Recurse on each half
    '''
    if len(sorted_indices) <= 1:
        return weights

    # Split into two halves
    mid = len(sorted_indices) // 2
    left = sorted_indices[:mid]
    right = sorted_indices[mid:]

    # Variance of each half
    var_left = _get_cluster_var(sigma, left)
    var_right = _get_cluster_var(sigma, right)

    # Risk budget allocation: more risk to lower-variance cluster
    alpha = 1.0 - var_left / (var_left + var_right)

    # Update weights for each half
    weights[left] *= alpha
    weights[right] *= (1.0 - alpha)

    # Recurse
    weights = _recursive_bisection(sigma, left, weights)
    weights = _recursive_bisection(sigma, right, weights)

    return weights


def hierarchical_risk_parity(
    sigma: np.ndarray,
    corr: np.ndarray | None = None,
    method: str = 'ward',
) -> np.ndarray:
    '''
    Hierarchical Risk Parity (HRP) - López de Prado (2016).

    ===========================================================================
    INTUITION
    ===========================================================================

    Standard Markowitz optimization is unstable because:
      1. It inverts Σ (amplifies noise in small eigenvalues)
      2. It requires μ (expected returns), which is estimated with huge error
      3. Small changes in μ/Σ cause large weight swings

    HRP solves this by:
      1. NO μ NEEDED -- purely risk-based allocation
      2. NO MATRIX INVERSION -- uses hierarchical clustering
      3. Uses correlation structure (our Louvain clusters!) to define hierarchy
      4. Recursive bisection allocates risk budget top-down

    Algorithm:
      1. Hierarchical clustering on correlation matrix (using our cleaned corr)
      2. Get quasi-diagonal ordering (assets in same cluster adjacent)
      3. Recursive bisection: at each split, allocate risk budget inversely
         to the cluster's variance (using inverse-vol internally)
      5. Result: weights that respect correlation structure, no inversion

    This is used in production at many firms (AQR, Man Group, etc.) as a
    robust alternative to Markowitz.

    ===========================================================================
    PARAMETERS
    ===========================================================================

    sigma:   P x P covariance matrix (used for variance calculations).
    corr:    P x P correlation matrix (used for clustering). If None, derived from sigma.
    method:  Linkage method for hierarchical clustering: 'ward', 'complete', 'average', 'single'.
             'ward' is default (minimizes variance within clusters).

    ===========================================================================
    RETURNS
    ===========================================================================

    P-vector of HRP weights (long-only, sum to 1).

    ===========================================================================
    REFERENCES
    ===========================================================================

    López de Prado, M. (2016). "Building Diversified Portfolios that Outperform
    Out-of-Sample." Journal of Portfolio Management, 42(4), 59-69.

    López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley.
    Chapter 11: Hierarchical Risk Parity.
    '''
    sigma = np.asarray(sigma, dtype=float)
    P = sigma.shape[0]

    # Use correlation for clustering (more stable)
    if corr is None:
        stds = np.sqrt(np.diag(sigma))
        stds = np.maximum(stds, 1e-8)
        corr = sigma / np.outer(stds, stds)
        np.fill_diagonal(corr, 1.0)

    corr = np.asarray(corr, dtype=float)

    # Convert correlation to distance for clustering
    # d_ij = sqrt(0.5 * (1 - corr_ij))  -- metric distance
    dist = np.sqrt(0.5 * (1.0 - corr))
    np.fill_diagonal(dist, 0.0)

    # Hierarchical clustering
    linkage_matrix = linkage(dist, method=method)

    # Get quasi-diagonal order
    sorted_idx = _get_quasi_diagonal(linkage_matrix)

    # Start with inverse-vol weights (as if one big cluster)
    vols = np.sqrt(np.diag(sigma))
    weights = inverse_volatility_weights(vols)

    # Recursive bisection
    weights = _recursive_bisection(sigma, sorted_idx, weights)

    return weights


# =============================================================================
# CLUSTERED RISK PARITY (Raffinot 2018) - uses Louvain clusters
# =============================================================================

def clustered_risk_parity(
    sigma: np.ndarray,
    cluster_map: dict[int, int] | np.ndarray,
    corr: np.ndarray | None = None,
) -> np.ndarray:
    '''
    Clustered Risk Parity - Raffinot (2018).

    ===========================================================================
    INTUITION
    ===========================================================================

    HRP builds a hierarchy from pairwise correlations. But we ALREADY HAVE
    a natural clustering from Louvain on the cleaned correlation matrix!
    Clustered Risk Parity uses this directly:

      1. Within each cluster: allocate weights via ERC (equal risk contribution)
      2. Between clusters: allocate cluster-level risk budget equally (or by
         some criterion like inverse cluster variance)

    This is more intuitive than HRP because clusters have economic meaning
    (Louvain finds communities that "move together"). It's also faster.

    ===========================================================================
    PARAMETERS
    ===========================================================================

    sigma:       P x P covariance matrix.
    cluster_map: Dict {asset_idx: cluster_id} OR array of cluster_ids per asset.
    corr:        Optional correlation matrix (for diagnostics).

    ===========================================================================
    RETURNS
    ===========================================================================

    P-vector of weights (long-only, sum to 1).
    '''
    sigma = np.asarray(sigma, dtype=float)
    P = sigma.shape[0]

    # Normalize cluster_map
    if isinstance(cluster_map, dict):
        clusters = np.array([cluster_map.get(i, -1) for i in range(P)], dtype=int)
    else:
        clusters = np.asarray(cluster_map, dtype=int)

    unique_clusters = np.unique(clusters)
    unique_clusters = unique_clusters[unique_clusters >= 0]
    n_clusters = len(unique_clusters)

    if n_clusters == 0:
        return inverse_volatility_weights(np.sqrt(np.diag(sigma)))

    # Step 1: Within each cluster, compute ERC weights
    weights = np.zeros(P)
    cluster_vars = np.zeros(n_clusters)
    cluster_assets = {}

    for i, cl in enumerate(unique_clusters):
        asset_idx = np.where(clusters == cl)[0]
        cluster_assets[cl] = asset_idx

        if len(asset_idx) == 1:
            weights[asset_idx] = 1.0
            cluster_vars[i] = sigma[asset_idx[0], asset_idx[0]]
        else:
            sub_sigma = sigma[np.ix_(asset_idx, asset_idx)]
            w_cluster = equal_risk_contribution_weights(sub_sigma)
            weights[asset_idx] = w_cluster
            cluster_vars[i] = float(w_cluster @ sub_sigma @ w_cluster)

    # Step 2: Allocate risk budget across clusters (equal risk contribution
    # at cluster level, using cluster variances)
    # Each cluster gets equal share of risk budget: 1/n_clusters
    target_rc = 1.0 / n_clusters

    # Current risk contributions of clusters
    cluster_rc = np.zeros(n_clusters)
    for i, cl in enumerate(unique_clusters):
        asset_idx = cluster_assets[cl]
        # RC of cluster = w_cluster' * Sigma_cluster * w_cluster / port_var
        # But easier: total weight in cluster * average MRC
        cluster_weight = np.sum(weights[asset_idx])
        if cluster_weight > 0:
            # Marginal contribution of cluster weight
            mrc_cluster = cluster_vars[i] / cluster_weight
            cluster_rc[i] = cluster_weight * mrc_cluster

    # Normalize to equal RC
    if np.sum(cluster_rc) > 0:
        cluster_rc = cluster_rc / np.sum(cluster_rc)
        # Scale cluster weights to achieve equal RC
        for i, cl in enumerate(unique_clusters):
            if cluster_rc[i] > 0:
                scale = target_rc / cluster_rc[i]
                asset_idx = cluster_assets[cl]
                weights[asset_idx] *= scale

    # Renormalize to sum = 1
    weights = weights / np.sum(weights)

    return weights


# =============================================================================
# MEAN-VARIANCE OPTIMIZATION
# =============================================================================

def mean_variance_optimize(
    mu: np.ndarray,
    sigma: np.ndarray,
    target_vol: float | None = None,
    target_return: float | None = None,
    max_gross: float = 1.0,
    long_only: bool = False,
    max_weight: float | None = None,
    min_weight: float | None = None,
    risk_aversion: float | None = None,
) -> np.ndarray:
    '''
    Mean-Variance Optimization (Markowitz) with constraints.

    ===========================================================================
    INTUITION
    ===========================================================================

    The classic Markowitz problem:
        max_w  w' μ - (λ/2) w' Σ w
        s.t.   sum w = 1 (or sum |w| = max_gross for dollar-neutral)
               constraints...

    We solve this via scipy.optimize. Two formulations:
      - Risk budget: target_vol given -> find max return for that vol
      - Return target: target_return given -> find min vol for that return
      - Risk aversion: λ given -> unconstrained max Sharpe (with constraints)

    CRITICAL: This requires a WELL-CONDITIONED sigma. Use the shrunk/cleaned
    covariance from covariance.py, NOT the raw sample covariance.

    ===========================================================================
    PARAMETERS
    ===========================================================================

    mu:              P-vector of expected returns (alpha signal).
    sigma:           P x P covariance matrix (must be PSD, well-conditioned).
    target_vol:      Target portfolio volatility (annualized). Mutually exclusive
                     with target_return and risk_aversion.
    target_return:   Target portfolio return. Mutually exclusive.
    max_gross:       Max gross exposure (sum |w|). Default 1.0.
    long_only:       If True, w_i >= 0. Default False (long-short).
    max_weight:      Max per-asset weight (absolute).
    min_weight:      Min per-asset weight (absolute, for long-only).
    risk_aversion:   λ in w'μ - λ/2 w'Σw. If given, target_vol/target_return ignored.

    ===========================================================================
    RETURNS
    ===========================================================================

    P-vector of optimal weights.

    ===========================================================================
    REFERENCES
    ===========================================================================

    Markowitz, H. (1952). "Portfolio Selection." Journal of Finance, 7(1), 77-91.
    Grinold & Kahn (2000), Ch. 3-4.
    '''
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    P = len(mu)

    if sigma.shape != (P, P):
        raise ValueError(f"sigma shape {sigma.shape} != mu length {P}")

    # Ensure PSD
    eigenvalues = np.linalg.eigvalsh(sigma)
    if eigenvalues.min() < -1e-8:
        sigma = sigma + np.eye(P) * (abs(eigenvalues.min()) + 1e-6)

    # Objective function
    if risk_aversion is not None:
        # max w'μ - (λ/2) w'Σw  ->  min (λ/2) w'Σw - w'μ
        lambda_ = risk_aversion
        def objective(w):
            return 0.5 * lambda_ * w @ sigma @ w - mu @ w
    elif target_vol is not None:
        # max w'μ s.t. w'Σw <= target_vol^2
        # We use a penalty method: min -w'μ + penalty * max(0, w'Σw - target_vol^2)^2
        target_var = target_vol ** 2
        def objective(w):
            port_var = w @ sigma @ w
            penalty = 1e6 * max(0, port_var - target_var) ** 2
            return -mu @ w + penalty
    elif target_return is not None:
        # min w'Σw s.t. w'μ >= target_return
        def objective(w):
            penalty = 1e6 * max(0, target_return - mu @ w) ** 2
            return 0.5 * w @ sigma @ w + penalty
    else:
        raise ValueError("Must specify one of: target_vol, target_return, risk_aversion")

    # Constraints
    constraints = []

    # Gross exposure: sum |w| <= max_gross (implemented via bounds + equality)
    # For dollar-neutral: sum w = 0
    if long_only:
        # sum w = 1 (fully invested)
        constraints.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = [(0.0, max_weight if max_weight else 1.0) for _ in range(P)]
    else:
        # Dollar-neutral: sum w = 0, sum |w| = max_gross
        # We enforce sum w = 0 and bound |w_i| <= max_gross
        constraints.append({'type': 'eq', 'fun': lambda w: np.sum(w)})
        # Gross constraint: sum |w| = max_gross -> use bounds and let optimizer find
        max_w = max_weight if max_weight else max_gross
        bounds = [(-max_w, max_w) for _ in range(P)]
        # Add gross constraint via penalty in objective (or as constraint)
        def gross_constraint(w):
            return np.sum(np.abs(w)) - max_gross
        constraints.append({'type': 'eq', 'fun': gross_constraint})

    if min_weight is not None and not long_only:
        # For long-short, min_weight applies to absolute value
        bounds = [(max(-max_w, -min_weight), max(max_w, min_weight)) for _ in range(P)]

    # Initial guess: inverse vol
    vols = np.sqrt(np.diag(sigma))
    w0 = inverse_volatility_weights(vols)
    if not long_only:
        w0 = w0 - np.mean(w0)  # dollar-neutral

    # Solve
    result = minimize(
        objective, w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )

    if not result.success:
        import warnings
        warnings.warn(f"Optimization failed: {result.message}", UserWarning)

    w_opt = result.x

    # Post-process: enforce exact constraints
    w_opt = np.clip(w_opt, -max_w, max_w) if not long_only else np.clip(w_opt, 0, max_w)
    if not long_only:
        w_opt = w_opt - np.mean(w_opt)  # enforce dollar-neutral
        gross = np.sum(np.abs(w_opt))
        if gross > 0:
            w_opt = w_opt / gross * max_gross

    return w_opt


# =============================================================================
# BLACK-LITTERMAN
# =============================================================================

def black_litterman_weights(
    sigma: np.ndarray,
    mu_prior: np.ndarray,
    views: np.ndarray,
    pick_matrix: np.ndarray,
    omega: np.ndarray | None = None,
    tau: float = 0.05,
    risk_aversion: float = 1.0,
) -> np.ndarray:
    '''
    Black-Litterman (1992) portfolio optimization.

    ===========================================================================
    INTUITION
    ===========================================================================

    Problem: Markowitz needs μ, but μ is estimated with huge error.
    BL solution: Start with a market equilibrium prior μ_prior (from CAPM),
    then tilt it toward your views (your alpha signal) with confidence Ω.

    Formula:
        μ_BL = [ (τ Σ)^{-1} + P' Ω^{-1} P ]^{-1} [ (τ Σ)^{-1} μ_prior + P' Ω^{-1} Q ]

    Where:
      - τ: scalar scaling the prior uncertainty (small = trust prior more)
      - P: pick matrix (K x P), maps assets to views
      - Q: K-vector of view returns (your signal's expected returns)
      - Ω: K x K covariance of view uncertainties (diagonal = view confidence)

    Then optimize with μ_BL and Σ as in standard mean-variance.

    In our context:
      - μ_prior = equilibrium returns from market cap weights (or CAPM beta)
      - Q = our signal's predicted returns (e.g., from Elastic Net)
      - P = identity (we have a view on every asset) or selector matrix
      - Ω = diagonal with variances reflecting signal confidence (e.g., q-values)

    ===========================================================================
    PARAMETERS
    ===========================================================================

    sigma:          P x P covariance matrix.
    mu_prior:       P-vector of prior expected returns (equilibrium).
    views:          K-vector of view expected returns (our signal).
    pick_matrix:    K x P matrix mapping assets to views.
    omega:          K x K view covariance. If None, uses τ * P Σ P' (default).
    tau:            Prior scaling (default 0.05, small = trust prior).
    risk_aversion:  λ for final mean-variance optimization.

    ===========================================================================
    RETURNS
    ===========================================================================

    P-vector of BL-optimal weights.
    '''
    sigma = np.asarray(sigma, dtype=float)
    mu_prior = np.asarray(mu_prior, dtype=float)
    views = np.asarray(views, dtype=float)
    P = np.asarray(pick_matrix, dtype=float)

    K = P.shape[0]
    P_dim = P.shape[1]

    if omega is None:
        # Default: Ω = τ * P Σ P'
        omega = tau * P @ sigma @ P.T

    omega = np.asarray(omega, dtype=float)

    # BL formula
    tau_sigma_inv = np.linalg.inv(tau * sigma)
    P_omega_inv = P.T @ np.linalg.inv(omega)

    mu_bl = np.linalg.inv(tau_sigma_inv + P_omega_inv @ P) @ \
            (tau_sigma_inv @ mu_prior + P_omega_inv @ views)

    # Final optimization with BL posterior mean
    w_bl = mean_variance_optimize(
        mu_bl, sigma, risk_aversion=risk_aversion, long_only=False
    )

    return w_bl


# =============================================================================
# COST-AWARE OPTIMIZATION (Gârleanu & Pedersen 2013)
# =============================================================================

def cost_aware_optimize(
    mu: np.ndarray,
    sigma: np.ndarray,
    w_prev: np.ndarray,
    Lambda: np.ndarray,  # diagonal cost matrix from costs.py
    target_vol: float | None = None,
    risk_aversion: float | None = None,
    max_gross: float = 1.0,
    long_only: bool = False,
) -> np.ndarray:
    '''
    Cost-aware optimization: max w'μ - 0.5 w' (Σ + Λ) w + w' Λ w_prev.

    ===========================================================================
    INTUITION
    ===========================================================================

    Standard MVO ignores turnover costs. Gârleanu & Pedersen (2013) show that
    the optimal dynamic trading problem with quadratic costs reduces to a
    static MVO with modified risk matrix:

        Max: w' μ - 0.5 w' (Σ + Λ) w + w' Λ w_prev

    Where Λ is a diagonal matrix with Λ_ii encoding the cost of changing
    position in asset i (from costs.cost_penalty_quadratic_form).

    Intuition: The cost penalty adds Λ to the risk matrix (making risky/
    illiquid assets look riskier) and adds a "pull" toward w_prev (the
    linear term Λ w_prev).

    This is the standard approach for cost-aware optimization at top firms.

    ===========================================================================
    PARAMETERS
    ===========================================================================

    mu:          P-vector of expected returns.
    sigma:       P x P covariance matrix.
    w_prev:      P-vector of current weights.
    Lambda:      P-vector of diagonal cost penalty (from costs.py).
    target_vol:  Target portfolio vol (if None, uses risk_aversion).
    risk_aversion: Risk aversion λ.
    max_gross:   Max gross exposure.
    long_only:   Long-only constraint.

    ===========================================================================
    RETURNS
    ===========================================================================

    P-vector of optimal weights.
    '''
    sigma = np.asarray(sigma, dtype=float)
    mu = np.asarray(mu, dtype=float)
    w_prev = np.asarray(w_prev, dtype=float)
    Lambda = np.asarray(Lambda, dtype=float)

    P = len(mu)
    if sigma.shape != (P, P) or w_prev.shape != (P,) or Lambda.shape != (P,):
        raise ValueError("Dimension mismatch")

    # Modified risk matrix: Σ + Λ (Λ is diagonal)
    sigma_tilde = sigma + np.diag(Lambda)

    # Modified expected returns: μ + Λ w_prev
    mu_tilde = mu + Lambda * w_prev

    # Solve with modified parameters
    w_opt = mean_variance_optimize(
        mu_tilde, sigma_tilde,
        target_vol=target_vol,
        risk_aversion=risk_aversion,
        max_gross=max_gross,
        long_only=long_only,
    )

    return w_opt


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def risk_contributions(weights: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    '''
    Marginal and proportional risk contributions per asset.

    RC_i = w_i * (Σ w)_i / (w' Σ w)
    '''
    weights = np.asarray(weights, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    mrc = sigma @ weights
    port_var = weights @ mrc
    if port_var <= 0:
        return np.zeros_like(weights)

    rc = weights * mrc / port_var
    return rc


def factor_contributions(weights: np.ndarray, sigma: np.ndarray, factor_loadings: np.ndarray) -> np.ndarray:
    '''
    Factor risk contributions using factor loadings B (P x K).
    Factor RC = (B' w)_k^2 / (w' Σ w)
    '''
    weights = np.asarray(weights, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    B = np.asarray(factor_loadings, dtype=float)

    port_var = weights @ sigma @ weights
    if port_var <= 0:
        return np.zeros(B.shape[1])

    factor_weights = B.T @ weights  # K-vector
    fc = (factor_weights ** 2) / port_var
    return fc


def portfolio_var_decomposition(weights: np.ndarray, sigma: np.ndarray) -> dict:
    '''
    Decompose portfolio variance into contributions.
    '''
    weights = np.asarray(weights, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    mrc = sigma @ weights
    port_var = weights @ mrc
    rc = weights * mrc / port_var

    # Top contributors
    sorted_idx = np.argsort(rc)[::-1]

    return {
        'total_variance': float(port_var),
        'total_vol': float(np.sqrt(port_var)),
        'risk_contributions': rc,
        'top_contributors': [(int(i), float(rc[i])) for i in sorted_idx[:10]],
    }