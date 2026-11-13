'''
Covariance estimation for portfolio construction: shrinkage and factor models.

WHY THIS FILE EXISTS
--------------------
The sample covariance of P stocks over N days is noisy when N is not much larger than P.
Its extreme eigenvalues are too large or too small, and the optimiser then bets on noise.
Every method below returns a covariance (or correlation) matrix that is more stable.

THE METHODS (all take returns as an array of shape P x N: one row per asset)

    Method                      Function                          Shrinks toward
    --------------------------  --------------------------------  ---------------------------------
    Ledoit-Wolf (2004, 2003)    ledoit_wolf_shrinkage             identity, or constant correlation
    OAS (Chen et al. 2010)      oracle_approximating_shrinkage    identity
    Factor model (PCA)          factor_model_covariance           K common factors + own idio. variance
    Fixed intensity             shrink_towards_constant_correlation   constant correlation, user delta

SHRINKAGE IN ONE PICTURE

    sample correlation C  +  target F (identity, or constant correlation)
                    |
                    v
        C_shrunk = rho * F + (1 - rho) * C          rho in [0, 1]

        rho = 0  no shrinkage              rho = 1  only the target
        rho is computed from the data (Ledoit-Wolf, OAS) so that the expected error is minimal.

WHERE THE INTENSITY IS COMPUTED: ON STANDARDISED RETURNS
    Each return series is divided by its own standard deviation first. The sample covariance of the
    standardised series is the sample correlation. The intensity is then the optimal one for the
    CORRELATION matrix. This matters: the shrinkage is applied to the correlation, so the intensity
    must be optimal for the correlation. A covariance-space intensity would give more weight to the
    stocks with large variance.

CONVENTIONS
    - Sample covariance uses the denominator N (not N - 1), as in the Ledoit-Wolf and OAS papers.
    - Returns must be finite. Handle missing values before (see src/risk/risk_model.py).
    - A series with zero variance gets a zero row in the standardised data (correlation 0 with the rest).
    - Covariances returned have the same units as the input returns (daily in, daily out).

HISTORY OF FIXES (read this if you compare with an older version)
    - Ledoit-Wolf: the intensity was missing the division by N, so it was clipped to 1 almost always.
    - OAS: the formula was not the OAS formula (the intensity was about 0.003).
    - Factor model: the number of factors used a wrong penalty, and all stocks received the same
      idiosyncratic variance, which distorted the volatilities. Now the diagonal of the sample
      covariance is kept exactly.

REFERENCES
    Ledoit, O., & Wolf, M. (2003). Improved estimation of the covariance matrix of stock returns with an
        application to portfolio selection. Journal of Empirical Finance, 10(5), 603-621.
        (constant-correlation target)
    Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices.
        Journal of Multivariate Analysis, 88(2), 365-411.  (identity target)
    Chen, Y., Wiesel, A., Eldar, Y., & Hero, A. (2010). Shrinkage algorithms for MMSE covariance estimation.
        IEEE Transactions on Signal Processing, 58(10), 5016-5029.  (OAS)
    Bai, J., & Ng, S. (2002). Determining the number of factors in approximate factor models.
        Econometrica, 70(1), 191-221.
    Connor, G., & Korajczyk, R. (1986). Performance measurement with the arbitrage pricing theory.
        Journal of Financial Economics, 15(2), 373-394.
'''

import numpy as np


# =============================================================================
# HELPERS
# =============================================================================

def _symmetrize(M: np.ndarray) -> np.ndarray:
    '''(M + M') / 2. Removes the tiny asymmetries (1e-16) that eigen-decompositions can create.'''
    return 0.5 * (M + M.T)


def _psd_project(M: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    '''
    Nearest positive semi-definite matrix by eigenvalue clipping: eigen-decompose, raise every eigenvalue
    below `floor` to `floor`, rebuild. A matrix that is already PSD comes back unchanged (to 1e-16).
    A covariance that is not PSD makes a negative variance possible, so the optimiser needs this guarantee.
    '''
    M = _symmetrize(M)
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    eigenvalues = np.clip(eigenvalues, floor, None)
    return _symmetrize((eigenvectors * eigenvalues) @ eigenvectors.T)


def _as_returns(returns) -> np.ndarray:
    '''Return a float array P x N (a 1-D input is one asset).'''
    X = np.asarray(returns, dtype=float)
    return X.reshape(1, -1) if X.ndim == 1 else X


def _standardise(returns) -> np.ndarray:
    '''
    Centre each row and divide it by its standard deviation (denominator N).
    Result Z has Z @ Z' / N = sample correlation (unit diagonal for every non-constant row).
    '''
    X = _as_returns(returns)
    Xc = X - X.mean(axis=1, keepdims=True)
    std = np.sqrt((Xc ** 2).mean(axis=1, keepdims=True))
    return np.divide(Xc, std, out=np.zeros_like(Xc), where=std > 1e-14)


# =============================================================================
# 1. LEDOIT-WOLF
# =============================================================================

def _lw_intensity_identity(Z: np.ndarray) -> float:
    '''
    Optimal intensity toward the identity (Ledoit-Wolf 2004), for standardised returns Z (P x N).

        S      sample correlation = Z Z' / N
        delta  = sum_{i != j} S_ij^2                   distance between S and the target I
        gamma  = (1 / N) * sum_{i != j} pi_ij          estimation noise of S
        pi_ij  = mean_k (z_ik z_jk - S_ij)^2           variance of one product z_i z_j
        rho    = gamma / delta, clipped to [0, 1]

    The division by N is essential: pi_ij is the variance of ONE observation. The noise of the AVERAGE of
    N observations is N times smaller. The diagonal is left out because the sample correlation has an exact
    unit diagonal: it carries no estimation error.
    '''
    n_obs = Z.shape[1]
    if n_obs < 2:
        return 1.0
    S = (Z @ Z.T) / n_obs
    Z2 = Z ** 2
    pi = np.maximum((Z2 @ Z2.T) / n_obs - S ** 2, 0.0)
    off = ~np.eye(S.shape[0], dtype=bool)
    delta = np.sum(S[off] ** 2)
    gamma = pi[off].sum() / n_obs
    if not np.isfinite(delta) or delta <= 0:
        return 0.0
    return float(np.clip(gamma / delta, 0.0, 1.0))


def _lw_intensity_constant_correlation(Z: np.ndarray) -> float:
    '''
    Optimal intensity toward the constant-correlation matrix (Ledoit-Wolf 2003), standardised returns Z.

    Target F: unit diagonal, every off-diagonal equal to rbar, the average sample correlation.
    The target is estimated from the same data, so its error is correlated with the error of S.
    The formula has three parts (kappa = (pi - rho) / gamma, intensity = kappa / N):
        pi     sum of the variances of the entries of S
        rho    sum of the covariances between the entries of S and the entries of F
        gamma  squared distance between S and F
    '''
    p, n_obs = Z.shape
    if n_obs < 2 or p < 2:
        return 1.0
    S = (Z @ Z.T) / n_obs
    off = ~np.eye(p, dtype=bool)
    rbar = S[off].mean()
    F = np.full((p, p), rbar)
    np.fill_diagonal(F, 1.0)

    Z2 = Z ** 2
    phi = np.maximum((Z2 @ Z2.T) / n_obs - S ** 2, 0.0)          # variance of each entry of S (times N)
    pi_hat = phi.sum()

    # theta_ij = Cov(sqrt(N) S_ii, sqrt(N) S_ij), with S_ii = 1 for standardised data
    theta = ((Z ** 3) @ Z.T) / n_obs - S                         # variances are 1: the term is S_ij itself
    np.fill_diagonal(theta, 0.0)
    rho_hat = np.trace(phi) + rbar * theta.sum()                 # sqrt(var_j / var_i) = 1 after standardising
    # theta is not symmetric: the sum over all (i, j) already counts both orders, as the formula requires.

    gamma_hat = np.sum((S - F) ** 2)
    if gamma_hat <= 0 or not np.isfinite(gamma_hat):
        return 0.0
    kappa = (pi_hat - rho_hat) / gamma_hat
    return float(np.clip(kappa / n_obs, 0.0, 1.0))


def ledoit_wolf_shrinkage(
    corr: np.ndarray,
    returns: "np.ndarray | None" = None,
    target: str = 'identity',
) -> "tuple[np.ndarray, float]":
    '''
    Ledoit-Wolf linear shrinkage of a correlation matrix.

        C_shrunk = rho * F + (1 - rho) * C

    corr:    P x P correlation matrix to shrink (raw, or already cleaned by random matrix theory).
    returns: P x N returns. Needed to compute rho. If None, rho = 0.5 (a conservative default).
    target:  'identity'             F = I. All off-diagonal correlations pulled to 0.
             'constant_correlation' F = average correlation everywhere. Keeps the market factor.
             Use the second one if the stocks share a strong common factor (equities do).

    Returns (C_shrunk, rho). The diagonal of C_shrunk is exactly 1, the matrix is PSD.
    The intensity rho is estimated on the standardised returns (see the file header).
    '''
    corr = _symmetrize(np.asarray(corr, dtype=float))
    p = corr.shape[0]
    if target not in ('identity', 'constant_correlation'):
        raise ValueError(f"unknown target: {target!r}. Use 'identity' or 'constant_correlation'.")

    if target == 'identity':
        F = np.eye(p)
    else:
        F = np.full((p, p), corr[~np.eye(p, dtype=bool)].mean())
        np.fill_diagonal(F, 1.0)

    if returns is None:
        rho = 0.5
    else:
        Z = _standardise(returns)
        rho = _lw_intensity_identity(Z) if target == 'identity' else _lw_intensity_constant_correlation(Z)

    shrunk = rho * F + (1.0 - rho) * corr
    shrunk = _psd_project(shrunk)
    np.fill_diagonal(shrunk, 1.0)
    return shrunk, float(rho)


# =============================================================================
# 2. OAS (ORACLE APPROXIMATING SHRINKAGE)
# =============================================================================

def oracle_approximating_shrinkage(
    corr: np.ndarray,
    returns: "np.ndarray | None" = None,
) -> "tuple[np.ndarray, float]":
    '''
    OAS shrinkage toward the identity (Chen, Wiesel, Eldar, Hero 2010).

    OAS has a closed form that needs only two numbers of the sample correlation S (P x P, N observations):
        tr(S^2) = sum_ij S_ij^2        tr(S)^2 = P^2   (unit diagonal)

        rho = ((1 - 2/P) * tr(S^2) + tr(S)^2) / ((N + 1 - 2/P) * (tr(S^2) - tr(S)^2 / P))     clipped to [0, 1]

    It assumes Gaussian returns, and it is derived for small N relative to P. Fat tails make it
    less accurate than Ledoit-Wolf; compare the two on your data.

    corr:    P x P correlation matrix to shrink.
    returns: P x N returns. Needed for rho. If None, rho = 0.5.
    Returns (C_shrunk, rho). The diagonal is exactly 1, the matrix is PSD.
    '''
    corr = _symmetrize(np.asarray(corr, dtype=float))
    p = corr.shape[0]

    if returns is None:
        rho = 0.5
    else:
        Z = _standardise(returns)
        n_obs = Z.shape[1]
        S = (Z @ Z.T) / n_obs
        tr_s2 = float(np.sum(S ** 2))
        tr_s_sq = float(np.trace(S) ** 2)
        num = (1.0 - 2.0 / p) * tr_s2 + tr_s_sq
        den = (n_obs + 1.0 - 2.0 / p) * (tr_s2 - tr_s_sq / p)
        rho = float(np.clip(num / den, 0.0, 1.0)) if den > 0 and np.isfinite(num / den) else 1.0

    shrunk = _psd_project(rho * np.eye(p) + (1.0 - rho) * corr)
    np.fill_diagonal(shrunk, 1.0)
    return shrunk, float(rho)


# =============================================================================
# 3. FACTOR MODEL (PCA)
# =============================================================================

def _bai_ng_n_factors(eigenvalues_desc: np.ndarray, n_assets: int, n_obs: int, max_k: int = 20) -> int:
    '''
    Number of factors by the Bai-Ng (2002) criterion IC_p2.

        IC(k) = ln V(k) + k * (N + T) / (N * T) * ln(min(N, T))
        V(k)  = (1 / N) * sum of the eigenvalues beyond the first k   (average residual variance)

    The k with the lowest IC wins. k = 0 is not allowed: a covariance needs at least one factor here.
    '''
    max_k = int(max(1, min(max_k, n_assets - 1, n_obs - 1)))
    penalty = (n_assets + n_obs) / (n_assets * n_obs) * np.log(min(n_assets, n_obs))
    tail = np.cumsum(eigenvalues_desc[::-1])[::-1]                    # tail[k] = sum of eigenvalues k..end
    best_k, best_ic = 1, np.inf
    for k in range(1, max_k + 1):
        v = tail[k] / n_assets
        if v <= 0:
            continue
        ic = np.log(v) + k * penalty
        if ic < best_ic:
            best_k, best_ic = k, ic
    return best_k


def factor_model_covariance(
    returns: np.ndarray,
    n_factors: "int | None" = None,
    method: str = 'fixed',
) -> "tuple[np.ndarray, np.ndarray, int]":
    '''
    Covariance from K principal-component factors plus a diagonal idiosyncratic part.

        S      = sample covariance (P x P, denominator N)
        B, F   = the K largest eigenvectors (P x K) and eigenvalues (K x K diagonal) of S
        D      = diag(S - B F B')            what the K factors leave unexplained, stock by stock
        Sigma  = B F B' + D

    Properties: Sigma is PSD (BFB' is PSD and D >= 0), and diag(Sigma) = diag(S) exactly,
    so every stock keeps its own sample variance. Only the correlations are regularised: they are
    forced into a K-factor structure. The noisy small eigenvalues of S are gone.

    returns:   P x N. Same units come out (daily in, daily out).
    n_factors: K. None = chosen by the Bai-Ng criterion.
    method:    'fixed' (use n_factors) or 'bic' (always choose K by Bai-Ng, even if n_factors is given).

    Returns (Sigma, loadings B (P x K), K used).
    '''
    X = _as_returns(returns)
    p, n_obs = X.shape
    Xc = X - X.mean(axis=1, keepdims=True)
    S = (Xc @ Xc.T) / n_obs

    eigenvalues, eigenvectors = np.linalg.eigh(S)                     # ascending
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    if n_factors is None or method == 'bic':
        n_factors = _bai_ng_n_factors(eigenvalues[::-1], p, n_obs)
    k = int(max(1, min(n_factors, p, n_obs)))

    B = eigenvectors[:, -k:]
    F = np.diag(eigenvalues[-k:])
    common = B @ F @ B.T
    idio = np.clip(np.diag(S) - np.diag(common), 0.0, None)
    sigma = _symmetrize(common + np.diag(idio))
    return sigma, B, k


# =============================================================================
# 4. FIXED-INTENSITY SHRINKAGE TOWARD CONSTANT CORRELATION
# =============================================================================

def shrink_towards_constant_correlation(corr: np.ndarray, delta: float) -> np.ndarray:
    '''
    C_shrunk = delta * F + (1 - delta) * C, F = constant-correlation matrix (average off-diagonal of C).
    delta is chosen by the user. For a data-driven delta use ledoit_wolf_shrinkage(target='constant_correlation').
    Returns a PSD matrix with unit diagonal.
    '''
    corr = _symmetrize(np.asarray(corr, dtype=float))
    p = corr.shape[0]
    delta = float(np.clip(delta, 0.0, 1.0))
    F = np.full((p, p), corr[~np.eye(p, dtype=bool)].mean())
    np.fill_diagonal(F, 1.0)
    shrunk = _psd_project(delta * F + (1.0 - delta) * corr)
    np.fill_diagonal(shrunk, 1.0)
    return shrunk


# =============================================================================
# 5. CORRELATION + VOLATILITIES -> COVARIANCE
# =============================================================================

def build_covariance_matrix(corr: np.ndarray, vols: np.ndarray) -> np.ndarray:
    '''
    Sigma = D C D with D = diag(vols):  Sigma_ij = vol_i * vol_j * C_ij.
    Correlation is more stable over time than volatility, and the two can be estimated separately.

    corr: P x P correlation. vols: P standard deviations (same time unit as the covariance wanted).
    A volatility below 1e-8 is raised to 1e-8. Returns a symmetric PSD matrix.
    '''
    corr = _symmetrize(np.asarray(corr, dtype=float))
    vols = np.asarray(vols, dtype=float).reshape(-1)
    if vols.shape[0] != corr.shape[0]:
        raise ValueError(f'vols length {vols.shape[0]} != corr dimension {corr.shape[0]}')
    vols = np.maximum(vols, 1e-8)
    return _psd_project(corr * np.outer(vols, vols))


# =============================================================================
# 6. FROM RETURNS TO COVARIANCE IN ONE CALL
# =============================================================================

def estimate_covariance(
    returns: np.ndarray,
    method: str = 'ledoit_wolf',
    target: str = 'identity',
    n_factors: "int | None" = None,
    delta: "float | None" = None,
) -> "tuple[np.ndarray, dict]":
    '''
    Returns (P x N) -> covariance (P x P) and a dict of diagnostics.

        returns -> sample correlation C and volatilities
                -> shrink C   ('ledoit_wolf', 'oas', 'constant_corr')   or   factor model ('factor')   or   nothing ('sample')
                -> Sigma = D C_shrunk D

    method:    'sample', 'ledoit_wolf', 'oas', 'constant_corr' or 'factor'.
    target:    for 'ledoit_wolf': 'identity' or 'constant_correlation' (None is read as 'identity').
    n_factors: for 'factor' (None = Bai-Ng).
    delta:     for 'constant_corr' (None = 0.5).

    info: {'method', 'n_assets', 'n_obs', 'rho' (intensity, or delta, or None), 'target', 'n_factors'}.
    Variances of the sample are kept on the diagonal for every method (Sigma_ii = sample variance).
    '''
    X = _as_returns(returns)
    if not np.isfinite(X).all():
        raise ValueError('returns contain NaN or inf: clean them before estimating the covariance')
    p, n_obs = X.shape
    Xc = X - X.mean(axis=1, keepdims=True)
    S = (Xc @ Xc.T) / n_obs
    stds = np.maximum(np.sqrt(np.diag(S)), 1e-8)
    corr = S / np.outer(stds, stds)
    np.fill_diagonal(corr, 1.0)

    info = {'method': method, 'n_assets': p, 'n_obs': n_obs, 'rho': None, 'target': None, 'n_factors': None}

    if method == 'sample':
        info['rho'] = 0.0
        sigma = build_covariance_matrix(corr, stds)

    elif method == 'ledoit_wolf':
        target = target or 'identity'
        corr_s, rho = ledoit_wolf_shrinkage(corr, returns=Xc, target=target)
        info.update(rho=rho, target=target)
        sigma = build_covariance_matrix(corr_s, stds)

    elif method == 'oas':
        corr_s, rho = oracle_approximating_shrinkage(corr, returns=Xc)
        info.update(rho=rho, target='identity')
        sigma = build_covariance_matrix(corr_s, stds)

    elif method == 'constant_corr':
        delta = 0.5 if delta is None else delta
        corr_s = shrink_towards_constant_correlation(corr, delta)
        info.update(rho=float(delta), target='constant_correlation')
        sigma = build_covariance_matrix(corr_s, stds)

    elif method == 'factor':
        sigma, _, k = factor_model_covariance(Xc, n_factors=n_factors)
        info['n_factors'] = k
        sigma = _psd_project(sigma)

    else:
        raise ValueError(f"unknown method: {method!r}. Use 'sample', 'ledoit_wolf', 'oas', 'constant_corr' or 'factor'.")

    return sigma, info