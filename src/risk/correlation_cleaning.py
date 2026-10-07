#  TO CHECK

import numpy as np
import polars as pl


def marchenko_pastur_bounds(n_assets: int, n_obs: int, variance: float = 1.0) -> tuple[float, float]:
    '''
    Theoretical (lambda_min, lambda_max) eigenvalue bounds for a correlation matrix built from pure noise, 
        given the assets-to-observations ratio q = n_assets / n_obs.
    Eigenvalues falling inside this range are statistically indistinguishable from noise and should be shrunk/removed when cleaning an empirical
    correlation matrix.

    Reference: Laloux, Cizeau, Bouchaud, Potters (1999). Noise dressing of
    financial correlation matrices. Physical Review Letters, 83(7), 1467.
    Marcenko & Pastur (1967) originally derived the limiting eigenvalue
    distribution of a Wishart matrix built from n_obs i.i.d. draws of
    n_assets random variables, Laloux et al. were among the first to
    apply this directly to empirical financial correlation matrices: any
    eigenvalue that a matrix of PURE NOISE could equally well have
    produced carries no genuine cross-sectional information, and with a
    real equity universe q = n_assets/n_obs is rarely small (a few
    hundred names, a few years of daily history), so a large fraction of
    the empirical spectrum typically falls inside this band, not a
    corner case to special-case around, the expected common case.

    variance=1.0 is the correct default for a CORRELATION matrix
    specifically (unit diagonal, average eigenvalue exactly 1 by
    construction, trace = n_assets). If this were ever called on a raw
    COVARIANCE matrix instead, variance would need to be the average
    variance across assets, not 1, worth remembering if this function
    ever gets reused outside clean_correlation_matrix() below.
    '''
    q = n_assets / n_obs
    lambda_min = variance * (1 - np.sqrt(q)) ** 2
    lambda_max = variance * (1 + np.sqrt(q)) ** 2
    return lambda_min, lambda_max


def clean_correlation_matrix(corr: np.ndarray, n_obs: int) -> np.ndarray:
    '''
    Denoise an empirical correlation matrix via RMT eigenvalue filtering: eigenvalues below the Marchenko-Pastur upper bound are replaced
        (commonly with their average, to preserve the trace / total variance), eigenvalues above it are kept as genuine signal.

    Only lambda_max is used, not lambda_min: financial correlation
    matrices are constrained to be positive semi-definite (all
    eigenvalues >= 0) by construction, MP's lower bound can in principle
    dip below zero for large q, not a real constraint to enforce here,
    the interesting boundary for "is this eigenvalue distinguishable
    from noise" is entirely on the upper side. This asymmetry is why the
    function signature above returns both bounds but this one only
    consumes lambda_max, worth knowing it's not an oversight.

    Eigenvalue replacement, not deletion: zeroing out the noise
    eigenvalues entirely would shrink the matrix's trace (total
    variance) below n_assets, silently claiming the noisy components
    carry zero variance, which isn't true, they carry variance, just not
    variance that's DISTINGUISHABLE FROM RANDOM across assets.
    Replacing them with their shared average preserves the trace exactly
    while removing the (spurious) claim that any one of them is more
    informative than the others, standard practice, matches the
    docstring's own stated convention.
    '''
    n_assets = corr.shape[0]
    _, lambda_max = marchenko_pastur_bounds(n_assets, n_obs, variance=1.0)

    eigenvalues, eigenvectors = np.linalg.eigh(corr)  # eigh: corr is symmetric, ascending order guaranteed
    is_noise = eigenvalues <= lambda_max

    if is_noise.all():
        raise ValueError(
            f'every eigenvalue (max {eigenvalues.max():.4f}) falls at or below the MP upper bound '
            f'({lambda_max:.4f}), n_obs={n_obs} may be too small relative to n_assets={n_assets} '
            'for this matrix to contain any detectable signal, check the inputs before proceeding'
        )

    cleaned_eigenvalues = eigenvalues.copy()
    cleaned_eigenvalues[is_noise] = eigenvalues[is_noise].mean()

    corr_clean = eigenvectors @ np.diag(cleaned_eigenvalues) @ eigenvectors.T

    # eigenvalue replacement perturbs the diagonal away from exactly 1
    # (a correlation matrix's defining property), rescale back to a true
    # correlation matrix rather than leaving a "correlation-like" object
    d = np.sqrt(np.diag(corr_clean))
    corr_clean = corr_clean / np.outer(d, d)

    return corr_clean