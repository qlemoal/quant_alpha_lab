import numpy as np
import polars as pl


def marchenko_pastur_bounds(n_assets: int, n_obs: int, variance: float = 1.0) -> tuple[float, float]:
    '''
    Theoretical (lambda_min, lambda_max) eigenvalue bounds for a correlation matrix built from pure noise, 
        given the assets-to-observations ratio q = n_assets / n_obs.
    Eigenvalues falling inside this range are statistically indistinguishable from noise and should be shrunk/removed when cleaning an empirical
    correlation matrix.
    '''

    '''
    TODO: implement.
    Reference: Laloux, Cizeau, Bouchaud, Potters (1999/2000).
    '''

    raise NotImplementedError


def clean_correlation_matrix(corr: np.ndarray, n_obs: int) -> np.ndarray:
    '''
    Denoise an empirical correlation matrix via RMT eigenvalue filtering: eigenvalues below the Marchenko-Pastur upper bound are replaced
        (commonly with their average, to preserve the trace / total variance), eigenvalues above it are kept as genuine signal.
    '''

    '''
    TODO: implement. Depends on marchenko_pastur_bounds().
    '''

    raise NotImplementedError

