# src/risk/covariance_cleaning.py

import numpy as np
import polars as pl


def absorption_ratio(corr: np.ndarray, n_factors: int) -> float:
    '''
    Kritzman & Li (2010) Absorption Ratio: proportion of total variance explained by the top `n_factors` eigenvalues of the correlation matrix.

    A rising absorption ratio over time indicates the universe's return structure is collapsing onto fewer common factors,
        historically associated with elevated systemic risk / preceding market stress.
    '''

    '''
    TODO: implement. Intended to be computed on a rolling basis over
    the cross-sectional correlation matrix of `logret` across tickers.
    '''
    
    raise NotImplementedError