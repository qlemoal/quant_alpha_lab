'''Tests for src/risk/covariance.py: shrinkage intensities, factor model, and the pipeline wrapper.'''

import numpy as np
import pytest

from src.risk import covariance as cv


def _returns(p=60, n=200, seed=0, common=1.0):
    '''Gaussian returns with one common factor: P x N.'''
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.01, (p, n)) + common * rng.normal(0, 0.01, (1, n))


def test_lw_identity_matches_sklearn():
    from sklearn.covariance import ledoit_wolf_shrinkage as sk_lw
    Z = cv._standardise(_returns(80, 300))
    # sklearn includes the diagonal in the noise term: the two agree to O(1/P)
    assert cv._lw_intensity_identity(Z) == pytest.approx(sk_lw(Z.T, assume_centered=True), rel=0.05)


def test_oas_matches_sklearn():
    from sklearn.covariance import oas as sk_oas
    X = _returns(150, 250)
    _, rho = cv.oracle_approximating_shrinkage(np.corrcoef(X), X)
    _, rho_sk = sk_oas(cv._standardise(X).T, assume_centered=True)
    assert rho == pytest.approx(rho_sk, rel=0.01)


def test_intensity_not_clipped_to_one():
    '''Regression test: the old Ledoit-Wolf lacked the division by N and returned 1 almost always.'''
    X = _returns(50, 250, common=1.0)           # true correlation 0.5: far from the identity target
    _, rho = cv.ledoit_wolf_shrinkage(np.corrcoef(X), X)
    assert 0.0 < rho < 0.5
    X0 = _returns(50, 250, common=0.0)          # independent series: the identity IS the truth, rho must be large
    assert cv.ledoit_wolf_shrinkage(np.corrcoef(X0), X0)[1] > 0.8


def test_constant_correlation_target_detects_constant_structure():
    '''True correlation is constant: the constant-correlation target is exact, so rho must be about 1.'''
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 200)) + rng.normal(size=(1, 200))
    _, rho = cv.ledoit_wolf_shrinkage(np.corrcoef(X), X, target='constant_correlation')
    assert rho > 0.8


def test_shrunk_matrix_is_valid_correlation():
    X = _returns()
    for fn in (lambda C: cv.ledoit_wolf_shrinkage(C, X)[0],
               lambda C: cv.ledoit_wolf_shrinkage(C, X, 'constant_correlation')[0],
               lambda C: cv.oracle_approximating_shrinkage(C, X)[0]):
        S = fn(np.corrcoef(X))
        assert np.abs(np.diag(S) - 1).max() < 1e-12
        assert np.abs(S - S.T).max() < 1e-12
        assert np.linalg.eigvalsh(S).min() > 0


def test_shrinkage_reduces_error_against_known_truth():
    rng = np.random.default_rng(3)
    p, n = 80, 120
    b = rng.uniform(0.2, 0.8, p)
    C = np.outer(b, b); np.fill_diagonal(C, 1.0)
    L = np.linalg.cholesky(C)
    err_sample, err_lw = [], []
    for _ in range(20):
        X = L @ rng.normal(size=(p, n))
        S = np.corrcoef(X)
        err_sample.append(np.sum((S - C) ** 2))
        err_lw.append(np.sum((cv.ledoit_wolf_shrinkage(S, X, 'constant_correlation')[0] - C) ** 2))
    assert np.mean(err_lw) < np.mean(err_sample)


def test_factor_model_keeps_variances_and_is_psd():
    X = _returns(40, 300) * np.random.default_rng(2).uniform(0.5, 2, (40, 1))
    sigma, B, k = cv.factor_model_covariance(X, n_factors=3)
    S = np.cov(X, ddof=0)
    assert k == 3 and B.shape == (40, 3)
    assert np.abs(np.diag(sigma) - np.diag(S)).max() < 1e-12
    assert np.linalg.eigvalsh(sigma).min() > -1e-12


def test_bai_ng_finds_one_strong_factor():
    _, _, k = cv.factor_model_covariance(_returns(60, 400, common=3.0))
    assert k == 1


@pytest.mark.parametrize('method', ['sample', 'ledoit_wolf', 'oas', 'constant_corr', 'factor'])
def test_estimate_covariance_all_methods(method):
    X = _returns()
    sigma, info = cv.estimate_covariance(X, method=method)
    assert sigma.shape == (60, 60) and info['method'] == method
    assert np.linalg.eigvalsh(sigma).min() > -1e-12
    assert np.abs(np.diag(sigma) - X.var(axis=1)).max() < 1e-10      # sample variances kept


def test_estimate_covariance_rejects_nan_and_unknown_method():
    X = _returns(); X[0, 0] = np.nan
    with pytest.raises(ValueError):
        cv.estimate_covariance(X)
    with pytest.raises(ValueError):
        cv.estimate_covariance(_returns(), method='nope')


def test_constant_series_does_not_crash():
    X = _returns(); X[0] = 0.0
    sigma, info = cv.estimate_covariance(X, method='ledoit_wolf')
    assert np.isfinite(sigma).all()


def test_build_covariance_matrix():
    C = np.array([[1, .5], [.5, 1.]]); v = np.array([.2, .3])
    S = cv.build_covariance_matrix(C, v)
    assert S[0, 1] == pytest.approx(.5 * .2 * .3) and S[1, 1] == pytest.approx(.09)
    with pytest.raises(ValueError):
        cv.build_covariance_matrix(C, np.ones(3))