'''
Tests for the beta neutralisation of stage 08 (src/risk/weights_pipeline.py, step 3b).

What is checked:
  1. project_out: exposures are exactly zero, the change is the smallest possible, collinear rows do not crash
  2. betas: shrink toward 1, missing -> 1; covariance betas average 1
  3. build_weights: with beta_neutral the exposure is 0 and the book stays dollar-neutral; without it, it is not
  4. load_betas_by_date reads the right values
  5. a long-high-beta / short-low-beta book is beta-neutral after the projection and still follows the score
'''

import datetime as dt

import numpy as np
import polars as pl
import pytest

from src.risk.weights_pipeline import (
    project_out, beta_neutralise, shrink_betas, market_betas, build_weights, load_betas_by_date,
)

CFG = {
    'signal_method': 'proportional', 'base_gross': 1.0, 'max_cluster_multiple': 2.0, 'cap_method': 'cap',
    'target_vol': 0.10, 'vol_method': 'scale', 'max_gross_leverage': 4.0, 'beta_neutral': True,
}


def universe(n=60, seed=0):
    rng = np.random.default_rng(seed)
    beta = rng.uniform(0.5, 1.5, n)
    vol = rng.uniform(0.15, 0.40, n)
    cov = np.outer(beta, beta) * 0.04 + np.diag(vol ** 2)
    return beta, cov, rng


# ---------------------------------------------------------------- projection

def test_project_out_gives_zero_exposures():
    rng = np.random.default_rng(1)
    w, beta = rng.normal(size=50), rng.uniform(0.5, 1.5, 50)
    new = project_out(w, np.vstack([np.ones(50), beta]))
    assert new.sum() == pytest.approx(0, abs=1e-12)
    assert beta @ new == pytest.approx(0, abs=1e-12)


def test_project_out_is_the_smallest_change():
    '''The change must lie in the row space of the constraints: change = A' y. Any other direction is waste.'''
    rng = np.random.default_rng(2)
    w, beta = rng.normal(size=40), rng.uniform(0.5, 1.5, 40)
    A = np.vstack([np.ones(40), beta])
    change = w - project_out(w, A)
    y = np.linalg.lstsq(A.T, change, rcond=None)[0]
    assert np.allclose(A.T @ y, change)


def test_project_out_does_not_move_a_book_that_already_satisfies_the_constraints():
    beta = np.array([1.0, 1.0, 1.0, 1.0])
    w = np.array([1.0, -1.0, 1.0, -1.0])
    assert np.allclose(project_out(w, np.vstack([np.ones(4), beta])), w)


def test_project_out_survives_collinear_constraints():
    w = np.array([0.3, -0.1, 0.2, -0.5])
    new = project_out(w, np.vstack([np.ones(4), np.ones(4)]))      # all betas equal: both rows identical
    assert np.isfinite(new).all() and new.sum() == pytest.approx(0, abs=1e-12)


# ---------------------------------------------------------------- betas

def test_shrink_betas():
    out = shrink_betas(np.array([2.0, np.nan, 0.4]), 1 / 3)
    assert out == pytest.approx([1 / 3 * 1 + 2 / 3 * 2.0, 1.0, 1 / 3 + 2 / 3 * 0.4])


def test_covariance_betas_average_one():
    _, cov, _ = universe()
    assert market_betas(cov).mean() == pytest.approx(1.0)


def test_covariance_betas_follow_true_betas_in_a_one_factor_world():
    beta, cov, _ = universe()
    assert np.corrcoef(market_betas(cov), beta)[0, 1] > 0.97


# ---------------------------------------------------------------- build_weights

def make(n=60, seed=0, **over):
    beta, cov, rng = universe(n, seed)
    scores = -beta + 0.3 * rng.normal(size=n)          # the signal favours LOW-beta stocks: a market-short book
    return beta, cov, scores, {**CFG, **over}


def test_beta_neutral_book_has_zero_beta_and_zero_net():
    beta, cov, scores, cfg = make()
    res = build_weights(scores, cov, np.full(len(scores), -1), cfg, betas=beta)
    w = res['final']
    assert w.sum() == pytest.approx(0, abs=1e-9)
    assert beta @ w == pytest.approx(0, abs=1e-9)
    assert res['beta_after'] == pytest.approx(0, abs=1e-9)


def test_without_neutralisation_the_same_book_is_short_the_market():
    beta, cov, scores, cfg = make(beta_neutral=False)
    res = build_weights(scores, cov, np.full(len(scores), -1), cfg, betas=beta)
    assert res['beta_before'] == pytest.approx(res['beta_after'])
    assert res['beta_after'] < -0.1                      # long low beta, short high beta: net short the market


def test_neutralised_book_still_follows_the_score():
    beta, cov, rng = universe()
    scores = -0.3 * beta + rng.normal(size=len(beta))     # mostly stock picking, a little beta tilt
    w = build_weights(scores, cov, np.full(len(scores), -1), CFG, betas=beta)['final']
    assert np.corrcoef(w, scores)[0, 1] > 0.8            # the projection moves weights a little, it does not invert them


def test_fallback_uses_covariance_betas():
    beta, cov, scores, cfg = make()
    res = build_weights(scores, cov, np.full(len(scores), -1), cfg)       # no betas given
    assert market_betas(cov) @ res['final'] == pytest.approx(0, abs=1e-9)


def test_vol_target_is_applied_after_the_projection():
    beta, cov, scores, cfg = make()
    res = build_weights(scores, cov, np.full(len(scores), -1), cfg, betas=beta)
    assert res['ex_ante_vol'] == pytest.approx(0.10, rel=1e-6) or res['leverage_capped']


# ---------------------------------------------------------------- loader

def test_load_betas_by_date(tmp_path):
    d1, d2 = dt.date(2020, 1, 2), dt.date(2020, 1, 3)
    pl.DataFrame({
        'date': [d1, d1, d1, d2], 'ticker': ['A', 'B', 'C', 'A'],
        'beta252': [1.2, None, float('nan'), 0.8],
    }).write_parquet(tmp_path / 'f.parquet')
    out = load_betas_by_date(tmp_path / 'f.parquet', 'beta252', ['2020-01-02'])
    assert out == {'2020-01-02': {'A': 1.2}}              # null and NaN are absent; other dates are not loaded
