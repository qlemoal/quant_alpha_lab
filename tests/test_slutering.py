'''
Tests for src/risk/clustering.py: the market-mode removal and Louvain.

Synthetic world: a market factor that moves every stock, plus K industry factors. Stocks of the same
industry share an industry factor. We KNOW the true industry of each stock, so we can measure how well the clusters
recover them (adjusted Rand index, 1 = perfect, 0 = random).
'''

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from src.risk.clustering import louvain_clustering, remove_market_mode, drop_small_clusters
from src.risk.risk_model import compute_raw_correlation, clean_correlation_rmt, cluster_correlation

K, PER, T = 6, 30, 500
N = K * PER


def world(seed=0, market_beta=1.0, industry_beta=0.7):
    '''Returns (returns N x T, true industry labels).'''
    rng = np.random.default_rng(seed)
    market = rng.normal(size=T)
    ind = rng.normal(size=(K, T))
    labels = np.repeat(np.arange(K), PER)
    R = market_beta * market + industry_beta * ind[labels] + rng.normal(size=(N, T))
    return R, labels


def clean_corr(R):
    corr, _, _ = clean_correlation_rmt(compute_raw_correlation(R), R.shape[1])
    return corr


def labels_of(cmap):
    return np.array([cmap[i] for i in range(N)])


# ---------------------------------------------------------------- remove_market_mode

def test_market_mode_is_found_and_removed():
    corr = clean_corr(world()[0])
    res, info = remove_market_mode(corr)
    assert info['market_mode_ok'] and info['removed']
    assert info['same_sign_share'] > 0.99 and info['gap'] > 2
    assert np.allclose(np.diag(res), 1.0)
    assert np.allclose(res, res.T, atol=1e-10)
    # the average off-diagonal correlation, driven by the market, must fall a lot
    off = ~np.eye(N, dtype=bool)
    assert res[off].mean() < 0.3 * corr[off].mean()


def test_no_market_mode_leaves_the_matrix_unchanged():
    '''Independent stocks: no common factor, so the top eigenvector is not a market mode.'''
    rng = np.random.default_rng(1)
    corr = np.corrcoef(rng.normal(size=(60, 400)))
    res, info = remove_market_mode(corr)
    assert not info['market_mode_ok'] and not info['removed']
    assert np.array_equal(res, corr)


def test_bad_shape_and_tiny_input():
    with pytest.raises(ValueError):
        remove_market_mode(np.ones((3, 4)))
    _, info = remove_market_mode(np.eye(2))
    assert not info['removed']


# ---------------------------------------------------------------- the point of it all

def unequal_world(seed, industry_beta):
    '''Heterogeneous market betas and industries of unequal strength: closer to real data.'''
    rng = np.random.default_rng(seed)
    market = rng.normal(size=T)
    ind = rng.normal(size=(K, T))
    labels = np.repeat(np.arange(K), PER)
    beta = rng.uniform(0.3, 1.8, size=N)
    strength = industry_beta * rng.uniform(0.5, 1.5, size=K)[labels]
    return beta[:, None] * market[None, :] + strength[:, None] * ind[labels] + rng.normal(size=(N, T)), labels


def test_removal_helps_on_average_when_industries_are_moderate():
    '''Measured on 6 seeds. The gain is real but modest (about +0.1 ARI), not dramatic.'''
    before_ari, after_ari = [], []
    for seed in range(6):
        R, labels = unequal_world(seed, 0.55)
        corr = clean_corr(R)
        b, _ = cluster_correlation(corr, 1.0, remove_market=False)
        a, info = cluster_correlation(corr, 1.0, remove_market=True)
        assert info['removed']
        before_ari.append(adjusted_rand_score(labels, labels_of(b)))
        after_ari.append(adjusted_rand_score(labels, labels_of(a)))
    print(f'ARI {np.mean(before_ari):.2f} -> {np.mean(after_ari):.2f}')
    assert np.mean(after_ari) > np.mean(before_ari) + 0.03


def test_strong_industries_are_found_with_or_without_removal():
    '''Honest check: when industries are strong, the cleaned matrix already works. The removal must not hurt.'''
    R, labels = world()
    corr = clean_corr(R)
    for flag in (False, True):
        cmap, _ = cluster_correlation(corr, 1.0, remove_market=flag)
        assert adjusted_rand_score(labels, labels_of(cmap)) > 0.95


def test_weak_industries_give_many_tiny_clusters_unless_filtered():
    '''Industries below the noise edge: the residual is noise, Louvain returns many tiny clusters.
    min_size sends them to -1, so K stays small and the caps stay meaningful.'''
    R, _ = world(industry_beta=0.25)
    corr = clean_corr(R)
    raw, _ = cluster_correlation(corr, 1.0, remove_market=True, min_size=1)
    kept, _ = cluster_correlation(corr, 1.0, remove_market=True, min_size=5)
    assert len(set(raw.values())) > 20
    valid = [c for c in kept.values() if c >= 0]
    assert len(set(valid)) < 0.5 * len(set(raw.values()))
    sizes = np.bincount(valid) if valid else np.array([5])
    assert (sizes >= 5).all()


def test_drop_small_clusters_relabels_by_size():
    cmap = {0: 7, 1: 7, 2: 7, 3: 2, 4: 2, 5: 9}
    out = drop_small_clusters(cmap, min_size=2)
    assert out == {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: -1}
    assert drop_small_clusters(cmap, min_size=4) == {i: -1 for i in range(6)}


def test_removal_is_deterministic():
    corr = clean_corr(world(seed=3)[0])
    a, _ = cluster_correlation(corr, 1.0)
    b, _ = cluster_correlation(corr, 1.0)
    assert a == b


# ---------------------------------------------------------------- louvain

def test_louvain_two_blocks():
    C = np.full((6, 6), -0.1)
    C[:3, :3] = 0.8
    C[3:, 3:] = 0.8
    np.fill_diagonal(C, 1.0)
    m = louvain_clustering(C)
    assert m[0] == m[1] == m[2] and m[3] == m[4] == m[5] and m[0] != m[3]


def test_louvain_negative_correlation_is_never_an_edge():
    C = np.array([[1, -0.9], [-0.9, 1.0]])
    m = louvain_clustering(C)
    assert m[0] != m[1]                         # opposite stocks are NOT grouped


def test_louvain_isolated_stock_gets_its_own_cluster():
    C = np.eye(3)
    m = louvain_clustering(C)
    assert len(set(m.values())) == 3


def test_louvain_resolution_more_clusters_when_higher():
    R, _ = world(seed=5)
    res, _ = remove_market_mode(clean_corr(R))
    low = len(set(louvain_clustering(res, resolution=0.3).values()))
    high = len(set(louvain_clustering(res, resolution=3.0).values()))
    assert high >= low