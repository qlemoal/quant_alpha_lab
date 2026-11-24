'''
GOAL: group stocks that move together, so that stage 08 can cap the exposure of each group.

    cleaned correlation C (07)
        |  [1] remove_market_mode        subtract the part of C that every stock shares (the market)
        v
    residual correlation C_res
        |  [2] louvain_clustering        graph of positive correlations -> communities
        v
    {stock index: cluster id}

WHY REMOVE THE MARKET MODE?
    Almost all stocks rise and fall with the market, so almost every pair has a positive correlation.
    The graph is then one dense blob, and Louvain returns about 3 big groups. Caps on 3 groups protect little.
    The market part of C is one eigenvector (v1, all entries of the same sign) with a very large eigenvalue
    (lambda_1). Subtracting lambda_1 * v1 v1' leaves what the market does NOT explain: sector and industry
    co-movement. Louvain on that matrix finds those groups.

    This is not beta neutralisation. Beta neutralisation (stage 08, planned) changes the weights.
    This step only changes how the GROUPS are found.

A SECOND RISK: MANY TINY CLUSTERS
    When the industries are weak, the residual correlation is close to noise and Louvain returns dozens of
    clusters of 1 to 3 stocks. A cap per cluster would then be meaningless (the cap is 2 x gross / K). So
    drop_small_clusters sends every cluster below min_size stocks to id -1 ("no cluster", never capped,
    not counted in K).

THE CHECK
    The top eigenvector is a market mode only if (a) nearly all its entries have the same sign and
    (b) lambda_1 is clearly above lambda_2. remove_market_mode reports both. If a check fails, it does NOT
    remove anything and says so in the returned info (market_mode_ok = False). Stage 07 counts these dates.
'''

import numpy as np
import networkx as nx
from networkx.algorithms.community import louvain_communities


# A market mode needs: at least this share of the top eigenvector entries with the same sign,
# and lambda_1 at least this multiple of lambda_2.
MIN_SAME_SIGN_SHARE = 0.9
MIN_EIGENVALUE_GAP = 2.0


def remove_market_mode(corr: np.ndarray) -> tuple:
    '''
    Subtract the market mode (top eigenvector) from a correlation matrix and rescale to a unit diagonal.

        C_res = C - lambda_1 v1 v1'            (v1: eigenvector of the largest eigenvalue)
        C_res_ij / sqrt(C_res_ii C_res_jj)     (back to correlations)

    Returns (corr_res, info). info has:
        lambda_1, lambda_2, gap (lambda_1 / lambda_2), variance_share (lambda_1 / n),
        same_sign_share (share of v1 entries with the majority sign), market_mode_ok (bool), removed (bool).
    If market_mode_ok is False the input is returned unchanged and removed is False.
    '''
    C = np.asarray(corr, dtype=float)
    n = C.shape[0]
    if C.ndim != 2 or C.shape[1] != n:
        raise ValueError(f'corr must be square, got {C.shape}')
    if n < 3:
        return C.copy(), {'market_mode_ok': False, 'removed': False, 'reason': 'fewer than 3 assets'}

    values, vectors = np.linalg.eigh((C + C.T) / 2)          # ascending order
    lam1, lam2 = float(values[-1]), float(values[-2])
    v1 = vectors[:, -1]
    pos = float((v1 > 0).mean())
    same = max(pos, 1.0 - pos)
    gap = lam1 / lam2 if lam2 > 0 else float('inf')
    ok = bool(same >= MIN_SAME_SIGN_SHARE and gap >= MIN_EIGENVALUE_GAP)
    info = {'lambda_1': lam1, 'lambda_2': lam2, 'gap': gap, 'variance_share': lam1 / n,
            'same_sign_share': same, 'market_mode_ok': ok, 'removed': False}
    if not ok:
        return C.copy(), info

    res = C - lam1 * np.outer(v1, v1)
    d = np.sqrt(np.clip(np.diag(res), 1e-8, None))
    res = res / np.outer(d, d)
    np.fill_diagonal(res, 1.0)
    info['removed'] = True
    return res, info


def drop_small_clusters(cluster_map: dict, min_size: int) -> dict:
    '''
    Clusters with fewer than min_size stocks get id -1. The other ids are renumbered 0, 1, 2, ... by
    decreasing size (ties: smallest old id first), so the output does not depend on Louvain's numbering order.
    '''
    sizes = {}
    for c in cluster_map.values():
        sizes[c] = sizes.get(c, 0) + 1
    keep = sorted((c for c, n in sizes.items() if n >= min_size), key=lambda c: (-sizes[c], c))
    new_id = {c: i for i, c in enumerate(keep)}
    return {t: new_id.get(c, -1) for t, c in cluster_map.items()}


def louvain_clustering(corr: np.ndarray, resolution: float = 1.0) -> dict:
    '''
    Louvain community detection on the correlation network: build a graph from
        the correlation matrix, partition tickers into clusters maximizing
        modularity. Operates on whatever matrix is passed in, raw or cleaned,
        no cleaning happens here. Chaining RMT cleaning before this is a
        deliberate separate step (see src/risk/pipeline.py, not built yet),
        not automatic, so it's never silently skipped or silently applied.

    resolution: >1 favors more, smaller clusters; <1 favors fewer, larger ones.

    Returns: dict {ticker_index: cluster_id}.

    EDGE WEIGHT CONVENTION, a real design decision, not mechanical:
    only POSITIVE correlations become edges, negative correlations are
    dropped entirely rather than taken as |corr| or shifted to [0, 2].
    The purpose here is grouping tickers that move together, so that a
    downstream risk step can avoid stacking correlated exposures inside
    one cluster. Two tickers with correlation -0.8 are the OPPOSITE of a
    diversification risk, holding both is what reduces portfolio
    variance, not what concentrates it, treating them as "belonging
    together" by folding their negative correlation into edge weight via
    absolute value would be actively wrong for the purpose this function
    exists for, not just a rounding choice. If a future use case needs
    "assets that move in either strong positive or strong negative
    lockstep" (e.g. a pairs-trading context, a genuinely different
    question), that would need a different, explicitly named function,
    not a silent option flipped on this one.

    Reference: Blondel, Guillaume, Lambiotte, Lefebvre (2008). Fast
    unfolding of communities in large networks. Journal of Statistical
    Mechanics: Theory and Experiment, 2008(10), P10008. Louvain
    iteratively moves nodes between communities to greedily maximize
    modularity (the fraction of edge weight within communities minus
    the expected fraction under a random-graph null), then coarsens the
    graph (each community becomes one node) and repeats, giving it
    near-linear time complexity on sparse graphs, part of why it scales
    to a few hundred tickers without issue here.
    '''
    n_assets = corr.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(n_assets))  # ensures isolated tickers (no positive correlation to anything) still get their own singleton cluster, not silently dropped

    rows, cols = np.triu_indices(n_assets, k=1)
    keep = corr[rows, cols] > 0                      # positive correlations only
    graph.add_weighted_edges_from(
        (int(i), int(j), float(w)) for i, j, w in zip(rows[keep], cols[keep], corr[rows, cols][keep])
    )

    communities = louvain_communities(graph, weight='weight', resolution=resolution, seed=0)

    return {
        ticker_idx: cluster_id
        for cluster_id, cluster in enumerate(communities)
        for ticker_idx in cluster
    }