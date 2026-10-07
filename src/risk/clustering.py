# TO CHECK

import numpy as np
import polars as pl
import networkx as nx
from networkx.algorithms.community import louvain_communities


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

    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            if corr[i, j] > 0:
                graph.add_edge(i, j, weight=corr[i, j])

    communities = louvain_communities(graph, weight='weight', resolution=resolution, seed=0)

    return {
        ticker_idx: cluster_id
        for cluster_id, cluster in enumerate(communities)
        for ticker_idx in cluster
    }