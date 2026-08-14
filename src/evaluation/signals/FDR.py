'''
False discovery rate control across multiple candidate signals tested in the same round. 
    Testing many signals (or many parameter variants of one signal) at a fixed 5% threshold produces false positives by construction, 
    this corrects for that.
'''

import numpy as np
import polars as pl


def benjamini_hochberg(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    '''
    BH (1995) step-up procedure. Valid under independence or positive regression dependence (PRDS) among test statistics. 
        Standard first choice, but candidate signals built off the same overlapping return panel are not independent, 
        so treat BH as an upper bound on how many signals survive, not the final word, see benjamini_yekutieli below.

    Returns a list of booleans, same order/length as pvalues, True = survives.
    '''
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]

    thresholds = alpha * (np.arange(1, m + 1) / m)
    below = sorted_p <= thresholds

    # largest k such that p_(k) <= alpha*k/m; everything up to k survives
    if not below.any():
        k = 0
    else:
        k = np.max(np.where(below)[0]) + 1

    survives_sorted = np.zeros(m, dtype=bool)
    survives_sorted[:k] = True

    survives = np.zeros(m, dtype=bool)
    survives[order] = survives_sorted
    return survives.tolist()


def benjamini_yekutieli(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    '''
    BY (2001) step-up procedure. Valid under arbitrary dependence, no PRDS assumption needed. 
        Same mechanics as BH, but the threshold is divided by the harmonic number H_m = sum(1/i for i in 1..m), 
        which grows with m and makes this strictly more conservative than BH. 
        The right choice when candidate signals are correlated, which is the normal case here:
        different lookback windows of the same feature, or different transform methods (zscore/rank/decile) 
        of the same underlying signal, are built from the same overlapping return history.
    '''
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    harmonic = np.sum(1.0 / np.arange(1, m + 1))

    order = np.argsort(p)
    sorted_p = p[order]

    thresholds = alpha * (np.arange(1, m + 1) / (m * harmonic))
    below = sorted_p <= thresholds

    if not below.any():
        k = 0
    else:
        k = np.max(np.where(below)[0]) + 1

    survives_sorted = np.zeros(m, dtype=bool)
    survives_sorted[:k] = True

    survives = np.zeros(m, dtype=bool)
    survives[order] = survives_sorted
    return survives.tolist()



#  Non-parametric method now with Storey's q-values:

from scipy.interpolate import UnivariateSpline


def estimate_pi0(pvalues, lambdas=None):
    '''
    Storey & Tibshirani (2003) automatic pi0 estimate: the proportion of
        candidates that are genuinely null, estimated from the data itself,
        no alpha or lambda chosen by hand for this specific batch. Under the
        null, p-values are uniform, so as lambda -> 1, #{p_i > lambda} /
        (m*(1-lambda)) converges to pi0. A cubic spline is fit across a fixed
        lambda grid (the grid itself is a standard, dataset-independent
        convention, not tuned) and evaluated at the right edge, smoothing
        across the whole grid instead of trusting one noisy single-lambda
        estimate.
    '''
    p = np.asarray(pvalues, dtype=float)
    p = p[~np.isnan(p)]
    n = len(p)

    if n < 20:
        # too few candidates for a stable spline fit, assume the worst case
        # (all null) rather than trust a noisy estimate
        return 1.0

    if lambdas is None:
        lambdas = np.arange(0.05, 0.96, 0.05)

    pi0_hat = np.array([np.mean(p > lam) / (1 - lam) for lam in lambdas])
    spline = UnivariateSpline(lambdas, pi0_hat, k=3, s=len(lambdas))
    pi0 = float(spline(lambdas[-1]))

    return float(np.clip(pi0, 0.0, 1.0))


def qvalues(pvalues):
    '''
    Storey q-values: for each candidate, the minimum FDR at which it would be
        called significant. No alpha, no per-signal cutoff decision baked in
        here, pi0 comes from the data (estimate_pi0), everything after that
        is a fixed, order-preserving transform of the p-values. Reported as
        a continuous number per signal rather than a survives/doesn't-survive
        boolean, on purpose, any cutoff is a separate downstream decision
        (e.g. feeding the Elastic Net combiner a continuous confidence
        weight instead of a hard pass/fail), not baked into the test itself.
    '''
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    valid = ~np.isnan(p)

    q = np.full(m, np.nan)
    if valid.sum() == 0:
        return q.tolist()

    p_valid = p[valid]
    n = len(p_valid)
    pi0 = estimate_pi0(p_valid)

    order = np.argsort(p_valid)
    p_sorted = p_valid[order]

    q_sorted = np.empty(n)
    q_sorted[-1] = pi0 * p_sorted[-1]
    for i in range(n - 2, -1, -1):
        q_sorted[i] = min(pi0 * n * p_sorted[i] / (i + 1), q_sorted[i + 1])

    q_valid = np.empty(n)
    q_valid[order] = q_sorted
    q[valid] = q_valid
    return q.tolist()


#  Aggregate

def fdr_report(signal_names, pvalues, alpha=0.05):
    p = np.asarray(pvalues, dtype=float)
    valid = ~np.isnan(p)
    # NaN entries stay False: excluded from m for everyone else, and never survive themselves, don't let an uncomputable signal ride along.

    survives_bh = np.zeros(len(p), dtype=bool)
    survives_by = np.zeros(len(p), dtype=bool)
    if valid.sum() > 0:
        survives_bh[valid] = benjamini_hochberg(p[valid].tolist(), alpha)
        survives_by[valid] = benjamini_yekutieli(p[valid].tolist(), alpha)

    return pl.DataFrame({
        'signal': signal_names, 'pvalue': pvalues,
        'survives_bh': survives_bh, 'survives_by': survives_by,
    }).sort('pvalue', nulls_last=True)