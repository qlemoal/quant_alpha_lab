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