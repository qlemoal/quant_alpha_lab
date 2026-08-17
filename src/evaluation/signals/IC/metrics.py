#  Heavier, occasionally-invoked statistical refinements, built from core.py
#  Newey-West particularly important to take into account autocorrelation in the 
#      estimation of the IC. Max lag is still to be determined systematically, 
#      instead of using the standard rule of thumb.

import numpy as np
import pandas as pd
import polars as pl
from polars import col as c
from scipy.stats import norm

from src.signals.combine import make_signal
from src.features.returns import add_fwdret_horizon
from src.evaluation.signals.IC.core import compute_ic, summarize_ic


def newey_west_ic_tstat(ic_values:pl.LazyFrame|pl.DataFrame|list, max_lag=None) -> dict:
    '''
    Newey-West (HAC, heteroskedastic autocorrelated consistent) corrected t-stat for the mean of a daily IC series.
        HAC is for the covariance matrix,   See https://en.wikipedia.org/wiki/Newey%E2%80%93West_estimator 

        The naive t-stat in summarize_ic() assumes independent daily observations, 
        which overlapping-window features violate (mom20's 20-day window means consecutive days' IC values are correlated). 
        This corrects for that autocorrelation instead of just flagging it as a caveat.

        Variance must be increased when positive autocorrelation is present, preventing you from overstating the statistical significance of your IC.

    Step-by-step: 
        - Compute unadjusted variance of IC series gamma0
        - Compute the autocovariance up to lag max_lag gamma_k
        - For each lag k, compute the Bartlett Kernel Weights w:=1-k/(max_lag+1)
        - Combine the variance with 2x the weighted autocovariance to get the NW variance
        - Use the NW variance to compute the standard error of the mean
        - Use the SE to get the t-stat

    max_lag defaults to the standard Newey-West (1994) rule of thumb, 4 * (n/100)^(2/9), if not given explicitly.

    IMPORTANT: Feed in only the values, not the date!!! 
        Temporary fix: forcing to be a pandas Series, which will be forced to be 1 dimensional. I then get the values of it to remove the dates.
    '''
    x = np.asarray(ic_values, dtype=float)
    if x.ndim != 1:
        raise ValueError(f'newey_west_ic_tstat expects 1D IC values, as a list, pl.LazyFrame or pl.DataFrame, but got shape {x.shape}')
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return {'mean': None, 'se': None, 't_stat': None, 'max_lag': None}

    # The max lag represents the maximum number of lag periods to include in the autocorrelation window.
    if max_lag is None:
        max_lag = max(1, int(4 * (n / 100) ** (2 / 9)))

    mean = x.mean()
    demeaned = x - mean
    gamma0 = np.sum(demeaned ** 2) / n

    var = gamma0
    gammas = []
    for k in range(1, max_lag + 1):
        w = 1 - k / (max_lag + 1)
        gamma_k = np.sum(demeaned[k:] * demeaned[:-k]) / n
        gammas.append(gamma_k)
        var += 2 * w * gamma_k

    se = np.sqrt(var / n) if var > 0 else None
    t_stat = mean / se if se else None

    return {'mean': mean, 'se': se, 't_stat': t_stat, 'max_lag': max_lag}#, 'n':n, 'gamma0':gamma0, 'gammas':gammas, 'deameaned':demeaned, 'x':x}





def ic_pvalue_nw(nw_result:dict) -> float | None:
    '''
    Two-sided p-value from a Newey-West-corrected t-stat, via the normal
        approximation (the NW t-stat is asymptotically normal, not exactly
        Student-t, unlike the naive t-stat which assumes a known small-sample
        distribution). Takes the dict returned by newey_west_ic_tstat directly,
        so the two functions stay paired.
    '''
    t = nw_result.get('t_stat')
    if t is None:
        return None
    return 2 * (1 - norm.cdf(abs(t)))



def ic_decay(lf, signal_col, horizons=(1, 5, 10, 20)):
    '''
    IC (Spearman) of a signal against forward returns at several horizons.
        Shows how long the signal's predictive power actually persists,
        directly informs what rebalance frequency makes sense: 
        a signal that only shows IC at horizon 1 wants daily rebalancing (expensive),
        one that holds up to horizon 20 can rebalance far less often.
    Note: No Newey-West compensation for autocorrelation is used here. TODO?
    '''

    rows = []
    for h in horizons:
        lf_h = add_fwdret_horizon(lf, horizon=h)
        ic_lf = compute_ic(lf_h, signal_col, f'fwd_ret_{h}')
        stats = summarize_ic(ic_lf)  # returns a dict
        stats['horizon'] = h
        rows.append(stats)

    return pl.DataFrame(rows).select(['horizon', 'mean', 'std', 'ir', 'hit_rate', 'n_days'])



if __name__=='__main__':
    from src.signals.combine import make_signal
    with pl.Config(tbl_cols=-1):  # that's to print the whole report instead of truncating some columns

        lf = pl.scan_parquet('data/processed/features.parquet')
        lf = make_signal(lf, ['mom5', 'mom10', 'mom20', 'mom60', 'mom120', 'mom252'], method='decile')
        # lf = make_signal(lf, ['mom5', 'mom10', 'mom20', 'mom60', 'mom252'], method='zscore_tanh')
        # print(signal_report(lf, signal_col='mom20_decile'))
        print(ic_decay(lf, 'mom60'))
