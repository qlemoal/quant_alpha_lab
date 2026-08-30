'''

Systematic, evidence-based procedure for picking the EMBARGO constant, instead of a hand-picked guess. 
I look at the autocorrelation present in time series. The embargo window should allow the autocorrelation
    to drop and therefore give more independence to cross-validation datasets.
Recompute the embargo only if the market returns have changed behaviour.

1. TO BE RAN ON MARKET RETURNS.
    Not on IC for example because it depends on each signal. 
    Return-series autocorrelation is a structural property of the market
    itself, independent of which signal you're about to cross-validate.
    We then put our max autocorrelation into config.constants.py

2. TEST EVERY LAG UP TO max_lag, FDR-CORRECT ACROSS THEM, TAKE THE LARGEST SURVIVING LAG, NOT THE FIRST INSIGNIFICANT ONE.
   Testing lags [1 : max_lag] one at a time is itself a multiple-testing
   problem, exactly the situation FDR.py already exists to handle, so this
   reuses it directly rather than inventing a parallel ad hoc rule. "First
   lag that crosses below significance" is fragile: autocorrelation series
   are noisy, a single lag can cross by chance while a later lag remains
   genuinely significant (e.g. weekly seasonality resurfacing at lag 5,
   10). Taking the LARGEST lag that survives FDR correction, within a
   stated search ceiling, is the conservative, defensible reading of "how
   far does dependence actually persist."

Reference for the significance test itself: under the null of no
autocorrelation, the sample autocorrelation at lag k is asymptotically
N(0, 1/n) (Bartlett's approximation, standard in Box, Jenkins & Reinsel,
Time Series Analysis: Forecasting and Control), giving a simple z-stat
r_k * sqrt(n). This is the same kind of asymptotic-normal t-stat machinery
FDR_report() already consumes in src.signals.fdr.py, no new pattern
introduced.
'''



import numpy as np
import polars as pl
from polars import col as c

from src.utils.stats import autocorrelation, compute_pval_from_tstat
from src.evaluation.signals.fdr import fdr_report



def equal_weighted_market_return( lf:pl.LazyFrame, ret_col:str='logret' ) -> pl.LazyFrame:
    '''
    Collapses the (date, ticker) panel into one flat row per date, the
    cross-sectional mean return. This is deliberately the SIMPLEST
    possible aggregate, equal-weighted, no market-cap weighting, no
    survivorship adjustment beyond what's already baked into the panel.
    Good enough for "does the market-wide process show persistent serial
    dependence", not intended as a tradable index proxy.
    '''
    return (
        lf.group_by('date')
        .agg(c(ret_col).mean().alias(ret_col))
        .sort('date')
    )


def select_embargo_from_autocorrelation( lf:pl.LazyFrame, col_name:str, max_lag:int, alpha:float=0.2 ) -> dict:
    '''
    lf_1d: flat series, one row per date, on which we compute the autocorrelation
    col_name: the column on which to compute the autocorrelation
    max_lag: search ceiling. A real, stated bound, not searched unboundedly, for FDR.
    alpha: the FDR alpha, reusing the previous convention from fdr_report, like 0.2.

    Returns a dict: 
        'embargo' : the selected integer 
        'fdr_table' : the full FDR_report DataFrame across all tested lags, to be inspected before
            trusting the single number, particularly whether significant lags are scattered or contiguous.
    '''

    lags = tuple(range(1, max_lag+1))
    ac = autocorrelation(lf, col_name, lags=lags, group=None)
    n_obs = lf.select(pl.len()).collect().item()

    tstats, pvalues = [], []
    for k in lags:
        r = ac[f'lag_{k}'][0]
        n_k = n_obs - k  # k leading rows become null after the shift
        if r is None or n_k <= 1:
            tstats.append(float('nan'))
            pvalues.append(float('nan'))
            continue
        t = r * np.sqrt(n_k)
        tstats.append(t)
        pvalues.append(compute_pval_from_tstat(t))

    fdr_table = fdr_report(
        signal_names = [f'lag_{k}' for k in lags],
        tstats = tstats,
        pvalues = pvalues,
        alpha = alpha,
        sort_pvalues = False
    )

    surviving_lags = [
        int(name.split('_')[1])
        for name, survives in zip(fdr_table['signal'], fdr_table['survives_bh'])
        if survives
    ]

    embargo = max(surviving_lags) if surviving_lags else 0

    # Plot qvalues to check how dispersed they are
    import matplotlib.pyplot as plt
    f, ax = plt.subplots(1, figsize=(12, 6))
    plt.plot(fdr_table['qvalue'], '-o')
    plt.axhline(alpha)
    plt.show()

    return {'embargo': embargo, 'fdr_table': fdr_table}




if __name__ == '__main__':
    lf = pl.scan_parquet('data/processed/features.parquet')
    market_lf = equal_weighted_market_return(lf, ret_col='logret')
    result = select_embargo_from_autocorrelation(
        market_lf, 'logret', max_lag=252, alpha=0.2
    )
    print(result['embargo'])
    print(result['fdr_table'])