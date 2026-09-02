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
    Computes market returns from cross-sectional mean of log-returns.
    Current S&P 500 tickers => big bias in the market returns in the early years.
    '''
    return (
        lf.group_by('date')
        .agg(c(ret_col).mean().alias(ret_col))
        .sort('date')
    )


def select_embargo_from_autocorrelation( lf:pl.LazyFrame, col_name:str, max_lag:int, alpha:float=0.05, 
                                         criterion:str='survives_by', rule:str='first_contiguous_run',
                                         _plot = False) -> dict:
    '''
    lf:         flat series on which to compute the autocorrelation, one row per date (e.g. output of equal_weighted_market_return above).
    max_lag:    search ceiling. A real, stated bound, not searched unboundedly. 252 seems like a logical choice here.
    alpha:      0.05 by default, DELIBERATELY NOT the project's 0.2 signal-screening alpha. That 0.2 exists to avoid missing weak-but-real
        candidate signals, a lenient choice for a specific reason. Here, being lenient means flagging MORE lags significant, which makes
        embargo LARGER, the opposite kind of conservatism, for a completely different reason. 0.05 also matches the standard
        +-1.96/sqrt(n) ACF significance band convention, not arbitrary.
    criterion:  'survives_by' (default), 'survives_bh', or 'qvalue' (thresholded at `alpha`). 
        Lag-k and lag-(k+1) autocorrelation are computed from the same series shifted by one step, genuinely
        dependent tests, not the roughly-independent cross-sectional signal tests BH/q-value are normally used on (in FDR e.g.). 
        BY is specifically valid under arbitrary dependence between tests, that's the actual reason to prefer it in THIS context, not
        just consistency with FDR.py's defaults. 'qvalue' is closer to this project's general stated preference, but it's not identical to survives_bh, because it uses
        an estimated pi0 and is typically less conservative than BH at the same alpha. 
    rule:       'first_contiguous_run' (default) takes the last lag in the first unbroken run of significant lags starting at lag 1, robust
        to an isolated significant lag appearing far out by chance (more likely a seasonal/periodic effect worth its own writeup in
        docs/findings.md than genuine boundary-contamination risk). 'largest_surviving_lag' is a rougher rule, kept available for comparison, vulnerable to exactly that failure
        mode, I would prefer the default unless I have a specific reason not to.

    Math logic: sample autocorrelation r_k is asymptotically N(0, 1/n) (Bartlett's approximation), so its standard error is ≈1/√n, 
        and a 95% confidence band on the raw correlation coefficient r_k itself is ±1.96/√n. 
        It's mathematically identical to the uncorrected z-test computed here ( z = r_k√n_k, then p = 2(1-Φ(|z|)), |z|>1.96 ⟺ p<0.05 ⟺ |r_k|>1.96/√n_k )

    Returns: a dict with 'embargo' (the selected integer), 'fdr_table' (the full FDR_report DataFrame across all tested lags, inspect this before
        trusting the single number either way, particularly whether significant lags are contiguous from lag 1 or scattered).
    '''
    lags = tuple(range(1, max_lag + 1))
    ac = autocorrelation(lf, col_name, lags=lags, group=None)

    n_total = lf.select(pl.len()).collect().item()

    tstats, pvalues = [], []
    for k in lags:
        r = ac[f'lag_{k}'][0]
        n_k = n_total - k  # k leading rows become null after the shift, reducing the number of obs for the t-stat
        if r is None or n_k <= 1:
            tstats.append(float('nan'))
            pvalues.append(float('nan'))
            continue
        t = r * np.sqrt(n_k)  # The t-stat of the ACF is apparently the autocorrelation times the number of obs
        tstats.append(t)
        pvalues.append(compute_pval_from_tstat(t))

    fdr_table = fdr_report(
        signal_names=[f'lag_{k}' for k in lags],
        tstats=tstats,
        pvalues=pvalues,
        alpha=alpha,
        sort_pvalues=False
    )

    if criterion == 'qvalue':
        is_significant = (fdr_table['qvalue'] <= alpha).to_list()
    elif criterion in ('survives_bh', 'survives_by'):
        is_significant = fdr_table[criterion].to_list()
    else:
        raise ValueError(f'unknown criterion: {criterion}')

    # edit: I added the sort_pvalues=False option in fdr_table, so no need to re-sort but... it's here...
    lag_by_position = list(lags)  # fdr_table is sorted by pvalue, not by lag, realign first
    sig_by_lag = dict(zip(
        [int(name.split('_')[1]) for name in fdr_table['signal']],
        is_significant,
    ))
    ordered_flags = [sig_by_lag[k] for k in lag_by_position]

    if rule == 'largest_surviving_lag':
        surviving = [k for k, flag in zip(lag_by_position, ordered_flags) if flag]
        embargo = max(surviving) if surviving else 0
    elif rule == 'first_contiguous_run':
        embargo = 0
        for k, flag in zip(lag_by_position, ordered_flags):
            if not flag:
                break
            embargo = k
    else:
        raise ValueError(f'unknown rule: {rule}')

    if _plot:
        import matplotlib.pyplot as plt
        _, _ = plt.subplots(1, figsize=(12, 6))
        plt.plot(fdr_table['qvalue'], '-o')
        plt.axhline(1.96/np.sqrt(n_total))
        plt.axhline(0.05/np.sqrt(n_total))
        plt.xlabel('lag')
        plt.ylabel('qvalue')
        plt.grid()
        plt.show()

    return {'embargo': embargo, 'fdr_table': fdr_table}







if __name__ == '__main__':
    lf = pl.scan_parquet('data/processed/features.parquet')
    market_lf = equal_weighted_market_return(lf, ret_col='logret')
    result = select_embargo_from_autocorrelation(
        market_lf, 'logret', max_lag=252, alpha=0.05, criterion='survives_by',rule='largest_surviving_lag',
        _plot=True
    )
    print(result['embargo'])
    print(result['fdr_table'])