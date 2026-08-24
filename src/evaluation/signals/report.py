'''
Single numeric aggregator, with statistics, from perfoamrnce.py and IC folder stats (similarly to signal_plots.py for plots)
'''

import polars as pl
from polars import col as c
import time

from src.evaluation.signals.ic.core import compute_ic, summarize_ic
from src.evaluation.signals.ic.metrics import newey_west_ic_tstat, ic_pvalue_nw, ic_decay
from src.evaluation.signals.performance import max_drawdown, compute_stability, decile_longshort_returns, sharpe_ratio
from src.utils.stats import autocorrelation, get_recent_coverage, compute_pval_from_tstat




def signal_report(lf, signal_col, fwd_ret_col='fwdret'):#, decay_horizons=(1, 5, 10, 20)):
    ic_lf = compute_ic(lf, signal_col, fwd_ret_col)
    ic_collected = ic_lf.collect()

    basic = summarize_ic(ic_collected)
    nw = newey_west_ic_tstat(ic_collected['ic'].to_numpy())
    pval = compute_pval_from_tstat(nw['t_stat'])

    ac = autocorrelation(ic_collected, 'ic', lags=(1,))
    
    turnover_df = compute_stability(lf, signal_col).collect()
    spread_df = decile_longshort_returns(lf, signal_col, fwd_ret_col).collect()
    spread_returns = spread_df['spread_ret'].to_list()

    recent_coverage = get_recent_coverage(lf, signal_col, n_days=365)
    

    row = {
        'signal': signal_col,
        'ic_mean': basic['mean'], 'ic_std': basic['std'], 'ic_ir': basic['ir'],
        'ic_tstat_naive': basic['t_stat'], 'ic_tstat_nw': nw['t_stat'], 'ic_pval_nw': pval,
        'ic_lag1_autocorr': ac['lag_1'][0],
        'hit_rate': basic['hit_rate'], 'n_days': basic['n_days'],
        'recent_coverage': recent_coverage,
        'avg_turnover': turnover_df['turnover'].mean(),
        'long_short_sharpe': sharpe_ratio(spread_returns),
        'long_short_max_dd': max_drawdown(spread_returns),
    }

    # if I want to show the ic_decay for multiple horizons, use that, but felt useless
    # decay_df = ic_decay(lf, signal_col, horizons=decay_horizons)
    # for h, m, ir in zip(decay_df['horizon'].to_list(), decay_df['mean'].to_list(), decay_df['ir'].to_list()):
    #     row[f'ic_mean_h{h}'] = m
    #     row[f'ic_ir_h{h}'] = ir

    return row





def compare_reports(lf, signal_cols, fwdret_col='fwdret'):
    '''
    signal_report() for several signals at once, one row per signal, 
    for a quick systematic side-by-side without opening a notebook.
    '''
    rows = [signal_report(lf, col, fwdret_col) for col in signal_cols]
    return pl.DataFrame(rows)



if __name__=='__main__':
    from src.signals.combine import make_signal
    with pl.Config(tbl_cols=-1):  # that's to print the whole report instead of truncating some columns

        lf = pl.scan_parquet('data/processed/features.parquet')
        lf = make_signal(lf, ['mom5', 'mom10', 'mom20', 'mom60', 'mom120', 'mom252'], method='decile')
        # lf = make_signal(lf, ['mom5', 'mom10', 'mom20', 'mom60', 'mom252'], method='zscore_tanh')
        # print(signal_report(lf, signal_col='mom20_decile'))
        print(compare_reports(lf, signal_cols=['mom5_decile', 'mom10_decile', 'mom20_decile', 'mom60_decile', 'mom120_decile', 'mom252_decile'], fwdret_col='fwdret'))
