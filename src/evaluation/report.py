import polars as pl

from src.evaluation.ic.core import compute_ic, summarize_ic
from src.evaluation.ic.metrics import newey_west_ic_tstat
from src.evaluation.performance import compute_turnover, quantile_spread_returns, sharpe_ratio


def signal_report(lf, signal_col, fwd_ret_col='fwd_ret', group='date'):
    """
    Single-call, numeric-only health check for a signal. No plots.
    Meant to run immediately after defining any new signal, before
    deciding whether it's worth a closer look via signal_diagnostics
    or evaluation/ic/plots.

    Returns a dict, one row's worth of numbers:
        ic_mean, ic_std, ic_ir       -- from summarize_ic
        ic_tstat_naive               -- assumes independent daily observations
        ic_tstat_nw                  -- Newey-West corrected, more trustworthy
                                         given overlapping-window features
        hit_rate, n_days             -- from summarize_ic
        avg_turnover                 -- mean day-over-day |signal change|
        long_short_sharpe            -- naive top-vs-bottom-decile paper Sharpe,
                                         pre-cost, directional sanity check only
    """
    ic_lf = compute_ic(lf, signal_col, fwd_ret_col, group)
    ic_collected = ic_lf.collect()
    ic_series = ic_collected['ic'].to_list()

    basic = summarize_ic(ic_collected)
    nw = newey_west_ic_tstat(ic_series)

    turnover_df = compute_turnover(lf, signal_col).collect()
    avg_turnover = turnover_df['turnover'].mean()

    spread_df = quantile_spread_returns(lf, signal_col, fwd_ret_col, group=group).collect()
    ls_sharpe = sharpe_ratio(spread_df['spread_ret'].to_list())

    return {
        'signal': signal_col,
        'ic_mean': basic['mean'],
        'ic_std': basic['std'],
        'ic_ir': basic['ir'],
        'ic_tstat_naive': basic['t_stat'],
        'ic_tstat_nw': nw['t_stat'],
        'hit_rate': basic['hit_rate'],
        'n_days': basic['n_days'],
        'avg_turnover': avg_turnover,
        'long_short_sharpe': ls_sharpe,
    }


def compare_reports(lf, signal_cols, fwd_ret_col='fwd_ret', group='date'):
    """
    signal_report() for several signals at once, one row per signal,
    for a quick systematic side-by-side without opening a notebook.
    """
    rows = [signal_report(lf, col, fwd_ret_col, group) for col in signal_cols]
    return pl.DataFrame(rows)