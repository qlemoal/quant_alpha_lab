'''
Created autocorrelation outside evaluation folder because we can apply it to any series, not just features. That may be useful later on.
'''


import polars as pl
from polars import col as c
from src.utils.helpers import as_list



def get_recent_coverage(lf, col, n_days=365):
    '''
    Get the fraction of non-null days in the last n_days of the column col in lf.
    '''
    recent_cutoff = (
        lf.select(
            c('date').max()
        ).collect().item() - pl.duration(days=n_days)
        )
    return (
        lf.filter(c('date') >= recent_cutoff)
        .select(c(col).is_not_null().mean())
        .collect()
        .item()
    )


def compute_pval_from_tstat(t_stats):
    '''
    t_stats can be a list of t-stats or a unique t-stat
    '''
    from scipy.stats import norm

    t_stats = as_list(t_stats)
    if len(t_stats) == 1:
        return  2 * (1 - norm.cdf(abs(t_stats[0]))) if t_stats[0] is not None else float('nan')
    else: 
        return [2 * (1 - norm.cdf(abs(tstat))) if t_stats[0] is not None else float('nan') for tstat in t_stats]



def autocorrelation(lf:pl.LazyFrame, col_name:str, lags=(1, 5, 10, 20), group=None):
    '''
    Lag-k autocorrelation of 'col_name', for each lag in 'lags'.

    group=None: 'col_name' is one flat series, one row per observation
        already (e.g. a daily IC series, one value per date). Returns one
        row: {lag_k: autocorr}.
    group='ticker' (or any grouping column): computes lag-k autocorrelation
        independently within each group's own time series, then reports the
        cross-group mean and std. A single "the" autocorrelation doesn't
        exist across a panel, some tickers are more/less persistent than
        others, the std tells you how much that varies.
    '''


    if group is None:
        lf_sorted = lf.sort('date')
        exprs = [pl.corr(c(col_name), c(col_name).shift(k)).alias(f'lag_{k}') for k in lags]
        result = lf_sorted.select(exprs)
        return result.collect() if isinstance(result, pl.LazyFrame) else result


    lf_shifted = lf.sort([group, 'date']).with_columns(
        [ c(col_name).shift(k).over(group).alias(f'_lag{k}') for k in lags ]
    )

    per_group = lf_shifted.group_by(group).agg(
        [pl.corr(c(col_name), c(f'_lag{k}')).alias(f'lag_{k}') for k in lags]
    )

    pg = per_group.collect() if isinstance(per_group, pl.LazyFrame) else per_group

    rows = []
    for k in lags:
        vals = pg[f'lag_{k}'].drop_nulls()
        rows.append({'lag': k, 'mean': vals.mean(), 'std': vals.std(), 'n_groups': vals.len()})
    return pl.DataFrame(rows)