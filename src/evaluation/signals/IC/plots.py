import matplotlib.pyplot as plt
import polars as pl
from polars import col as c

from evaluation.signals.ic.core import compute_ic


def plot_ic_time_series(ic_dict, ax=None):
    '''
    ic_dict: {method_name: pl.DataFrame with 'date', 'ic'}
    One line per method, daily Rank IC over time. Noisy day to day by
    nature, look for whether the sign/level is broadly consistent
    across sub-periods, not for a clean trend.
    '''
    _, ax = (plt.subplots(figsize=(12, 4)) if ax is None else (None, ax))
    for name, ic_df in ic_dict.items():
        ax.plot(ic_df['date'], ic_df['ic'], label=name, alpha=0.6, lw=0.8)
    ax.axhline(0, color='grey', lw=0.5)
    ax.legend()
    ax.set_title('Daily Rank IC by method')
    return ax


def plot_cumulative_ic(ic_dict, ax=None):
    '''
    Cumulative sum of daily IC. The slope is what matters, a
    consistently upward slope means the signal kept working, a flat
    or downward stretch means it stopped (or never worked) in that
    period. More informative than the single aggregate mean.    
    '''
    _, ax = (plt.subplots(figsize=(12, 4)) if ax is None else (None, ax))
    for name, ic_df in ic_dict.items():
        cum = ic_df['ic'].fill_null(0).cum_sum()
        ax.plot(ic_df['date'], cum, label=name)
    ax.legend()
    ax.set_title('Cumulative IC (consistency over time)')
    return ax


def plot_ic_distribution(ic_dict, ax=None):
    '''
    Boxplot of daily IC per method. Compares spread, not just the mean,
    two methods can have the same average IC with very different
    day-to-day reliability.
    '''
    _, ax = (plt.subplots(figsize=(8, 4)) if ax is None else (None, ax))
    data = [ic_df['ic'].drop_nulls().to_list() for ic_df in ic_dict.values()]
    ax.boxplot(data, labels=list(ic_dict.keys()))
    ax.axhline(0, color='grey', lw=0.5)
    ax.set_title('Distribution of daily IC by method')
    return ax


def plot_decile_spread(lf, signal_col, fwd_ret_col='fwdret', n_buckets=10, ax=None):
    '''
    Average forward return by cross-sectional bucket of signal_col.
    Classic monotonicity check: a genuinely useful signal should show
    a roughly increasing bar chart from bucket 0 (lowest) to bucket
    n_buckets-1 (highest). A flat or non-monotonic chart is a bad sign
    even if the aggregate IC looks okay.
    Note: the make_signal function with decile already splits into -1, 1 signal, so we lose the buckets in between.
    '''
    bucketed = (
        lf
        .with_columns(
            c(signal_col)
            .qcut(n_buckets, labels=[str(i) for i in range(n_buckets)])
            .over('date')
            .cast(pl.Utf8)
            .cast(pl.Int32)
            .alias('bucket')
        )
        .group_by('bucket')
        .agg(c(fwd_ret_col).mean().alias('mean_fwd_ret'))
        .sort('bucket')
        .collect()
    )
    _, ax = (plt.subplots(figsize=(6, 4)) if ax is None else (None, ax))
    ax.bar(bucketed['bucket'], bucketed['mean_fwd_ret'])
    ax.set_xlabel('bucket (0 = lowest signal, N-1 = highest)')
    ax.set_ylabel('mean forward return')
    ax.set_title(f'Decile spread: {signal_col}')
    return ax



def build_ic_dict(lf, feature_col, methods, fwd_ret_col='fwd_ret', **kwargs):
    '''
    Convenience wrapper: build a signal under each method, compute its IC time series, 
    return {method_name: ic_df}, ready to feed straight into any of the plot_* functions above.
    '''
    from src.signals.transforms import make_signal

    ic_dict = {}
    for method in methods:
        signal_lf = make_signal(lf, feature_col, method=method, **kwargs)
        signal_col = f'{feature_col}_{method}'
        ic_dict[method] = compute_ic(signal_lf, signal_col, fwd_ret_col).collect()
    return ic_dict