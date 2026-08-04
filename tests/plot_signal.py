import matplotlib.pyplot as plt
from polars import col as c
import pandas as pd
import polars as pl
from src.signals.combine import make_signal


def plot_signals(df_wide:pl.DataFrame|pd.DataFrame, signal_name=None, ax=None, legend=False):
    '''
    Plots the signals of tickers of a wide dataframe, i.e., each column is one ticker's signal, each row is one date.
    '''
    f, ax = (plt.subplots(figsize=(12, 3)) if ax is None else (None, ax))
    if isinstance(df_wide, pl.DataFrame):
        df_wide = df_wide.to_pandas().set_index('date')
    df_wide.iloc[:].plot(ax=ax, legend=legend)
    ax.set_title(f'Evolution of {signal_name} signal for {df_wide.shape[1]} tickers')
    ax.set_ylabel('Signal')
    return ax

def plot_signal_coverage(lf, signal_col, ax=None):
    """Non-null signal count per date. Sudden drops flag early-history
    or grouping issues, exactly the kind of thing that breaks qcut."""
    counts = (
        lf.group_by('date')
        .agg(c(signal_col).is_not_null().sum().alias('n'))
        .sort('date')
        .collect()
    )
    f, ax = (plt.subplots(figsize=(12, 3)) if ax is None else (None, ax))
    ax.plot(counts['date'], counts['n'])
    ax.set_title(f'{signal_col}: non-null coverage per date')
    ax.set_ylabel('Number of signals')
    return ax


def plot_signal_moments(lf, signal_col, ax=None):
    '''
    Per-date mean and std of the signal. 
    Should be stable near 0 for mean (1 for std) for a zscore-based signal, drift or spikes flag a problem.
    '''
    moments = (
        lf.group_by('date')
        .agg(
                c(signal_col).mean().alias('mean'), 
                c(signal_col).std().alias('std')
            )
        .sort('date')
        .collect()
    )
    fig, ax = (plt.subplots(figsize=(12, 3)) if ax is None else (None, ax))
    ax.plot(moments['date'], moments['mean'], label='mean', lw=2)
    ax.axhline(0, color='grey', lw=0.5)
    ax.legend(loc=2)
    ax.set_ylabel('Signal mean')
    ax2 = ax.twinx()
    ax2.plot(moments['date'], moments['std'], label='std', color='C1', alpha=0.7)
    ax2.legend(loc=1)
    ax2.set_ylabel('Signal std')
    ax.set_title(f'{signal_col} mean and std')
    return ax, ax2


def plot_signal_heatmap(wide_df, signal_name=None, ax=None):
    '''
    wide_df: output of pivot_wide(). Ticker x date grid, imshow.
    Stripes or blocks of a single value are a structural-bug signature,
    same pattern as catching the earlier sorting bug visually.
    '''
    data = wide_df.drop('date').to_numpy()
    fig, ax = (plt.subplots(figsize=(18, 8)) if ax is None else (None, ax))
    im = ax.imshow(data.T, aspect='auto', cmap='PRGn', vmin=-1, vmax=1)
    ax.set_yticks(range(len(wide_df.columns) - 1))
    ax.set_yticklabels(wide_df.columns[1:])
    plt.colorbar(im, ax=ax)
    ax.set_title(f'Signal distribution of {signal_name} over time')
    return ax


def pivot_wide(df, value_col, index='date', on='ticker'):
    '''
    Long (ticker, date, value) panel -> wide (date rows, ticker columns). Requires an already-collected pl.DataFrame. 
        This is useful for a per-ticker time series plot, or a ticker-ticker correlation matrix of a signal.
    '''
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    return df.pivot(index=index, on=on, values=value_col, aggregate_function='first')






feature = 'std20'
signal_method = 'zscore_tanh'
n_random_tickers = 10

if __name__ == '__main__':
    INPUT = 'data/processed/features.parquet'
    lf = pl.scan_parquet(INPUT).sort(['ticker', 'date'])
    lf = lf.filter(
        c('ticker').is_in( c('ticker').unique().sample(n_random_tickers).implode() )
    )
    lf = make_signal(lf, feature, signal_method)
    wide_df = pivot_wide(
                            make_signal(lf, feature, signal_method), 
                            f'{feature}_{signal_method}', 
                            index='date',
                            on='ticker',
                        )

    ax = plot_signals(wide_df, signal_name=f'{feature}_{signal_method}', ax=None, legend=False)
    plt.show()
    ax = plot_signal_coverage(lf, signal_col=f'{feature}_{signal_method}', ax=None)
    plt.show()
    ax, ax2 = plot_signal_moments(lf, signal_col=f'{feature}_{signal_method}', ax=None)
    plt.show()

    
    ax = plot_signal_heatmap(wide_df, ax=None)
    plt.show()