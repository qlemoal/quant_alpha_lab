#  pre-IC signal sanity visuals, coverage/moments/heatmap/sample lines.


import matplotlib.pyplot as plt
import polars as pl
from polars import col as c

from src.utils.panels import pivot_wide


def _sample_ticker_lines(ax, wide_df, n=4, last_n_dates=750):
    '''
    Fewer tickers, recent window only, small multiples would be even better for n > ~4, but this keeps it to one axis for the one-pager
    '''
    tickers = [col for col in wide_df.columns if col != 'date'][:n]
    recent = wide_df.tail(last_n_dates) if last_n_dates else wide_df
    for t in tickers:
        ax.plot(recent['date'], recent[t], alpha=0.8, lw=1, label=t)
    ax.axhline(0, color='grey', lw=0.5)
    ax.legend(fontsize=7)
    ax.set_title(f'sample ticker signals (last {last_n_dates} dates)' if last_n_dates else 'sample ticker signals')


def _coverage(ax, lf, signal_col):
    counts = (
        lf.group_by('date')
        .agg(c(signal_col).is_not_null().sum().alias('n'))
        .sort('date')
        .collect()
    )
    ax.plot(counts['date'], counts['n'])
    ax.set_title('non-null coverage per date')


def _moments(ax, lf, signal_col):
    moments = (
        lf.group_by('date')
        .agg(c(signal_col).mean().alias('mean'), c(signal_col).std().alias('std'))
        .sort('date')
        .collect()
    )
    ax.plot(moments['date'], moments['mean'], label='mean')
    ax.plot(moments['date'], moments['std'], label='std')
    ax.axhline(0, color='grey', lw=0.5)
    ax.legend(fontsize=7)
    ax.set_title('per-date mean / std')


def _heatmap(ax, fig, wide_df, n_date_ticks=8):
    data = wide_df.drop('date').to_numpy()
    im = ax.imshow(data.T, aspect='auto', cmap='RdBu', vmin=-1, vmax=1)

    dates = wide_df['date'].to_list()
    n = len(dates)
    step = max(1, n // n_date_ticks)
    tick_idx = list(range(0, n, step))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([str(dates[i]) for i in tick_idx], rotation=45, ha='right', fontsize=7)

    tickers = wide_df.columns[1:]
    if len(tickers) <= 30:
        ax.set_yticks(range(len(tickers)))
        ax.set_yticklabels(tickers, fontsize=6)
    else:
        ax.set_ylabel('ticker')

    ax.set_title('signal heatmap')
    fig.colorbar(im, ax=ax, fraction=0.02)


def one_pager(lf, signal_col, n_sample_tickers=6, fs=(16, 10)):
    '''
    All signal-sanity diagnostics in one figure: sample ticker lines, coverage, per-date mean/std, and the ticker x date heatmap. 
        Meant to catch structural bugs at a glance before trusting anything to IC evaluation.
    '''
    wide_df = pivot_wide(lf, signal_col)

    f, axes = plt.subplots(2, 2, figsize=fs)
    _sample_ticker_lines(axes[0, 0], wide_df, n=n_sample_tickers)
    _coverage(axes[0, 1], lf, signal_col)
    _moments(axes[1, 0], lf, signal_col)
    _heatmap(axes[1, 1], f, wide_df)

    f.suptitle(signal_col)
    f.tight_layout()
    return f