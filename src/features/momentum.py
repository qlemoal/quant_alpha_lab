'''
Momentum features: the past return over w trading days.

    mom{w}[t] = close[t] / close[t-w] - 1        (adjusted close, so splits and dividends are included)

Uses the close of day t at the latest: known when the signal is computed. The first w rows of each
ticker are null. No "skip the last month" convention is applied: add one more column if you want it.
'''

from polars import col as c

from src.utils.helpers import as_list


def add_momentum(lf, window=20):
    '''Add mom{w} for one window or a list of windows.'''
    return lf.sort(['ticker', 'date']).with_columns(
        [c('close').pct_change(w).over('ticker').alias(f'mom{w}') for w in as_list(window)]
    )