'''
Volatility features: the standard deviation of the daily log-returns over w days.

    vol{w}[t] = std(logret[t-w+1 .. t])          DAILY, not annualised (multiply by sqrt(252) for annual)

Needs the column 'logret' (src/features/returns.py). Uses the return of day t at the latest.
The first w rows of each ticker are null (a full window is required).
'''

from polars import col as c

from src.utils.helpers import as_list


def add_volatility(lf, window=20):
    '''Add vol{w} for one window or a list of windows.'''
    return lf.sort(['ticker', 'date']).with_columns(
        [c('logret').rolling_std(w).over('ticker').alias(f'vol{w}') for w in as_list(window)]
    )