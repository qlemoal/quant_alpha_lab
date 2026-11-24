'''
Average dollar volume (ADV): the liquidity of a stock.

    adv{w}[t]     = mean(dollar_volume[t-w+1 .. t])
    log_adv{w}[t] = log(adv{w}[t])

Needs 'dollar_volume' (src/features/liquidity.py). Uses day t at the latest. The first w-1 rows of each
ticker are null. The backtest reads log_adv60 as the liquidity of a stock when it computes market impact.
'''

from polars import col as c

from src.utils.helpers import as_list


def add_adv(lf, window=20):
    '''Add adv{w} (dollars) for one window or a list of windows.'''
    return lf.sort(['ticker', 'date']).with_columns(
        [c('dollar_volume').rolling_mean(w).over('ticker').alias(f'adv{w}') for w in as_list(window)]
    )


def add_log_adv(lf, window=20):
    '''Add log_adv{w} for one window or a list of windows. Dollar volume is > 0, so the log is defined.'''
    return lf.sort(['ticker', 'date']).with_columns(
        [c('dollar_volume').rolling_mean(w).log().over('ticker').alias(f'log_adv{w}') for w in as_list(window)]
    )