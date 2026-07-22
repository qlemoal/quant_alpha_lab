import polars as pl
from polars import col as c


def add_adv(lf, window=20):

    if isinstance(window, list):
        exprs = []
        for w in window:
            exprs.append(
                c('dollar_volume')
                .rolling_mean(w)
                .over('ticker')
                .alias(f'adv{w}')
            )
        return lf.with_columns(exprs).sort(['ticker', 'date'])
    
    elif isinstance(window, int):

        return lf.with_columns(
                    (
                        c('dollar_volume')
                        .rolling_mean(window)
                        .over('ticker')
                        .alias(f'adv{window}')
                    )
                )
    
def add_log_adv(lf, window=20):

    if isinstance(window, list):
        exprs = []
        for w in window:
            exprs.append(
                c('dollar_volume')
                .rolling_mean(w)
                .over('ticker')
                .log()
                .alias(f'log_adv{w}')
            )
        return lf.with_columns(
                    exprs
                ).sort('ticker', 'date')
    
    elif isinstance(window, int):

        return lf.with_columns(
                    (
                        c('dollar_volume')
                        .rolling_mean(window)
                        .over('ticker')
                        .log()
                        .alias(f'log_adv{window}')
                    )
                ).sort('ticker', 'date')