from polars import col as c


def add_volatility(lf, window=20):
    
    if isinstance(window, list):
        exprs = []
        for w in window:
            exprs.append(
                c('logret')
                .rolling_std(w)
                .over('ticker')
                .alias(f'std{w}')
            )
        return lf.with_columns(exprs).sort('ticker', 'date')
    
    elif isinstance(window, int):
        return lf.with_columns(
                    (
                        c('logret')
                        .rolling_std(window)
                        .over('ticker')
                        .alias(f'std{window}')
                    )
                ).sort('ticker', 'date')