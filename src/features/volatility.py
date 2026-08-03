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
        return lf.sort(['ticker', 'date']).with_columns(exprs)
    
    elif isinstance(window, int):
        return lf.sort(['ticker', 'date']).with_columns(
                    (
                        c('logret')
                        .rolling_std(window)
                        .over('ticker')
                        .alias(f'std{window}')
                    )
                )