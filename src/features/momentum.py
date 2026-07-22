from polars import col as c


def add_momentum(lf, window=20):

    if isinstance(window, list):
        exprs = []
        for w in window:
            exprs.append(
                c('close')
                .pct_change(w)
                .over('ticker')
                .alias(f'mom{w}')
            )
        return lf.with_columns(exprs).sort('ticker', 'date')
    
    elif isinstance(window, int):
        return lf.with_columns(
                    (
                        c('close')
                        .pct_change(window)
                        .over('ticker')
                        .alias(f'mom{window}')
                    )
                ).sort('ticker', 'date')
    