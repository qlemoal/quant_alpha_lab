from polars import col as c


def add_momentum(lf, window=20):
    return lf.with_columns(
                (
                    c('close')
                    .pct_change(window)
                    .over('ticker')
                    .alias(f'mom{window}')
                )
            )