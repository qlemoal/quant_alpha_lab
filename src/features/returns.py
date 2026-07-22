from polars import col as c


def add_log_returns(lf):
    return lf.with_columns(
                logret = c('close').log().diff().over('ticker')
            ).sort('ticker', 'date')