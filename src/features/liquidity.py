from polars import col as c


def add_dollar_volume(lf):
    return lf.with_columns(
                dollar_volume = c('close') * c('volume')
            ).sort('ticker', 'date')