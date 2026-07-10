import polars as pl
from polars import col as c

INPUT = 'data/processed/prices.parquet'
OUTPUT = 'data/processed/base_features.parquet'

lf = pl.scan_parquet(INPUT)

lf = lf.sort(
    ['ticker', 'date']
)

lf = lf.with_columns(
    dollar_volume = c('volume') * c('close')
)

lf = lf.with_columns(
    logret = c('close').log().diff().over('ticker')
)

lf = lf.with_columns(
    ret5 = c('close').pct_change(5).over('ticker'),
    ret20 = c('close').pct_change(20).over('ticker'),
    ret60 = c('close').pct_change(60).over('ticker')
)

lf = lf.with_columns(
    std5 = c('close').rolling_std(5).over('ticker'),
    std20 = c('close').rolling_std(20).over('ticker'),
    std60 = c('close').rolling_std(60).over('ticker')
)

lf = lf.with_columns(
    adv5 = c('dollar_volume').rolling_mean(5).over('ticker'),
    adv20 = c('dollar_volume').rolling_mean(20).over('ticker'),
    adv60 = c('dollar_volume').rolling_mean(60).over('ticker')
)

lf = lf.drop_nulls(
    ['logret', 'ret5', 'ret20', 'ret60', 'std5', 'std20', 'std60', 'adv5', 'adv20', 'adv60']
)

lf.sink_parquet(OUTPUT)
