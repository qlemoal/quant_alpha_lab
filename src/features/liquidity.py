'''
Dollar volume of one day: what was traded, in dollars.

    dollar_volume[t] = close_raw[t] * volume[t]

WHY close_raw AND NOT close
    In ohlcv.parquet, 'close' is the ADJUSTED close (splits and dividends, scaled to today's prices).
    'close_raw' is the close adjusted for splits only (Yahoo "Close"). The volume is split-adjusted too, so the product is right.
    Dollars traded = price as traded * shares traded. With the adjusted close, the dollar volume of
    old years would be too small for every stock that paid dividends since. The backtest uses it
    to size the market impact, so a bias here is a bias in the costs.
'''

from polars import col as c


def add_dollar_volume(lf):
    '''Add dollar_volume = close_raw * volume. Raises if close_raw is missing (rerun 02_etl_prices.py).'''
    if 'close_raw' not in lf.collect_schema().names():
        raise ValueError("column 'close_raw' not found: rerun scripts/pipeline/02_etl_prices.py")
    return lf.with_columns((c('close_raw') * c('volume')).alias('dollar_volume'))
