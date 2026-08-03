import polars as pl
from polars import col as c

from setup import *
from src.features.returns import *
from src.features.momentum import *
from src.features.volatility import *
from src.features.liquidity import *
from src.features.adv import *
from src.features.beta import *


print('--> Start')

INPUT = 'data/processed/prices.parquet'
OUTPUT = 'data/processed/features.parquet'

print('    Importing prices from ', INPUT)

lf = pl.scan_parquet(INPUT)

lf = lf.sort(
    ['ticker', 'date']
)

print('    Building features')

lf = add_log_returns(lf)

lf = add_fwd_returns(lf)

lf = add_beta(lf, window=[60, 252])  #  Maybe one window is usually enough, but I choose two for comparison

lf = add_momentum(lf, window=[5, 10, 20, 60, 120, 252])

lf = add_volatility(lf, window=[5, 10, 20, 60, 120, 252])

lf = add_dollar_volume(lf)

lf = add_log_adv(lf, window=[5, 10, 20, 60, 120, 252])


# Do not drop nulls as we would not remove whole rows that could have useful 
# features depending on the alpha we want to build
# lf = lf.drop_nulls(
#     ['logret', 'ret5', 'ret20', 'ret60', 'std5', 'std20', 'std60', 'adv5', 'adv20', 'adv60']
# )
#
# We sort the lf after each feature we add because we see issues arising sometimes.

print('    Saving features to ', OUTPUT)

lf.sink_parquet(OUTPUT) #  Note that sink_parquet does not keep the order, it will save in chunks in parallel. 

print('--> Done')