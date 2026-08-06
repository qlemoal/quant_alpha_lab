import matplotlib.pyplot as plt
from polars import col as c
import pandas as pd
import polars as pl
from src.signals.combine import make_signal
from src.evaluation.IC.signal_diagnostics import one_pager





feature = 'mom252'
signal_method = 'zscore_tanh'
n_random_tickers = 10

if __name__ == '__main__':
    INPUT = 'data/processed/features.parquet'
    lf = pl.scan_parquet(INPUT).sort(['ticker', 'date'])
    lf = make_signal(lf, feature, method=signal_method, descending=False, scale=1)
    
    one_pager(lf, signal_col=f'{feature}_{signal_method}', n_sample_tickers=5)