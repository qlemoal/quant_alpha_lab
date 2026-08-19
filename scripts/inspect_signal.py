


#  Standalone diagnostic runner: edit MODE/FEATURES/METHOD below, then run
#  "python scripts/inspect_signal.py". Top-of-file parameters, not a CLI,
#  same pattern as the rest of scripts/, on purpose.



import polars as pl
from setup import *
from src.signals.combine import make_signal
from src.evaluation.signals.signal_plots import one_pager
from src.evaluation.signals.report import compare_reports






MODE = 'single'                          # 'single' or 'compare'
FEATURES = ['mom20']#, 'mom60', 'mom252']   # single mode uses FEATURES[0] only
METHOD = 'zscore_tanh'
DESCENDING = False


INPUT = 'data/processed/features.parquet'

if __name__ == '__main__':
    lf = pl.scan_parquet(INPUT).sort(['ticker', 'date'])

    if MODE == 'single':
        feature = FEATURES[0]
        lf = make_signal(lf, feature, method=METHOD, descending=DESCENDING)
        one_pager(lf, signal_col=f'{feature}_{METHOD}', n_sample_tickers=5)

    elif MODE == 'compare':
        signal_cols = []
        for feat in FEATURES:
            lf = make_signal(lf, feat, method=METHOD, descending=DESCENDING)
            signal_cols.append(f'{feat}_{METHOD}')
        with pl.Config(tbl_cols=-1):  # that's to print the whole report instead of truncating some columns
            print(compare_reports(lf, signal_cols, fwdret_col='fwdret'))

    else:
        raise ValueError(f'unrecognized MODE: {MODE!r}, expected "single" or "compare"')