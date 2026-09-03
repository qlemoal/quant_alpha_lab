'''
Demo, not a merged design: shows the EN combiner fit through CPCV folds
(for the alpha/l1_ratio search itself) with a final, genuinely untouched
holdout chunk for the causal-realism sanity check, per the nested design
discussed. Reuses build_design_matrix / apply_q_value_weighting /
coefficient_stability_by_fold / combined_score_column unchanged, only the
FOLD GENERATION step (Step 1 in elastic_net_combiner.py) is swapped.
'''

import numpy as np
import polars as pl
import datetime

from src.models.elastic_net_combiner import (
    build_design_matrix, apply_q_value_weighting, combined_score_column,
)
from src.validation.cpcv import cpcv
from src.evaluation.signals.report import signal_report
from sklearn.linear_model import ElasticNetCV


N_DATES = 700
N_TICKERS = 30
TRUE_COEF = 0.8
NOISE_STD = 3.0

# CPCV search region config
N_BLOCKS = 10
N_TEST_BLOCKS = 2
PURGE_W = 6      # horizon + 1, small here for a fast demo
EMBARGO_W = 1    # matches the derived EMBARGO=1 from earlier

HOLDOUT_DATES = 150  # genuinely untouched, never enters the CPCV search


def build_synthetic_panel(seed=0):
    rng = np.random.default_rng(seed)
    calendar_dates = [datetime.date(2015, 1, 1) + datetime.timedelta(days=i) for i in range(N_DATES)]
    dates = [d for d in calendar_dates for _ in range(N_TICKERS)]
    tickers = np.tile([f'T{i:02d}' for i in range(N_TICKERS)], N_DATES)
    good_signal = rng.normal(0, 1, N_DATES * N_TICKERS)
    noise_signal = rng.normal(0, 1, N_DATES * N_TICKERS)
    fwdret = TRUE_COEF * good_signal + rng.normal(0, NOISE_STD, N_DATES * N_TICKERS)
    return pl.DataFrame({
        'date': dates, 'ticker': tickers,
        'good_signal': good_signal, 'noise_signal': noise_signal, 'fwdret': fwdret,
    }).lazy()


def cpcv_row_index_folds(panel_dates, unique_dates, n_blocks, n_test_blocks, purge_w, embargo_w):
    '''
    Same job as build_row_index_folds() in the combiner, but wraps cpcv()
    instead of rolling_purged_embargoed_splits(). cpcv() yields DATE
    POSITION indices (0..n_dates-1) into unique_dates, not row indices
    into the panel, so this still needs the date-value lookup step,
    fold_to_row_indices() in the combiner works on Fold namedtuples of
    actual dates, cpcv()'s output shape is different (raw index arrays,
    not named start/end), hence a separate small adapter rather than reuse.
    '''
    n_dates = len(unique_dates)
    folds = []
    for train_date_idx, test_date_idx, _test_ids in cpcv(n_dates, n_blocks, n_test_blocks, purge_w, embargo_w):
        train_dates = unique_dates[train_date_idx]
        test_dates = unique_dates[test_date_idx]
        train_row_idx = np.flatnonzero(np.isin(panel_dates, train_dates))
        test_row_idx = np.flatnonzero(np.isin(panel_dates, test_dates))
        folds.append((train_row_idx, test_row_idx))
    return folds


if __name__ == '__main__':
    lf = build_synthetic_panel()
    signal_cols = ['good_signal', 'noise_signal']
    q_values = {'good_signal': 0.01, 'noise_signal': 0.6}

    panel = build_design_matrix(lf, signal_cols, fwd_ret_col='fwdret')
    unique_dates = panel['date'].unique().sort().to_numpy()

    # --- split off a genuinely untouched final holdout, never seen by CPCV ---
    holdout_start = unique_dates[-HOLDOUT_DATES]
    gap_cutoff = unique_dates[-HOLDOUT_DATES - (PURGE_W + EMBARGO_W)]  # search region stops here, before the gap

    search_panel = panel.filter(pl.col('date') <= gap_cutoff)
    holdout_panel = panel.filter(pl.col('date') >= holdout_start)
    search_unique_dates = search_panel['date'].unique().sort().to_numpy()

    print(f'Search region: {len(search_unique_dates)} dates, {len(search_panel)} rows')
    print(f'Holdout: {HOLDOUT_DATES} dates, {len(holdout_panel)} rows, gap of {PURGE_W + EMBARGO_W} dates between them\n')

    # --- CPCV folds for the alpha/l1_ratio search, search region only ---
    search_panel_dates = search_panel['date'].to_numpy()
    cv_folds = cpcv_row_index_folds(
        search_panel_dates, search_unique_dates, N_BLOCKS, N_TEST_BLOCKS, PURGE_W, EMBARGO_W
    )
    print(f'CPCV generated {len(cv_folds)} combinatorial folds (vs. ~5-6 under plain walk-forward)\n')

    X = search_panel.select(signal_cols).to_numpy()
    y = search_panel['fwdret'].to_numpy()
    X_weighted = apply_q_value_weighting(X, signal_cols, q_values)

    model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=cv_folds, n_jobs=-1)
    model.fit(X_weighted, y)

    print(f'CPCV-selected alpha={model.alpha_:.4f}, l1_ratio={model.l1_ratio_}')
    for col, coef in zip(signal_cols, model.coef_):
        print(f'  {col}: {coef:.4f}')

    # --- final sanity check: evaluate THIS fixed model on the untouched holdout ---
    fit_result = {'model': model, 'signal_cols': signal_cols, 'q_values': q_values}
    holdout_lf = holdout_panel.lazy()
    combined_lf = combined_score_column(holdout_lf, fit_result)
    report = signal_report(combined_lf, 'combined_score', 'fwdret')

    print('\nHoldout (never touched during CPCV search), signal_report():')
    for k, v in report.items():
        print(f'  {k}: {v}')