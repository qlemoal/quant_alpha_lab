'''

TO INSPECT

Compares the Elastic Net and GBM combiners honestly, via PBO, not by
eyeballing which has the higher headline IC. See the conversation for the
full step-by-step explanation of what PBO is and why headline comparison
isn't enough, summary here: whichever model looks better on one number is
exactly the kind of comparison PBO exists to stress-test, a single
comparison can't tell you if that ranking would hold up under a different
slice of the same data.

DESIGN, deliberately scoped down from full textbook CSCV, stated
explicitly rather than silently:
    True CSCV operates on an already-computed T x N performance matrix
    built from genuinely out-of-sample predictions at every sub-period,
    which in principle needs each candidate refit inside every sub-period
    combination. Doing that rigorously for two full hyperparameter
    searches would be expensive and, more importantly, a much bigger
    piece of code to get right. This script instead: (1) tunes both
    models ONCE via CPCV on a shared search region, using IDENTICAL row-
    index folds for both (see gbm_combiner.py's docstring on why that
    matters for fairness), (2) evaluates both FIXED, already-chosen
    models on a single shared holdout, genuinely untouched by either
    model's fitting, split into T contiguous sub-blocks for the PBO
    matrix. This is still honest, genuinely out-of-sample data feeds the
    PBO computation, it's just a lighter-weight adaptation, not the full
    nested-refit version. Worth upgrading later if this becomes a
    recurring, high-stakes comparison rather than a one-off check.
'''

import numpy as np
import polars as pl
import datetime

from src.models.elastic_net_combiner import (
    build_design_matrix, apply_q_value_weighting, combined_score_column,
)
from src.models.gbm_combiner import fit_gbm_combiner, combined_score_from_gbm
from src.validation.cpcv import cpcv, total_embargo_window, probability_of_backtest_overfitting
from sklearn.linear_model import ElasticNetCV


N_DATES = 900
N_TICKERS = 30
TRUE_COEF = 0.8
NOISE_STD = 3.0

N_BLOCKS, N_TEST_BLOCKS = 10, 2
HORIZON = 5
PURGE_W = HORIZON + 1
AUTOCORR_EMBARGO = 1  # placeholder, would come from scripts/embargo_selection.py on real data
EMBARGO_W = total_embargo_window(PURGE_W, AUTOCORR_EMBARGO, combine='sum')

HOLDOUT_DATES = 200
N_HOLDOUT_BLOCKS = 8  # T in the PBO matrix, must divide evenly per probability_of_backtest_overfitting()'s own requirement


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
    n_dates = len(unique_dates)
    folds = []
    for train_date_idx, test_date_idx, _ in cpcv(n_dates, n_blocks, n_test_blocks, purge_w, embargo_w):
        train_dates = unique_dates[train_date_idx]
        test_dates = unique_dates[test_date_idx]
        train_row_idx = np.flatnonzero(np.isin(panel_dates, train_dates))
        test_row_idx = np.flatnonzero(np.isin(panel_dates, test_dates))
        folds.append((train_row_idx, test_row_idx))
    return folds


def per_block_sharpe(returns: np.ndarray, n_blocks: int) -> np.ndarray:
    '''
    Splits a 1D daily-return series into n_blocks equal contiguous
    chunks, returns one Sharpe per chunk. This is the raw material for
    probability_of_backtest_overfitting()'s performance matrix, one
    column of this per candidate.
    '''
    blocks = np.array_split(returns, n_blocks)
    sharpes = []
    for b in blocks:
        mu, sigma = b.mean(), b.std(ddof=1)
        sharpes.append(mu / sigma if sigma > 0 else 0.0)
    return np.array(sharpes)


if __name__ == '__main__':
    lf = build_synthetic_panel()
    signal_cols = ['good_signal', 'noise_signal']
    q_values = {'good_signal': 0.01, 'noise_signal': 0.6}

    panel = build_design_matrix(lf, signal_cols, fwd_ret_col='fwdret')
    unique_dates = panel['date'].unique().sort().to_numpy()

    holdout_start = unique_dates[-HOLDOUT_DATES]
    gap_cutoff = unique_dates[-HOLDOUT_DATES - EMBARGO_W]
    search_panel = panel.filter(pl.col('date') <= gap_cutoff)
    holdout_panel = panel.filter(pl.col('date') >= holdout_start)
    search_unique_dates = search_panel['date'].unique().sort().to_numpy()
    search_panel_dates = search_panel['date'].to_numpy()

    cv_folds = cpcv_row_index_folds(
        search_panel_dates, search_unique_dates, N_BLOCKS, N_TEST_BLOCKS, PURGE_W, EMBARGO_W
    )
    print(f'Search region: {len(search_unique_dates)} dates, {len(cv_folds)} CPCV folds (shared by both models)\n')

    # --- Elastic Net, tuned via CPCV ---
    X = search_panel.select(signal_cols).to_numpy()
    y = search_panel['fwdret'].to_numpy()
    X_weighted = apply_q_value_weighting(X, signal_cols, q_values)
    en_model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=cv_folds, n_jobs=-1)
    en_model.fit(X_weighted, y)
    en_fit = {'model': en_model, 'signal_cols': signal_cols, 'q_values': q_values}
    print(f'EN:  alpha={en_model.alpha_:.4f}, l1_ratio={en_model.l1_ratio_}, coefs={dict(zip(signal_cols, en_model.coef_))}')

    # --- GBM, tuned via the SAME CPCV folds ---
    gbm_fit = fit_gbm_combiner(search_panel, signal_cols, 'fwdret', cv_folds)
    print(f"GBM: best_params={gbm_fit['best_params']}\n")

    # --- shared, untouched holdout: build both candidates' daily long-short return series ---
    en_holdout = combined_score_column(holdout_panel.lazy(), en_fit).collect()
    gbm_holdout = combined_score_from_gbm(holdout_panel, gbm_fit)

    def daily_long_short_return(df: pl.DataFrame, score_col: str) -> np.ndarray:
        # simplest possible long-short: mean fwdret of top-half score minus bottom-half, per date
        daily = (
            df.with_columns(
                (pl.col(score_col) > pl.col(score_col).median().over('date')).alias('long')
            )
            .group_by(['date', 'long'])
            .agg(pl.col('fwdret').mean())
            .pivot(values='fwdret', index='date', on='long')
            .sort('date')
        )
        return (daily['true'] - daily['false']).to_numpy()

    en_returns = daily_long_short_return(en_holdout, 'combined_score')
    gbm_returns = daily_long_short_return(gbm_holdout, 'combined_score_gbm')

    en_block_sharpes = per_block_sharpe(en_returns, N_HOLDOUT_BLOCKS)
    gbm_block_sharpes = per_block_sharpe(gbm_returns, N_HOLDOUT_BLOCKS)

    performance_matrix = np.column_stack([en_block_sharpes, gbm_block_sharpes])
    print('Per-block holdout Sharpe (rows=blocks, cols=[EN, GBM]):')
    print(performance_matrix)

    result = probability_of_backtest_overfitting(performance_matrix, n_splits=N_HOLDOUT_BLOCKS)
    print(f"\nPBO = {result['pbo']:.3f}  ({result['n_combinations']} IS/OOS splits)")