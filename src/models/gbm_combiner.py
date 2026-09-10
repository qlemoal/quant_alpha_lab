'''
TO INSPECT

Structurally parallel to elastic_net_combiner.py, same fold-generation and
design-matrix machinery reused directly, not duplicated. The real
difference, and the thing worth understanding before using this, is that
GBM breaks the trick that made EN's combined_score_column() a one-liner:
a linear model's prediction is just a weighted sum, expressible as a
polars expression (coef_[0]*signal_0 + coef_[1]*signal_1 + ...), so the
combined score could be computed lazily, in polars, without ever leaving
the lazyframe. A gradient-boosted tree ensemble's prediction is hundreds
of sequential tree traversals, there's no algebraic expression to hand to
polars. combined_score_from_gbm() below has to materialize the feature
matrix, call model.predict(), and attach the result back as a column,
an eager, not lazy, step. Worth knowing this if you're chaining this into
a larger lazy pipeline elsewhere, this breaks the chain at this point,
on purpose, not an oversight.

Why compare against GBM specifically, not some other alternative: it's
the natural nonlinear counterpoint to Elastic Net. If GBM meaningfully
outperforms EN, that's evidence of nonlinear interactions between
candidate signals real enough to matter (e.g. momentum only working
conditional on low volatility, a threshold or interaction effect a linear
model can't express regardless of regularization). If it doesn't, that's
itself informative: no found evidence of interaction effects worth the
added complexity, interpretability, and overfitting risk, GBM's OWN
literature (Friedman, 2001, Greedy Function Approximation: A Gradient
Boosting Machine, Annals of Statistics 29(5), 1189-1232) is explicit that
more flexible function classes buy predictive power ONLY if the true
relationship actually has that shape, otherwise they mostly buy variance.
That's precisely the comparison PBO exists to adjudicate honestly rather
than by eye.
'''

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV


# Small, fixed grid, a stated convention decided before looking at
# results, same principle as the EN combiner's l1_ratio grid. Kept
# deliberately small: GBM has more hyperparameters than EN and each
# CPCV fold refits the model from scratch, a large grid gets expensive
# fast, this is a starting point to argue with, not a settled choice.
DEFAULT_PARAM_GRID = {
    'max_depth': [2, 3, 4],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_iter': [50, 100],
}


def fit_gbm_combiner(
    panel: pl.DataFrame,
    signal_cols: list[str],
    fwd_ret_col: str,
    cv_folds: list[tuple[np.ndarray, np.ndarray]],
    param_grid: dict = DEFAULT_PARAM_GRID,
) -> dict:
    '''
    cv_folds: the SAME row-index folds used for the EN combiner's search,
    passed in rather than rebuilt here. This matters for the comparison
    to be fair: if GBM and EN were tuned on different folds, any
    difference in their selected hyperparameters could partly reflect
    different data splits rather than a genuine difference in what each
    model family can learn. Reusing identical folds removes that
    confound. No q-value weighting step here, unlike the EN combiner,
    tree splits aren't sensitive to input SCALE the way a linearly
    penalized model is, rescaling a column by (1-q) wouldn't change
    which splits a tree chooses, only where the threshold falls
    numerically. If you want q-value information to influence GBM, it
    would need to enter as a sample_weight or a training-row inclusion
    threshold instead, a genuinely different mechanism, not implemented
    here, worth a real conversation before adding it.
    '''
    X = panel.select(signal_cols).to_numpy()
    y = panel[fwd_ret_col].to_numpy()

    search = GridSearchCV(
        HistGradientBoostingRegressor(random_state=0),
        param_grid,
        cv=cv_folds,
        n_jobs=-1,
    )
    search.fit(X, y)

    return {
        'model': search.best_estimator_,
        'best_params': search.best_params_,
        'signal_cols': signal_cols,
    }


def combined_score_from_gbm(
    panel_or_lf,
    fit_result: dict,
    out_col: str = 'combined_score_gbm',
) -> pl.DataFrame:
    '''
    Deliberately returns an eager DataFrame, not a lazyframe, see the
    module docstring for why: model.predict() has no lazy/polars-native
    form, this step has to materialize.
    '''
    df = panel_or_lf.collect() if isinstance(panel_or_lf, pl.LazyFrame) else panel_or_lf
    X = df.select(fit_result['signal_cols']).to_numpy()
    preds = fit_result['model'].predict(X)
    return df.with_columns(pl.Series(out_col, preds))