'''
GOAL: combine several already-evaluated candidate signals (the survivors of signal_report() + fdr.fdr_report()) 
    into a single "score" via Elastic Net regression against forward returns, with alpha/l1_ratio chosen using 
    the leakage-safe walk-forward CV (src/validation/walk_forward.py), or the combinatorial purged CV (src/validation/cpcv.py), 
    not sklearn's default random k-fold, which would silently reintroduce exactly the leakage the CV module exists to prevent.


References:
    Zou, H. & Hastie, T. (2005). Regularization and variable selection via
        the elastic net. JRSS B, 67(2), 301-320. Why EN ->
        Lasso alone: candidate signals here are correlated by construction
            (mom20 / mom60 / mom252 share overlapping return history). Pure
            Lasso handles correlated groups badly, tends to arbitrarily keep
            one member of a correlated group and zero the rest, unstable
            across resamples ("if predictors are correlated, lasso arbitrarily
            selects one" - Zou & Hastie's own framing). 
        Ridge: keeps everything, never zeros anything, no sparsity, harder to say which signals
            survived combination. 
        Elastic Net: L1+L2 mix is the standard answer to exactly this setup.
    Zou, H. (2006). The adaptive lasso and its oracle properties. JASA, 101(476), 1418-1429.
        Theoretical grounding for the per-feature penalty-weighting trick
        used in apply_q_value_weighting() below.
    Lopez de Prado, M. (2018). Advances in Financial Machine Learning.
        Wiley, Ch. 7 (purge/embargo, reused directly via walk_forward.py and cpcv.py).
'''

import numpy as np
import polars as pl
from polars import col as c
from sklearn.linear_model import ElasticNet, ElasticNetCV

from src.validation.walk_forward_cv import walk_forward_cv, Fold


# =============================================================================
# STEP 1: fold-to-row-index adapter
# =============================================================================




# rolling_purged_embargoed_splits() yields Fold(train_start, train_end, test_start, test_end) as DATES. 
# But the design matrix here has one row per (date, ticker), not one row per date, sklearn's `cv` parameter needs
# integer ROW indices into that matrix. This adapter is the direct payoff of having built CV myself: 
# everything downstream reuses it unchanged, no separate leakage logic duplicated here.

def fold_to_row_indices(panel_dates:np.ndarray, fold:Fold) -> tuple[np.ndarray, np.ndarray]:
    '''
    panel_dates: 1D array, one entry per ROW of the design matrix (i.e. not unique, repeats once per ticker per date). 
        Must be date-typed, same dtype as what walk_forward_cv() was called with.
    fold: a single Fold namedtuple from walkforward_cv.py.

    Returns (train_row_idx, test_row_idx), both int arrays, suitable for
        a single entry in the list you pass to ElasticNetCV(cv=...).
    '''
    train_mask = (panel_dates >= fold.train_start) & (panel_dates <= fold.train_end)
    test_mask = (panel_dates >= fold.test_start) & (panel_dates <= fold.test_end)
    return np.flatnonzero(train_mask), np.flatnonzero(test_mask)  # flatnonzero returns the indices of the elements that are non-zero (flattened).


def build_row_index_folds(
    panel_dates: np.ndarray,
    unique_dates: np.ndarray,
    train_window: int,
    horizon: int,
    test_window: int,
    embargo: int,
    next_fold: str | int = 'consecutive',
) -> list[tuple[np.ndarray, np.ndarray]]:
    '''
    Wraps walk_forward_cv() + fold_to_row_indices() into the exact list-of-(train_idx, test_idx) format sklearn's cv= expects.

    unique_dates: the deduplicated, sorted date array splits.py operates
        on (what you'd pass to rolling_purged_embargoed_splits directly).
    panel_dates: the (date, ticker)-row-level date array from the actual
        design matrix, used only for the row-index lookup.

    NOTE which next_fold mode to use here is a real, unsettled choice
    (see conversation): 'consecutive' gives ~5-6 properly-embargoed,
    close-to-independent folds, appropriate for THIS use (averaging fold
    validation scores into a single alpha/l1_ratio choice). An int
    next_fold (e.g. next_fold=test_window for dense back-to-back tiling)
    gives many more folds but does NOT respect embargo between them and
    should not be averaged over for a hyperparameter decision, save that
    mode for rolling-diagnostic plots instead. Defaulting to 'consecutive'
    here on purpose, for that reason.
    '''
    folds = list(walk_forward_cv(
        unique_dates, train_window, horizon, test_window, embargo, next_fold
    ))
    return [fold_to_row_indices(panel_dates, f) for f in folds]


# =============================================================================
# STEP 2: build the panel design matrix
# =============================================================================



def build_design_matrix( signals_lf:pl.LazyFrame, signal_cols:list[str], fwdret_col:str, date_col: str='date', ticker_col:str='ticker' ) -> pl.DataFrame:
    '''
    Transforms lf to with multiple (date, ticker) rows to lf.DataFrame with one row per (date, ticker). 
    Columns: date, ticker, each candidate signal, and the forward-return label.

    POOLED PANEL, NOT PER-DATE REGRESSION:
    ElasticNetCV needs enough rows per CV fold to fit a stable model. A
    single date's cross-section is about 500 tickers, thin and noisy to
    fit several correlated candidate signals against. Pooling across a
    fold's ~800 trading days (train_window + horizon) gives tens of
    thousands of rows.
    Tradeoff: pooling implicitly assumes the relationship between signals and 
    forward returns is stable across the whole fold's history. Fama-MacBeth (fit
    per date, then average/summarize the resulting coefficient time
    series) is the literature alternative.

    MISSING-DATA POLICY:
    Rows where ANY signal or the label is null get dropped. Cost: candidate signals have different lookback windows
    (mom20 needs 20 days of history, mom252 needs 252), so including a
    long-lookback signal disproportionately drops early-history and
    recently-listed tickers, due to the survivorship-bias
    caveat already documented in README. 
    Cross-sectional per-date median imputation is the standard alternative. 

    Returns: the design matrix as a lf.Dataframe with [date, ticker, signal, fwdret] columns
    '''

    keep_cols = [date_col, ticker_col] + signal_cols + [fwdret_col]
    design_mat = (
        signals_lf
        .select(keep_cols)
        .drop_nulls()  # <- the explicit policy named above, to check though
        .sort([date_col, ticker_col])
        .collect()
    )
    return design_mat



# =============================================================================
# STEP 3: q-value as a continuous prior, applied as a per-column rescale
# =============================================================================



def apply_q_value_weighting( X:np.ndarray, signal_cols:list[str], q_values:dict[str, float] ) -> np.ndarray:
    '''
    Design decision (see methodology.md): chose q-values to be fed as continuous weights, 
    not a hard survives/not boolean like BH and BY, so no information is thrown away and no second free
    threshold parameter gets introduced.

    THE PROBLEM: there exists a function in sklearn's ElasticNet.fit(sample_weight=)
    which weights ROWS (observations). q-value is a property of a COLUMN (a
    candidate signal).

    THE TRICK: rescale each signal's column by a monotonic function of
    (1 - q) BEFORE fitting. Elastic Net's penalty acts uniformly on
    coefficient magnitude: shrinking a low-quality signal's raw input
    values means a larger true coefficient is needed to contribute the
    same amount to the prediction, an implicit differential penalty
    without adding a second regularization parameter to tune. This is the
    same idea as Zou's Adaptive Lasso (2006), per-feature penalty
    weights. It defined the Lasso error as 
        $$[Prediction error] + lambda sum_{j=1}^N w_j |beta_j|$$
    where N is the number of features, beta is the weights, and w_j are the penalties weights.
    Therefore a high penalty will make the feature weight go to zero more easily.

    OPEN QUESTIOON: Should we use linear penalization or squared or square-root or other?
    '''

    weights = np.array([1.0 - q_values.get(col, 0.0) for col in signal_cols])
    return X * weights[np.newaxis, :]


# =============================================================================
# STEP 4: fit, with the project's own CV folds, not sklearn's default
# =============================================================================

def fit_elastic_net_combiner(
    panel : pl.DataFrame,
    signal_cols : list[str],
    fwd_ret_col : str,
    date_col : str,
    unique_dates : np.ndarray,
    train_window : int,
    horizon : int,
    test_window:  int,
    embargo : int,
    q_values : dict[str, float] | None = None,
    l1_ratio_grid : list[float] = [.1, .5, .7, .9, .95, .99, 1],
) -> dict:
    '''
    l1_ratio_grid: sklearn's own standard default grid, a fixed convention here, not tuned to this dataset, consistent with the
        project's no-hand-picked-free-parameters rule. alphas left as None, auto-generated path, also not hand-picked.

    Returns a dict with the fitted ElasticNetCV, the row-index folds used (for reuse in step 5's stability check), and the feature order.
    '''
    panel_dates = panel[date_col].to_numpy()
    X = panel.select(signal_cols).to_numpy()
    y = panel[fwd_ret_col].to_numpy()

    if q_values is not None:
        X = apply_q_value_weighting(X, signal_cols, q_values)

    cv_folds = build_row_index_folds(
        panel_dates, unique_dates, train_window, horizon, test_window,
        embargo, next_fold='consecutive',
        # 'consecutive' specifically here, see the note in
        # build_row_index_folds(): this fit averages a validation score
        # across folds to pick alpha/l1_ratio, so the folds need to be
        # closer to independent, not densely tiled.
    )

    model = ElasticNetCV(l1_ratio=l1_ratio_grid, cv=cv_folds, n_jobs=-1)
    model.fit(X, y)

    return {
        'model': model,
        'cv_folds': cv_folds,
        'signal_cols': signal_cols,
        'q_values': q_values,
    }


# =============================================================================
# STEP 5: per-fold coefficient stability
# =============================================================================
#
# ElasticNetCV's internal CV only returns an MSE grid over (alpha, l1_ratio),
# not per-fold coefficients, that information is thrown away once the best
# hyperparameters are picked. To see WHICH signals survive combination, and
# whether that's stable across time, refit a plain ElasticNet at the
# SELECTED alpha_/l1_ratio_ separately on each fold's train set. This is a
# genuinely useful diagnostic beyond just "the combiner works": if a signal
# drops to zero in some folds and not others, that's real information about
# stability, complementary to what FDR already told you about significance,
# not redundant with it.

def coefficient_stability_by_fold(fit_result: dict, panel: pl.DataFrame) -> pl.DataFrame:
    model = fit_result['model']
    signal_cols = fit_result['signal_cols']
    X_full = panel.select(signal_cols).to_numpy()
    if fit_result['q_values'] is not None:
        X_full = apply_q_value_weighting(X_full, signal_cols, fit_result['q_values'])
    y_full = panel[fit_result.get('fwd_ret_col', 'fwdret')].to_numpy()

    rows = []
    for i, (train_idx, _test_idx) in enumerate(fit_result['cv_folds']):
        fold_model = ElasticNet(alpha=model.alpha_, l1_ratio=model.l1_ratio_)
        fold_model.fit(X_full[train_idx], y_full[train_idx])
        row = {'fold': i}
        row.update(dict(zip(signal_cols, fold_model.coef_)))
        rows.append(row)
    return pl.DataFrame(rows)


# =============================================================================
# STEP 6: turn the fit into a signal column, reusing the existing checklist
# =============================================================================
#
# Deliberately NOT a special-cased evaluation path. The combined score gets
# piped through signal_plots.one_pager(), report.signal_report(), ic_decay(),
# same four-step checklist as any other signal (methodology.md Section 4).
# This is a continuity choice worth keeping: if the combiner needed its own
# bespoke evaluation, that would be a sign something about the design is
# fighting the rest of the codebase.

def combined_score_column( lf:pl.LazyFrame, fit_result:dict, out_col:str='combined_score' ) -> pl.LazyFrame:
    
    model = fit_result['model']
    signal_cols = fit_result['signal_cols']
    q_values = fit_result['q_values']

    weights = (
        np.array([1.0 - (q_values or {}).get(col, 0.0) for col in signal_cols])
        if q_values is not None else np.ones(len(signal_cols))
    )
    # model.coef_[j] was fit against the WEIGHTED column, so the effective
    # per-signal weight applied to the raw (unweighted) signal at inference
    # time is coef_[j] * weight[j], folding the two together here rather
    # than re-deriving it wrong somewhere downstream.
    effective_coefs = model.coef_ * weights

    expr = pl.lit(model.intercept_)  # pl.lit makes sure that I pass the argument as a constant value and not a column name
    for col, coef in zip(signal_cols, effective_coefs):
        expr = expr + coef * c(col)

    return lf.with_columns(expr.alias(out_col))


if __name__ == '__main__':
    
    # from src.signals.combine import make_signal
    
    lf = pl.scan_parquet('data/processed/features.parquet')

    #  Test with fake signals and qvalues
    survivor_signals = ['mom20_zscore_tanh', 'mom252_zscore_tanh', 'vol60_rank']
    q_values = {'mom20_zscore_tanh': 0.02, 'mom252_zscore_tanh': 0.01, 'vol60_rank': 0.15}
    
    panel = build_design_matrix(lf, survivor_signals, fwdret_col='fwdret')
    unique_dates = panel['date'].unique().sort().to_numpy()
    
    fit_result = fit_elastic_net_combiner(
        panel, survivor_signals, 'fwdret', 'date', unique_dates,
        train_window=756, horizon=20, test_window=63, embargo=20,  # embargo: see Q1, derive don't guess
        q_values=q_values,
    )
    fit_result['fwd_ret_col'] = 'fwdret'
    
    print(coefficient_stability_by_fold(fit_result, panel))
    
    combined_lf = combined_score_column(lf, fit_result)
    from src.evaluation.signals.report import signal_report
    print(signal_report(combined_lf, 'combined_score', 'fwdret'))