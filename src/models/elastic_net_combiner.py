'''
GOAL: combine several already-evaluated candidate signals (the survivors of signal_report() + fdr.fdr_report()) 
    into a single "score" via Elastic Net regression against forward returns, with alpha/l1_ratio chosen using 
    the leakage-safe combinatorial purged CV (src/validation/cpcv.py) by default, or plain walk-forward CV
    (src/validation/walk_forward_cv.py) as an alternative, not sklearn's default random k-fold, which would
    silently reintroduce exactly the leakage the CV module exists to prevent.

WHY CPCV, NOT WALK-FORWARD, AS THE DEFAULT HERE, a real reconsideration, not the original position:
    Earlier reasoning favored walk-forward for hyperparameter selection specifically because it's causally
    realistic, train always strictly before test, matching real deployment. Still true, but incomplete:
    ElasticNetCV isn't fitting one model, it's searching a GRID of (alpha, l1_ratio) and picking whichever
    looks best on average across the folds. With only ~5-6 walk-forward folds, that search has real freedom
    to land on a combination that happens to look good on those particular historical windows by chance,
    there's no way to check whether the winning hyperparameter is robust or a lucky artifact of that one
    historical path. That's exactly the failure mode PBO exists to detect, and it applies to hyperparameter
    SEARCH just as much as to comparing finished model families (Section 4/5 of methodology.md). CPCV gives
    many more, still-properly-purged-and-embargoed folds to search across (see cpcv.py), directly addressing
    this. The causal-realism argument for walk-forward doesn't disappear, it just moves: use it for a final,
    genuinely untouched holdout check on the ALREADY-CHOSEN model (see the __main__ block below), not for the
    search itself. This is the nested design demonstrated in scripts/demos/, now the actual default here,
    not just an exploratory script no longer connected to the real module.

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
        Wiley, Ch. 7 (purge/embargo, reused directly via walk_forward_cv.py and cpcv.py).
'''






import numpy as np
import polars as pl
from polars import col as c
from sklearn.linear_model import ElasticNet, ElasticNetCV

from config.constants import EMBARGO_WINDOW
from src.validation.walk_forward_cv import walk_forward_cv, Fold
from src.validation.cpcv import cpcv







# =============================================================================
# STEP 1: fold-to-row-index adapters, one per CV scheme, same output contract
# =============================================================================




# Both walk_forward_cv() and cpcv() describe folds in terms of DATES (or date
# positions), but the design matrix here has one row per (date, ticker), not
# one row per date, sklearn's `cv` parameter needs integer ROW indices into
# that matrix. These adapters are the direct payoff of having built the CV
# machinery separately: everything downstream (fit_elastic_net_combiner and
# beyond) only ever consumes a plain list[(train_idx, test_idx)], it doesn't
# know or care which scheme produced it.

def fold_to_row_indices(panel_dates:np.ndarray, fold:Fold) -> tuple[np.ndarray, np.ndarray]:
    '''
    panel_dates: 1D array, one entry per ROW of the design matrix (i.e. not unique, repeats once per ticker per date). 
        Must be date-typed, same dtype as what walk_forward_cv() was called with.
    fold: a single Fold namedtuple from walk_forward_cv.py.

    Returns (train_row_idx, test_row_idx), both int arrays, suitable for
        a single entry in the list you pass to ElasticNetCV(cv=...).
    '''
    train_mask = (panel_dates >= fold.train_start) & (panel_dates <= fold.train_end)
    test_mask = (panel_dates >= fold.test_start) & (panel_dates <= fold.test_end)
    return np.flatnonzero(train_mask), np.flatnonzero(test_mask)  # flatnonzero returns the indices of the elements that are non-zero (flattened).





def build_row_index_folds(  panel_dates:np.ndarray, unique_dates:np.ndarray, train_window:int,
                            horizon:int, test_window:int, embargo:int, next_fold:str='consecutive'
) -> list[tuple[np.ndarray, np.ndarray]]:
    '''
    Wraps walk_forward_cv() + fold_to_row_indices() into the exact list of (train_idx, test_idx) format sklearn's cv= expects.

    unique_dates: the deduplicated, sorted date array walk_forward_cv.py operates on.

    panel_dates: the (date, ticker)-row-level date array from the actual
        design matrix, used only for the row-index lookup.

    next_fold: 'consecutive' or 'dense', see walk_forward_cv.py's own docstring, the old free-form int option
        was removed there (silently decoupled from `embargo`, a real bug, not a style choice).

    No longer the default fold source for fit_elastic_net_combiner() below (see module docstring for why),
    kept available and still useful on its own: a genuinely realistic, causally-ordered final holdout check
    on an already-chosen model benefits from exactly this scheme, see the __main__ block.
    '''
    
    folds = list(walk_forward_cv(
        unique_dates, train_window, horizon, test_window, embargo, next_fold
    ))
    return [fold_to_row_indices(panel_dates, f) for f in folds]




def build_cpcv_row_index_folds(
    panel_dates:np.ndarray, unique_dates:np.ndarray,
    n_blocks:int, n_test_blocks:int, purge_w:int, embargo_w:int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    '''
    Same job as build_row_index_folds() above, wraps cpcv() instead of
    walk_forward_cv(). cpcv() yields DATE POSITION indices (0..n_dates-1)
    into unique_dates, not row indices into the panel and not Fold
    namedtuples of actual dates, a different output shape than
    walk_forward_cv(), hence a separate adapter rather than reusing
    fold_to_row_indices() here.

    It's a bit finicky to have different formats to have to deal with here. 
    But it works nonetheless now...
    '''
    n_dates = len(unique_dates)
    folds = []
    for train_date_idx, test_date_idx, _test_ids in cpcv(n_dates, n_blocks, n_test_blocks, purge_w, embargo_w):  # test_ids are the block ids of test sets
        train_dates = unique_dates[train_date_idx]
        test_dates = unique_dates[test_date_idx]
        train_row_idx = np.flatnonzero(np.isin(panel_dates, train_dates))
        test_row_idx = np.flatnonzero(np.isin(panel_dates, test_dates))
        folds.append((train_row_idx, test_row_idx))
    return folds





# =============================================================================
# STEP 2: build the panel design matrix
# =============================================================================





def build_design_matrix( signals_lf:pl.LazyFrame, signal_cols:list[str], fwdret_col:str='fwdret', date_col: str='date', ticker_col:str='ticker' ) -> pl.DataFrame:
    '''
    Transforms LazyFrame with multiple (date, ticker) rows to lf.DataFrame with one row per (date, ticker). 
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
    series) is the literature alternative. But relevant here ?

    MISSING-DATA POLICY:
    Rows where ANY signal or the label is null get dropped. A row is just for 1 ticker FYI. 
    Cost: candidate signals have different lookback windows
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
        .drop_nulls()       # <- the explicit policy named above, to check though 
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
        $$ [Prediction error] + lambda sum_{j=1}^N w_j |beta_j| $$
    where N is the number of features, beta is the weights, and w_j are the penalties weights.
    Therefore a high penalty will make the feature weight go to zero more easily.

    OPEN QUESTIOON: Should we use linear penalization or squared or square-root or other?
    '''

    weights = np.array([1.0 - q_values.get(col, 0.0) for col in signal_cols])
    return X * weights[np.newaxis, :]





# =============================================================================
# STEP 4: fit, with the project's own CV folds (walk_forward or CPCV), not sklearn's default
# =============================================================================






def fit_elastic_net_combiner( panel:pl.DataFrame, signal_cols:list[str], fwdret_col:str,
                              cv_folds:list[tuple[np.ndarray, np.ndarray]],
                              q_values:dict[str, float]|None=None, l1_ratio_grid:list[float]=[.1, .5, .7, .9, .95, .99, 1],
) -> dict:
    '''
    cv_folds: pre-built list[(train_idx, test_idx)], built by the CALLER via build_cpcv_row_index_folds()
        (the recommended default, see module docstring) or build_row_index_folds() (walk-forward). This
        function doesn't build folds itself anymore, doesn't know or care which scheme produced them, a
        deliberate refactor: the earlier version hard-coded walk-forward internally, which meant the more
        carefully-reasoned CPCV-search design (demonstrated in scripts/demos/) never actually fed into this
        "official" module, two divergent, silently-inconsistent code paths. Passing folds in directly closes
        that gap, and makes the nested CPCV-search + untouched-holdout design (see __main__ below) a single,
        explicit pipeline instead of logic split across a module and separate one-off scripts.

    l1_ratio_grid: sklearn's own standard default grid, a fixed convention here, not tuned to this dataset, consistent with the
        project's no-hand-picked-free-parameters rule. alphas left as None, auto-generated path, also not hand-picked.

    Returns a dict with the fitted ElasticNetCV, the row-index folds used (for reuse in step 5's stability check), and the feature order.
    '''
    X = panel.select(signal_cols).to_numpy()
    y = panel[fwdret_col].to_numpy()

    if q_values is not None:
        X = apply_q_value_weighting(X, signal_cols, q_values)

    # cv must be an iterable, so that's what we have as cv_folds
    # n_jobs is the number of CPUs to use, -1 uses all available
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





def coefficient_stability_by_fold(fit_result:dict, panel:pl.DataFrame) -> pl.DataFrame:
    '''
    ElasticNetCV's internal CV only returns an MSE grid over (alpha, l1_ratio),
    not per-fold coefficients, that information is thrown away once the best
    hyperparameters are picked. To see WHICH signals survive combination, and
    whether that's stable across time, refit a plain ElasticNet at the
    SELECTED alpha_/l1_ratio_ separately on each fold's train set. This is a
    genuinely useful diagnostic beyond just "the combiner works":  if a signal
    drops to zero in some folds and not others, that's real information about
    stability, complementary to what FDR already told us about significance,
    not redundant with it.
    '''
    model = fit_result['model']
    signal_cols = fit_result['signal_cols']
    X_full = panel.select(signal_cols).to_numpy()
    if fit_result['q_values'] is not None:
        X_full = apply_q_value_weighting(X_full, signal_cols, fit_result['q_values'])
    y_full = panel[fit_result.get('fwd_ret_col', 'fwdret')].to_numpy()

    rows = []
    for i, (train_idx, _test_idx) in enumerate(fit_result['cv_folds']):
        fold_model = ElasticNet(alpha=model.alpha_, l1_ratio=model.l1_ratio_) # alpha's the Constant that multiplies the penalty terms, before the l1_ratio
        fold_model.fit(X_full[train_idx], y_full[train_idx])
        row = {'fold': i}
        row.update(dict(zip(signal_cols, fold_model.coef_)))
        rows.append(row)
    return pl.DataFrame(rows)





# =============================================================================
# STEP 6: turn the fit into a signal column, reusing the existing checklist
# =============================================================================
#
# Deliberately NOT a special-cased evaluation path for the EN combiner. 
# The combined score gets piped through signal_plots.one_pager(), report.signal_report(), ic_decay(),
# same four-step checklist as any other signal (methodology.md Section 4).





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
    
    from src.signals.combine import make_signal
    
    lf = pl.scan_parquet('data/processed/features.parquet')

    lf = make_signal(lf, ['mom20', 'mom252'], method='zscore_tanh')
    lf = make_signal(lf, ['vol60'], method='rank')


    #  Test with fake signals and qvalues
    survivor_signals = ['mom20_zscore_tanh', 'mom252_zscore_tanh', 'vol60_rank']
    q_values = {'mom20_zscore_tanh': 0.02, 'mom252_zscore_tanh': 0.01, 'vol60_rank': 0.15}

    print('Building design matrix')
    panel = build_design_matrix(lf, survivor_signals, fwdret_col='fwdret')
    unique_dates = panel['date'].unique().sort().to_numpy()

    # --- nested design: CPCV search on a search region, genuine holdout for the final check ---
    # numbers below are illustrative, not tuned, real EMBARGO_WINDOW/horizon come from
    # config.constants and scripts/research/embargo_selection.py, see methodology.md Section 1.7
    horizon = 1
    purge_w = horizon + 1
    embargo_w = purge_w + EMBARGO_WINDOW  # purge_w + EMBARGO_WINDOW(16), sum convention, methodology.md Section 1.7
    n_blocks, n_test_blocks = 10, 2
    holdout_dates = 200

    holdout_start_date = unique_dates[-holdout_dates]
    gap_cutoff = unique_dates[-holdout_dates - embargo_w]
    search_panel = panel.filter(pl.col('date') <= gap_cutoff)
    holdout_panel = panel.filter(pl.col('date') >= holdout_start_date)
    search_unique_dates = search_panel['date'].unique().sort().to_numpy()
    search_panel_dates = search_panel['date'].to_numpy()

    print('Building the CPCV folds')
    cv_folds = build_cpcv_row_index_folds(
        search_panel_dates, search_unique_dates, n_blocks, n_test_blocks, purge_w, embargo_w
    )

    print('Fitting the ElsticNetCV on the CPCV folds')
    fit_result = fit_elastic_net_combiner(
        search_panel, survivor_signals, 'fwdret', cv_folds, q_values=q_values,
    )
    fit_result['fwd_ret_col'] = 'fwdret'

    print('Computing the coefficient stability by fold (by re-fitting an ElasticNet on each fold -not CPCV)')
    print(coefficient_stability_by_fold(fit_result, search_panel))

    print('Evaluating the signal resulting from EN')
    # fixed, already-chosen model, evaluated ONLY on the untouched holdout, never seen during the CPCV search
    combined_lf = combined_score_column(holdout_panel.lazy(), fit_result)
    from src.evaluation.signals.report import signal_report
    print(signal_report(combined_lf, 'combined_score', 'fwdret'))