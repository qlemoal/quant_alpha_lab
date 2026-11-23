'''
GOAL: combine several already-evaluated candidate signals (the survivors of 05: IC + Newey-West + FDR)
    into ONE score, using an Elastic Net regression against forward returns.
    The hyper-parameters (alpha, l1_ratio) are chosen with the leakage-safe combinatorial purged CV
    (src/validation/cpcv.py), NOT with sklearn's random k-fold (which would leak overlapping labels).

THE FLOW IN ONE PICTURE
-----------------------
    signals.parquet
        |  build_design_matrix        drop rows with any null; one row per (date, ticker)
        v
    panel  --split by date-->  search_panel (dates <= gap_cutoff)   | gap (embargo) | holdout_panel (last N dates)
        |
        |  fit_elastic_net_combiner
        |     1. standardise each signal column (mean 0, std 1)           <- NEW: penalty is scale-free
        |     2. multiply column j by (1 - q_j) ** q_power                <- q-value prior (mild, see below)
        |     3. target = fwdret minus the same-date cross-sectional mean <- NEW: remove the market-wide part
        |     4. ElasticNetCV with CPCV folds  -> alpha, l1_ratio, coefficients
        |     5. out-of-fold (OOF) scores for every search row            <- NEW: no in-sample scoring
        v
    combined_score_column  -> one column "combined_score" for ALL dates
        search dates : average of the OOF predictions (the model never saw those rows when predicting them)
        other dates  : prediction of the final model (gap + holdout + latest dates)

WHAT CHANGED vs the previous version (and why)
----------------------------------------------
1. IN-SAMPLE SCORING FIXED.
   Before: one model fitted on the whole search period scored the whole history.
   For dates inside the search period, the score had "seen" those labels -> a backtest on it is look-ahead.
   Now: search-period scores are out-of-fold. Use the holdout dates (and later) for any honest evaluation.
2. STANDARDISED INPUTS.
   Elastic Net penalises coefficient size, so the answer depends on the scale of each column.
   Now each column is z-scored first (statistics from the search panel only).
3. TARGET = CROSS-SECTIONALLY DEMEANED RETURN (default).
   The strategy is dollar-neutral. The market-wide part of the return on a date is noise for it, and it
   dominates the variance. Now: y = fwdret - mean(fwdret on the same date). 'rank' and 'raw' are available.
4. q-VALUE WEIGHTING KEPT, BUT HONEST.
   Scaling a standardised column by (1 - q) is a per-feature penalty (Adaptive-Lasso-like idea, Zou 2006).
   The effect is MILD unless q is far from 0. It is a soft prior, not a feature filter.
   A NaN q is treated as q = 1 (weight 0).
5. WIDER l1_ratio GRID by default (adds 0.05 and 0.3).

References:
    Zou, H. & Hastie, T. (2005). Regularization and variable selection via the elastic net. JRSS B, 67(2).
        Why EN and not Lasso: candidate signals are correlated by construction (mom20 / mom60 / mom252
        share return history). Lasso keeps one of a correlated group arbitrarily. Ridge never zeros anything.
        EN (L1 + L2) is the standard answer.
    Zou, H. (2006). The adaptive lasso and its oracle properties. JASA, 101(476).
    Lopez de Prado, M. (2018). Advances in Financial Machine Learning. Wiley, Ch. 7 (purge, embargo, CPCV).
'''

import numpy as np
import polars as pl
from polars import col as c
from sklearn.linear_model import ElasticNet, ElasticNetCV, enet_path

from config.constants import EMBARGO_WINDOW
from src.validation.walk_forward_cv import walk_forward_cv, Fold
from src.validation.cpcv import cpcv
from src.features.returns import add_fwd_returns, add_fwdret_horizon

# Default grid for the L1/L2 mix. 1 = pure Lasso, small = mostly Ridge.
DEFAULT_L1_GRID = [.05, .1, .3, .5, .7, .9, .95, .99, 1]


# =============================================================================
# STEP 1: fold-to-row-index adapters
# =============================================================================
# walk_forward_cv() and cpcv() describe folds with DATES (or date positions).
# The design matrix has one row per (date, ticker), and sklearn needs integer ROW indices.
# These adapters convert one into the other. Everything downstream only sees list[(train_idx, test_idx)].

def fold_dates_to_indices(panel_dates: np.ndarray, fold: Fold) -> tuple[np.ndarray, np.ndarray]:
    '''
    panel_dates: one date per ROW of the design matrix (repeats once per ticker).
    fold:        one Fold namedtuple from walk_forward_cv.py.
    Returns (train_row_idx, test_row_idx).
    '''
    train_mask = (panel_dates >= fold.train_start) & (panel_dates <= fold.train_end)
    test_mask = (panel_dates >= fold.test_start) & (panel_dates <= fold.test_end)
    return np.flatnonzero(train_mask), np.flatnonzero(test_mask)


def build_walk_forward_idx(panel_dates: np.ndarray, unique_dates: np.ndarray, train_window: int,
                           horizon: int, test_window: int, embargo: int, next_fold: str = 'consecutive'
                           ) -> list[tuple[np.ndarray, np.ndarray]]:
    '''
    walk_forward_cv() + fold_dates_to_indices() in the (train_idx, test_idx) format of sklearn's cv=.
    Not the default (CPCV is), but good for a causal, realistic check of an already-chosen model.
    '''
    folds = list(walk_forward_cv(unique_dates, train_window, horizon, test_window, embargo, next_fold))
    return [fold_dates_to_indices(panel_dates, f) for f in folds]


def build_cpcv_idx(panel_dates: np.ndarray, unique_dates: np.ndarray,
                   n_blocks: int, n_test_blocks: int, purge_w: int, embargo_w: int,
                   ) -> list[tuple[np.ndarray, np.ndarray]]:
    '''
    Same job for CPCV. cpcv() yields positions in unique_dates; we turn them into row indices.

    Speed: instead of np.isin per fold on millions of rows, we map each row to its date position once
    (searchsorted) and then test membership with a boolean lookup table per fold.
    '''
    n_dates = len(unique_dates)
    row_pos = np.searchsorted(unique_dates, panel_dates)       # date position of every row
    folds = []
    for train_date_idx, test_date_idx, _test_ids in cpcv(n_dates, n_blocks, n_test_blocks, purge_w, embargo_w):
        in_train = np.zeros(n_dates, dtype=bool)
        in_train[train_date_idx] = True
        in_test = np.zeros(n_dates, dtype=bool)
        in_test[test_date_idx] = True
        folds.append((np.flatnonzero(in_train[row_pos]), np.flatnonzero(in_test[row_pos])))
    return folds


# =============================================================================
# STEP 2: the panel (design matrix)
# =============================================================================

def build_design_matrix(signals_lf: pl.LazyFrame, signal_cols: list[str], fwdret_col: str = 'fwdret',
                        date_col: str = 'date', ticker_col: str = 'ticker') -> pl.DataFrame:
    '''
    One row per (date, ticker). Columns: date, ticker, each signal, the forward-return label.

    POOLED PANEL, NOT ONE REGRESSION PER DATE:
        One date has about 500 tickers: too thin for several correlated signals. Pooling over many dates
        gives tens of thousands of rows. Price: we assume one stable relation across the pooled period.
        (Fama-MacBeth, one fit per date, is the alternative.)

    MISSING-DATA POLICY:
        A row with ANY null (signal or label) is dropped. Long-lookback signals (mom252) therefore drop
        early-history and recently listed tickers.
    '''
    keep_cols = [date_col, ticker_col] + signal_cols + [fwdret_col]
    return (
        signals_lf
        .select(keep_cols)
        .drop_nulls()
        .sort([date_col, ticker_col])
        .collect()
    )


# =============================================================================
# STEP 3: small, testable helpers (inputs, weights, target)
# =============================================================================

def _q_weights(signal_cols: list[str], q_values: "dict[str, float] | None", q_power: float = 1.0) -> np.ndarray:
    '''
    Per-column weight (1 - q) ** q_power.
      q near 0 (strong signal)  -> weight near 1  -> column keeps its size -> coefficient is penalised normally.
      q near 1 (weak signal)    -> weight near 0  -> column is shrunk       -> needs a huge coefficient to matter.
    A missing q counts as 0 (weight 1). A NaN q counts as 1 (weight 0). q is clipped to [0, 1].
    With q_values=None, all weights are 1.
    '''
    if q_values is None:
        return np.ones(len(signal_cols))
    q = np.array([q_values.get(col, 0.0) for col in signal_cols], dtype=float)
    q = np.where(np.isnan(q), 1.0, q)
    q = np.clip(q, 0.0, 1.0)
    return (1.0 - q) ** q_power


def apply_q_value_weighting(X: np.ndarray, signal_cols: list[str], q_values: dict[str, float],
                            q_power: float = 1.0) -> np.ndarray:
    '''
    Multiply each column by its q-weight (see _q_weights).
    sklearn's sample_weight weights ROWS. A q-value belongs to a COLUMN, hence this rescale trick.
    Call it on STANDARDISED columns, otherwise the original scale of the column hides the effect.
    '''
    return X * _q_weights(signal_cols, q_values, q_power)[np.newaxis, :]


def _make_target(panel: pl.DataFrame, fwdret_col: str, target: str = 'csdemean') -> np.ndarray:
    '''
    The regression target y, one value per row.
      'csdemean': fwdret - mean(fwdret over the tickers of the SAME date). Default.
                  Uses only same-date information, so no look-ahead. Removes the market-wide move.
      'rank'    : cross-sectional rank of fwdret on that date, scaled to [-0.5, +0.5]. Robust to outliers.
      'raw'     : fwdret as it is (the old behaviour).
    '''
    if target == 'raw':
        return panel[fwdret_col].to_numpy()
    if target == 'csdemean':
        expr = c(fwdret_col) - c(fwdret_col).mean().over('date')
    elif target == 'rank':
        expr = (c(fwdret_col).rank().over('date') - 1) / (c(fwdret_col).count().over('date') - 1).clip(lower_bound=1) - 0.5
    else:
        raise ValueError(f"unknown target {target!r}, use 'csdemean', 'rank' or 'raw'")
    return panel.select(expr.alias('_y'))['_y'].to_numpy()


def _fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''Column mean and std on the SEARCH panel. Std is floored so a constant column cannot divide by zero.'''
    mean = X.mean(axis=0)
    std = np.maximum(X.std(axis=0), 1e-12)
    return mean, std


def _prepare_X(panel: pl.DataFrame, fit_result: dict) -> np.ndarray:
    '''Raw signals -> standardised -> q-weighted. The exact matrix the model was fitted on.'''
    X = panel.select(fit_result['signal_cols']).to_numpy()
    X = (X - fit_result['x_mean']) / fit_result['x_std']
    return X * fit_result['col_weights'][np.newaxis, :]


def score_contributions(fit_result: dict, panel: pl.DataFrame) -> "dict[str, float]":
    '''
    Share of the variance of the combined score that each signal brings: Cov(term_j, score) / Var(score).
    term_j = coefficient_j x (standardised, q-weighted signal j). The shares add up to 1 (they can be
    negative for a signal that hedges the others). This is a better size of a signal's role than its
    coefficient: the coefficient depends on the units, the share does not.
    Rows with a missing signal are ignored. Returns {signal: share}; all zeros if the score is constant.
    '''
    X = _prepare_X(panel, fit_result)
    ok = np.isfinite(X).all(axis=1)
    terms = X[ok] * fit_result['model'].coef_[np.newaxis, :]
    score = terms.sum(axis=1)
    var = float(score.var())
    cols = fit_result['signal_cols']
    if var <= 0 or len(score) < 2:
        return {c: 0.0 for c in cols}
    centred = terms - terms.mean(axis=0)
    cov = (centred * (score - score.mean())[:, np.newaxis]).mean(axis=0)
    return {c: float(v / var) for c, v in zip(cols, cov)}


def _fold_path_correlations(X: np.ndarray, y: np.ndarray, cv_folds: list, l1_ratio: float,
                            alphas: np.ndarray, positive: bool) -> np.ndarray:
    '''
    Correlation between prediction and target on the TEST rows of each fold, for every alpha of the grid.
    Returns an array (n_folds, n_alphas). One enet_path call per fold (warm starts), no refit per alpha.
    A constant prediction (all coefficients zero) has correlation 0.
    '''
    out = np.zeros((len(cv_folds), len(alphas)))
    for f, (tr, te) in enumerate(cv_folds):
        xm = X[tr].mean(axis=0)
        ym = y[tr].mean()
        _, coefs, _ = enet_path(X[tr] - xm, y[tr] - ym, l1_ratio=l1_ratio, alphas=alphas, positive=positive)
        pred = (X[te] - xm) @ coefs                              # (n_test, n_alphas)
        yt = y[te] - y[te].mean()
        pc = pred - pred.mean(axis=0)
        sd = np.sqrt((pc ** 2).mean(axis=0))
        cov = (pc * yt[:, None]).mean(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            r = cov / (sd * yt.std())
        out[f] = np.where(sd > 1e-15, r, 0.0)
    return out


def apply_one_se_rule(model: ElasticNetCV, X: np.ndarray, y: np.ndarray, l1_ratio_grid: list,
                      positive: bool = False, cv_folds: "list | None" = None) -> float:
    '''
    One-standard-error rule (Hastie, Tibshirani, Friedman, "Elements of Statistical Learning", 7.10),
    with the CORRELATION between prediction and target as the CV score, not the MSE.

    WHY NOT THE MSE?
        A return has a huge noise part. A real signal lowers the MSE by a tiny amount, smaller than the
        spread of the MSE across folds. The one-SE rule on the MSE then always picks the largest alpha: all
        coefficients zero, even for a real signal (seen on a synthetic test with IC = 0.03).
        The strategy only uses the RANKING of the score inside each date. Correlation measures that and
        ignores the scale of the prediction.

    THE RULE
        1. keep the l1_ratio that ElasticNetCV chose, and its alpha grid;
        2. for each alpha: mean and standard error over the folds of the test correlation;
        3. best = alpha with the highest mean;   threshold = mean(best) - se(best);
        4. take the LARGEST alpha whose mean correlation >= threshold.
    Then refit a plain ElasticNet at that alpha on all rows and overwrite model.alpha_, coef_, intercept_
    (the rest of the code reads these attributes and needs no change).

    NOISE CASE: if no alpha beats zero by more than one standard error, the largest alpha wins and all
    coefficients are zero. 06 then stops with "no usable information". This is the wanted behaviour.
    The folds of CPCV share data, so their scores are correlated and se is a little too small.
    Returns the chosen alpha.
    '''
    if cv_folds is None:
        raise ValueError('apply_one_se_rule needs the cv_folds')
    l1 = np.atleast_1d(np.asarray(l1_ratio_grid, dtype=float))
    i = int(np.argmin(np.abs(l1 - model.l1_ratio_)))
    alphas = np.atleast_2d(model.alphas_)[i]
    corr = _fold_path_correlations(X, y, cv_folds, float(model.l1_ratio_), alphas, positive)
    m = corr.mean(axis=0)
    se = corr.std(axis=0, ddof=1) / np.sqrt(corr.shape[0])
    best = int(np.argmax(m))
    chosen = float(alphas[m >= m[best] - se[best]].max())
    refit = ElasticNet(alpha=chosen, l1_ratio=float(model.l1_ratio_), positive=positive, max_iter=5000).fit(X, y)
    model.alpha_ = chosen
    model.coef_ = refit.coef_
    model.intercept_ = refit.intercept_
    return chosen


# =============================================================================
# STEP 4: fit (hyper-parameters chosen with the project's own CV folds)
# =============================================================================

def fit_elastic_net_combiner(panel: pl.DataFrame, signal_cols: list[str], fwdret_col: str,
                             cv_folds: list[tuple[np.ndarray, np.ndarray]],
                             q_values: "dict[str, float] | None" = None,
                             l1_ratio_grid: "list[float] | None" = None,
                             target: str = 'csdemean', q_power: float = 1.0,
                             compute_oof: bool = True, positive: bool = False,
                             one_se: bool = False) -> dict:
    '''
    Fit ElasticNetCV on the (standardised, q-weighted) panel with the given CV folds.

    cv_folds:     list[(train_idx, test_idx)] built by the CALLER (build_cpcv_idx or build_walk_forward_idx).
    l1_ratio_grid: None -> DEFAULT_L1_GRID. alphas are auto-generated by sklearn (not hand-picked).
    target:       'csdemean' (default), 'rank' or 'raw'. See _make_target.
    q_power:      exponent on (1 - q). 1 = linear. Larger = stronger prior.
    compute_oof:  also compute out-of-fold scores for every row of `panel` (needed for honest scoring).
    positive:     force every coefficient >= 0. Use it when the columns are SIGN-ALIGNED with an ex-ante prior
                  (family scores of 06): a negative weight would then go against the economic story.
    one_se:       one-standard-error rule: among the alphas whose CV correlation is within one standard error
                  of the best, take the LARGEST (the strongest shrinkage). See apply_one_se_rule.

    Returns a dict with the fitted ElasticNetCV ('model'), the folds, the standardiser (x_mean, x_std),
    the column weights, and 'oof' (DataFrame[date, ticker, oof_score]) if requested.
    NOTE: coefficients live in STANDARDISED, q-weighted units. Use combined_score_column() to score raw data.
    '''
    if l1_ratio_grid is None:
        l1_ratio_grid = DEFAULT_L1_GRID

    X_raw = panel.select(signal_cols).to_numpy()
    x_mean, x_std = _fit_standardizer(X_raw)
    col_weights = _q_weights(signal_cols, q_values, q_power)
    X = ((X_raw - x_mean) / x_std) * col_weights[np.newaxis, :]
    y = _make_target(panel, fwdret_col, target)

    # cv= accepts a list of (train_idx, test_idx); n_jobs=-1 uses all CPUs
    model = ElasticNetCV(l1_ratio=l1_ratio_grid, cv=cv_folds, n_jobs=-1, positive=positive)
    model.fit(X, y)
    if one_se:
        apply_one_se_rule(model, X, y, l1_ratio_grid, positive, cv_folds)

    result = {
        'model': model,
        'cv_folds': cv_folds,
        'signal_cols': signal_cols,
        'q_values': q_values,
        'q_power': q_power,
        'fwdret_col': fwdret_col,
        'target': target,
        'x_mean': x_mean,
        'x_std': x_std,
        'col_weights': col_weights,
        'positive': positive,
        'one_se': one_se,
    }
    if compute_oof:
        result['oof'] = out_of_fold_scores(panel, result)
    return result


# =============================================================================
# STEP 5: out-of-fold scores, and per-fold coefficients
# =============================================================================

def _refit_folds(panel: pl.DataFrame, fit_result: dict):
    '''
    Generator: for each CV fold, refit a plain ElasticNet at the SELECTED (alpha_, l1_ratio_)
    on the fold's train rows. Yields (fold_index, fitted_model, X, test_idx).
    Shared by the OOF scores and the stability diagnostic, so both use the same fits.
    '''
    model = fit_result['model']
    X = _prepare_X(panel, fit_result)
    y = _make_target(panel, fit_result['fwdret_col'], fit_result['target'])
    for i, (train_idx, test_idx) in enumerate(fit_result['cv_folds']):
        fold_model = ElasticNet(alpha=model.alpha_, l1_ratio=model.l1_ratio_, max_iter=5000,
                                positive=fit_result.get('positive', False))
        fold_model.fit(X[train_idx], y[train_idx])
        yield i, fold_model, X, test_idx


def out_of_fold_scores(panel: pl.DataFrame, fit_result: dict) -> pl.DataFrame:
    '''
    Score every row of `panel` with models that did NOT train on it.

    In CPCV each row lies in the TEST part of several folds. For each such fold we predict the row with the
    model fitted on that fold's (purged, embargoed) train rows, then average the predictions of the row.

    Remaining weakness (small, documented): alpha and l1_ratio were chosen with these same folds.
    The hyper-parameters therefore saw the labels, the coefficients did not. The holdout stays the clean test.

    Returns DataFrame[date, ticker, oof_score]. Rows that are never in a test fold are not returned.
    '''
    n = panel.height
    pred_sum = np.zeros(n)
    pred_cnt = np.zeros(n)
    for _i, fold_model, X, test_idx in _refit_folds(panel, fit_result):
        pred_sum[test_idx] += fold_model.predict(X[test_idx])
        pred_cnt[test_idx] += 1
    mask = pred_cnt > 0
    sub = panel.select(['date', 'ticker'])
    return pl.DataFrame({
        'date': sub['date'].to_numpy()[mask],
        'ticker': sub['ticker'].to_numpy()[mask],
        'oof_score': pred_sum[mask] / pred_cnt[mask],
    }).with_columns(c('date').cast(sub.schema['date']), c('ticker').cast(sub.schema['ticker']))


def coefficient_stability_by_fold(fit_result: dict, panel: pl.DataFrame) -> pl.DataFrame:
    '''
    ElasticNetCV only returns an error grid, not per-fold coefficients.
    We refit a plain ElasticNet at the chosen (alpha_, l1_ratio_) on each fold's train rows.
    If a signal is non-zero in some folds and zero in others, the combination is unstable
    (complementary to FDR, which only speaks about significance).
    Coefficients are in standardised, q-weighted units: compare them across signals and folds.
    '''
    signal_cols = fit_result['signal_cols']
    rows = []
    for i, fold_model, _X, _test_idx in _refit_folds(panel, fit_result):
        row = {'fold': i}
        row.update(dict(zip(signal_cols, fold_model.coef_)))
        rows.append(row)
    return pl.DataFrame(rows)


# =============================================================================
# STEP 6: turn the fit into a score column
# =============================================================================
# Not a special evaluation path: the combined score goes through the same checklist as any signal
# (signal_report, ic_decay, ...).

def raw_unit_coefficients(fit_result: dict) -> "tuple[dict, float]":
    '''
    The final model written on the RAW signal columns:  score = const + sum_j eff_j * x_j.
        eff_j = coef_j * q_weight_j / std_j      const = intercept - sum_j eff_j * mean_j
    Returns ({signal: eff_j}, const). Same formula as combined_score_column. Use it for reporting.
    '''
    model = fit_result['model']
    eff = model.coef_ * fit_result['col_weights'] / fit_result['x_std']
    const = float(model.intercept_ - np.sum(eff * fit_result['x_mean']))
    return {k: float(v) for k, v in zip(fit_result['signal_cols'], eff)}, const


def combined_score_column(lf: pl.LazyFrame, fit_result: dict, out_col: str = 'combined_score',
                          use_oof: bool = True) -> pl.LazyFrame:
    '''
    Add the combined score to a LazyFrame that holds the RAW signal columns.

    Model score (final model, raw input):
        score = intercept + sum_j coef_j * w_j * (x_j - mean_j) / std_j
        (mean_j, std_j from the search panel; w_j the q-weight)

    If use_oof=True and fit_result holds 'oof', then for the rows that have an OOF score
    (the search period) the OOF score replaces the model score. Other rows keep the model score.
    This is what removes the in-sample look-ahead.
    '''
    model = fit_result['model']
    signal_cols = fit_result['signal_cols']
    eff = model.coef_ * fit_result['col_weights'] / fit_result['x_std']          # coefficient on the RAW column
    const = float(model.intercept_ - np.sum(eff * fit_result['x_mean']))         # constant after un-standardising

    expr = pl.lit(const)
    for col, coef in zip(signal_cols, eff):
        expr = expr + float(coef) * c(col)
    out = lf.with_columns(expr.alias('_model_score'))

    oof = fit_result.get('oof')
    if use_oof and oof is not None and oof.height > 0:
        out = (
            out.join(oof.lazy(), on=['date', 'ticker'], how='left', maintain_order='left')   # keep the original row order
               .with_columns(pl.coalesce(['oof_score', '_model_score']).alias(out_col))
               .drop(['oof_score', '_model_score'])
        )
    else:
        out = out.with_columns(c('_model_score').alias(out_col)).drop('_model_score')
    return out


# =============================================================================
# STEP 7: the high-level wrapper
# =============================================================================

def run_en_combiner(
    lf: pl.LazyFrame,
    signal_cols: list[str],
    horizon: int = 20,
    embargo_window: int = EMBARGO_WINDOW,
    n_blocks: int = 10,
    n_test_blocks: int = 2,
    holdout_dates: int = 200,
    q_values: "dict[str, float] | None" = None,
    l1_ratio_grid: "list[float] | None" = None,
    fwdret_col: "str | None" = None,
    target: str = 'csdemean',
    q_power: float = 1.0,
    positive: bool = False,
    one_se: bool = False,
) -> dict:
    '''
    CPCV search + untouched holdout.

    Date layout (same rule as 05, so both stages cut at the same date):

        |------------ search_panel ------------|--- gap ---|------ holdout ------|
        dates <= gap_cutoff                      embargo_w     last `holdout_dates`

        purge_w   = horizon + 1                    (labels that overlap a test block are removed from train)
        embargo_w = purge_w + embargo_window       (extra safety gap after a test block, also the search/holdout gap)
        gap_cutoff = unique_dates[-holdout_dates - embargo_w]

    Returns everything from fit_elastic_net_combiner plus: horizon, purge_w, embargo_w, search_panel,
    holdout_panel, holdout_report (signal_report of the combined score on the holdout), train_end, holdout_start.
    '''
    if fwdret_col is None:
        fwdret_col = f'fwdret{horizon}'

    purge_w = horizon + 1
    embargo_w = purge_w + embargo_window

    print('    Building the design matrix')
    panel = build_design_matrix(lf, signal_cols, fwdret_col=fwdret_col)
    unique_dates = panel['date'].unique().sort().to_numpy()

    print('    Separating the train/holdout sets')
    holdout_start = unique_dates[-holdout_dates]
    gap_cutoff = unique_dates[-holdout_dates - embargo_w]
    search_panel = panel.filter(pl.col('date') <= gap_cutoff)
    holdout_panel = panel.filter(pl.col('date') >= holdout_start)
    search_unique_dates = search_panel['date'].unique().sort().to_numpy()
    search_panel_dates = search_panel['date'].to_numpy()

    print('    Building the CPCV folds')
    cv_folds = build_cpcv_idx(search_panel_dates, search_unique_dates, n_blocks, n_test_blocks, purge_w, embargo_w)

    print('    Fitting the EN (CPCV search) and computing out-of-fold scores')
    fit_result = fit_elastic_net_combiner(
        search_panel, signal_cols, fwdret_col, cv_folds,
        q_values=q_values, l1_ratio_grid=l1_ratio_grid, target=target, q_power=q_power,
        positive=positive, one_se=one_se,
    )

    from src.evaluation.signals.report import signal_report
    print('    Combining the EN into one signal')
    # Holdout rows are not in the OOF table, so they get the final model's score: a true out-of-sample test.
    combined_lf = combined_score_column(holdout_panel.lazy(), fit_result)
    print('    Computing the signal report for this combined signal (holdout)')
    holdout_report = signal_report(combined_lf, 'combined_score', fwdret_col, horizon=horizon)

    return {
        **fit_result,
        'horizon': horizon,
        'purge_w': purge_w,
        'embargo_w': embargo_w,
        'search_panel': search_panel,
        'holdout_panel': holdout_panel,
        'holdout_report': holdout_report,
        'train_end': gap_cutoff,
        'holdout_start': holdout_start,
    }


if __name__ == '__main__':
    from src.signals.combine import make_signal

    lf = pl.scan_parquet('data/processed/features.parquet')
    lf = make_signal(lf, ['mom20', 'mom252'], method='zscore_tanh')
    lf = make_signal(lf, ['vol60'], method='rank')

    survivor_signals = ['mom20_zscore_tanh', 'mom252_zscore_tanh', 'vol60_rank']
    q_values = {'mom20_zscore_tanh': 0.02, 'mom252_zscore_tanh': 0.01, 'vol60_rank': 0.15}

    result = run_en_combiner(lf, survivor_signals, horizon=20, embargo_window=EMBARGO_WINDOW, q_values=q_values,
                             n_blocks=10, n_test_blocks=2, holdout_dates=200)

    print(f"alpha={result['model'].alpha_:.6f}, l1_ratio={result['model'].l1_ratio_}")
    print(dict(zip(survivor_signals, result['model'].coef_)), 'intercept=', result['model'].intercept_)
    print(coefficient_stability_by_fold(result, result['search_panel']))
    print(result['holdout_report'])