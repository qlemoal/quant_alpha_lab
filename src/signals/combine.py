# For a new signal, we should check the following list:
# - Build the signal via make_signal().
# - signal_plots.one_pager() — visual sanity check, catches structural bugs (coverage gaps, heatmap stripes) before anything downstream is trusted.
# - report.signal_report() — numeric health check: IC mean/std/IR, Newey-West-corrected t-stat, hit rate, turnover, naive long-short Sharpe.
# - If it clears both, ic.metrics.ic_decay() to see how far out the predictive power actually holds, informs rebalance frequency.
# - report.compare_reports() across every candidate signal you're considering, one table, side by side.
# - FDR across all candidates tested this round, not yet built, this is genuinely the next real piece of infrastructure, distinct from everything above.
# - Survivors move into the Elastic Net combiner / portfolio construction stage.


import polars as pl
from polars import col as c
from src.utils.helpers import as_list

from src.signals.normalize import zscore_expr
from src.signals.transforms import tanh_expr, clip_expr
from src.signals.ranking import rank_scaled_expr, decile_bucket_expr




def make_signal(lf, cols, method='zscore_tanh', **kwargs):
    '''
    Turn one or more raw features into signals, in a single with_columns call. Composes expressions in memory, adds exactly one '{col}_{method}'
        column per input, never touches or overwrites the original feature, never leaves intermediate columns behind,
        thanks to the "expr" version of our transform functions.
 
    cols: single column name or list of column names.
 
    method:
        'zscore_tanh'    -- z-score, then tanh. Default for well-behaved features (e.g., mom, adv, std).
        'zscore_clip'    -- z-score, then hard clip.
        'rank'           -- rank transform straight to [-1, 1]. Default for noisy features (e.g., beta for now).
        'decile_bucket'  -- long-short top-bottom deciles of a feature.
 
    kwargs:
        scale (for zscore_tanh), low/high (for zscore_clip), descending (for rank).
    '''

    cols = as_list(cols)
    exprs = []
 
    for col in cols:
        base = c(col)

        if method == 'zscore_tanh':
            e = tanh_expr( zscore_expr(base, kwargs.get('descending', False)), kwargs.get('scale', 1.0) )
        elif method == 'zscore':
            e =  zscore_expr(base, kwargs.get('descending', False))
        elif method == 'zscore_clip':
            e = clip_expr( zscore_expr(base, kwargs.get('descending', False)), kwargs.get('low', -3.0), kwargs.get('high', 3.0) )
        elif method == 'rank':
            e = rank_scaled_expr( base, kwargs.get('descending', False) )
        elif method == 'decile':
            e = decile_bucket_expr( base , kwargs.get('n_buckets', 10) )
        else:
            raise ValueError(f'unknown method: {method}')
 
        exprs.append(e.alias(f'{col}_{method}'))
 
    return lf.with_columns(exprs)