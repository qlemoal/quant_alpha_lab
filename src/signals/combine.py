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
            e = tanh_expr( zscore_expr(base), kwargs.get('scale', 1.0) )
        elif method == 'zscore':
            e =  zscore_expr(base)
        elif method == 'zscore_clip':
            e = clip_expr( zscore_expr(base), kwargs.get('low', -3.0), kwargs.get('high', 3.0) )
        elif method == 'rank':
            e = rank_scaled_expr( base, kwargs.get('descending', False) )
        elif method == 'decile':
            e = decile_bucket_expr( base , kwargs.get('n_buckets', 10) )
        else:
            raise ValueError(f'unknown method: {method}')
 
        exprs.append(e.alias(f'{col}_{method}'))
 
    return lf.with_columns(exprs)