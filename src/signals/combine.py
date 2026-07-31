import polars as pl
from polars import col as c
from src.utils.helpers import as_list

from normalize import zscore, apply_tanh, apply_clip
from ranking import rank_into_minus1_1



def make_signal(lf, cols, method='zscore_tanh', group='date', **kwargs):
    '''
    Turn one or more raw features into signals, in a single call.
        Systematic entry point meant to be called from the signal library,
        rather than chaining the individual transforms by hand each time.
 
    cols: single column name or list of column names.
 
    method:
        'zscore_tanh' = z-score, then tanh. Default for well-behaved, trustworthy-magnitude features (mom, adv, std).
        'zscore_clip' = z-score, then hard clip.
        'rank'        = rank transformed into [-1, 1]. Default for noisy features (e.g., beta for now).
 
    kwargs:
        scale (for zscore_tanh), lower/upper (for zscore_clip).
 
    Returns lf with new '{col}_signal' columns added, originals kept.
    '''

    cols = as_list(cols)
 
    if method == 'zscore_tanh':
        lf = zscore(lf, cols, group)
        lf = apply_tanh(lf, [f'{col}_z' for col in cols], kwargs.get('scale', 1.0))
        rename_map = {f'{col}_z_tanh': f'{col}_signal' for col in cols}
 
    elif method == 'zscore_clip':
        lf = zscore(lf, cols, group)
        lf = apply_clip(
            lf,
            [f'{col}_z' for col in cols],
            kwargs.get('lower', -3.0),
            kwargs.get('upper', 3.0),
        )
        rename_map = {f'{col}_z_clipped': f'{col}_signal' for col in cols}
 
    elif method == 'rank':
        lf = rank_into_minus1_1(lf, cols, group)
        rename_map = {f'{col}_rank': f'{col}_signal' for col in cols}
 
    else:
        raise ValueError(f'unknown method: {method}')
 
    return lf.rename(rename_map)