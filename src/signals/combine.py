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
from src.signals.ranking import rank_scaled_expr, decile_expr


# Method registry for extensibility (e.g., cheat signals)
_METHOD_REGISTRY = {
    'zscore_tanh': lambda base, **kw: tanh_expr(zscore_expr(base, kw.get('descending', False)), kw.get('scale', 1.0)),
    'zscore': lambda base, **kw: zscore_expr(base, kw.get('descending', False)),
    'zscore_clip': lambda base, **kw: clip_expr(zscore_expr(base, kw.get('descending', False)), kw.get('low', -3.0), kw.get('high', 3.0)),
    'rank': lambda base, **kw: rank_scaled_expr(base, kw.get('descending', False)),
    'decile': lambda base, **kw: decile_expr(base, kw.get('n_buckets', 10)),
}


def register_method(name: str, func):
    """
    Register a custom signal transformation method.

    Args:
        name: Method name (e.g., 'cheat_perfect')
        func: Function with signature (base_expr: pl.Expr, **kwargs) -> pl.Expr

    Example:
        from src.signals.combine import register_method
        from src.signals.cheat import cheat_perfect_expr
        register_method('cheat_perfect', cheat_perfect_expr)
    """
    _METHOD_REGISTRY[name] = func


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
        'decile'         -- long-short top-bottom deciles of a feature.
        'cheat_perfect'  -- normalized forward return (requires fwdret{h} column)
        'cheat_rank'     -- rank by future return, scaled to [-1, 1]
        'cheat_decile'   -- +1 top decile, -1 bottom decile, 0 otherwise
        'cheat_noisy'    -- forward return + noise (requires signal_to_noise kwarg)

    kwargs:
        scale (for zscore_tanh), low/high (for zscore_clip), descending (for rank),
        n_buckets (for decile), signal_to_noise (for cheat_noisy).
    '''

    cols = as_list(cols)
    exprs = []

    for col in cols:
        base = c(col)

        if method in _METHOD_REGISTRY:
            e = _METHOD_REGISTRY[method](base, **kwargs)
        else:
            raise ValueError(f'unknown method: {method}. Available: {list(_METHOD_REGISTRY.keys())}')

        exprs.append(e.alias(f'{col}_{method}'))

    return lf.with_columns(exprs)