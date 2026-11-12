"""
Cheating Signals for Benchmarking
=================================

These signals use FUTURE information (forward returns) to establish
theoretical upper bounds on performance. Use ONLY for:
- Sanity checking your pipeline
- Understanding maximum achievable IC/Sharpe
- Debugging evaluation metrics

NEVER use in production or real backtests!

Integration with make_signal:
-----------------------------
Add to SIGNAL_SPECS in 04a_generate_signals.py:
    SIGNAL_SPECS = [
        # ... your real signals ...
        ('fwdret20', 'cheat_perfect', {}),
        ('fwdret20', 'cheat_rank', {}),
        ('fwdret20', 'cheat_decile', {}),
        ('fwdret20', 'cheat_noisy', {'signal_to_noise': 1.0}),
        ('fwdret20', 'cheat_noisy', {'signal_to_noise': 2.0}),
    ]

Note: Forward returns (fwdret1, fwdret5, fwdret10, fwdret20) must be added
BEFORE calling make_signal for cheat signals to work.
"""

from __future__ import annotations
import polars as pl
from polars import col as c
import numpy as np


# -----------------------------------------------------------------------------
# Expr-based transforms (compatible with make_signal)
# -----------------------------------------------------------------------------

def cheat_perfect_expr(base_expr: pl.Expr, **kwargs) -> pl.Expr:
    """
    Perfect foresight: normalized forward return.

    Usage in SIGNAL_SPECS:
        ('fwdret20', 'cheat_perfect', {})

    Output column: fwdret20_cheat_perfect
    """
    # Cross-sectional z-score normalization
    return (base_expr - base_expr.mean().over('date')) / base_expr.std().over('date')


def cheat_rank_expr(base_expr: pl.Expr, **kwargs) -> pl.Expr:
    """
    Rank-based perfect signal: ranks stocks by future return each day.
    Scaled to [-1, 1] like rank_scaled_expr.

    Usage in SIGNAL_SPECS:
        ('fwdret20', 'cheat_rank', {})

    Output column: fwdret20_cheat_rank
    """
    # Rank within each date (1 = best/lowest return if descending=False)
    # We want highest future return = highest rank = long signal
    rank = base_expr.rank(method='ordinal', descending=True).over('date')
    max_rank = rank.max().over('date')
    # Scale to [-1, 1]: 2 * (rank - 1) / (max_rank - 1) - 1
    return (2 * (rank - 1) / (max_rank - 1) - 1)


def cheat_decile_expr(base_expr: pl.Expr, n_buckets: int = 10, **kwargs) -> pl.Expr:
    """
    Decile-based perfect signal: +1 for top decile, -1 for bottom, 0 otherwise.

    Usage in SIGNAL_SPECS:
        ('fwdret20', 'cheat_decile', {'n_buckets': 10})

    Output column: fwdret20_cheat_decile
    """
    decile = base_expr.qcut(n_buckets, labels=range(n_buckets)).over('date')
    return pl.when(decile == n_buckets - 1).then(1.0) \
             .when(decile == 0).then(-1.0) \
             .otherwise(0.0)


def cheat_noisy_expr(base_expr: pl.Expr, signal_to_noise: float = 1.0, **kwargs) -> pl.Expr:
    """
    Noisy perfect signal: forward return + deterministic pseudo-noise.

    signal_to_noise = 1.0 -> expected IC ~ 0.71
    signal_to_noise = 2.0 -> expected IC ~ 0.89
    signal_to_noise = 5.0 -> expected IC ~ 0.98

    Usage in SIGNAL_SPECS:
        ('fwdret20', 'cheat_noisy', {'signal_to_noise': 1.0})

    Output column: fwdret20_cheat_noisy
    """
    # Deterministic pseudo-noise from ticker hash (reproducible)
    noise = (c('ticker').hash() % 10000 / 10000.0 - 0.5)
    # Scale noise relative to signal std
    signal_std = base_expr.std().over('date')
    noisy_signal = base_expr + noise * signal_std / signal_to_noise
    # Cross-sectional z-score
    return (noisy_signal - noisy_signal.mean().over('date')) / noisy_signal.std().over('date')


# -----------------------------------------------------------------------------
# Register with make_signal
# -----------------------------------------------------------------------------

# Map method names to expr functions
_CHEAT_METHODS = {
    'cheat_perfect': cheat_perfect_expr,
    'cheat_rank': cheat_rank_expr,
    'cheat_decile': cheat_decile_expr,
    'cheat_noisy': cheat_noisy_expr,
}


def register_cheat_methods():
    """
    Register cheat methods with make_signal.
    Call this before using cheat methods in SIGNAL_SPECS.

    Example:
        from src.signals.cheat import register_cheat_methods
        register_cheat_methods()

        # Now you can use in SIGNAL_SPECS:
        SIGNAL_SPECS = [
            ('fwdret20', 'cheat_perfect', {}),
            ('fwdret20', 'cheat_rank', {}),
            ('fwdret20', 'cheat_noisy', {'signal_to_noise': 1.0}),
        ]
    """
    from src.signals.combine import _METHOD_REGISTRY
    _METHOD_REGISTRY.update(_CHEAT_METHODS)


# -----------------------------------------------------------------------------
# Convenience functions for batch adding
# -----------------------------------------------------------------------------

def add_all_cheat_signals(lf: pl.LazyFrame, horizons: list[int] = None,
                           include_perfect: bool = True,
                           include_rank: bool = True,
                           include_decile: bool = True,
                           include_noisy: bool = True,
                           noisy_snrs: list[float] = None) -> pl.LazyFrame:
    """
    Add all cheating signals at once for a comprehensive benchmark.

    Requires fwdret{h} columns to exist in the LazyFrame.

    Args:
        lf: LazyFrame with fwdret columns
        horizons: List of horizons to create signals for (default: [1, 5, 10, 20])
        include_perfect: Add perfect normalized signal
        include_rank: Add rank-based perfect signal
        include_decile: Add decile-based perfect signal
        include_noisy: Add noisy perfect signals
        noisy_snrs: Signal-to-noise ratios for noisy signals (default: [0.5, 1.0, 2.0, 5.0])

    Returns:
        LazyFrame with all cheat signals added
    """
    if horizons is None:
        horizons = [1, 5, 10, 20]
    if noisy_snrs is None:
        noisy_snrs = [0.5, 1.0, 2.0, 5.0]

    # Check which forward return columns exist
    schema = lf.collect_schema()
    available_horizons = [h for h in horizons if f'fwdret{h}' in schema.names()]

    if not available_horizons:
        available = [c for c in schema.names() if c.startswith('fwdret')]
        raise ValueError(f'No fwdret columns found for horizons {horizons}. Available: {available}')

    print(f'Adding cheat signals for horizons: {available_horizons}')

    for h in available_horizons:
        fwd_col = f'fwdret{h}'
        if include_perfect:
            lf = lf.with_columns(cheat_perfect_expr(c(fwd_col)).alias(f'{fwd_col}_cheat_perfect'))
            print(f'  Added {fwd_col}_cheat_perfect')
        if include_rank:
            lf = lf.with_columns(cheat_rank_expr(c(fwd_col)).alias(f'{fwd_col}_cheat_rank'))
            print(f'  Added {fwd_col}_cheat_rank')
        if include_decile:
            lf = lf.with_columns(cheat_decile_expr(c(fwd_col)).alias(f'{fwd_col}_cheat_decile'))
            print(f'  Added {fwd_col}_cheat_decile')
        if include_noisy:
            for snr in noisy_snrs:
                lf = lf.with_columns(cheat_noisy_expr(c(fwd_col), signal_to_noise=snr).alias(f'{fwd_col}_cheat_noisy_sn{snr:.1f}'))
                print(f'  Added {fwd_col}_cheat_noisy_sn{snr:.1f}')

    return lf


def theoretical_ic(signal_to_noise: float) -> float:
    """
    Theoretical IC for a noisy perfect signal.

    If signal = true_return + noise, where noise ~ N(0, sigma_n)
    and true_return ~ N(0, sigma_s), then:
    IC = corr(signal, true_return) = sigma_s / sqrt(sigma_s^2 + sigma_n^2)
       = SNR / sqrt(1 + SNR^2) where SNR = sigma_s / sigma_n
    """
    return signal_to_noise / np.sqrt(1 + signal_to_noise**2)


# Expected ICs for default noisy SNRs
EXPECTED_ICS = {snr: theoretical_ic(snr) for snr in [0.5, 1.0, 2.0, 5.0]}
# {0.5: 0.447, 1.0: 0.707, 2.0: 0.894, 5.0: 0.981}