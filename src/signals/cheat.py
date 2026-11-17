'''
Cheating signals: a benchmark with a KNOWN information coefficient (IC).

A cheating signal is built from the FUTURE return, plus noise. It cannot be traded.
Use it for one thing only: to see what the pipeline does with a signal of a known strength.

    IC = 0.90   near-perfect foresight     the upper bound: the pipeline must find it at once
    IC = 0.30   very strong                far above any real signal
    IC = 0.10   strong                     above any real daily signal
    IC = 0.05   good                       about the best real signals
    IC = 0.02   weak                       a typical real signal

HOW IT IS BUILT (for one date, across the stocks)

    forward return of the stock   fwdret{h}
            |  rank across the stocks, then normal scores          -> z_true   (mean 0, std 1)
            v
    signal = rho * z_true  +  sqrt(1 - rho^2) * noise                    noise: N(0, 1), independent
            |
            v
    rho = 2 sin(pi * ic / 6)  makes the RANK correlation (Spearman) of the signal with fwdret{h} equal to `ic`
    (for normal variables, Spearman = 6/pi * asin(rho / 2)). The IC measured in 04b is a Spearman IC.

WHY THE NOISE IS BUILT FROM (ticker, date)
    A new noise value for every stock AND every date, so the noise is independent across days.
    A noise made from the ticker alone would be a fixed bias of the stock, and it would not lower the IC the
    way a real error does. The noise is a hash of (ticker, date, seed), so a run can be repeated exactly.

WHERE IT IS USED
    04a_generate_signals.py with the option --cheat. See CHEAT_SIGNALS in that script.

NEVER READ A RESULT FROM A CHEAT RUN AS A STRATEGY RESULT.
    It uses the future. The run must have `cheat` in its tag (04a checks this).
'''

import math

import numpy as np
import polars as pl

_TWO32 = 4294967296.0
MAX_IC = 0.99


def cheat_name(horizon: int, ic: float) -> str:
    '''Column name of a cheat signal. Example: (20, 0.05) -> cheat_h20_ic050 ; (20, 0.9) -> cheat_h20_ic900.'''
    return f'cheat_h{horizon}_ic{int(round(ic * 1000)):03d}'


def validate_cheat_specs(specs: list, horizons: list) -> list:
    '''
    Check the list [(horizon, ic), ...]. Raises ValueError on a mistake. Returns the list as tuples.
    The horizon must be one of the label horizons of 04a, and 0 < ic <= 0.99. Two equal names are an error.
    IC 1 is refused: the IC would be the same on every date (std 0), so its t-statistic does not exist and 04b drops it.
    '''
    out, seen = [], set()
    for spec in specs:
        h, ic = spec
        if h not in horizons:
            raise ValueError(f'cheat horizon {h} is not in HORIZONS {horizons}')
        if not 0 < ic <= MAX_IC:
            raise ValueError(f'cheat IC must be in (0, {MAX_IC}], got {ic}')
        name = cheat_name(h, ic)
        if name in seen:
            raise ValueError(f'cheat signal {name} is defined twice')
        seen.add(name)
        out.append((int(h), float(ic)))
    return out


def _uniform(seed: int) -> pl.Expr:
    '''Pseudo-random numbers in (0, 1), one per (ticker, date). The same input and seed give the same number.'''
    h = pl.struct(['ticker', 'date']).hash(seed=seed)
    return ((h % 4294967296).cast(pl.Float64) + 0.5) / _TWO32


def _normal_noise(seed: int) -> pl.Expr:
    '''Standard normal noise per (ticker, date), from two uniforms (Box-Muller).'''
    u1, u2 = _uniform(2 * seed + 1), _uniform(2 * seed + 2)
    return (-2.0 * u1.log()).sqrt() * (2.0 * math.pi * u2).cos()


def _probit(u: pl.Expr) -> pl.Expr:
    '''Inverse normal CDF of u in (0, 1). Abramowitz and Stegun 26.2.23: absolute error below 5e-4.'''
    p = pl.min_horizontal(u, 1.0 - u)
    s = (-2.0 * p.log()).sqrt()
    x = s - (2.515517 + 0.802853 * s + 0.010328 * s * s) / (1.0 + 1.432788 * s + 0.189269 * s * s + 0.001308 * s * s * s)
    return pl.when(u < 0.5).then(-x).otherwise(x)


def cheat_expr(horizon: int, ic: float, seed: int = 0) -> pl.Expr:
    '''
    Expression of one cheat signal. The column fwdret{horizon} must exist in the frame.
    Rows without a label (the last rows of a ticker) get no signal (null).
    '''
    fwd = pl.col(f'fwdret{horizon}')
    n = fwd.count().over('date')                                    # stocks with a label on this date
    u = (fwd.rank(method='average').over('date') - 0.5) / n         # uniform rank in (0, 1)
    z_true = _probit(u)                                              # normal scores: N(0, 1), monotone in the return
    rho = 2.0 * math.sin(math.pi * ic / 6.0)                        # Pearson rho that gives a Spearman IC of `ic`
    return rho * z_true + math.sqrt(1.0 - rho * rho) * _normal_noise(seed)


def add_cheat_signals(lf: pl.LazyFrame, specs: list, seed: int = 0) -> "tuple[pl.LazyFrame, list]":
    '''
    Add one cheat column per (horizon, ic). Lazy. The labels fwdret{h} must already be in `lf`.
    Every signal has its OWN noise (the stream depends on the position in `specs`). This is important: with a shared
    noise, a linear model could subtract one signal from another, cancel the noise and rebuild the future return.
    Use another `seed` for another draw.
    Returns (lf, names).
    '''
    have = set(lf.collect_schema().names())
    exprs, names = [], []
    for i, (h, ic) in enumerate(specs):
        if f'fwdret{h}' not in have:
            raise ValueError(f'fwdret{h} is missing: add the labels before the cheat signals')
        name = cheat_name(h, ic)
        exprs.append(cheat_expr(h, ic, seed=1000 * seed + i).alias(name))
        names.append(name)
    return lf.with_columns(exprs), names