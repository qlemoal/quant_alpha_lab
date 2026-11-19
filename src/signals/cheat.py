'''
Cheating signals: a benchmark with a KNOWN information coefficient (IC).

A cheating signal is built from the FUTURE return, plus noise. It cannot be traded.
Use it for one thing only: to see what the pipeline does with a signal of a known strength.

    IC = 0.90   near-perfect foresight     the upper bound: the pipeline must find it at once
    IC = 0.30   very strong                far above any real signal
    IC = 0.10   strong                     above any real daily signal
    IC = 0.05   good                       about the best real signals
    IC = 0.02   weak                       a typical real signal
    IC = 0.01   very weak                  the pipeline should still see it, with 25 years of data

PLACEBO SIGNALS (the other half of this file)
    A placebo is a signal of pure noise. It uses NO return, past or future, so it is not a cheat.
    Its true IC is exactly 0. A sound pipeline must NOT select it, and must give it no weight.
    Use it to measure the false discoveries: how many placebos pass the FDR step? How much weight do they get?

HOW IT IS BUILT (for one date, across the stocks)

    forward return of the stock   fwdret{h}
            |  rank across the stocks, then normal scores          -> z_true   (mean 0, std 1)
            v
    signal = rho * z_true  +  sqrt(1 - rho^2) * noise                    noise: N(0, 1), independent
            |
            v
    rho = 2 sin(pi * ic / 6)  makes the RANK correlation (Spearman) of the signal with fwdret{h} equal to `ic`
    (for normal variables, Spearman = 6/pi * asin(rho / 2)). The IC measured in 05 is a Spearman IC.

HOW SLOWLY THE NOISE CHANGES (noise_days)
    noise_days = 1: a new noise every date. The IC of the signal is then almost independent from one date to the
    next, so 25 years give a huge t-statistic, and the signal reshuffles every day (huge turnover).
    A real signal is persistent. With noise_days = k, the noise is the average of the last k daily draws
    (still N(0, 1) across the stocks). The signal changes slowly, its IC series is autocorrelated, the t-statistic is
    smaller, and the turnover is realistic. The mean IC over a SHORT period then differs more from the wanted IC
    (the IC series has fewer independent dates). Over 25 years it is close.

WHY THE NOISE IS BUILT FROM (ticker, date)
    A new noise value for every stock AND every date, so the noise is independent across days.
    A noise made from the ticker alone would be a fixed bias of the stock, and it would not lower the IC the
    way a real error does. The noise is a hash of (ticker, date, seed), so a run can be repeated exactly.

WHERE IT IS USED
    04_generate_signals.py with the options --cheat and --placebo. See CHEAT_SIGNALS and PLACEBO_COUNT in that script.

NEVER READ A RESULT FROM A CHEAT RUN AS A STRATEGY RESULT.
    It uses the future. The run must have `cheat` in its tag (04 checks this). A run with placebos must have
    `placebo` in its tag. Both options together: a tag with both words, for example `cheat_placebo`.
'''

import math

import numpy as np
import polars as pl

_TWO32 = 4294967296.0
MAX_IC = 0.99
CHEAT_PREFIX = 'cheat_'
PLACEBO_PREFIX = 'placebo_'
PLACEBO_SEED_OFFSET = 1_000_000          # placebo noise streams never share a seed with the cheat noise streams


def cheat_name(horizon: int, ic: float) -> str:
    '''Column name of a cheat signal. Example: (20, 0.05) -> cheat_h20_ic050 ; (20, 0.9) -> cheat_h20_ic900.'''
    return f'cheat_h{horizon}_ic{int(round(ic * 1000)):03d}'


def validate_cheat_specs(specs: list, horizons: list) -> list:
    '''
    Check the list [(horizon, ic), ...]. Raises ValueError on a mistake. Returns the list as tuples.
    The horizon must be one of the label horizons of 04, and 0 < ic <= 0.99. Two equal names are an error.
    IC 1 is refused: the IC would be the same on every date (std 0), so its t-statistic does not exist and 05 drops it.
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


def _normal_noise(seed: int, days: int = 1) -> pl.Expr:
    '''
    Standard normal noise per (ticker, date), from two uniforms (Box-Muller).
    days = 1: a new draw every date (independent across dates).
    days = k > 1: the average of the last k daily draws of the ticker, times sqrt(k). Still N(0, 1) across the
    stocks of a date, but it changes slowly: the autocorrelation at lag 1 is (k - 1) / k, like a slow real signal.
    The first k - 1 rows of a ticker have no value (null). Needs the frame sorted by (ticker, date).
    '''
    u1, u2 = _uniform(2 * seed + 1), _uniform(2 * seed + 2)
    base = (-2.0 * u1.log()).sqrt() * (2.0 * math.pi * u2).cos()
    if days <= 1:
        return base
    return base.rolling_mean(window_size=days, min_samples=days).over('ticker') * math.sqrt(days)


def _check_days(days) -> int:
    if int(days) != days or days < 1:
        raise ValueError(f'noise_days must be an integer >= 1, got {days}')
    return int(days)


def _probit(u: pl.Expr) -> pl.Expr:
    '''Inverse normal CDF of u in (0, 1). Abramowitz and Stegun 26.2.23: absolute error below 5e-4.'''
    p = pl.min_horizontal(u, 1.0 - u)
    s = (-2.0 * p.log()).sqrt()
    x = s - (2.515517 + 0.802853 * s + 0.010328 * s * s) / (1.0 + 1.432788 * s + 0.189269 * s * s + 0.001308 * s * s * s)
    return pl.when(u < 0.5).then(-x).otherwise(x)


def cheat_expr(horizon: int, ic: float, seed: int = 0, noise_days: int = 1) -> pl.Expr:
    '''
    Expression of one cheat signal. The column fwdret{horizon} must exist in the frame.
    Rows without a label (the last rows of a ticker) get no signal (null).
    noise_days: how slowly the noise changes (see _normal_noise).
    '''
    fwd = pl.col(f'fwdret{horizon}')
    n = fwd.count().over('date')                                    # stocks with a label on this date
    u = (fwd.rank(method='average').over('date') - 0.5) / n         # uniform rank in (0, 1)
    z_true = _probit(u)                                              # normal scores: N(0, 1), monotone in the return
    rho = 2.0 * math.sin(math.pi * ic / 6.0)                        # Pearson rho that gives a Spearman IC of `ic`
    return rho * z_true + math.sqrt(1.0 - rho * rho) * _normal_noise(seed, noise_days)


def add_cheat_signals(lf: pl.LazyFrame, specs: list, seed: int = 0, noise_days: int = 1) -> "tuple[pl.LazyFrame, list]":
    '''
    Add one cheat column per (horizon, ic). Lazy. The labels fwdret{h} must already be in `lf`.
    Every signal has its OWN noise (the stream depends on the position in `specs`). This is important: with a shared
    noise, a linear model could subtract one signal from another, cancel the noise and rebuild the future return.
    Use another `seed` for another draw.
    noise_days: 1 = new noise every date. 20 = a noise that changes slowly (a real signal is persistent, and
    so is its IC series: the effective number of independent dates is smaller, and so is the turnover).
    The frame is sorted by (ticker, date) first. Returns (lf, names).
    '''
    noise_days = _check_days(noise_days)
    lf = lf.sort(['ticker', 'date'])
    have = set(lf.collect_schema().names())
    exprs, names = [], []
    for i, (h, ic) in enumerate(specs):
        if f'fwdret{h}' not in have:
            raise ValueError(f'fwdret{h} is missing: add the labels before the cheat signals')
        name = cheat_name(h, ic)
        exprs.append(cheat_expr(h, ic, seed=1000 * seed + i, noise_days=noise_days).alias(name))
        names.append(name)
    return lf.with_columns(exprs), names


# =============================================================================
# PLACEBO SIGNALS: pure noise, no return used
# =============================================================================

def placebo_names(n: int) -> list:
    '''Column names of n placebo signals: placebo_01, placebo_02, ... (zero-padded to the same width).'''
    width = max(2, len(str(n)))
    return [f'{PLACEBO_PREFIX}{i + 1:0{width}d}' for i in range(n)]


def add_placebo_signals(lf: pl.LazyFrame, n: int, seed: int = 0, noise_days=(1, 5, 20, 60)) -> "tuple[pl.LazyFrame, list]":
    '''
    Add n placebo columns: N(0, 1) noise across the stocks of each date. Lazy.
    No label and no return is read, so this can run before or after the labels.
    Every placebo has its own noise stream. The same seed gives the same noise.
    noise_days: an integer, or a list that the placebos use in turn. A real signal that is pure junk is still
    persistent (a 60-day momentum changes slowly), and its IC series is autocorrelated: that is the case where a
    wrong Newey-West lag gives false discoveries. So the default mixes fast and slow placebos: 1, 5, 20, 60 days.
    The frame is sorted by (ticker, date) first. Returns (lf, names).
    '''
    if n < 1:
        raise ValueError(f'placebo count must be at least 1, got {n}')
    days = [noise_days] if isinstance(noise_days, int) else list(noise_days)
    days = [_check_days(d) for d in days]
    lf = lf.sort(['ticker', 'date'])
    names = placebo_names(n)
    exprs = [_normal_noise(PLACEBO_SEED_OFFSET + 1000 * seed + i, days[i % len(days)]).alias(name)
             for i, name in enumerate(names)]
    return lf.with_columns(exprs), names


# =============================================================================
# GROUPS AND TAG RULES (used by 04, 05 and 06)
# =============================================================================

def signal_group(name: str) -> str:
    '''"cheat", "placebo" or "real". Decided by the column name only.'''
    if name.startswith(CHEAT_PREFIX):
        return 'cheat'
    if name.startswith(PLACEBO_PREFIX):
        return 'placebo'
    return 'real'


def check_benchmark_tag(tag: str, cheat: bool, placebo: bool) -> None:
    '''
    A run with benchmark signals must say so in its tag, so that it never replaces a real run.
    --cheat needs "cheat" in the tag. --placebo needs "placebo" in the tag. Raises ValueError.
    '''
    low = tag.lower()
    if cheat and 'cheat' not in low:
        raise ValueError(f'--cheat needs a tag that contains "cheat" (got {tag!r}). Example: --tag cheat')
    if placebo and 'placebo' not in low:
        raise ValueError(f'--placebo needs a tag that contains "placebo" (got {tag!r}). '
                         f'Example: --tag placebo, or --tag cheat_placebo with --cheat')