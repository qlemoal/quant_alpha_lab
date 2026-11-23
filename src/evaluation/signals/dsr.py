'''
GOAL: the Deflated Sharpe Ratio (DSR) of Bailey and Lopez de Prado (2014), applied to a signal.
    It answers: "after I tried N signals and kept the best, is the best one still better than luck?"

THE PROBLEM
    Try 100 signals with true IC = 0. The best one will show a t-stat near 2.5 by pure chance.
    The more you try, the higher the best chance result. A Sharpe ratio (or IR) alone cannot tell.

THE IDEA IN THREE STEPS

    [1] Probabilistic Sharpe Ratio (PSR): the chance that the TRUE Sharpe is above a benchmark SR0,
        given a short sample, skewed and fat-tailed returns.

            PSR = Phi( (SR - SR0) * sqrt(T - 1) / sqrt(1 - skew*SR + (kurt - 1)/4 * SR^2) )

        SR, SR0 are per-observation (not annualised). T = number of observations.
        kurt is the plain kurtosis (a normal law has 3). Phi is the standard normal CDF.

    [2] SR0 = the best Sharpe you expect from N skill-less trials (the "expected maximum"):

            SR0 = sqrt(V) * ( (1 - g) * Phi^-1(1 - 1/N)  +  g * Phi^-1(1 - 1/(N e)) )

        V = variance of the Sharpe ratios of the N trials. g = 0.5772 (Euler-Mascheroni). e = 2.718.
        More trials -> higher SR0. With N = 1 there is no selection: SR0 = 0.

    [3] DSR = PSR evaluated at SR0.   DSR >= 0.95 means: 95% confidence that the signal beats the best luck.

WHAT SERIES DO WE FEED IT?
    The per-date rank IC of the signal, SIGN-ALIGNED with the prior, sampled every h days (h = label horizon)
    so that the labels do not overlap (an overlapping series has a false sample size T). "Sharpe" here is
    mean(IC) / std(IC) per observation: the information ratio of the IC series.

HONEST LIMITS
    - N counts the signals of this run PLUS n_trials_extra (variants you tried before and removed).
      Signals are correlated, so the true number of independent trials is lower: N is conservative.
    - V: by default the sampling variance of a Sharpe estimate under no skill (1 / (T - 1)). The paper
      uses the variance across the trials of the run. That one contains the real signals, which spreads V and
      raises SR0 (a true Sharpe of 0.25 per observation among 30 trials is not enough, 0.35 is). It is
      available as var_mode='trials', as a conservative check.
    - It fixes the multiple-testing bias, not survivorship or a wrong sample (see docs/findings.md).

Reference:
    Bailey, D. and Lopez de Prado, M. (2014). The deflated Sharpe ratio: correcting for selection bias,
    backtest overfitting and non-normality. Journal of Portfolio Management, 40(5).
'''

import numpy as np
from scipy.stats import norm, skew as _skew, kurtosis as _kurtosis

EULER_GAMMA = 0.5772156649015329


def sharpe_stats(x, step: int = 1) -> dict:
    '''
    Per-observation Sharpe, sample size, skewness and kurtosis of a series.
    step: keep one value every `step` points (use the label horizon for overlapping labels).
    NaN values are dropped first. A series with fewer than 4 points or zero variance gives sr = nan.
    '''
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)][::max(1, int(step))]
    n = len(v)
    if n < 4 or v.std(ddof=1) == 0:
        return {'sr': float('nan'), 'n': n, 'skew': float('nan'), 'kurt': float('nan')}
    return {
        'sr': float(v.mean() / v.std(ddof=1)),
        'n': n,
        'skew': float(_skew(v)),
        'kurt': float(_kurtosis(v, fisher=False)),     # plain kurtosis: normal = 3
    }


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    '''SR0: the expected best Sharpe of n_trials skill-less trials whose Sharpe ratios have variance var_sr.'''
    if n_trials <= 1 or not np.isfinite(var_sr) or var_sr <= 0:
        return 0.0
    g = EULER_GAMMA
    return float(np.sqrt(var_sr) * ((1 - g) * norm.ppf(1 - 1 / n_trials)
                                    + g * norm.ppf(1 - 1 / (n_trials * np.e))))


def probabilistic_sharpe(sr: float, n: int, skew: float = 0.0, kurt: float = 3.0, sr0: float = 0.0) -> float:
    '''PSR: probability that the true Sharpe exceeds sr0. nan if the inputs are not usable.'''
    if not np.isfinite(sr) or n < 2:
        return float('nan')
    skew = 0.0 if not np.isfinite(skew) else skew
    kurt = 3.0 if not np.isfinite(kurt) else kurt
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    if denom <= 0:                                     # extreme inputs: the formula breaks, say "not usable"
        return float('nan')
    return float(norm.cdf((sr - sr0) * np.sqrt(n - 1) / np.sqrt(denom)))


def deflated_sharpe(sr: float, n: int, skew: float, kurt: float, n_trials: int, var_sr: float) -> float:
    '''DSR = PSR at the expected maximum of n_trials skill-less trials.'''
    return probabilistic_sharpe(sr, n, skew, kurt, sr0=expected_max_sharpe(n_trials, var_sr))


def deflated_sharpe_table(ic_series: dict, signs: dict, step: int, n_trials_extra: int = 0,
                          var_mode: str = 'null') -> dict:
    '''
    DSR of every signal of a run.

    ic_series: {signal: array of daily IC}      signs: {signal: +1/-1/None}  (None -> +1)
    step:      label horizon (one IC every `step` days, so the labels do not overlap)
    n_trials_extra: signals you tried earlier and removed (they count as trials too)

    N = number of signals + n_trials_extra.
    V (var_mode) is the variance of the Sharpe ratios of SKILL-LESS trials:
        'null'   (default) its sampling variance, 1 / (T - 1) with T the median sample size. This is the exact
                 variance of a Sharpe estimate when the true Sharpe is 0, so SR0 answers the question
                 "how high can the best of N pure-noise trials get?".
        'trials' the variance of the Sharpe ratios of the signals of this run (the paper's choice). It also
                 contains the REAL signals, so V and SR0 grow when real signals exist: stricter, and unstable
                 with few trials. Use it as a conservative check.
    Returns {signal: {sr, n, skew, kurt, sr0, dsr, n_trials, var_sr}}.
    DSR is computed on the SIGN-ALIGNED series, so a signal with the wrong sign gets a very low DSR.
    '''
    raw = {k: sharpe_stats(v, step) for k, v in ic_series.items()}
    srs = np.array([s['sr'] for s in raw.values() if np.isfinite(s['sr'])])
    n_trials = int(len(ic_series) + max(0, n_trials_extra))
    if var_mode == 'trials':
        var_sr = float(srs.var(ddof=1)) if len(srs) > 1 else float('nan')
    elif var_mode == 'null':
        ns = [s['n'] for s in raw.values() if s['n'] > 1]
        var_sr = 1.0 / (float(np.median(ns)) - 1.0) if ns else float('nan')
    else:
        raise ValueError(f"var_mode must be 'null' or 'trials', got {var_mode!r}")
    sr0 = expected_max_sharpe(n_trials, var_sr)

    out = {}
    for k, s in raw.items():
        sign = signs.get(k) or 1
        sr_al = s['sr'] * sign if np.isfinite(s['sr']) else float('nan')
        skew_al = s['skew'] * sign if np.isfinite(s['skew']) else float('nan')   # flipping the sign flips the skew
        out[k] = {
            'sr': sr_al, 'n': s['n'], 'skew': skew_al, 'kurt': s['kurt'],
            'sr0': sr0, 'n_trials': n_trials, 'var_sr': var_sr,
            'dsr': probabilistic_sharpe(sr_al, s['n'], skew_al, s['kurt'], sr0),
        }
    return out