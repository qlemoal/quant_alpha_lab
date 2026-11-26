'''
GOAL: measure how often each test of stage 05 says "significant", on simulated IC series with a KNOWN truth.
    true IC = 0   ->  the rejection rate is the SIZE (false-positive rate). It must be close to the nominal 5%.
    true IC > 0   ->  the rejection rate is the POWER. Higher is better.
A test whose size is 12% at a nominal 5% admits more than twice the junk it promises. This file shows which test
keeps its promise. scripts/research/calibrate_tests.py prints the table; tests/test_bootstrap.py checks it.

THE SIMULATED IC SERIES (it copies the two sources of memory of a real daily IC series)
    e_t                 independent N(0, 1) shocks, one per day
    persistence p       the signal changes slowly: average of the last p shocks
    overlap h           the label is an h-day return: average of the last h values again
    IC_t = mean_ic + ic_std * (the result, rescaled to variance 1)
    So the series is autocorrelated up to lag about h + p - 2, like a real IC series with fwdret{h}.
'''

import numpy as np

from src.evaluation.signals.ic.metrics import newey_west_ic_tstat, ic_pvalue_nw, ic_tstat_non_overlapping
from src.evaluation.signals.bootstrap import bootstrap_test_table
from scipy.stats import norm


def simulate_ic_series(n: int, horizon: int, mean_ic: float = 0.0, ic_std: float = 0.15,
                       persistence: int = 1, rng: "np.random.Generator | None" = None) -> np.ndarray:
    '''One daily IC series of length n (see the header). rng: numpy Generator (default: a new one, seed 0).'''
    rng = rng or np.random.default_rng(0)
    p, h = max(1, int(persistence)), max(1, int(horizon))
    e = rng.normal(size=n + p + h)
    x = np.convolve(e, np.ones(p) / p, mode='valid')
    x = np.convolve(x, np.ones(h) / h, mode='valid')[:n]
    x = x / np.sqrt(np.sum((np.convolve(np.ones(p) / p, np.ones(h) / h)) ** 2))   # exact variance 1 (do NOT demean)
    return mean_ic + ic_std * x


# -----------------------------------------------------------------------------
# One p-value per test (two-sided for Newey-West and non-overlap, as stage 05 reports them;
# one-sided in the + direction for the bootstrap, as stage 05 uses it with a + prior).
# -----------------------------------------------------------------------------

def p_newey_west(ic: np.ndarray, horizon: int) -> float:
    return ic_pvalue_nw(newey_west_ic_tstat(ic, horizon=horizon))


def p_non_overlap(ic: np.ndarray, horizon: int) -> float:
    t = ic_tstat_non_overlapping(ic, horizon=horizon)['t_stat']
    return float('nan') if t is None else float(2 * (1 - norm.cdf(abs(t))))


def p_bootstrap(ic: np.ndarray, horizon: int, n_boot: int = 199, seed: int = 0) -> float:
    dates = list(range(len(ic)))
    tab = bootstrap_test_table({'s': (dates, ic)}, {'s': +1}, horizon, n_boot=n_boot, seed=seed)
    return tab['signals']['s']['p_single']


TESTS = {
    'newey_west (two-sided)': p_newey_west,
    'non_overlap (two-sided)': p_non_overlap,
    'bootstrap (one-sided)': p_bootstrap,
}


def rejection_rates(n_sims: int, n: int, horizon: int, mean_ic: float = 0.0, persistence: int = 1,
                    level: float = 0.05, ic_std: float = 0.15, seed: int = 0, tests: "dict | None" = None) -> dict:
    '''
    Share of n_sims simulated series where each test gives p < level.
    For the two-sided tests under a positive true IC, a rejection counts only if the t-stat is positive.
    Returns {test name: rate}.
    '''
    tests = tests or TESTS
    rng = np.random.default_rng(seed)
    hits = {k: 0 for k in tests}
    for i in range(n_sims):
        ic = simulate_ic_series(n, horizon, mean_ic, ic_std, persistence, rng)
        for name, fn in tests.items():
            p = fn(ic, horizon) if 'bootstrap' not in name else fn(ic, horizon, seed=i)
            right_side = ic.mean() > 0 if mean_ic > 0 else True
            hits[name] += bool(np.isfinite(p) and p < level and right_side)
    return {k: v / n_sims for k, v in hits.items()}
