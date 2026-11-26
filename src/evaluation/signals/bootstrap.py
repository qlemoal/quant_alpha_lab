'''
GOAL: a test of "mean IC > 0" whose false-positive rate is KNOWN, and a multiple-testing control that uses the
REAL dependence between the signals. These two numbers decide the admission of a signal in stage 05.

WHY NOT ONLY NEWEY-WEST?
    The daily IC series is autocorrelated: labels overlap by h - 1 days, and a slow signal adds more memory.
    Newey-West with lag 2(h - 1) still gives about 7-8% false positives at a nominal 5% (simulation in
    tests/test_bootstrap.py and scripts/research/calibrate_tests.py). A bootstrap that resamples BLOCKS of days keeps
    this memory inside each block, so its null distribution has the right width.

WHY ROMANO-WOLF AND NOT BENJAMINI-YEKUTIELI?
    BY assumes the worst possible dependence between the tests: very strict, and blind to the real one.
    Our signals are strongly dependent (mom20 and mom60 share 20 days of returns). Romano-Wolf resamples the SAME
    blocks of days for all signals at once, so the bootstrap world has the same correlation between signals as the
    data. It controls the family-wise error rate (FWER): the chance of admitting AT LEAST ONE false signal.

THE PIPELINE OF THIS FILE
    ic_dated {signal: (dates, IC)}
        |  align_ic_series            one matrix: rows = dates (union), columns = signals, NaN where missing
        |  common_block_length        Politis-White automatic block length, max over signals, at least h
        |  stationary_bootstrap_indices   B resampled date orders (blocks of random length, mean = block length)
        |  bootstrap_t_stats          t = mean / se,  t* = (mean* - mean) / se*   (se by batch means, recomputed in
        |                             every resample: the "studentized" or "percentile-t" bootstrap)
        |  align                      one-sided in the prior direction, or two-sided without a prior
        |  single_pvalues             one test per signal:  p = share of t* >= t
        |  romano_wolf                adjusted p for the whole family (step-down on the max of t*)
        v
    bootstrap_test_table  ->  {signal: mean_ic, se, t, p_single, p_rw, ...}

ONE-SIDED OR TWO-SIDED
    A signal with a prior sign (+1 or -1, src/signals/priors.py) is tested ONE-SIDED in the prior direction.
    A signal without a prior is tested TWO-SIDED (|t|). Exploration only.

References:
    Politis, D. and Romano, J. (1994). The stationary bootstrap. JASA, 89(428).
    Politis, D. and White, H. (2004). Automatic block-length selection for the dependent bootstrap. Econometric
        Reviews, 23(1). Correction: Patton, Politis and White (2009), Econometric Reviews, 28(4).
    Romano, J. and Wolf, M. (2005). Stepwise multiple testing as formalized data snooping. Econometrica, 73(4).
'''

import warnings

import numpy as np


# =============================================================================
# 1. ALIGN THE IC SERIES
# =============================================================================

def align_ic_series(ic_dated: dict) -> "tuple[np.ndarray, list, np.ndarray]":
    '''
    ic_dated: {signal: (list of dates, array of IC)}.
    Returns (dates (n,), names (K,), M (n, K)). M[i, k] = IC of signal k on date i, NaN if the signal has no IC
    on that date. Dates are the sorted union of all dates.
    '''
    names = list(ic_dated)
    all_dates = sorted({d for dates, _ in ic_dated.values() for d in dates})
    pos = {d: i for i, d in enumerate(all_dates)}
    M = np.full((len(all_dates), len(names)), np.nan)
    for k, name in enumerate(names):
        dates, ic = ic_dated[name]
        M[[pos[d] for d in dates], k] = np.asarray(ic, dtype=float)
    return np.array(all_dates), names, M


# =============================================================================
# 2. BLOCK LENGTH (Politis-White 2004, corrected 2009)
# =============================================================================

def _flat_top(s: np.ndarray) -> np.ndarray:
    '''Flat-top kernel: 1 on |s| <= 1/2, then a straight line down to 0 at |s| = 1.'''
    a = np.abs(s)
    return np.where(a <= 0.5, 1.0, np.where(a <= 1.0, 2.0 * (1.0 - a), 0.0))


def politis_white_block_length(x) -> float:
    '''
    Optimal MEAN block length of the stationary bootstrap for the mean of x (NaN are dropped).
    Steps:
      1. autocorrelations up to m_max = sqrt(n) + K_n   (K_n = 5)
      2. m_hat = first lag after which K_n autocorrelations in a row are inside +-2 sqrt(log10(n) / n)
      3. M = 2 m_hat;  G = sum over |k| <= M of w(k/M) |k| R(k);  g = sum of w(k/M) R(k)   (w: flat-top kernel)
      4. b = (2 G^2 / (2 g^2))^(1/3) n^(1/3), clipped to [1, min(3 sqrt(n), n/3)]
    A series with no memory gives b near 1. A series with long memory gives a long block.
    '''
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 20 or v.std() == 0:
        return 1.0
    v = v - v.mean()
    kn = 5
    m_max = int(np.ceil(np.sqrt(n))) + kn
    b_max = float(np.ceil(min(3.0 * np.sqrt(n), n / 3.0)))
    acov = np.array([v[k:] @ v[:n - k] / n for k in range(m_max + 1)])
    rho = acov[1:] / acov[0]
    thr = 2.0 * np.sqrt(np.log10(n) / n)

    m_hat = m_max
    for m in range(0, m_max - kn + 1):                       # lags m+1 .. m+kn all small
        if np.all(np.abs(rho[m:m + kn]) < thr):
            m_hat = m
            break
    M = int(min(max(2 * m_hat, 1), m_max))
    k = np.arange(-M, M + 1)
    w = _flat_top(k / M)
    R = acov[np.abs(k)]
    G = np.sum(w * np.abs(k) * R)
    g = np.sum(w * R)
    if g <= 0:
        return 1.0
    b = (2.0 * G ** 2 / (2.0 * g ** 2)) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    return float(np.clip(b, 1.0, b_max))


def common_block_length(M: np.ndarray, horizon: int) -> float:
    '''
    One block length for ALL signals (they must be resampled together to keep their correlation).
    The largest Politis-White length over the signals, and at least `horizon` (the label overlap is h - 1 days).
    '''
    per_signal = [politis_white_block_length(M[:, k]) for k in range(M.shape[1])]
    return float(max([float(horizon)] + per_signal))


# =============================================================================
# 3. RESAMPLING
# =============================================================================

def stationary_bootstrap_indices(n: int, mean_block: float, n_boot: int, seed: int = 0) -> np.ndarray:
    '''
    (n_boot, n) array of row indices. Each row is one resample of the n dates:
    start at a random date, go forward one date at a time, and with probability p = 1 / mean_block jump to a new
    random date (the series wraps around at the end). Block lengths are geometric with mean `mean_block`.
    '''
    rng = np.random.default_rng(seed)
    p = 1.0 / max(1.0, float(mean_block))
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=n_boot)
    jump = rng.random((n_boot, n)) < p
    fresh = rng.integers(0, n, size=(n_boot, n))
    for t in range(1, n):
        idx[:, t] = np.where(jump[:, t], fresh[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def batch_means_se(X: np.ndarray, batch: int) -> np.ndarray:
    '''
    Standard error of the mean of each column, by BATCH MEANS: cut the n rows into n // batch consecutive batches,
    average each batch, and use  se = std(batch means) / sqrt(number of batches).
    If a batch is longer than the memory of the series, the batch means are almost independent, so this se is honest.
    X: (..., n, K). Returns (..., K). NaN are skipped inside a batch.
    '''
    n = X.shape[-2]
    nb = max(2, n // batch)
    m = n // nb
    Y = X[..., :nb * m, :].reshape(X.shape[:-2] + (nb, m, X.shape[-1]))
    with np.errstate(invalid='ignore', divide='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)        # a batch with no IC at all (warm-up): NaN, skipped
        bm = np.nanmean(Y, axis=-2)
        return np.nanstd(bm, axis=-2, ddof=1) / np.sqrt(np.sum(np.isfinite(bm), axis=-2))


def default_batch_length(horizon: int, block_length: float) -> int:
    '''Batch length of the standard error: at least 4 label lengths and 2 bootstrap blocks (longer than the memory).'''
    return int(max(4 * horizon, np.ceil(2 * block_length)))


# =============================================================================
# 4. TEST STATISTICS AND P-VALUES
# =============================================================================

def bootstrap_t_stats(M: np.ndarray, idx: np.ndarray, batch: int, chunk: int = 25
                      ) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    '''
    STUDENTIZED ("percentile-t") bootstrap. Returns (mean (K,), se (K,), t (K,), t_star (n_boot, K)), NOT aligned:
        t       = mean / se                      se: batch means on the data
        t_star  = (mean* - mean) / se*           se*: batch means on each resample (recomputed every time)
    t_star is centred on the sample mean: it is the distribution of t when the TRUE mean IC is 0.
    Why recompute se* in every resample: with few independent blocks, the se itself is noisy. A fixed se ignores
    that noise, and the test then says "significant" too often (about 10% instead of 5% in a simulation with
    10 correlated signals and 6 years of data). With se* recomputed, it is 5%.
    '''
    with np.errstate(invalid='ignore'):
        mean = np.nanmean(M, axis=0)
    se = batch_means_se(M, batch)
    with np.errstate(invalid='ignore', divide='ignore'):
        t = mean / np.where(se > 0, se, np.nan)
    t_star = np.empty((idx.shape[0], M.shape[1]))
    for s in range(0, idx.shape[0], chunk):
        X = M[idx[s:s + chunk]]
        with np.errstate(invalid='ignore', divide='ignore'):
            ms = np.nanmean(X, axis=1)
            ses = batch_means_se(X, batch)
            t_star[s:s + chunk] = (ms - mean) / np.where(ses > 0, ses, np.nan)
    return mean, se, t, t_star


def align(t: np.ndarray, t_star: np.ndarray, signs: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    '''
    signs: (K,) +1 / -1 (one-sided in that direction) or 0 (two-sided).
    One-sided: t -> sign * t (large = good in the prior direction). Two-sided: t -> |t|. Same for t_star.
    '''
    two = signs == 0
    s = np.where(two, 1.0, signs)
    return np.where(two, np.abs(t), s * t), np.where(two, np.abs(t_star), s * t_star)


def single_pvalues(t: np.ndarray, t_star: np.ndarray) -> np.ndarray:
    '''One test per signal: p_k = (1 + #{t*_k >= t_k}) / (B + 1). NaN where t is NaN.'''
    B = t_star.shape[0]
    p = (1.0 + np.sum(t_star >= t[np.newaxis, :], axis=0)) / (B + 1.0)
    return np.where(np.isfinite(t), p, np.nan)


def romano_wolf(t: np.ndarray, t_star: np.ndarray) -> np.ndarray:
    '''
    Romano-Wolf step-down adjusted p-values (family-wise error rate).
        sort the signals by t, largest first
        step j: p_j = share of resamples where the MAX of t* over the signals not yet treated is >= t_(j)
        adjusted p_(j) = max(p_(j), adjusted p_(j-1))        (never smaller than the one before)
    Admitting every signal with adjusted p < alpha keeps the chance of ANY false admission below alpha.
    Signals with a NaN t get NaN and do not take part.
    '''
    B, K = t_star.shape
    ok = np.where(np.isfinite(t))[0]
    order = ok[np.argsort(-t[ok])]
    ts = np.where(np.isfinite(t_star), t_star, -np.inf)
    adj = np.full(K, np.nan)
    prev = 0.0
    for j, k in enumerate(order):
        max_star = ts[:, order[j:]].max(axis=1)
        p = (1.0 + np.sum(max_star >= t[k])) / (B + 1.0)
        prev = max(prev, p)
        adj[k] = prev
    return adj


# =============================================================================
# 5. THE WRAPPER USED BY STAGE 05
# =============================================================================

def bootstrap_test_table(ic_dated: dict, signs: dict, horizon: int, n_boot: int = 1000, seed: int = 0) -> dict:
    '''
    ic_dated: {signal: (dates, IC)}.  signs: {signal: +1 / -1 / None}  (None -> two-sided).
    Returns {'block_length': b, 'batch_length': m, 'n_boot': B, 'n_dates': n,
             'signals': {signal: {mean_ic, se, t, p_single, p_rw, sided}}}.
    '''
    dates, names, M = align_ic_series(ic_dated)
    b = common_block_length(M, horizon)
    batch = default_batch_length(horizon, b)
    idx = stationary_bootstrap_indices(len(dates), b, n_boot, seed)
    sg = np.array([float(signs.get(k) or 0) for k in names])
    mean, se, t_raw, t_star_raw = bootstrap_t_stats(M, idx, batch)
    t, t_star = align(t_raw, t_star_raw, sg)
    p1 = single_pvalues(t, t_star)
    prw = romano_wolf(t, t_star)
    out = {}
    for i, k in enumerate(names):
        out[k] = {'mean_ic': float(mean[i]), 'se': float(se[i]), 't': float(t[i]),
                  'p_single': float(p1[i]), 'p_rw': float(prw[i]),
                  'sided': 'two-sided' if sg[i] == 0 else ('one-sided +' if sg[i] > 0 else 'one-sided -')}
    return {'block_length': b, 'batch_length': batch, 'n_boot': int(n_boot), 'n_dates': int(len(dates)), 'signals': out}
