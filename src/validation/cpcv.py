'''
Added after wakl_forward_cv.py because realised it is much more used in practice,
and will yield more accurate results statistcally.

Two related but distinct pieces, kept separate on purpose (see the section
headers below for why they're not the same function):

1. purged_embargoed_kfold_splits() / cpcv_splits()
   The actual leakage-safe TRAIN/TEST row-index generator for TRUE K-fold
   and Combinatorial Purged CV. Unlike src/validation/walk_forward_cv.py's
   walk-forward scheme, train blocks here can sit BOTH before and after a
   given test block chronologically, which is what makes true k-fold
   different from walk-forward, and why embargo becomes a within-fold
   necessity here (protecting THIS fold's own validity), not just a
   cross-fold aggregation nicety like it was in the walk-forward file.
   Combinatorial purged CV will take all combinations of K blocks among N
   as test sets. The whole rest will be the train set every time (accounting
   for purge and embargo). We then have multiple OOS estimates of any stat for
   each time block. We can then aggregate as we wish, e.g. using PBO.

2. probability_of_backtest_overfitting()
   The CSCV/PBO statistic from Bailey, Borwein, Lopez de Prado & Zhu
   (2017), operating on an already-computed performance matrix (one
   column per CANDIDATE STRATEGY/config you are choosing between, one row
   per sub-period). This does NOT need (1) as an input, it's a distinct,
   simpler combinatorial split over a performance matrix, not over raw
   dates/features. You'll typically produce the performance matrix BY
   running several candidate configs through splits built with (1) (or
   with the walk-forward splits, either works), then hand the resulting
   matrix to this function.
   The idea is to get the OOS rank of the best in-sample performing strat, 
   for each train-test split. We then average the ranks and possibly transform 
   into logits to get the PBO.

NOT IMPLEMENTED HERE: AFML's full CPCV "path reconstruction" (stitching
per-combination OOS slices into phi = C(N,k)*k/N complete alternate
history-length paths, ch. 12). That's a real combinatorial-assignment
problem in its own right, and PBO (the thing actually being asked for)
doesn't need it, Bailey et al.'s original CSCV formulation works directly
off per-combination OOS performance values, no path stitching required.
Worth building later specifically if you want smooth full-history OOS
equity curves for a report figure, not needed for the PBO number itself.

References:
    Lopez de Prado, M. (2018). Advances in Financial Machine Learning. 
        Wiley, Ch. 7 (purged k-fold), Ch. 12 (CPCV, full path version).
    Bailey, D. H., Borwein, J. M., Lopez de Prado, M., & Zhu, Q. J.
        (2017). The probability of backtest overfitting. Journal of
        Computational Finance, 20(4), 39-69. (CSCV / PBO algorithm used
        directly below, this is the primary source for Section 2, not
        just a secondary citation of the AFML textbook's summary of it.)
'''

from itertools import combinations
from collections import namedtuple

import numpy as np




# =============================================================================
# SECTION 1: true purged-embargoed K-fold / CPCV split generator
# =============================================================================



BlockTuple = namedtuple('Block', ['start_idx', 'end_idx'])  # inclusive date-index bounds of train/test blocks

def get_raw_train_test_blocks(n_dates: int, n_blocks: int) -> list[BlockTuple]:
    '''
    Partition n_dates indices into n_blocks of roughly-equal size. 
    Sizes differ by at most 1 date when n_dates doesn't divide evenly, remainder distributed to the earliest groups.

    returns: a list of namedtuples with the start and end indices, representing the train/test groups.
    '''

    base, remainder = divmod(n_dates, n_blocks)
    blocks = []
    start = 0

    for i in range(n_blocks):
        size = base + (1 if i < remainder else 0)  # add 1 to the first blocks to account for the remainder
        blocks.append(BlockTuple(start, start + size - 1))  # don't we need to give a name to each namedtuple ? -> No the name is Block
        start += size
    return blocks




def _merge_blocks(block_ids: tuple[int, ...]) -> list[tuple[int, int]]:
    '''
    Certain combinations of K test blocks in N total blocks have some consecutive
        test blocks. In this case we do not need to remove the purge and embargo windows 
        for all test sets. 
    _merge_blocks identifies the start and end of consecutive test blocks.

    block_ids: IDs of test blocks

    returns: list of 2-tuples giving the start and end of consecutive blocks (the same index twice if a block is alone)
    '''

    sorted_ids = sorted(block_ids)
    runs = []
    run_start = run_end = sorted_ids[0]

    for block_id in sorted_ids[1:]:
        if block_id == run_end + 1:
            run_end = block_id
        else:
            runs.append((run_start, run_end))
            run_start = run_end = block_id
    runs.append((run_start, run_end))
    return runs




def cpcv(n_dates: int, n_blocks: int, n_test_blocks: int, purge_w: int, embargo_w: int):
    '''
    Generates every C(n_groups, n_test_groups) combination of test groups, 
        accounting for purge and embargo windows around consecutive test blocks.
    For each combination:
        test_idx  = every date index inside a selected test group.
        train_idx = every date index inside a non-selected group, MINUS
            a purge slice of length `purge_w` immediately before each
            contiguous run of selected test groups (labels there could
            extend forward into the test run), MINUS an embargo slice of
            length `embargo_w` immediately after each run (serial
            correlation proximity to the test run, see module docstring
            and the Section 1 note above on why this now matters for a
            SINGLE fold's validity, unlike in walk-forward).

    n_test_groups=1 is exactly "true purged-embargoed k-fold CV", not combinatorial. 
    n_test_groups>1 is CPCV, more combinations, and importantly, WIDER test blocks per combination
        (k contiguous-ish groups' worth), which is the actual lever CPCV
        uses to get more effective train/test evaluations without shrinking
        any individual test window down to something too short to be meaningful.

    Yields (train_idx, test_idx, test_group_ids) as numpy int arrays, 
        plus the raw tuple of which groups were held out, useful for building the
        T x N performance matrix for PBO later on.
        (T = one row per group- or combination-level result).
    '''

    raw_blocks = get_raw_train_test_blocks(n_dates, n_blocks)  # a list of namedtuples with start_idx and end_idx 
    all_ids = list(range(n_blocks))

    for test_ids in combinations(all_ids, n_test_blocks):  # generate all combinations of n_test_blocks items of all_ids 
        test_idx_parts = [np.arange(raw_blocks[id].start_idx, raw_blocks[id].end_idx + 1) for id in test_ids]  # the full test set indices
        test_idx = np.concatenate(test_idx_parts)

        excluded = set()  # date indices to remove from train beyond the test groups themselves
        for run_start, run_end in _merge_blocks(test_ids):
            run_start_date = raw_blocks[run_start].start_idx
            run_end_date = raw_blocks[run_end].end_idx

            purge_lo = max(0, run_start_date - purge_w)  # purge cannot be before we the first block
            excluded.update(range(purge_lo, run_start_date))

            embargo_hi = min(n_dates - 1, run_end_date + embargo_w)  #  embargo cannot be after the last block
            excluded.update(range(run_end_date + 1, embargo_hi + 1))

        # we remove the purge and embargo windows from the train sets, not test sets.
        train_idx = np.array([
            i for i in range(n_dates) if i not in set(test_idx.tolist()) and i not in excluded
        ])  

        yield train_idx, test_idx, test_ids






# =============================================================================
# SECTION 2: PBO via CSCV (Bailey, Borwein, Lopez de Prado & Zhu, 2017)
# =============================================================================

def probability_of_backtest_overfitting(performance_matrix: np.ndarray, n_splits: int, metric: str = 'sharpe') -> dict:
    '''
    PBO is computed by comparing the bset performing strat in sample to its rank OOS.
        The OOS rank is put in (0,1) and transformed into logits.

    performance_matrix: shape (T, N). 
        T = sub-periods (NOT necessarily the same object as number of blocks previously, though you can build
        them the same way, one row per contiguous time block). 
        N = candidate strategies/configs you are choosing between (different
        alpha/l1_ratio settings, different signal sets, Elastic Net vs
        GBM later, etc). Each entry is that candidate's realized return
        (or whatever the base series is) during that sub-period. 
    This function does not care how the matrix was produced, it operates
        purely on the T x N array, per Bailey et al.'s original CSCV
        formulation, no reference back to dates/features/purge/embargo
        at this stage, that all already happened when the matrix was
        built.
    n_splits: S in the paper, T must be evenly divisible by S for the
        even-split-in-half combinatorics below (T // S rows per block).
        S is a real, fixed convention to pick before looking at results,
        not tuned to whichever S makes PBO look better or worse, same
        principle as everywhere else in this project.
    metric: 'sharpe' (mean/std of the sub-period returns making up an
        IS or OOS half) is the standard choice in the paper. Any monotone
        performance statistic works, the algorithm itself doesn't care,
        only the ranking of candidates under it matters.

    Returns dict with 'pbo' (the headline number, in [0, 1], probability
    the in-sample-best candidate underperforms the OOS median) and
    'logits' (raw per-combination logit values, useful for sanity-checking
    the distribution isn't secretly bimodal/degenerate before trusting a
    single summary number, this is the kind of thing worth eyeballing on
    a real run, not something the function should hide).
    '''
    T, N = performance_matrix.shape
    if T % n_splits != 0:
        raise ValueError(
            f'T={T} not evenly divisible by n_splits={n_splits}, '
            'CSCV as specified needs equal-sized blocks.'
        )
    block_size = T // n_splits
    blocks = [performance_matrix[i * block_size:(i + 1) * block_size] for i in range(n_splits)]

    def sharpe(returns_2d: np.ndarray) -> np.ndarray:
        # returns_2d: (rows, N), returns one Sharpe per candidate column.
        # No annualization factor applied, PBO only needs the RANKING of
        # candidates under this statistic, not its absolute scale.
        mu = returns_2d.mean(axis=0)
        sigma = returns_2d.std(axis=0, ddof=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(sigma > 0, mu / sigma, 0.0)

    if metric != 'sharpe':
        raise NotImplementedError("only 'sharpe' wired up, swap the sharpe() closure above for another monotone stat if needed")

    half = n_splits // 2
    if n_splits % 2 != 0:
        raise ValueError('CSCV as specified in Bailey et al. splits S into exactly two equal halves, n_splits must be even.')

    logits = []
    for is_block_ids in combinations(range(n_splits), half):
        oos_block_ids = tuple(i for i in range(n_splits) if i not in is_block_ids)

        is_returns = np.concatenate([blocks[i] for i in is_block_ids], axis=0)
        oos_returns = np.concatenate([blocks[i] for i in oos_block_ids], axis=0)

        is_perf = sharpe(is_returns)
        oos_perf = sharpe(oos_returns)

        n_star = np.argmax(is_perf)  # the in-sample "winner" for this split

        # relative rank of n_star's OOS performance among ALL N candidates'
        # OOS performance, 1-indexed, per the paper's own convention
        oos_rank = 1 + np.sum(oos_perf < oos_perf[n_star])
        omega = oos_rank / (N + 1)  # relative rank in (0, 1)

        # logit <= 0 means the in-sample winner fell at or below the OOS
        # median, exactly the overfitting signature the statistic is
        # built to catch
        logit = np.log(omega / (1 - omega))
        logits.append(logit)

    logits = np.array(logits)
    pbo = float(np.mean(logits <= 0))

    return {'pbo': pbo, 'logits': logits, 'n_combinations': len(logits)}
