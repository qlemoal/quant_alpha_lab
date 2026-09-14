'''
Following Lopez de Prado's recommentdations, we do not use plain cross-validation, but rolling train/test-sets, 
    with a purge window between them, and an embargo window between a test set and the next train set. 
    This should prevent data leakage and data snooping.
Embargo may seem useless here since no data after a test set is used, we only use the data before a test set as a train set.
    However, we average the results over multiple test sets, so to avoid dependence between the test sets, 
    we add an embargo window after a test, so that test sets have a buffer between them.

There is the package mlfinlab from Lopez de Prado, maybe worth a look at some point.

Unpack sets using "for train_start, train_end, test_start, test_end in rolling_purged_embargoed_splits(...):"

Usually:
    train_window = 756
    test_window = 63
    max_horizon = purge_window = 20
    embargo = 16
    step = test_window + embargo = 83
'''


from collections import namedtuple



Fold = namedtuple('Fold', ['train_start', 'train_end', 'test_start', 'test_end'])


def walk_forward_cv(dates, train_window, purge, test_window, embargo, next_fold='consecutive'):
    '''
    Non-overlapping train -> purge -> test -> embargo cycles, tiled backward
        from the most recent date so the newest fold uses the latest data
        available, any leftover history that doesn't fill a full cycle gets
        dropped at the oldest end, not the newest.
        Reference for purge and embargo
        both: López de Prado (2018), Advances in Financial Machine
        Learning, Ch. 7.

    train_window: length of the training block. Rolling-window CV convention
        (see docs/methodology.md), rolling rather than expanding specifically
        so the documented survivorship-biased pre-2003 history eventually
        ages out of training.
    purge_window: forward-return span of the label actually being trained
        against this round, not an independent choice, purge is fully
        determined by it (max(horizons) if training against several at once).
    test_window: length of the held-out evaluation block.
    embargo: gap enforced after a fold's test block, before the next (more
        recent) fold's train block resumes (or test block if next_fold != consecutive). Dampens correlation between
        different folds' out-of-sample results when aggregating across
        folds, doesn't make any single fold more causally valid, purge
        alone already guarantees that. 
        We set it by default to config.constants.EMBARGO_WINDOW, 
        a too big embargo will make the number of folds too small.
    next_fold = 'consecutive' or an int. 
        If consecutive, no data is used by multiple sets --> very conservative.
        If an int, next train_set starts after `next_fold` dates after the previous.
            The obvious choice is to set it to test_window+embargo. Note that embargo is useless in that case.


    Yields Fold namedtuples in chronological order, oldest first. 
        Use with "for train_start, train_end, test_start, test_end in walk_forward_cv(...):"
    '''

    n = len(dates)
    folds = []
    test_end_idx = n - 1

    while True:
        test_start_idx = test_end_idx - test_window + 1
        train_end_idx = test_start_idx - purge - 1
        train_start_idx = train_end_idx - train_window + 1

        if train_start_idx < 0 or train_end_idx < 0 or test_start_idx < 0:
            break

        folds.append(Fold(
            train_start = dates[train_start_idx],
            train_end = dates[train_end_idx],
            test_start =  dates[test_start_idx],
            test_end = dates[test_end_idx],
        ))

        if next_fold == 'consecutive':
            test_end_idx = train_start_idx - embargo - 1
        elif isinstance(next_fold, int):
            test_end_idx = test_end_idx - next_fold
        else: raise ValueError

    yield from reversed(folds)






if __name__ == '__main__':
    # from datetime import date, timedelta
    import polars as pl
    from config.constants import EMBARGO_WINDOW

    lf = pl.scan_parquet('data/processed/features.parquet').sort(['ticker', 'date'])

    dates = lf.select(
        pl.col('date').unique()
    ).collect().to_numpy()[:, 0]

    print('n dates:', len(dates), 'min_date:', min(dates), 'max_date:', max(dates))
    for train_start, train_end, test_start, test_end in walk_forward_cv( dates, train_window=500, test_window=10, 
                                                                         purge=100, embargo=EMBARGO_WINDOW, next_fold='consecutive' ):
        print('Train from', train_start, 'to', train_end, '  -   Test from', test_start, 'to', test_end)