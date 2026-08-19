



from collections import namedtuple




Fold = namedtuple('Fold', ['train_start', 'train_end', 'test_start', 'test_end'])


def rolling_purged_embargoed_splits(dates, train_window, test_window, step, horizon, embargo):
    '''
    Rolling, purged, embargoed date-range splits. Operates on the unique
        sorted date list only, callers filter their own lazyframe by the
        returned date boundaries.

    Precondition: step >= test_window + embargo. This guarantees embargo
        can never be violated between folds, by construction, not by
        checking and skipping at runtime. If this raises, it's the wrong
        parameter combination for this train_window, not something to
        silently work around.

    Yields Fold namedtuples, unpack directly:
        for train_start, train_end, test_start, test_end in rolling_purged_embargoed_splits(...):
    '''
    if step < test_window + embargo:
        raise ValueError(
            f'step ({step}) must be >= test_window + embargo '
            f'({test_window} + {embargo} = {test_window + embargo})'
        )

    n = len(dates)
    test_start_idx = train_window + horizon

    while test_start_idx + test_window <= n:
        train_end_idx = test_start_idx - horizon
        train_start_idx = train_end_idx - train_window
        test_end_idx = test_start_idx + test_window

        if train_start_idx >= 0:
            yield Fold(
                train_start=dates[train_start_idx],
                train_end=dates[train_end_idx - 1],
                test_start=dates[test_start_idx],
                test_end=dates[test_end_idx - 1],
            )

        test_start_idx += step