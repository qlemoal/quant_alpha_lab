'''
Musee des horreurs: a small zoo of deliberately awkward panels, reused across tests for any feature/signal function. 
    Each one exercises a specific way real data (or a real bug) can break an otherwise-correct implementation. 
    New feature/signal functions should get tested against the relevant subset of these before being trusted on the full dataset.

Note: it seems the file must be named conftest.py for pytest to recognize it.
'''

import polars as pl
import pytest


@pytest.fixture
def panel_normal():
    '''4 tickers, one date, distinct values. The boring baseline.'''
    return pl.DataFrame({
        'ticker': ['A', 'B', 'C', 'D'],
        'date':   [1, 1, 1, 1],
        'feat':   [10.0, 20.0, 30.0, 40.0],
    })


@pytest.fixture
def panel_normal_multidate():
    '''4 tickers, 3 dates, clean and sorted. The real baseline, most functions should be tested against this before anything trickier.'''
    return pl.DataFrame({
        'ticker': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'D', 'D', 'D'],
        'date':   [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
        'feat':   [10.0, 11.0, 9.0, 20.0, 22.0, 19.0, 30.0, 33.0, 28.0, 40.0, 44.0, 38.0],
    })


@pytest.fixture
def panel_unsorted():
    '''Same data as panel_normal_multidate, rows shuffled. 
    Anything that silently relies on row order without an explicit .sort() first should fail against this, this is what the mom20/std252 sorting bug looked like from the inside.'''
    return pl.DataFrame({
        'ticker': ['C', 'A', 'D', 'B', 'A', 'C', 'B', 'D', 'A', 'B', 'C', 'D'],
        'date':   [1, 3, 2, 2, 1, 3, 1, 1, 2, 3, 2, 3],
        'feat':   [30.0, 10.0, 44.0, 20.0, 11.0, 33.0, 20.0, 40.0, 9.0, 19.0, 28.0, 38.0],
    })


@pytest.fixture
def panel_ragged():
    '''Tickers not present on every date, e.g. C only exists from date 2 nward (IPO), D disappears after date 2 (delisting). 
    Realistic, and stresses anything using shift()/.over('ticker') or assuming every date has the same ticker set.'''
    return pl.DataFrame({
        'ticker': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'D', 'D'],
        'date':   [1, 2, 3, 1, 2, 3, 2, 3, 1, 2],
        'feat':   [10.0, 11.0, 9.0, 20.0, 22.0, 19.0, 33.0, 28.0, 40.0, 44.0],
    })


@pytest.fixture
def panel_single_ticker():
    '''Group of size 1. std() is degenerate, rank has nothing to rank against.'''
    return pl.DataFrame({
        'ticker': ['A'],
        'date':   [1],
        'feat':   [10.0],
    })


@pytest.fixture
def panel_single_ticker_multidate():
    '''One ticker, several dates. No real cross-section exists on any date, tests that per-date operations degrade sensibly rather than crashing, 
    and that rolling/positional operations still work fine even when the cross-sectional ones don't have much to work with.'''
    return pl.DataFrame({
        'ticker': ['A', 'A', 'A', 'A'],
        'date':   [1, 2, 3, 4],
        'feat':   [10.0, 12.0, 9.0, 15.0],
    })


@pytest.fixture
def panel_identical_values():
    '''All tickers share the same value on this date. std = 0, every rank tied.'''
    return pl.DataFrame({
        'ticker': ['A', 'B', 'C', 'D'],
        'date':   [1, 1, 1, 1],
        'feat':   [5.0, 5.0, 5.0, 5.0],
    })


@pytest.fixture
def panel_with_nulls():
    '''Some tickers missing the feature on this date (e.g. not enough history yet for a rolling window).'''
    return pl.DataFrame({
        'ticker': ['A', 'B', 'C', 'D'],
        'date':   [1, 1, 1, 1],
        'feat':   [10.0, None, 30.0, None],
    })


@pytest.fixture
def panel_uneven_group():
    '''7 tickers, doesn't divide evenly into 10 deciles.'''
    return pl.DataFrame({
        'ticker': list('ABCDEFG'),
        'date':   [1] * 7,
        'feat':   [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    })


@pytest.fixture
def panel_with_ties():
    '''Several tickers sharing the exact same value on the same date, what tripped up qcut with allow_duplicates=False.'''
    return pl.DataFrame({
        'ticker': ['A', 'B', 'C', 'D', 'E', 'F'],
        'date':   [1, 1, 1, 1, 1, 1],
        'feat':   [1.0, 1.0, 1.0, 2.0, 2.0, 3.0],
    })




##  TEST OF P-VALUES

@pytest.fixture
def pvals_all_significant():
    '''Everything genuinely significant, should survive both BH and BY.'''
    return [0.0001, 0.0005, 0.001, 0.002, 0.003]

@pytest.fixture
def pvals_all_null():
    '''Nothing significant, nothing should survive either method.'''
    return [0.4, 0.5, 0.6, 0.7, 0.9]

@pytest.fixture
def pvals_single_candidate():
    '''m=1, BH and BY should agree exactly (harmonic number H_1 = 1).'''
    return [0.01]

@pytest.fixture
def pvals_identical():
    '''All tests share the same p-value, exercises tie-handling in sort/threshold comparison.'''
    return [0.03, 0.03, 0.03, 0.03]

@pytest.fixture
def pvals_at_threshold_boundary():
    '''Values sitting close to typical BH/BY thresholds at alpha=0.05, m=5, worth checking against hand-computed thresholds.'''
    return [0.008, 0.019, 0.031, 0.044, 0.09]

@pytest.fixture
def pvals_mixed_with_nan():
    '''One candidate's t-stat failed to compute (e.g. degenerate variance), should not corrupt sorting/threshold logic for the rest.'''
    return [0.001, 0.02, float('nan'), 0.15, 0.4]

@pytest.fixture
def pvals_empty():
    '''No candidates tested this round, degenerate but should not crash.'''
    return []

@pytest.fixture
def pvals_bh_by_disagree():
    '''Engineered so BH survives more than BY, illustrates why reporting both matters instead of picking one.'''
    return [0.004, 0.009, 0.014, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3]