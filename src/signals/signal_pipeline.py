'''
Plumbing for stage 04a (signal generation). The script 04a_generate_signals.py only calls these functions.
The transforms themselves (z-score, tanh, rank...) are in src/signals/combine.py and its helpers.

WHAT STAGE 04a DOES, AND WHAT IT NEVER DOES
-------------------------------------------

    features.parquet  (one row per date and ticker: mom20, vol60, log_adv60, open_adj, ...)
          |
          |  [1] clean_features      NaN and inf -> null  (one NaN would spoil the mean of a whole date)
          |  [2] build_signals       one transform per spec, ALWAYS across the stocks of the same date
          |  [3] add_labels          fwdret{h} = log(open_adj[t+1+h]) - log(open_adj[t+1])
          v
    signals.parquet   [date, ticker, <signals>, fwdret1, fwdret5, ...]

    04a never measures how well a signal predicts returns. No IC, no Sharpe, no drawdown.
    That is the job of 04b, which uses the search period only. If 04a printed IC on all dates, the holdout
    would leak into the choice of signals through the eyes of the researcher.
    04a only checks that the signals and the labels are well formed (coverage, scale, label alignment).

THE CLOCK
---------
    close of t        open of t+1          open of t+1+h
    features known -> trade (entry)  ...   exit.      label of row t = the return between the two opens.
    A signal on row t uses data up to the close of t. The label uses prices after t. No overlap.
'''

import numpy as np
import polars as pl

from src.features.returns import add_fwdret_horizon
from src.signals.combine import make_signal

SIGNAL_METHODS = ('zscore_tanh', 'zscore', 'zscore_clip', 'rank', 'decile')


# =============================================================================
# 1. SPECS, CLEANING, SIGNALS, LABELS
# =============================================================================

def validate_specs(specs: list, available_cols: set) -> "tuple[list, list]":
    '''
    Split the signal specs into (usable, skipped).

    spec = (feature_col, method, params). A spec is skipped, with a reason, if its feature is not in the
    features file. An unknown method, or two specs that give the same signal name, is an error:
    these are mistakes in the script, not in the data.
    Returns (usable specs, list of {'spec': ..., 'reason': ...}).
    '''
    usable, skipped, seen = [], [], set()
    for feat, method, params in specs:
        if method not in SIGNAL_METHODS:
            raise ValueError(f'unknown method {method!r} for {feat}. Use one of {SIGNAL_METHODS}.')
        name = f'{feat}_{method}'
        if name in seen:
            raise ValueError(f'signal {name} is defined twice')
        seen.add(name)
        if feat not in available_cols:
            skipped.append({'spec': [feat, method, params], 'reason': 'feature not in features file'})
            continue
        usable.append((feat, method, params))
    return usable, skipped


def clean_features(lf: pl.LazyFrame, feature_cols: list) -> pl.LazyFrame:
    '''
    NaN and +-inf become null in the feature columns that are used.
    Why: a null is skipped by the mean and the std of a date, a NaN is not. One NaN feature would turn the
    cross-sectional mean of the whole date into NaN, and every signal of that date with it.
    '''
    return lf.with_columns([
        pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c) for c in feature_cols
    ])


def build_signals(lf: pl.LazyFrame, specs: list) -> "tuple[pl.LazyFrame, list]":
    '''
    Add one signal column per spec (name = '{feature}_{method}'). Lazy: nothing is computed yet.
    All transforms normalise across the stocks of one date (see src/signals/normalize.py and ranking.py).
    Returns (lf, signal column names).
    '''
    names = []
    for feat, method, params in specs:
        lf = make_signal(lf, cols=feat, method=method, **params)
        names.append(f'{feat}_{method}')
    return lf, names


def add_labels(lf: pl.LazyFrame, horizons: list) -> "tuple[pl.LazyFrame, list]":
    '''Add fwdret{h} for every horizon. Open-to-open, per ticker (see src/features/returns.py).'''
    for h in horizons:
        lf = add_fwdret_horizon(lf, horizon=h)
    return lf, [f'fwdret{h}' for h in horizons]


def restrict_dates(lf: pl.LazyFrame, start=None, end=None) -> pl.LazyFrame:
    '''
    Keep start <= date <= end (either bound may be None).
    The labels are computed AFTER this cut by the script, so the last rows before `end` have no label.
    That is correct: a label must not use prices after the cut.
    '''
    if start is not None:
        lf = lf.filter(pl.col('date') >= pl.lit(start).str.to_date('%Y-%m-%d'))
    if end is not None:
        lf = lf.filter(pl.col('date') <= pl.lit(end).str.to_date('%Y-%m-%d'))
    return lf


# =============================================================================
# 2. CHECKS (they raise on a broken file, and warn on a suspicious one)
# =============================================================================

def check_labels(df: pl.DataFrame, horizons: list) -> dict:
    '''
    Check the forward returns. Raises ValueError if they are misaligned. Returns a dict of numbers.

    1. TAIL RULE: for each ticker, the last h+1 rows must have fwdret{h} = null (the prices do not exist yet).
       A non-null value there means the label was built across the boundary of two tickers.
    2. SIZE: a 20-day log-return beyond +-2 (a x7 move) is not a normal stock. It points to a split that
       is not adjusted, or to a mix of two stocks. (Counted and reported, not an error.)
    '''
    report = {}
    ordered = df.sort(['ticker', 'date'])
    for h in horizons:
        col = f'fwdret{h}'
        # position of each row counted from the END of its ticker: 1 = last row
        tail = (
            ordered.select(['ticker', 'date', col])
            .with_columns(pl.col('date').rank('ordinal', descending=True).over('ticker').alias('_from_end'))
            .filter(pl.col('_from_end') <= h + 1)
        )
        n_bad_tail = int(tail[col].is_not_null().sum())
        if n_bad_tail > 0:
            raise ValueError(
                f'{col}: {n_bad_tail} rows in the last {h + 1} rows of a ticker have a label. '
                f'The label crosses a ticker boundary. Check src/features/returns.py.'
            )
        vals = df[col].drop_nulls().to_numpy()
        report[col] = {
            'null_pct': float(df[col].null_count() / df.height * 100),
            'std': float(np.std(vals)),
            'max_abs': float(np.max(np.abs(vals))) if len(vals) else None,
            'n_abs_above_2': int((np.abs(vals) > 2).sum()),
        }
    return report


def calendar_gaps(lf: pl.LazyFrame) -> dict:
    '''
    How many dates are missing inside the life of the tickers?
    A label uses row shifts (t+1, t+1+h), which equal trading days only if every ticker has a row for every
    trading day between its first and last date. Returns the share of missing rows and the worst ticker.
    '''
    per_ticker = (
        lf.select(['date', 'ticker']).group_by('ticker')
        .agg(pl.len().alias('n'), pl.col('date').min().alias('first'), pl.col('date').max().alias('last'))
        .collect()
    )
    date_idx = (
        lf.select(pl.col('date').unique().sort()).collect()
        .with_row_index('i')
    )
    expected = (
        per_ticker
        .join(date_idx.rename({'i': 'i_first'}), left_on='first', right_on='date')
        .join(date_idx.rename({'i': 'i_last'}), left_on='last', right_on='date')
        .with_columns((pl.col('i_last') - pl.col('i_first') + 1).alias('expected'))
        .with_columns((pl.col('expected') - pl.col('n')).alias('missing'))
    )
    total_missing = int(expected['missing'].sum())
    worst = expected.sort('missing', descending=True).head(1)
    return {
        'missing_rows': total_missing,
        'missing_share': float(total_missing / max(int(expected['expected'].sum()), 1)),
        'worst_ticker': worst['ticker'][0] if worst.height else None,
        'worst_ticker_missing': int(worst['missing'][0]) if worst.height else 0,
    }


# =============================================================================
# 3. DESCRIPTIVE STATISTICS OF THE SIGNALS (no returns used)
# =============================================================================

def signal_descriptives(df: pl.DataFrame, signal_cols: list) -> dict:
    '''
    For every signal: coverage and scale, with and without the cross-section.
        null_pct          share of rows without a value
        mean, std, min, max   over all rows
        avg_names_per_date    average number of stocks with a value on a date
        avg_cs_mean, avg_cs_std   average over dates of the cross-sectional mean and std
                                  (a z-score based signal: about 0 and about 1, before tanh or clipping)
    Returns {signal: {...}}. No return is used.
    '''
    if not signal_cols:
        return {}
    per_date = df.group_by('date').agg(
        [pl.col(s).mean().alias(f'{s}__mean') for s in signal_cols]
        + [pl.col(s).std().alias(f'{s}__std') for s in signal_cols]
        + [pl.col(s).count().alias(f'{s}__n') for s in signal_cols]
    )
    out = {}
    for s in signal_cols:
        col = df[s]
        out[s] = {
            'null_pct': float(col.null_count() / df.height * 100),
            'mean': _f(col.mean()), 'std': _f(col.std()), 'min': _f(col.min()), 'max': _f(col.max()),
            'avg_names_per_date': _f(per_date[f'{s}__n'].mean()),
            'avg_cs_mean': _f(per_date[f'{s}__mean'].mean()),
            'avg_cs_std': _f(per_date[f'{s}__std'].mean()),
        }
    return out


def _f(x):
    '''Python float, or None if the value is missing or not finite.'''
    return float(x) if x is not None and np.isfinite(x) else None


def signal_warnings(stats: dict, max_null_pct: float = 50.0) -> list:
    '''
    Plain-language warnings about signals that will not be usable:
    all null, constant, or mostly null. A warning is not an error: 04b decides what to keep.
    '''
    warns = []
    for s, st in stats.items():
        if st['null_pct'] >= 100:
            warns.append(f'{s}: all values are null')
        elif st['std'] is None or st['std'] == 0:
            warns.append(f'{s}: constant (std = 0)')
        elif st['null_pct'] > max_null_pct:
            warns.append(f'{s}: {st["null_pct"]:.0f}% null')
    return warns