from collections.abc import Iterable
import polars as pl
from polars import col as c
import subprocess
import hashlib
import json
import os
from datetime import date, timedelta


def as_list(x):
    if isinstance(x, (str, float, int)):
        return [x]
    elif isinstance(x, Iterable):
        return list(x)
    else:
        raise TypeError("features must be str or iterable")


def add_expr(lf, expr_func, colname, suffix=None, **kwargs):
    colname = as_list(colname)
    return lf.with_columns(
        [
            expr_func(c(cn), **kwargs).alias(f'{cn}{suffix}')
            for cn in colname
        ]
    )


def get_rebalance_dates(start_date: date, end_date: date, freq: str = 'weekly') -> list:
    '''Generate rebalance dates between start and end.
    
    Args:
        start_date: First rebalance date
        end_date: Last rebalance date (inclusive)
        freq: 'weekly' or 'monthly'
    
    Returns:
        List of date objects
    '''
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        if freq == 'weekly':
            current += timedelta(weeks=1)
        elif freq == 'monthly':
            # Approximate month step
            current = (current.replace(day=1) + timedelta(days=32)).replace(day=1)
        else:
            raise ValueError(f'Unknown freq: {freq}')
    return dates


# =============================================================================
# Provenance / reproducibility helpers
# =============================================================================

def get_git_commit(cwd='.'):
    '''Return current git commit hash, or None if not a git repo.'''
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd, text=True).strip()
    except Exception:
        return None


def get_git_diff(cwd='.'):
    '''Return git diff --shortstat, or None if clean/not a git repo.'''
    try:
        diff = subprocess.check_output(['git', 'diff', '--shortstat'], cwd=cwd, text=True).strip()
        return diff if diff else None
    except Exception:
        return None


def get_git_status(cwd='.'):
    '''Return dict with commit, diff, and dirty flag.'''
    commit = get_git_commit(cwd)
    diff = get_git_diff(cwd)
    return {
        'commit': commit,
        'diff': diff,
        'dirty': diff is not None and diff != '',
    }


def file_hash(path, algo='sha256', length=16):
    '''Return hash of file contents, e.g., "sha256:abc123...".'''
    try:
        h = hashlib.new(algo)
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return f'{algo}:{h.hexdigest()[:length]}'
    except Exception:
        return None


def params_hash(params, algo='sha256', length=16):
    '''Return hash of JSON-serializable params.'''
    try:
        h = hashlib.new(algo)
        h.update(json.dumps(params, sort_keys=True, default=str).encode())
        return f'{algo}:{h.hexdigest()[:length]}'
    except Exception:
        return None

