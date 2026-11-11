'''
Plumbing for stage 04d (risk model). The script 04d_estimate_risk_model.py only calls these functions.
The maths is NOT here: it lives in src/risk/risk_model.py (one date at a time).

WHAT THIS FILE DOES
-------------------

    features.parquet (long: date, ticker, logret)
            |
            |  make_returns_panel          pivot to a wide table: rows = dates, columns = tickers
            v
    ReturnsPanel  (dates, tickers, values[n_dates, n_tickers], NaN = no return that day)
            |
            |  pick_rebalance_dates        e.g. the last trading day of each week
            v
    rebalance dates
            |
            |  plan_rebalances             for each date: which rows (window) and which tickers (eligible)
            v
    plans  (small: only indices, no data copied)
            |
            |  run_risk_model              for each plan, in parallel:
            |        window matrix  ->  risk_model.estimate_risk_model_for_date  ->  save files
            v
    per-date summaries  (small dicts)

WINDOW FOR ONE REBALANCE DATE t  (no look-ahead)
------------------------------------------------

    trading days:   ... | t-251 | ...... | t-1 | t |  t+1 | ...
                        |<---- lookback_window rows ---->|
                        returns known at the CLOSE of t   |  not used

    A ticker is ELIGIBLE at t if
      (a) it has a return on day t itself   (it trades, so it can be held), and
      (b) it has at least `min_obs` finite returns inside the window.
    Tickers that are not eligible are simply not in that date's matrices. No fake zero rows.
'''

import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from tqdm import tqdm

from src.risk.risk_model import estimate_risk_model_for_date
from src.utils.risk_io import save_long_matrix, save_clusters, save_eigenvalues


# =============================================================================
# 1. THE RETURNS PANEL
# =============================================================================

@dataclass
class ReturnsPanel:
    '''Wide table of daily log returns.'''
    dates: np.ndarray      # (n_dates,) datetime64[D], sorted
    tickers: np.ndarray    # (n_tickers,) str, sorted
    values: np.ndarray     # (n_dates, n_tickers) float, NaN = missing


def make_returns_panel(returns_long: pl.DataFrame, date_col: str = 'date',
                       ticker_col: str = 'ticker', ret_col: str = 'logret') -> ReturnsPanel:
    '''
    Long table [date, ticker, logret] -> ReturnsPanel.

    - Tickers are sorted alphabetically, dates ascending: the order is deterministic.
    - Missing values and infinite values (log of a zero price) become NaN.
    - If a (date, ticker) pair appears twice, the first value is kept.
    '''
    wide = (
        returns_long
        .select([date_col, ticker_col, ret_col])
        .pivot(on=ticker_col, index=date_col, values=ret_col, aggregate_function='first')
        .sort(date_col)
    )
    tickers = sorted(col for col in wide.columns if col != date_col)
    wide = wide.select([date_col] + tickers)

    values = wide.select(tickers).to_numpy().astype(float)
    values = np.where(np.isfinite(values), values, np.nan)
    dates = wide[date_col].to_numpy().astype('datetime64[D]')
    return ReturnsPanel(dates=dates, tickers=np.asarray(tickers), values=values)


# =============================================================================
# 2. REBALANCE DATES
# =============================================================================

def pick_rebalance_dates(trading_dates: list, freq: str = 'weekly', first_date: "date | None" = None) -> list:
    '''
    One rebalance date per period = the LAST trading day of the period.

        freq='weekly'  -> last trading day of each ISO week   (usually a Friday)
        freq='monthly' -> last trading day of each month

    Why the last day: a holiday never makes a whole period disappear.
    The most recent period may be incomplete: then its date is simply the latest available day.

    trading_dates: list of datetime.date (any order, duplicates allowed).
    first_date:    drop rebalance dates before this day.
    '''
    if freq not in ('weekly', 'monthly'):
        raise ValueError(f"unknown rebalance frequency {freq!r}, use 'weekly' or 'monthly'")

    last_in_period = {}
    for d in sorted(set(trading_dates)):
        key = tuple(d.isocalendar())[:2] if freq == 'weekly' else (d.year, d.month)
        last_in_period[key] = d            # sorted ascending, so the last write is the last day

    result = list(last_in_period.values())
    if first_date is not None:
        result = [d for d in result if d >= first_date]
    return result


# =============================================================================
# 3. ONE PLAN PER REBALANCE DATE
# =============================================================================

@dataclass
class Plan:
    '''Which part of the panel one rebalance date will use. Holds indices only, no data.'''
    date_str: str            # ISO date of the rebalance
    start: int               # first row of the window (inclusive)
    end: int                 # last row of the window (exclusive); row end-1 is the rebalance date
    asset_idx: np.ndarray    # column indices of the eligible tickers


def plan_rebalances(panel: ReturnsPanel, rebalance_dates: list, lookback_window: int,
                    min_obs: int, min_assets: int) -> "tuple[list[Plan], list[dict]]":
    '''
    Decide, for each rebalance date, the window and the eligible tickers (rules in the file header).

    A date is SKIPPED (and listed in the second return value, with the reason) if
      - it is not a trading date of the panel,
      - the window has fewer than `min_obs` rows,
      - fewer than `min_assets` tickers are eligible.

    Returns (plans, skipped).
    '''
    plans, skipped = [], []
    for d in rebalance_dates:
        d64 = np.datetime64(d, 'D')
        end = int(np.searchsorted(panel.dates, d64, side='right'))      # rows with date <= d
        if end == 0 or panel.dates[end - 1] != d64:
            skipped.append({'date': d.isoformat(), 'reason': 'not a trading date in the returns panel'})
            continue

        start = max(0, end - lookback_window)
        if end - start < min_obs:
            skipped.append({'date': d.isoformat(), 'reason': f'window has {end - start} rows < min_obs={min_obs}'})
            continue

        window = panel.values[start:end]                                # (T, n_tickers)
        trades_today = np.isfinite(window[-1])                          # rule (a)
        enough_obs = np.isfinite(window).sum(axis=0) >= min_obs         # rule (b)
        asset_idx = np.flatnonzero(trades_today & enough_obs)

        if len(asset_idx) < min_assets:
            skipped.append({'date': d.isoformat(), 'reason': f'{len(asset_idx)} eligible tickers < min_assets={min_assets}'})
            continue

        plans.append(Plan(d.isoformat(), start, end, asset_idx))
    return plans, skipped


def plan_matrix(panel: ReturnsPanel, plan: Plan) -> "tuple[list[str], np.ndarray]":
    '''Cut the data of one plan out of the panel. Returns (tickers, matrix of shape (n_assets, n_obs)).'''
    window = panel.values[plan.start:plan.end][:, plan.asset_idx]       # (T, N)
    tickers = [str(t) for t in panel.tickers[plan.asset_idx]]
    return tickers, np.ascontiguousarray(window.T)


# =============================================================================
# 4. ONE DATE: estimate, check, save, summarise
# =============================================================================

def process_one_date(date_str: str, tickers: list, matrix: np.ndarray, config: dict, out_dir: str) -> dict:
    '''
    Runs in a worker process. Steps:
      1. estimate the risk model for this date (src/risk/risk_model.py)
      2. check the result (fail loudly, never save garbage)
      3. write the per-date files
      4. return a SMALL summary dict (the big matrices stay out of the main process)
    '''
    result = estimate_risk_model_for_date(matrix, tickers, date_str, config)

    # --- checks ---
    cov = result['cov']
    if not np.isfinite(cov).all():
        raise ValueError(f'{date_str}: covariance contains NaN or inf')
    if np.diag(cov).min() <= 0:
        raise ValueError(f'{date_str}: covariance has a non-positive variance')
    missing = set(range(len(tickers))) - set(result['cluster_map'])
    if missing:
        raise ValueError(f'{date_str}: {len(missing)} tickers have no cluster')

    # --- files ---
    out = Path(out_dir)
    if config.get('save_corr_raw', False):
        save_long_matrix(result['corr_raw'], tickers, date_str, out / 'corr_raw.parquet', 'corr')
    save_long_matrix(result['corr_clean'], tickers, date_str, out / 'corr_clean.parquet', 'corr')
    save_long_matrix(cov, tickers, date_str, out / 'cov.parquet', 'cov')
    save_clusters(result['cluster_map'], tickers, date_str, out / 'clusters.parquet')
    save_eigenvalues(result['eigenvalues'], date_str, out / 'eigenvalues.parquet')

    # --- summary ---
    sizes = sorted(result['cluster_sizes'].values())
    return {
        'date': date_str,
        'n_tickers': result['n_tickers'],
        'n_obs': result['n_obs'],
        'q_ratio': round(float(result['q_ratio']), 4),
        'lambda_max': round(float(result['lambda_max']), 4),
        'n_clusters': result['n_clusters'],
        'cluster_size_stats': {
            'min': sizes[0],
            'max': sizes[-1],
            'median': sizes[len(sizes) // 2],
            'mean': round(sum(sizes) / len(sizes), 2),
        },
        'shrinkage': result['shrinkage'],
        'cov_method': result['cov_method'],
        # Sanity number: typical annualised volatility implied by the covariance (equities: about 0.2 to 0.5)
        'median_vol': round(float(np.sqrt(np.median(np.diag(cov)))), 4),
    }


# =============================================================================
# 5. ALL DATES, IN PARALLEL
# =============================================================================

def run_risk_model(panel: ReturnsPanel, plans: list, config: dict, out_dir: Path,
                   n_jobs: "int | None" = None) -> list:
    '''
    Run process_one_date for every plan, in parallel, with a real progress bar.
    Returns the list of per-date summaries, in the order of `plans`.

    Memory: the data of a plan is cut from the panel only when the plan is submitted,
    and workers return only small dicts. The full history of matrices is never held in memory.
    '''
    if not plans:
        return []
    n_jobs = n_jobs or min(os.cpu_count() or 1, len(plans))

    tasks = (
        delayed(process_one_date)(p.date_str, *plan_matrix(panel, p), config, str(out_dir))
        for p in plans
    )
    summaries = []
    with Parallel(n_jobs=n_jobs, backend='loky', return_as='generator') as parallel:
        for summary in tqdm(parallel(tasks), total=len(plans), desc='  Risk model', unit='date'):
            summaries.append(summary)
    return summaries


# =============================================================================
# 6. THE SUMMARY FILE
# =============================================================================

def build_summary(summaries: list, skipped: list) -> dict:
    '''
    The content of summary.json.

    NOTE for readers (04e, 04f): this is a DICT. The list of dates is in summary['per_date'].
    '''
    if not summaries:
        raise ValueError('no rebalance date was processed, nothing to summarise')

    shrink = [s['shrinkage'] for s in summaries if s['shrinkage'] is not None]
    return {
        'n_rebalances': len(summaries),
        'date_range': {'start': summaries[0]['date'], 'end': summaries[-1]['date']},
        'avg_n_clusters': round(sum(s['n_clusters'] for s in summaries) / len(summaries), 1),
        'avg_lambda_max': round(sum(s['lambda_max'] for s in summaries) / len(summaries), 4),
        'avg_shrinkage': round(sum(shrink) / len(shrink), 4) if shrink else None,
        'avg_median_vol': round(sum(s['median_vol'] for s in summaries) / len(summaries), 4),
        'n_skipped': len(skipped),
        'skipped': skipped,
        'per_date': summaries,
    }


def write_json(path: Path, content: dict) -> None:
    with open(path, 'w') as f:
        json.dump(content, f, indent=2, default=str)