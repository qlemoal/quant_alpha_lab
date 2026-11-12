'''
Plumbing for stage 04e (portfolio weights). The script 04e_construct_weights.py only calls these functions.
The portfolio maths (sizing, caps, vol target) is in src/risk/portfolio_construction.py.
This file adds: reading the risk model, the order of the steps, the checks, and the diagnostics.

THE CHAIN FOR ONE REBALANCE DATE
--------------------------------

    scores (one per ticker)        covariance Σ (annualised)       clusters
            |                              |                          |
            v                              |                          |
   [1] raw weights                         |                          |
       proportional to the                 |                          |
       centred score                       |                          |
       sum(w) = 0,  sum|w| = base_gross   |                          |
            |                              |                          |
            v                              |                          v
   [2] cluster caps  -------------------------------------------- cluster of each ticker
       no cluster may hold more than max_cluster_multiple x its fair share
            |
            v
   [3] re-neutralise the legs
       the caps shrink some clusters, so sum(w) is no longer 0.
       Scale the long leg and the short leg so that long = short. Gross is kept.
            |
            v                              |
   [4] volatility target  <---------------- Σ
       multiply ALL weights by  target_vol / sqrt(w' Σ w)   (library cap: x10)
            |
            v
   [5] leverage cap
       if sum|w| > max_gross_leverage: multiply ALL weights down. Volatility ends below target.
            |
            v
        final weights

Steps 4 and 5 multiply every weight by the same number. They keep the legs balanced and keep
each cluster's share of the book.

UNITS
-----
Weights are fractions of capital (NAV). sum|w| is the GROSS exposure.
Example: sum|w| = 1 means 50% long and 50% short. sum|w| = 4 means 200% long and 200% short.
Volatilities and Σ are annualised (see src/risk/risk_model.py).
'''

import json
from pathlib import Path

import numpy as np
import polars as pl

from src.risk.portfolio_construction import raw_weights_from_score, apply_cluster_caps, apply_vol_target
from src.utils.risk_io import load_matrix_for_date

NET_TOLERANCE = 1e-8      # |sum of weights| above this is a bug, not rounding


# =============================================================================
# 1. THE MATHS FOR ONE DATE (pure function: arrays in, arrays out)
# =============================================================================

def neutralise_legs(weights: np.ndarray) -> np.ndarray:
    '''
    Make a long-short book exactly dollar-neutral without changing its gross exposure.

    L = sum of the long weights, S = sum of the absolute short weights, m = (L + S) / 2.
    Long weights are multiplied by m / L, short weights by m / S.
    Afterwards: long leg = short leg = m, and sum|w| is unchanged.
    If one leg is empty, no neutral book exists: returns zeros.
    '''
    w = np.asarray(weights, dtype=float).copy()
    longs, shorts = w > 0, w < 0
    L, S = w[longs].sum(), -w[shorts].sum()
    if L <= 0 or S <= 0:
        return np.zeros_like(w)
    m = 0.5 * (L + S)
    w[longs] *= m / L
    w[shorts] *= m / S
    return w


def build_weights(scores: np.ndarray, cov: np.ndarray, clusters: np.ndarray, cfg: dict) -> "dict | None":
    '''
    Scores -> final weights for ONE date, following the chain in the file header.

    scores:    (n,) alpha score per ticker, no NaN.
    cov:       (n, n) annualised covariance, same ticker order as scores.
    clusters:  (n,) integer cluster id per ticker (-1 = no cluster, never capped).
    cfg:       PORTFOLIO_CONFIG of the script.

    Returns a dict (raw, capped, neutral, final, ex_ante_vol, leverage_capped), or None if the
    scores cannot form a long-short book (for example all scores equal).
    '''
    n = len(scores)
    vols = np.sqrt(np.diag(cov)) if cfg['signal_method'] == 'inv_vol' else None

    # [1] raw weights: centred score, sum(w) = 0, sum|w| = base_gross
    raw = raw_weights_from_score(
        pl.Series('score', scores), max_gross=cfg['base_gross'],
        method=cfg['signal_method'], vols=vols,
    ).to_numpy()
    if not np.any(raw):
        return None

    # [2] cluster caps
    capped = apply_cluster_caps(
        pl.Series('w', raw), cluster_map=clusters,
        max_multiple=cfg['max_cluster_multiple'], method=cfg['cap_method'],
    ).to_numpy()

    # [3] back to dollar-neutral
    neutral = neutralise_legs(capped)
    if not np.any(neutral):
        return None

    # [4] volatility target (uniform scaling)
    targeted = apply_vol_target(
        pl.Series('w', neutral), sigma=cov, target_vol=cfg['target_vol'], method=cfg['vol_method'],
    ).to_numpy()

    # [5] leverage cap (uniform scaling)
    gross = np.abs(targeted).sum()
    leverage_capped = bool(gross > cfg['max_gross_leverage'])
    final = targeted * (cfg['max_gross_leverage'] / gross) if leverage_capped else targeted

    return {
        'raw': raw,
        'capped': capped,
        'final': final,
        'ex_ante_vol': float(np.sqrt(max(final @ cov @ final, 0.0))),
        'leverage_capped': leverage_capped,
    }


def check_weights(date_str: str, final: np.ndarray, cfg: dict) -> None:
    '''Fail loudly on a broken book. Never save garbage.'''
    if not np.isfinite(final).all():
        raise ValueError(f'{date_str}: weights contain NaN or inf')
    net = final.sum()
    if abs(net) > NET_TOLERANCE:
        raise ValueError(f'{date_str}: book is not dollar-neutral, net exposure = {net:.3e}')
    gross = np.abs(final).sum()
    if gross > cfg['max_gross_leverage'] * (1 + 1e-9):
        raise ValueError(f'{date_str}: gross {gross:.4f} above max_gross_leverage {cfg["max_gross_leverage"]}')


def diagnostics(date_str: str, final: np.ndarray, clusters: np.ndarray, result: dict, cfg: dict) -> dict:
    '''One row of weights_diag.parquet. (turnover is added later, it needs the previous date.)'''
    gross = float(np.abs(final).sum())
    valid_clusters = np.unique(clusters[clusters >= 0])
    cluster_gross = [np.abs(final[clusters == k]).sum() for k in valid_clusters]
    return {
        'date': date_str,
        'n_tickers': int(len(final)),
        'n_clusters': int(len(valid_clusters)),
        'gross': gross,
        'long_exposure': float(final[final > 0].sum()),
        'short_exposure': float(-final[final < 0].sum()),
        'net_exposure': float(final.sum()),
        'max_weight': float(final.max()),
        'min_weight': float(final.min()),
        'max_cluster_share': float(max(cluster_gross) / gross) if cluster_gross and gross > 0 else None,
        'ex_ante_vol': result['ex_ante_vol'],
        'vol_target': cfg['target_vol'],
        'leverage_capped': result['leverage_capped'],
    }


# =============================================================================
# 2. READING THE INPUTS
# =============================================================================

def risk_model_dates(summary_file: Path) -> list:
    '''
    Rebalance dates (ISO strings) of the risk model, from summary.json.
    04d writes a dict with the dates under 'per_date'. A plain list (old format) is accepted too.
    '''
    with open(summary_file) as f:
        summary = json.load(f)
    entries = summary['per_date'] if isinstance(summary, dict) else summary
    return [e['date'] for e in entries]


def load_scores_by_date(en_scores_file: Path, score_col: str, date_strs: list) -> dict:
    '''
    {date_str: (tickers list, scores array)}, only for the given dates and non-null scores.
    Tickers are sorted alphabetically inside each date.
    '''
    wanted = set(date_strs)
    df = (
        pl.scan_parquet(en_scores_file)
        .select(['date', 'ticker', score_col])
        .filter(pl.col(score_col).is_not_null() & pl.col(score_col).is_not_nan())
        .collect()
        .filter(pl.col('date').dt.to_string('%Y-%m-%d').is_in(list(wanted)))
        .sort(['date', 'ticker'])
    )
    out = {}
    for key, group in df.partition_by(['date'], as_dict=True).items():
        d = key[0] if isinstance(key, tuple) else key
        out[d.isoformat()] = (group['ticker'].to_list(), group[score_col].to_numpy())
    return out


class RiskModelReader:
    '''
    Reads one date of the 04d outputs at a time: the covariance (full symmetric matrix) and the clusters.
    Clusters are small, so they are loaded once.
    '''

    def __init__(self, risk_dir: Path):
        self.cov_file = Path(risk_dir) / 'cov.parquet'
        clusters = pl.read_parquet(Path(risk_dir) / 'clusters.parquet')
        self._clusters = {}
        for key, group in clusters.partition_by(['date'], as_dict=True).items():
            d = key[0] if isinstance(key, tuple) else key
            self._clusters[str(d)] = dict(zip(group['ticker'].to_list(), group['cluster_id'].to_list()))

    def get(self, date_str: str) -> "tuple[np.ndarray, list, dict]":
        '''Returns (cov, tickers, cluster_of_ticker). cov rows/columns follow `tickers` (sorted).'''
        cov, tickers = load_matrix_for_date(self.cov_file, date_str, 'cov')
        return cov, tickers, self._clusters.get(date_str, {})


# =============================================================================
# 3. ALL DATES
# =============================================================================

def one_way_turnover(prev: "dict | None", current: dict) -> "float | None":
    '''
    One-way turnover between two books given as {ticker: weight}: 0.5 * sum |w_new - w_old|.
    A ticker missing on one side counts as weight 0. Returns None for the first date.
    '''
    if prev is None:
        return None
    names = set(prev) | set(current)
    return float(0.5 * sum(abs(current.get(t, 0.0) - prev.get(t, 0.0)) for t in names))


def construct_all(dates: list, scores_by_date: dict, reader: RiskModelReader, cfg: dict, progress=None):
    '''
    Run build_weights on every date. Returns (weights_df, diag_df, skipped).

    For each date the tickers are those that have BOTH a score and a risk-model entry.
    A date is skipped (with a reason) if fewer than cfg['min_names'] tickers remain or no book can be built.
    `progress`: optional iterator wrapper (e.g. tqdm) around the dates.
    '''
    weight_frames, diag_rows, skipped = [], [], []
    prev_book = None

    for date_str in (progress(dates) if progress else dates):
        if date_str not in scores_by_date:
            skipped.append({'date': date_str, 'reason': 'no score on this date'})
            continue
        score_tickers, scores = scores_by_date[date_str]
        cov_full, cov_tickers, cluster_of = reader.get(date_str)

        # Common tickers, in the order of the covariance matrix (alphabetical)
        score_of = dict(zip(score_tickers, scores))
        keep = [i for i, t in enumerate(cov_tickers) if t in score_of]
        if len(keep) < cfg['min_names']:
            skipped.append({'date': date_str, 'reason': f'{len(keep)} common tickers < min_names={cfg["min_names"]}'})
            continue

        tickers = [cov_tickers[i] for i in keep]
        cov = cov_full[np.ix_(keep, keep)]
        sc = np.array([score_of[t] for t in tickers])
        clusters = np.array([cluster_of.get(t, -1) for t in tickers], dtype=int)

        result = build_weights(sc, cov, clusters, cfg)
        if result is None:
            skipped.append({'date': date_str, 'reason': 'scores cannot form a long-short book'})
            continue
        final = result['final']
        check_weights(date_str, final, cfg)

        book = dict(zip(tickers, final))
        row = diagnostics(date_str, final, clusters, result, cfg)
        row['turnover'] = one_way_turnover(prev_book, book)
        diag_rows.append(row)
        prev_book = book

        weight_frames.append(pl.DataFrame({
            'date': [date_str] * len(tickers),
            'ticker': tickers,
            'score': sc,
            'cluster_id': clusters,
            'raw_weight': result['raw'],
            'capped_weight': result['capped'],
            'weight': final,
        }))

    if not weight_frames:
        raise ValueError('No weights were built. Check the overlap of en_scores and risk_model, and min_names.')

    weights_df = pl.concat(weight_frames).with_columns(pl.col('date').str.to_date('%Y-%m-%d'))
    diag_df = pl.DataFrame(diag_rows).with_columns(pl.col('date').str.to_date('%Y-%m-%d'))
    return weights_df, diag_df, skipped