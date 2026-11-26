#!/usr/bin/env python
'''
===============================================
Phase 08: Construct Portfolio Weights
===============================================

QUESTION ANSWERED
    "At each rebalance date, how much capital goes long or short in each stock?"

THE STAGE IN ONE PICTURE

    en_scores.parquet            risk_model/ (from 07)
    (date, ticker, score)        cov.parquet, clusters.parquet, summary.json
            |                               |
            v                               v
     [1] rebalance dates = dates in BOTH the scores and the risk model
            |
            v          one book per date
     [2] src/risk/weights_pipeline.py  build_weights()
            raw weights        proportional to the centred score
              -> cluster caps          no correlated cluster dominates
              -> re-neutralise legs    long = short again
              -> volatility target     scale to target_vol
              -> leverage cap          never above max_gross_leverage
            |
            v
     [3] checks (fail loudly) + diagnostics (gross, net, vol, turnover)
            |
            v
     [4] weights.parquet + weights_diag.parquet + config.json

WHERE IS THE COMPLICATED PART?
    src/risk/portfolio_construction.py   sizing, caps, vol target (the maths)
    src/risk/weights_pipeline.py         reading inputs, order of steps, checks, diagnostics
    This script only calls them, in order.

UNITS
    A weight is a fraction of capital. gross = sum of |weight|.
    gross 1 = 50% long + 50% short.  gross 4 = 200% long + 200% short.

INPUTS
    {clean_base}/runs/{run_id}/en_scores/en_scores.parquet      (date, ticker, combined_score)
    {clean_base}/runs/{run_id}/risk_model/                      (07 outputs)
    {clean_base}/features/features.parquet                 (beta252, for the beta neutralisation)

OUTPUTS   {clean_base}/runs/{run_id}/weights/
    weights.parquet        [date, ticker, score, cluster_id, raw_weight, capped_weight, weight]
    weights_diag.parquet   one row per date: gross, long/short/net exposure, beta exposure, ex-ante vol, turnover, ...
    config.json

HONEST USE OF THE RESULT
    The scores of the search period are out-of-fold (see 06), the holdout scores are out-of-sample.
    Judge any later backtest on the holdout dates only.

USAGE
    python scripts/pipeline/08_construct_weights.py --date 2026-11-07 --tag default
'''

import argparse
import json
import sys
from pathlib import Path

# Make `import src...` and `import config...` work wherever the script is started from.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tqdm import tqdm

from config.paths import get_paths, run_id_from_args
from src.risk.weights_pipeline import (
    risk_model_dates,
    load_scores_by_date,
    load_betas_by_date,
    RiskModelReader,
    construct_all,
)
from src.utils import console as out


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_COL = 'combined_score'

PORTFOLIO_CONFIG = {
    'target_vol': 0.10,              # annualised ex-ante volatility to reach (before the leverage cap)
    'base_gross': 1.0,               # gross of the raw book (sum|w|), before the volatility scaling
    'max_gross_leverage': 4.0,       # hard cap on final gross. ASSUMPTION: review with your financing limits.
    'beta_neutral': True,            # project the book to zero exposure to the equal-weight market (beta' w = 0)
    'beta_source': 'features',       # 'features': feature beta (below). 'covariance': betas implied by the risk model
    'beta_feature': 'beta252',       # rolling beta of the features table (03)
    'beta_shrink': 1.0 / 3.0,        # Blume: beta_adj = (1 - k) beta + k. Missing beta -> 1
    'max_cluster_multiple': 2.0,     # a cluster may hold at most 2x its fair share of gross
    'signal_method': 'proportional', # 'proportional', 'inv_vol' or 'rank'
    'cap_method': 'cap',             # 'cap' or 'rescale'
    'vol_method': 'scale',           # 'scale'
    'min_names': 50,                 # a date needs this many tickers (score AND risk model), else it is skipped
}


def main(run_date: str, tag: str):
    run_id = run_id_from_args(run_date, tag)
    paths = get_paths(run_id=run_id)

    en_scores_file = paths.stage_dir('en_scores') / 'en_scores.parquet'
    risk_dir = paths.stage_dir('risk_model')
    out_dir = paths.stage_dir('weights')

    for f in (en_scores_file, risk_dir / 'cov.parquet', risk_dir / 'clusters.parquet', risk_dir / 'summary.json',
              paths.features_file):
        if not f.exists():
            raise FileNotFoundError(f'Missing input: {f}')
    out_dir.mkdir(parents=True, exist_ok=True)

    c = PORTFOLIO_CONFIG
    out.stage('08', 'Construct portfolio weights', 'How much do we hold long and short of each stock, '
              'on each rebalance date?')
    out.info('run', run_id)
    out.info('rules', f'dollar-neutral{", beta-neutral" if c["beta_neutral"] else ""}, cluster cap '
             f'{c["max_cluster_multiple"]}x fair share, target vol {c["target_vol"]:.0%}, gross cap {c["max_gross_leverage"]}')

    # ------------------------------------------------------------------
    # [1/4] Rebalance dates: the dates of the risk model that also have scores
    # ------------------------------------------------------------------
    out.step(1, 4, 'Read the rebalance dates and the combined scores')
    dates = risk_model_dates(risk_dir / 'summary.json')
    scores_by_date = load_scores_by_date(en_scores_file, SCORE_COL, dates)
    dates = [d for d in dates if d in scores_by_date]
    out.info('dates', f'{len(dates)} with both a risk model and scores ({dates[0]} to {dates[-1]})')

    # ------------------------------------------------------------------
    # [2/4] Build one book per date
    # ------------------------------------------------------------------
    out.step(2, 4, 'Build one book per date',
             'weights follow the score; then caps per cluster (no hidden sector bet), long = short, zero market beta '
             '(a book long high-beta stocks is long the market), and a scale to the target volatility')
    reader = RiskModelReader(risk_dir)
    betas_by_date = None
    if PORTFOLIO_CONFIG['beta_neutral'] and PORTFOLIO_CONFIG['beta_source'] == 'features':
        betas_by_date = load_betas_by_date(paths.features_file, PORTFOLIO_CONFIG['beta_feature'], dates)
        out.info('betas', f'{c["beta_feature"]} (rolling, past only), shrunk {c["beta_shrink"]:.2f} toward 1')
    weights_df, diag_df, skipped = construct_all(
        dates, scores_by_date, reader, PORTFOLIO_CONFIG, betas_by_date=betas_by_date,
        progress=lambda it: tqdm(it, desc='      books', unit='date', leave=False),
    )
    for s in skipped:
        out.note(f'skipped {s["date"]}: {s["reason"]}')

    # ------------------------------------------------------------------
    # [3/4] Save
    # ------------------------------------------------------------------
    out.step(3, 4, 'Save the weights and the per-date diagnostics')
    weights_df.write_parquet(out_dir / 'weights.parquet')
    diag_df.write_parquet(out_dir / 'weights_diag.parquet')
    with open(out_dir / 'config.json', 'w') as f:
        json.dump({
            'run_id': run_id,
            'tag': tag,
            'portfolio_config': PORTFOLIO_CONFIG,
            'score_col': SCORE_COL,
            'n_dates': diag_df.height,
            'n_skipped': len(skipped),
            'skipped': skipped,
        }, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # [4/4] What to look at
    # ------------------------------------------------------------------
    out.step(4, 4, 'Check the books', 'every book must be neutral, finite and below the leverage cap '
             '(the stage already stopped if one was not)')
    net_max = diag_df['net_exposure'].abs().max()
    out.verdict(net_max < 1e-6, f'dollar-neutral: largest |long - short| = {net_max:.1e} of NAV')
    if c['beta_neutral']:
        b_before, b_after = diag_df['beta_before_neutralising'].abs().mean(), diag_df['beta_exposure'].abs().mean()
        out.verdict(b_after < 1e-6, f'beta-neutral: mean |beta exposure| {b_before:.3f} before, {b_after:.1e} after')
    out.info('books', f'{diag_df.height} dates, {weights_df.height:,} weights ({len(skipped)} dates skipped)')
    out.info('gross exposure', f'mean {diag_df["gross"].mean():.2f} x NAV, max {diag_df["gross"].max():.2f} '
             f'(cap {c["max_gross_leverage"]})')
    out.info('ex-ante volatility', f'mean {diag_df["ex_ante_vol"].mean():.1%} (target {c["target_vol"]:.0%})')
    n_cap = int(diag_df['leverage_capped'].sum())
    if n_cap:
        out.note(f'the leverage cap binds on {n_cap} of {diag_df.height} dates: there the volatility is below target')
    out.info('turnover per rebalance', f'{diag_df["turnover"].mean():.1%} of NAV, one way, on average')
    out.done({'weights': out_dir / 'weights.parquet', 'diagnostics': out_dir / 'weights_diag.parquet'},
             f'python scripts/pipeline/09_run_backtest.py --date {run_date} --tag {tag}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Construct portfolio weights from scores and the risk model')
    parser.add_argument('--date', type=str, required=True, help='Run date, YYYY-MM-DD (used in the run_id)')
    parser.add_argument('--tag', type=str, default='default', help='Run tag (default: default)')
    args = parser.parse_args()
    main(args.date, args.tag)
