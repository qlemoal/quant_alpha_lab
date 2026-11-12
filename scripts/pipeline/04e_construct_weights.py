#!/usr/bin/env python
'''
===============================================
Phase 04e: Construct Portfolio Weights
===============================================

QUESTION ANSWERED
    "At each rebalance date, how much capital goes long or short in each stock?"

THE STAGE IN ONE PICTURE

    en_scores.parquet            risk_model/ (from 04d)
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
    {clean_base}/en_scores/{run_id}/en_scores.parquet      (date, ticker, combined_score)
    {clean_base}/risk_model/{run_id}/                      (04d outputs)

OUTPUTS   {clean_base}/weights/{run_id}/
    weights.parquet        [date, ticker, score, cluster_id, raw_weight, capped_weight, weight]
    weights_diag.parquet   one row per date: gross, long/short/net exposure, ex-ante vol, turnover, ...
    config.json

HONEST USE OF THE RESULT
    The scores of the search period are out-of-fold (see 04c), the holdout scores are out-of-sample.
    Judge any later backtest on the holdout dates only.

USAGE
    python scripts/pipeline/04e_construct_weights.py --date 2026-11-07 --tag default
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
    RiskModelReader,
    construct_all,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_COL = 'combined_score'

PORTFOLIO_CONFIG = {
    'target_vol': 0.10,              # annualised ex-ante volatility to reach (before the leverage cap)
    'base_gross': 1.0,               # gross of the raw book (sum|w|), before the volatility scaling
    'max_gross_leverage': 4.0,       # hard cap on final gross. ASSUMPTION: review with your financing limits.
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

    for f in (en_scores_file, risk_dir / 'cov.parquet', risk_dir / 'clusters.parquet', risk_dir / 'summary.json'):
        if not f.exists():
            raise FileNotFoundError(f'Missing input: {f}')
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print('PHASE 04e: CONSTRUCT PORTFOLIO WEIGHTS')
    print('=' * 70)
    print(f'Run ID:  {run_id}')
    print(f'Config:  {PORTFOLIO_CONFIG}')
    print(f'Output:  {out_dir}')

    # ------------------------------------------------------------------
    # [1/4] Rebalance dates: the dates of the risk model that also have scores
    # ------------------------------------------------------------------
    print('\n[1/4] Loading rebalance dates and scores...')
    dates = risk_model_dates(risk_dir / 'summary.json')
    scores_by_date = load_scores_by_date(en_scores_file, SCORE_COL, dates)
    dates = [d for d in dates if d in scores_by_date]
    print(f'  {len(dates)} dates with both a risk model and scores ({dates[0]} to {dates[-1]})')

    # ------------------------------------------------------------------
    # [2/4] Build one book per date
    # ------------------------------------------------------------------
    print('\n[2/4] Constructing weights...')
    reader = RiskModelReader(risk_dir)
    weights_df, diag_df, skipped = construct_all(
        dates, scores_by_date, reader, PORTFOLIO_CONFIG,
        progress=lambda it: tqdm(it, desc='  Weights', unit='date'),
    )
    for s in skipped:
        print(f'  Skipped {s["date"]}: {s["reason"]}')

    # ------------------------------------------------------------------
    # [3/4] Save
    # ------------------------------------------------------------------
    print('\n[3/4] Saving...')
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
    print('\n[4/4] Summary')
    print(f'  {weights_df.height:,} weights over {diag_df.height} dates ({len(skipped)} skipped)')
    print(f'  gross exposure   mean {diag_df["gross"].mean():.2f}   max {diag_df["gross"].max():.2f}'
          f'   (cap {PORTFOLIO_CONFIG["max_gross_leverage"]})')
    print(f'  ex-ante vol      mean {diag_df["ex_ante_vol"].mean():.3f}   (target {PORTFOLIO_CONFIG["target_vol"]})')
    print(f'  dates where the leverage cap binds (vol below target): '
          f'{diag_df["leverage_capped"].sum()} / {diag_df.height}')
    print(f'  max |net exposure|  {diag_df["net_exposure"].abs().max():.2e}   (must be about 0)')
    print(f'  one-way turnover per rebalance  mean {diag_df["turnover"].mean():.3f}')
    print(f'  output: {out_dir}/')
    print(f'\nNext: python scripts/pipeline/04f_run_backtest.py --date {run_date} --tag {tag}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Construct portfolio weights from scores and the risk model')
    parser.add_argument('--date', type=str, required=True, help='Run date, YYYY-MM-DD (used in the run_id)')
    parser.add_argument('--tag', type=str, default='default', help='Run tag (default: default)')
    args = parser.parse_args()
    main(args.date, args.tag)