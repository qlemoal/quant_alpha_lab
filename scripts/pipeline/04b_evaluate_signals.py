#!/usr/bin/env python
'''
===============================================
Phase 0b: Evaluate Signals (IC, FDR, Decay)
===============================================

Evaluates generated signals: Rank IC, Newey-West, IC decay, FDR (BH/BY/q-value).
Produces a report and a filtered signal list for the combiner.

Inputs:
  data/processed/signals_<tag>.parquet

Outputs:
  data/exports/signal_eval_<tag>/
    ├── ic_summary.parquet      # Per-signal IC stats
    ├── ic_decay.parquet        # IC vs horizon
    ├── fdr_report.parquet      # BH, BY, q-value per signal
    ├── signal_report.json      # Human-readable summary
    └── selected_signals.json   # List of signals passing FDR (for combiner)

Usage:
  python scripts/pipeline/04b_evaluate_signals.py --signal-tag default
'''

import argparse
import json
from pathlib import Path
import polars as pl

from src.evaluation.signals.ic.core import compute_ic, summarize_ic, ic_decay
from src.evaluation.signals.ic.metrics import newey_west_ic_tstat
from src.evaluation.signals.fdr import fdr_report
from src.evaluation.signals.performance import decile_longshort_returns

# =============================================================================
# CONFIGURATION
# =============================================================================

SIGNAL_TAG = 'default'  # Overridden by --signal-tag

INPUT_FILE = f'data/processed/signals_{SIGNAL_TAG}.parquet'
OUTPUT_DIR = f'data/exports/signal_eval_{SIGNAL_TAG}'

# Forward return column to evaluate against
FWD_RET_COL = 'fwd_ret_20d'  # Main evaluation horizon

# FDR settings
FDR_ALPHA = 0.2
FDR_METHODS = ['bh', 'by', 'qvalue']  # All reported

# IC decay horizons
DECAY_HORIZONS = [1, 5, 10, 20]

# Minimum IC IR to consider (pre-FDR filter)
MIN_IC_IR = 0.0

# Date range for evaluation
EVAL_START = None
EVAL_END = None


def main(signal_tag: str):
    global INPUT_FILE, OUTPUT_DIR
    INPUT_FILE = f'data/processed/signals_{signal_tag}.parquet'
    OUTPUT_DIR = f'data/exports/signal_eval_{signal_tag}'

    print('=' * 70)
    print('PHASE 0b: EVALUATE SIGNALS')
    print('=' * 70)
    print(f'Input:         {INPUT_FILE}')
    print(f'Output dir:    {OUTPUT_DIR}')
    print(f'FWD_RET_COL:   {FWD_RET_COL}')
    print(f'FDR alpha:     {FDR_ALPHA}')
    print(f'Decay horizons: {DECAY_HORIZONS}')
    print('=' * 70)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Load signals
    print('\n[1/5] Loading signals...')
    if not Path(INPUT_FILE).exists():
        raise FileNotFoundError(f'Signal file not found: {INPUT_FILE}. Run 04a_generate_signals.py first.')

    df = pl.read_parquet(INPUT_FILE)
    if EVAL_START or EVAL_END:
        df = df.filter(
            (pl.col('date') >= EVAL_START) & (pl.col('date') <= EVAL_END)
        )
    print(f'  Loaded {len(df):,} rows')

    # Identify signal columns (exclude date, ticker, fwd_ret_*)
    all_cols = df.columns
    exclude = {'date', 'ticker'} | {c for c in all_cols if c.startswith('fwd_ret_')}
    signal_cols = [c for c in all_cols if c not in exclude]
    print(f'  Found {len(signal_cols)} signals: {signal_cols}')

    # --- IC per signal ---
    print('\n[2/5] Computing Rank IC...')
    ic_results = []
    for col in signal_cols:
        ic_series = compute_ic(df, col, FWD_RET_COL)
        summary = summarize_ic(ic_series)
        nw_tstat = newey_west_ic_tstat(ic_series)

        ic_results.append({
            'signal': col,
            'mean_ic': summary['mean'],
            'std_ic': summary['std'],
            'ir': summary['ir'],
            'hit_rate': summary['hit_rate'],
            'n_obs': summary['n'],
            'nw_tstat': nw_tstat,
        })

    ic_df = pl.DataFrame(ic_results)
    print(f'  IC Summary (top 10 by |IR|):')
    for row in ic_df.sort('ir', descending=True).head(10).iter_rows(named=True):
        print(f'    {row["signal"]:25s} IC={row["mean_ic"]:7.4f} IR={row["ir"]:7.3f} HR={row["hit_rate"]:.2%} NW_t={row["nw_tstat"]:6.2f}')

    ic_df.write_parquet(f'{OUTPUT_DIR}/ic_summary.parquet')

    # --- IC Decay ---
    print('\n[3/5] Computing IC decay...')
    decay_results = []
    for col in signal_cols:
        decay = ic_decay(df, col, [f'fwd_ret_{h}d' for h in DECAY_HORIZONS])
        decay_results.append({'signal': col, **decay})

    decay_df = pl.DataFrame(decay_results)
    decay_df.write_parquet(f'{OUTPUT_DIR}/ic_decay.parquet')
    print(f'  Decay computed for {len(signal_cols)} signals')

    # --- FDR ---
    print('\n[4/5] Computing FDR...')
    # p-values from Neyman-Pearson (two-sided) using IC t-stat approximation
    from scipy import stats
    ic_df_pd = ic_df.to_pandas()
    ic_df_pd['p_value'] = 2 * (1 - stats.norm.cdf(abs(ic_df_pd['ir'] * (ic_df_pd['n_obs'] ** 0.5))))

    fdr_df = fdr_report(ic_df_pd, pval_col='p_value', alpha=FDR_ALPHA)
    fdr_pl = pl.from_pandas(fdr_df)
    fdr_pl.write_parquet(f'{OUTPUT_DIR}/fdr_report.parquet')

    print(f'  BH survivors:  {fdr_pl["survives_bh"].sum()} / {len(fdr_pl)}')
    print(f'  BY survivors:  {fdr_pl["survives_by"].sum()} / {len(fdr_pl)}')
    print(f'  q-value survivors: {fdr_pl["survives_q"].sum()} / {len(fdr_pl)} (q < {FDR_ALPHA})')

    # --- Select signals for combiner (q-value < alpha) ---
    selected = fdr_pl.filter(pl.col('survives_q'))['signal'].to_list()
    if MIN_IC_IR > 0:
        selected = [s for s in selected if ic_df.filter(pl.col('signal') == s)['ir'][0] >= MIN_IC_IR]

    print(f'\n  Selected for combiner: {selected}')

    # --- Save report ---
    report = {
        'signal_tag': signal_tag,
        'input_file': INPUT_FILE,
        'fwd_ret_col': FWD_RET_COL,
        'n_signals': len(signal_cols),
        'fdr_alpha': FDR_ALPHA,
        'selected_signals': selected,
        'ic_summary': ic_df.to_dicts(),
        'fdr_summary': {
            'bh': int(fdr_pl['survives_bh'].sum()),
            'by': int(fdr_pl['survives_by'].sum()),
            'qvalue': int(fdr_pl['survives_q'].sum()),
        },
    }
    with open(f'{OUTPUT_DIR}/signal_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    with open(f'{OUTPUT_DIR}/selected_signals.json', 'w') as f:
        json.dump({'signals': selected, 'fwd_ret_col': FWD_RET_COL}, f, indent=2)

    print(f'\n✓ Evaluation complete. Results in {OUTPUT_DIR}/')
    print(f'  Selected signals saved to {OUTPUT_DIR}/selected_signals.json')
    print(f'\nNext: python scripts/pipeline/04c_fit_combiner.py --signal-tag {signal_tag}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--signal-tag', type=str, default=SIGNAL_TAG, help='Signal tag from 04a')
    args = parser.parse_args()
    main(args.signal_tag)