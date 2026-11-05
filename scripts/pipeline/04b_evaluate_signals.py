#!/usr/bin/env python
'''
===============================================
Phase 04b: Evaluate Signals (IC, FDR, Decay)
===============================================

Evaluates generated signals: Rank IC, Newey-West, IC decay, FDR (BH/BY/q-value).
Produces a report and a filtered signal list for the combiner.

Inputs:
  data/clean/{asset}/{region}/{universe}/{freq}/signals/{run_id}/signals.parquet
  data/clean/{asset}/{region}/{universe}/{freq}/signals/{run_id}/config.json

Outputs:
  data/clean/{asset}/{region}/{universe}/{freq}/signal_eval/{run_id}/
    ├── ic_summary.parquet      # Per-signal IC stats
    ├── ic_decay.parquet        # IC vs horizon
    ├── fdr_report.parquet      # BH, BY, q-value per signal
    ├── signal_report.json      # Human-readable summary
    └── selected_signals.json   # List of signals passing FDR (for combiner)

Usage:
    Previous runs:
        python scripts/pipeline/04a_generate_signals.py --date 2026-10-29 --tag default
        
  python scripts/pipeline/04b_evaluate_signals.py --date 2026-10-29 --tag default --horizon 20 --alpha 0.2
'''

import argparse
import json
from pathlib import Path
import polars as pl
import numpy as np

from config.paths import get_data_config, run_id_from_args
from src.evaluation.signals.ic.core import compute_ic, summarize_ic
from src.evaluation.signals.ic.metrics import ic_decay
from src.evaluation.signals.ic.metrics import newey_west_ic_tstat
from src.evaluation.signals.fdr import fdr_report


def main(date: str, tag: str = 'default', horizon: int = 20, alpha: float = 0.2, min_ir: float = 0.0):
    run_id = run_id_from_args(date, tag)
    
    # Use centralized config
    cfg = get_data_config(run_id=run_id)
    
    signal_dir = cfg.signals_dir
    if not signal_dir.exists():
        raise FileNotFoundError(f'Signal directory not found: {signal_dir}')

    signals_file = cfg.signals_file
    config_file = cfg.signals_config_file

    if not signals_file.exists():
        raise FileNotFoundError(f'Signals file not found: {signals_file}. Run 04a_generate_signals.py first.')

    # Load config for metadata
    with open(config_file) as f:
        config = json.load(f)

    signal_tag = config.get('signal_tag', 'unknown')

    OUTPUT_DIR = cfg.signal_eval_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    FWD_RET_COL = f'fwdret{horizon}'

    print('=' * 70)
    print('PHASE 04b: EVALUATE SIGNALS')
    print('=' * 70)
    print(f'Run ID:        {run_id}')
    print(f'Date:          {date}')
    print(f'Tag:           {tag}')
    print(f'Signal tag:    {signal_tag}')
    print(f'Input:         {signals_file}')
    print(f'Output dir:    {OUTPUT_DIR}')
    print(f'FWD_RET_COL:   {FWD_RET_COL}')
    print(f'FDR alpha:     {alpha}')
    print(f'Min IC IR:     {min_ir}')
    print('=' * 70)

    # Load signals
    print('\n[1/5] Loading signals...')
    df = pl.read_parquet(signals_file)
    print(f'  Loaded {len(df):,} rows, {df["ticker"].n_unique()} tickers')
    print(f'  Date range: {df["date"].min()} to {df["date"].max()}')

    # Identify signal columns (exclude date, ticker, fwdret*)
    all_cols = df.columns
    exclude = {'date', 'ticker'} | {c for c in all_cols if c.startswith('fwdret')}
    signal_cols = [c for c in all_cols if c not in exclude]
    print(f'  Found {len(signal_cols)} signals: {signal_cols}')

    # Verify forward return column exists
    if FWD_RET_COL not in df.columns:
        available = [c for c in df.columns if c.startswith('fwdret')]
        raise ValueError(f'Forward return column {FWD_RET_COL} not found. Available: {available}')

    # --- IC per signal ---
    print('\n[2/5] Computing Rank IC...')
    ic_results = []
    for col in signal_cols:
        ic_series = compute_ic(df.lazy(), col, FWD_RET_COL)
        summary = summarize_ic(ic_series)
        # Extract IC values for Newey-West (collect LazyFrame and get 'ic' column)
        ic_df = ic_series.collect()
        ic_values = ic_df['ic'].to_numpy()
        nw_result = newey_west_ic_tstat(ic_values)
        nw_tstat = nw_result.get('t_stat')

        ic_results.append({
            'signal': col,
            'mean_ic': summary['mean'],
            'std_ic': summary['std'],
            'ir': summary['ir'],
            'hit_rate': summary['hit_rate'],
            'n_obs': summary['n_days'],
            'nw_tstat': nw_tstat,
        })

    ic_df = pl.DataFrame(ic_results)
    print(f'  IC Summary (top 10 by |IR|):')
    for row in ic_df.sort('ir', descending=True).head(10).iter_rows(named=True):
        print(f'    {row["signal"]:25s} IC={row["mean_ic"]:7.4f} IR={row["ir"]:7.3f} HR={row["hit_rate"]:.2%} NW_t={row["nw_tstat"]:6.2f}')

    ic_df.write_parquet(OUTPUT_DIR / 'ic_summary.parquet')

    # --- IC Decay ---
    print('\n[3/5] Computing IC decay...')
    decay_horizons = [1, 5, 10, 20, 60]  # Extended horizons
    # Filter to only horizons that exist in the data
    available_horizons = [h for h in decay_horizons if f'fwdret{h}' in df.columns]
    decay_results = []
    for col in signal_cols:
        decay_df = ic_decay(df.lazy(), col, available_horizons)
        # Add signal column to decay DataFrame
        decay_df = decay_df.with_columns(pl.lit(col).alias('signal'))
        decay_results.append(decay_df)

    decay_df = pl.concat(decay_results)
    decay_df.write_parquet(OUTPUT_DIR / 'ic_decay.parquet')
    print(f'  Decay computed for {len(signal_cols)} signals across {len(available_horizons)} horizons')

    # --- FDR ---
    print('\n[4/5] Computing FDR...')
    from scipy import stats
    ic_df_pd = ic_df.to_pandas()
    # Two-sided p-value from Newey-West t-stat (corrects for IC autocorrelation)
    ic_df_pd['p_value'] = 2 * (1 - stats.norm.cdf(abs(ic_df_pd['nw_tstat'])))

    fdr_df = fdr_report(
        signal_names=ic_df_pd['signal'].tolist(),
        tstats=ic_df_pd['ir'].tolist(),
        pvalues=ic_df_pd['p_value'].tolist(),
        alpha=alpha
    )
    fdr_pl = fdr_df  # fdr_report already returns polars DataFrame
    fdr_pl.write_parquet(OUTPUT_DIR / 'fdr_report.parquet')

    n_bh = int(fdr_pl['survives_bh'].sum())
    n_by = int(fdr_pl['survives_by'].sum())
    # fdr_report returns 'qvalue' column, not 'survives_q'
    n_q = int((fdr_pl['qvalue'] < alpha).sum())
    print(f'  BH survivors:     {n_bh:3d} / {len(fdr_pl)}')
    print(f'  BY survivors:     {n_by:3d} / {len(fdr_pl)}')
    print(f'  q-value survivors: {n_q:3d} / {len(fdr_pl)} (q < {alpha})')

    # --- Select signals for combiner ---
    selected = fdr_pl.filter(pl.col('qvalue') < alpha)['signal'].to_list()
    if min_ir > 0:
        before = len(selected)
        selected = [s for s in selected if ic_df.filter(pl.col('signal') == s)['ir'][0] >= min_ir]
        print(f'  After min_ic_ir filter: {len(selected)} (removed {before - len(selected)})')

    print(f'\n  Selected for combiner: {selected}')

    # --- Save report ---
    report = {
        'run_id': run_id,
        'signal_tag': signal_tag,
        'input_dir': str(signal_dir),
        'fwd_ret_col': FWD_RET_COL,
        'horizon': horizon,
        'n_signals': len(signal_cols),
        'fdr_alpha': alpha,
        'min_ic_ir': min_ir,
        'selected_signals': selected,
        'ic_summary': ic_df.to_dicts(),
        'fdr_summary': {
            'bh': n_bh,
            'by': n_by,
            'qvalue': n_q,
        },
        'config': config,  # Full upstream config for traceability
    }
    with open(OUTPUT_DIR / 'signal_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    with open(OUTPUT_DIR / 'selected_signals.json', 'w') as f:
        json.dump({'signals': selected, 'fwd_ret_col': FWD_RET_COL, 'horizon': horizon}, f, indent=2)

    print(f'\n✓ Evaluation complete. Results in {OUTPUT_DIR}/')
    print(f'  Selected signals saved to {OUTPUT_DIR}/selected_signals.json')
    print(f'\nNext: python scripts/pipeline/04c_fit_combiner.py --date {date} --tag {tag}')






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate signals for combiner selection')
    parser.add_argument('--date', required=True, help='Date in YYYY-MM-DD format (e.g., 2026-10-29)')
    parser.add_argument('--tag', type=str, default='default', help='Signal tag (default: default)')
    parser.add_argument('--horizon', type=int, default=20, help='Forward return horizon for evaluation')
    parser.add_argument('--alpha', type=float, default=0.2, help='FDR alpha threshold')
    parser.add_argument('--min_ir', type=float, default=0.0, help='Minimum IC IR for signal selection')
    args = parser.parse_args()
    main(args.date, args.tag, args.horizon, args.alpha, args.min_ir)