#!/usr/bin/env python
'''
===============================================
Phase 05: Evaluate Signals and Select (strict gates)
===============================================

Evaluates generated signals: Rank IC, Newey-West, IC decay, FDR (BH/BY/q-value), deflated Sharpe.
Then it ADMITS a signal to the combiner only if it passes every gate (src/evaluation/signals/selection.py):

    prior -> sign -> Newey-West t >= 3 -> non-overlapping t >= 2 -> q < 0.05 -> deflated Sharpe >= 0.95

The priors (family, sign, rationale) are written in src/signals/priors.py BEFORE the data are seen.
The default mode is 'strict'. The old rule (q < --alpha only) is kept as --mode legacy, for exploration:
a legacy selection is NOT evidence of a strategy.
Admitting no signal is a valid result. It means the data show no robust signal.

WHAT CHANGED vs the previous version (and why)
----------------------------------------------
1. NO HOLDOUT LEAKAGE.
   Before: IC, p-values and the selection used ALL dates, including the 200-date
   holdout that 06 uses as its final out-of-sample test.
   Now: only dates <= train_end are used. train_end is computed with the SAME rule as 06:
       train_end = unique_dates[-holdout_dates - embargo_w]
       embargo_w = (horizon + 1) + EMBARGO_WINDOW
   The gap between train_end and the holdout start (embargo_w dates) is larger than
   the label length, so no training label touches the holdout.
2. CORRECT AUTOCORRELATION CONTROL.
   Newey-West now gets horizon=h, so its lag covers the label overlap (see IC/metrics.py).
   A non-overlapping t-stat (one IC every h days) is added as a cross-check.
3. FDR gets the right numbers.
   Before: the Newey-West t-stat was replaced by the IR in the 'tstat' column.
   Now: 'tstat' = Newey-West t-stat. p-values come from the same t-stat.
4. No crash if a t-stat is missing (None): the p-value is NaN and the signal is ignored by FDR.

Inputs:
  data/clean/{asset}/{region}/{universe}/{freq}/signals/{run_id}/signals.parquet
  data/clean/{asset}/{region}/{universe}/{freq}/signals/{run_id}/config.json

Outputs:
  data/clean/{asset}/{region}/{universe}/{freq}/signal_eval/{run_id}/
    ├── ic_summary.parquet      # Per-signal IC stats (training dates only)
    ├── ic_decay.parquet        # IC vs horizon (training dates only)
    ├── fdr_report.parquet      # BH, BY, q-value per signal
    ├── selection_gates.parquet # one row per signal: statistics, one pass/fail per gate, first failed gate
    ├── signal_report.json      # Human-readable summary
    └── selected_signals.json   # Admitted signals + family and sign of each (for the combiner)

Usage:
    python scripts/pipeline/05_evaluate_signals.py --date 2026-11-06 --tag default --horizon 20
    python scripts/pipeline/05_evaluate_signals.py --date 2026-11-06 --tag default --mode legacy --alpha 0.2
'''

import argparse
import json
import math
import sys
from pathlib import Path

# Make `import src...` and `import config...` work wherever the script is started from.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import polars as pl
import numpy as np

from config.paths import get_paths, run_id_from_args
from config.constants import EMBARGO_WINDOW
from src.evaluation.signals.ic.core import compute_ic, summarize_ic
from src.evaluation.signals.ic.metrics import (
    ic_decay,
    newey_west_ic_tstat,
    ic_pvalue_nw,
    ic_tstat_non_overlapping,
)
from src.evaluation.signals.fdr import fdr_report
from src.evaluation.signals.benchmarks import benchmark_report, format_benchmark_report
from src.evaluation.signals.dsr import deflated_sharpe_table
from src.evaluation.signals.selection import SelectionConfig, apply_gates, gate_summary, config_dict
from src.signals.cheat import signal_group
from src.signals.priors import priors_table


def _fmt(x, spec):
    '''Format a number that may be None/NaN without crashing.'''
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 'nan'.rjust(len(format(0.0, spec)))
    return format(x, spec)


def main(date: str, tag: str = 'default', horizon: int = 20, alpha: float = 0.2,
         min_ir: float = 0.0, holdout_dates: int = 200, mode: str = 'strict',
         selection: "SelectionConfig | None" = None):
    if mode not in ('strict', 'legacy'):
        raise ValueError(f"mode must be 'strict' or 'legacy', got {mode!r}")
    selection = selection or SelectionConfig()
    run_id = run_id_from_args(date, tag)
    cfg = get_paths(run_id=run_id)

    signals_file = cfg.stage_dir('signals') / 'signals.parquet'
    config_file = cfg.stage_dir('signals') / 'config.json'

    if not signals_file.exists():
        raise FileNotFoundError(f'Signals file not found: {signals_file}. Run 04_generate_signals.py first.')

    with open(config_file) as f:
        config = json.load(f)

    signal_tag = config.get('signal_tag', 'unknown')

    signals_dir = cfg.stage_dir('signals')
    OUTPUT_DIR = cfg.stage_dir('signal_eval')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    FWD_RET_COL = f'fwdret{horizon}'

    # Same split rule as 06 (see src/models/elastic_net_combiner.py, run_en_combiner)
    purge_w = horizon + 1
    embargo_w = purge_w + EMBARGO_WINDOW

    print('=' * 70)
    print('PHASE 05: EVALUATE SIGNALS')
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
    print(f'Holdout dates: {holdout_dates}   embargo_w: {embargo_w}')
    print('=' * 70)

    # ------------------------------------------------------------------
    # [1/5] Load and cut the data: TRAINING DATES ONLY
    # ------------------------------------------------------------------
    print('\n[1/5] Loading signals...')
    df = pl.read_parquet(signals_file)
    print(f'  Loaded {len(df):,} rows, {df["ticker"].n_unique()} tickers')
    print(f'  Date range: {df["date"].min()} to {df["date"].max()}')

    # Signal columns = everything except date, ticker and the forward-return labels
    all_cols = df.columns
    exclude = {'date', 'ticker'} | {c for c in all_cols if c.startswith('fwdret')}
    signal_cols = [c for c in all_cols if c not in exclude]
    print(f'  Found {len(signal_cols)} signals: {signal_cols}')

    if FWD_RET_COL not in df.columns:
        available = [c for c in df.columns if c.startswith('fwdret')]
        raise ValueError(f'Forward return column {FWD_RET_COL} not found. Available: {available}')

    # Dates that have a label (the last h+1 dates have none)
    labelled_dates = df.filter(pl.col(FWD_RET_COL).is_not_null())['date'].unique().sort().to_list()
    if len(labelled_dates) <= holdout_dates + embargo_w + 100:
        raise ValueError(
            f'Not enough labelled dates ({len(labelled_dates)}) for holdout={holdout_dates} + embargo={embargo_w}.'
        )
    train_end = labelled_dates[-holdout_dates - embargo_w]
    holdout_start = labelled_dates[-holdout_dates]
    df = df.filter(pl.col('date') <= train_end)          # <-- the leakage fix: nothing after train_end is used below
    n_train_dates = df['date'].n_unique()
    print(f'  TRAINING dates only: {df["date"].min()} to {train_end}  ({n_train_dates} dates)')
    print(f'  Excluded: embargo gap + holdout (holdout starts {holdout_start})')

    # ------------------------------------------------------------------
    # [2/5] IC per signal
    # ------------------------------------------------------------------
    print('\n[2/5] Computing Rank IC...')
    ic_results = []
    ic_series = {}                                        # daily IC of each signal, for the deflated Sharpe
    for col in signal_cols:
        ic_df_col = compute_ic(df.lazy(), col, FWD_RET_COL).collect()   # one IC per date
        summary = summarize_ic(ic_df_col)
        ic_values = ic_df_col['ic'].to_numpy()
        ic_series[col] = ic_values

        # Main test: Newey-West with lag >= 2(h-1), covers the label overlap
        nw_result = newey_west_ic_tstat(ic_values, horizon=horizon)
        # Cross-check: one IC every h days, plain t-test
        no_result = ic_tstat_non_overlapping(ic_values, horizon=horizon)

        ic_results.append({
            'signal': col,
            'mean_ic': summary['mean'],
            'std_ic': summary['std'],
            'ir': summary['ir'],
            'hit_rate': summary['hit_rate'],
            'n_obs': summary['n_days'],
            'nw_tstat': nw_result.get('t_stat'),
            'nw_lag': nw_result.get('max_lag'),
            'p_value': ic_pvalue_nw(nw_result),                 # None if the t-stat is missing
            'nonoverlap_tstat': no_result.get('t_stat'),
        })

    ic_df = pl.DataFrame(ic_results, infer_schema_length=None)
    print('  IC Summary (top 10 by |IR|):')
    top = ic_df.with_columns(pl.col('ir').abs().alias('_abs_ir')).sort('_abs_ir', descending=True, nulls_last=True).head(10)
    for row in top.iter_rows(named=True):
        print(f'    {row["signal"]:25s} IC={_fmt(row["mean_ic"], "7.4f")} IR={_fmt(row["ir"], "7.3f")} '
              f'HR={_fmt(row["hit_rate"], ".2%")} NW_t={_fmt(row["nw_tstat"], "6.2f")} '
              f'nonoverlap_t={_fmt(row["nonoverlap_tstat"], "6.2f")}')

    ic_df.write_parquet(OUTPUT_DIR / 'ic_summary.parquet')

    # ------------------------------------------------------------------
    # [3/5] IC decay (diagnostic only, not used for selection)
    # ------------------------------------------------------------------
    print('\n[3/5] Computing IC decay...')
    decay_horizons = [1, 5, 10, 20, 60]
    available_horizons = [h for h in decay_horizons if f'fwdret{h}' in df.columns]
    decay_results = []
    for col in signal_cols:
        d = ic_decay(df.lazy(), col, available_horizons)
        decay_results.append(d.with_columns(pl.lit(col).alias('signal')))

    decay_df = pl.concat(decay_results)
    decay_df.write_parquet(OUTPUT_DIR / 'ic_decay.parquet')
    print(f'  Decay computed for {len(signal_cols)} signals across {len(available_horizons)} horizons')

    # ------------------------------------------------------------------
    # [4/5] FDR across all signals tested in this round
    # ------------------------------------------------------------------
    print('\n[4/5] Computing FDR...')
    if len(signal_cols) < 20:
        print(f'  NOTE: only {len(signal_cols)} signals. Storey pi0 needs >= 20 p-values, '
              'otherwise pi0 = 1 and the q-values equal the BH adjusted p-values.')
    p_values = [float('nan') if p is None else p for p in ic_df['p_value'].to_list()]
    t_stats = [float('nan') if t is None else t for t in ic_df['nw_tstat'].to_list()]

    fdr_pl = fdr_report(
        signal_names=ic_df['signal'].to_list(),
        tstats=t_stats,                      # the Newey-West t-stat (NOT the IR)
        pvalues=p_values,
        alpha=alpha,
    )
    fdr_pl.write_parquet(OUTPUT_DIR / 'fdr_report.parquet')

    # Benchmark signals (cheat, placebo)? Then also test the real signals ALONE, to see what the benchmarks change.
    fdr_real = None
    real_names = [s for s in ic_df['signal'].to_list() if signal_group(s) == 'real']
    if len(real_names) < ic_df.height:
        keep = ic_df['signal'].is_in(real_names)
        fdr_real = fdr_report(
            signal_names=ic_df.filter(keep)['signal'].to_list(),
            tstats=[float('nan') if x is None else x for x in ic_df.filter(keep)['nw_tstat'].to_list()],
            pvalues=[float('nan') if x is None else x for x in ic_df.filter(keep)['p_value'].to_list()],
            alpha=alpha,
        )
    bench = benchmark_report(fdr_pl, fdr_real, alpha)

    n_bh = int(fdr_pl['survives_bh'].sum())
    n_by = int(fdr_pl['survives_by'].sum())
    n_q = int((fdr_pl['qvalue'] < alpha).sum())
    print(f'  BH survivors:      {n_bh:3d} / {len(fdr_pl)}')
    print(f'  BY survivors:      {n_by:3d} / {len(fdr_pl)}')
    print(f'  q-value survivors: {n_q:3d} / {len(fdr_pl)} (q < {alpha})')
    if bench:
        print()
        for line in format_benchmark_report(bench):
            print(line)

    # ------------------------------------------------------------------
    # [5/5] Selection for the combiner
    # ------------------------------------------------------------------
    names = ic_df['signal'].to_list()
    meta = priors_table(names)                                          # family, sign, rationale (ex ante)
    signs = {k: (v['sign'] if v else None) for k, v in meta.items()}
    dsr_tab = deflated_sharpe_table(ic_series, signs, step=horizon, n_trials_extra=selection.n_trials_extra,
                                   var_mode=selection.dsr_variance)
    qmap = dict(zip(fdr_pl['signal'].to_list(), fdr_pl['qvalue'].to_list()))
    stats = []
    for row in ic_df.iter_rows(named=True):
        k = row['signal']
        stats.append({
            'signal': k, 'sign': signs[k], 'family': meta[k]['family'] if meta[k] else None,
            'mean_ic': row['mean_ic'], 'nw_tstat': row['nw_tstat'],
            'nonoverlap_tstat': row['nonoverlap_tstat'], 'qvalue': qmap.get(k), 'dsr': dsr_tab[k]['dsr'],
        })
    gates = apply_gates(stats, selection)
    gates.write_parquet(OUTPUT_DIR / 'selection_gates.parquet')
    summary = gate_summary(gates)

    if mode == 'strict':
        print(f'  STRICT selection: {config_dict(selection)}')
        print(f'  Trials in the deflated Sharpe: N = {len(signal_cols) + selection.n_trials_extra}, '
              f'expected best luck SR0 = {next(iter(dsr_tab.values()))["sr0"]:.3f} per observation')
        print('  Signals stopped by each gate: ' + ', '.join(f'{g} {n}' for g, n in summary.items() if g not in ('admitted', 'tested')))
        selected = gates.filter(pl.col('selected'))['signal'].to_list()
    else:
        print(f'  LEGACY selection: q < {alpha} only. Exploration. This is NOT evidence of a strategy.')
        selected = fdr_pl.filter(pl.col('qvalue') < alpha)['signal'].to_list()
        if min_ir > 0:
            before = len(selected)
            selected = [s for s in selected if abs(ic_df.filter(pl.col('signal') == s)['ir'][0]) >= min_ir]
            print(f'  After min_ic_ir filter: {len(selected)} (removed {before - len(selected)})')

    print('\n  Gate table (aligned t = t x prior sign):')
    for row in gates.sort('aligned_nw_t', descending=True, nulls_last=True).head(15).iter_rows(named=True):
        flags = ''.join('Y' if row[f'pass_{g}'] else '.' for g in ('prior', 'sign', 't_nw', 't_nonov', 'fdr', 'dsr'))
        print(f'    {row["signal"]:28s} [{flags}] t_nw {_fmt(row["aligned_nw_t"], "6.2f")}  t_no {_fmt(row["aligned_nonoverlap_t"], "6.2f")}'
              f'  q {_fmt(row["qvalue"], "6.3f")}  DSR {_fmt(row["dsr"], "5.2f")}  {row["failed_gate"] or "ADMITTED"}')
    print('    flags: prior sign t_nw t_nonov fdr dsr   (Y pass, . fail)')

    # Warn when the two tests disagree: sign of a possible overlap problem
    disagree = ic_df.filter(
        pl.col('signal').is_in(selected)
        & (pl.col('nonoverlap_tstat').abs() < 1.0)
    )['signal'].to_list()
    if disagree:
        print(f'  WARNING: selected but non-overlapping |t| < 1: {disagree}')

    if not selected:
        print('  NO SIGNAL ADMITTED. 06 will stop. This is a valid result: no signal passed every gate.')
        print('  To explore anyway (not evidence): --mode legacy, or lower a threshold on purpose and say so.')

    print(f'\n  Selected for combiner: {selected}')

    # --- Save report ---
    report = {
        'run_id': run_id,
        'signal_tag': signal_tag,
        'input_dir': str(signals_dir),
        'fwd_ret_col': FWD_RET_COL,
        'horizon': horizon,
        'n_signals': len(signal_cols),
        'fdr_alpha': alpha,
        'selection_mode': mode,
        'selection_config': config_dict(selection),
        'gate_summary': summary,
        'gates': gates.to_dicts(),
        'min_ic_ir': min_ir,
        'train_start': str(df['date'].min()),
        'train_end': str(train_end),
        'holdout_start': str(holdout_start),
        'holdout_dates': holdout_dates,
        'embargo_w': embargo_w,
        'n_train_dates': n_train_dates,
        'selected_signals': selected,
        'ic_summary': ic_df.to_dicts(),
        'fdr_summary': {'bh': n_bh, 'by': n_by, 'qvalue': n_q},
        'benchmark_report': bench,
        'config': config,
    }
    with open(OUTPUT_DIR / 'signal_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    with open(OUTPUT_DIR / 'selected_signals.json', 'w') as f:
        json.dump({
            'signals': selected,
            'mode': mode,                       # 06 uses the family combiner for 'strict', one-per-signal for 'legacy'
            'signal_meta': {k: meta[k] for k in selected},    # family, sign, rationale of each admitted signal
            'selection_config': config_dict(selection),
            'fwd_ret_col': FWD_RET_COL,
            'horizon': horizon,
            'train_end': str(train_end),        # 06 checks that it computes the same cut
            'holdout_start': str(holdout_start),
            'holdout_dates': holdout_dates,     # 06 reads these: one source of truth for the split
            'embargo_w': embargo_w,
        }, f, indent=2)

    print(f'\n✓ Evaluation complete. Results in {OUTPUT_DIR}/')
    print(f'  Selected signals saved to {OUTPUT_DIR}/selected_signals.json')
    print(f'\nNext: python scripts/pipeline/06_fit_combiner.py --date {date} --tag {tag}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate signals for combiner selection')
    parser.add_argument('--date', required=True, help='Date in YYYY-MM-DD format (e.g., 2026-10-29)')
    parser.add_argument('--tag', type=str, default='default', help='Signal tag (default: default)')
    parser.add_argument('--horizon', type=int, default=20, help='Forward return horizon for evaluation')
    parser.add_argument('--mode', choices=['strict', 'legacy'], default='strict',
                        help="strict = all gates (default). legacy = q < --alpha only (exploration, not evidence)")
    parser.add_argument('--alpha', type=float, default=0.2, help='FDR alpha for the report and for --mode legacy')
    parser.add_argument('--min_t', type=float, default=SelectionConfig.min_t, help='strict: min aligned Newey-West t')
    parser.add_argument('--min_nonoverlap_t', type=float, default=SelectionConfig.min_nonoverlap_t,
                        help='strict: min aligned non-overlapping t')
    parser.add_argument('--fdr_alpha', type=float, default=SelectionConfig.fdr_alpha, help='strict: q-value below this')
    parser.add_argument('--min_dsr', type=float, default=SelectionConfig.min_dsr, help='strict: min deflated Sharpe')
    parser.add_argument('--n_trials_extra', type=int, default=0,
                        help='strict: signals tried and removed earlier (they count as trials in the deflated Sharpe)')
    parser.add_argument('--dsr_variance', choices=['null', 'trials'], default='null',
                        help="strict: variance of skill-less Sharpe ratios in the DSR. null = 1/(T-1) (default), trials = across the run's signals (stricter)")
    parser.add_argument('--no_require_prior', action='store_true',
                        help='strict: let signals without a prior skip gates 1 and 2 (exploration only)')
    parser.add_argument('--min_ir', type=float, default=0.0, help='Minimum |IC IR| for signal selection')
    parser.add_argument('--holdout_dates', type=int, default=200,
                        help='Number of final dates reserved for the 06 holdout (must match 06)')
    args = parser.parse_args()
    sel = SelectionConfig(min_t=args.min_t, min_nonoverlap_t=args.min_nonoverlap_t, fdr_alpha=args.fdr_alpha,
                          min_dsr=args.min_dsr, require_prior=not args.no_require_prior,
                          n_trials_extra=args.n_trials_extra, dsr_variance=args.dsr_variance)
    main(args.date, args.tag, args.horizon, args.alpha, args.min_ir, args.holdout_dates, args.mode, sel)