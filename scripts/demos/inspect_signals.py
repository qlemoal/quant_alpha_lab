#!/usr/bin/env python
'''
===============================================
Inspect Pipeline Outputs - Signal Analysis
===============================================

Comprehensive inspection of signal runs with visualization and comparison.

Usage (with 2026-10-29_default as run ID):

    # List all runs
    python scripts/demos/inspect_signals.py list

    # Signal overview
    python scripts/demos/inspect_signals.py signals --run 2026-10-29_default

    # Detailed stats from config
    python scripts/demos/inspect_signals.py stats --run 2026-10-29_default

    # IC analysis (with optional plots)
    python scripts/demos/inspect_signals.py ic --run 2026-10-29_default --horizon 20 --plot

    # Decile long-short performance
    python scripts/demos/inspect_signals.py perf --run 2026-10-29_default --horizon 20 --plot

    # Turnover/stability
    python scripts/demos/inspect_signals.py turnover --run 2026-10-29_default --plot

    # Compare runs side-by-side
    python scripts/demos/inspect_signals.py compare --runs 2026-10-29_default 2026-10-29_mom_rev

    # Everything at once
    python scripts/demos/inspect_signals.py all --run 2026-10-29_default --horizon 20 --plot
'''

import argparse
import json
from pathlib import Path
from datetime import datetime

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from src.evaluation.signals.ic.core import compute_ic, summarize_ic
from src.evaluation.signals.performance import (
    decile_longshort_returns, sharpe_ratio, max_drawdown, compute_stability
)
from src.utils.helpers import file_hash


# New structure: processed/{step}/{run_id}/{data_file}.parquet + config.json
SIGNALS_BASE = Path('data/processed/signals')
SIGNAL_EVAL_BASE = Path('data/processed/signal_eval')
EN_SCORES_BASE = Path('data/processed/en_scores')
RISK_MODEL_BASE = Path('data/processed/risk_model')
WEIGHTS_BASE = Path('data/processed/weights')
BACKTEST_BASE = Path('data/processed/backtest')


def list_runs():
    '''List all available signal runs with key metadata.'''
    runs = []
    if SIGNALS_BASE.exists():
        for d in sorted(SIGNALS_BASE.iterdir()):
            if d.is_dir():
                config_file = d / 'config.json'
                signals_file = d / 'signals.parquet'
                if config_file.exists() and signals_file.exists():
                    with open(config_file) as f:
                        cfg = json.load(f)
                    runs.append({
                        'run_id': d.name,
                        'tag': cfg.get('signal_tag', 'unknown'),
                        'date': cfg.get('run_date', 'unknown'),
                        'n_signals': cfg.get('n_signals', 0),
                        'n_rows': cfg.get('n_rows', 0),
                        'git_commit': cfg.get('git_commit', '')[:8] if cfg.get('git_commit') else 'N/A',
                        'signals': [s[0] if isinstance(s, list) else s for s in cfg.get('signal_specs', [])],
                    })
    
    if not runs:
        print('No signal runs found.')
        return
    
    print(f'{"RUN ID":<25} {"TAG":<15} {"DATE":<12} {"SIGNALS":<20} {"ROWS":>10} {"COMMIT":<10}')
    print('-' * 95)
    for r in runs:
        sigs = ', '.join(r['signals'][:3])
        if len(r['signals']) > 3:
            sigs += f' +{len(r["signals"])-3} more'
        print(f'{r["run_id"]:<25} {r["tag"]:<15} {r["date"]:<12} {sigs:<20} {r["n_rows"]:>10,} {r["git_commit"]:<10}')


def find_run_dir(base: Path, run_id: str) -> Path | None:
    """Find the run directory for a given base path and run_id."""
    if not base.exists():
        return None
    exact = base / run_id
    if exact.exists():
        return exact
    # Fallback: find by tag suffix
    tag = run_id.split('_', 1)[1] if '_' in run_id else run_id
    matches = list(base.glob(f'*_{tag}'))
    if matches:
        return sorted(matches)[-1]
    return None


def load_run(run_id: str):
    '''Load signals data and config for a run.'''
    run_dir = find_run_dir(SIGNALS_BASE, run_id)
    if not run_dir:
        raise ValueError(f'Run not found: {run_id} in {SIGNALS_BASE}')
    
    config_file = run_dir / 'config.json'
    signals_file = run_dir / 'signals.parquet'
    
    if not config_file.exists():
        raise ValueError(f'Config not found: {config_file}')
    if not signals_file.exists():
        raise ValueError(f'Signals file not found: {signals_file}')
    
    with open(config_file) as f:
        config = json.load(f)
    
    df = pl.read_parquet(signals_file)
    return config, df


def load_signal_eval(run_id: str):
    '''Load signal evaluation data for a run.'''
    run_dir = find_run_dir(SIGNAL_EVAL_BASE, run_id)
    if not run_dir:
        raise ValueError(f'Signal eval run not found: {run_id} in {SIGNAL_EVAL_BASE}')
    
    config_file = run_dir / 'config.json'
    # Could also load IC data, FDR data, decay data
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        return config
    return {}


def get_signal_cols(df):
    '''Get signal column names (exclude date, ticker, forward returns).'''
    return [c for c in df.columns if c not in {'date', 'ticker'} and not c.startswith('fwdret')]


def get_fwd_cols(df):
    '''Get forward return column names.'''
    return [c for c in df.columns if c.startswith('fwdret')]


def cmd_signals(args):
    '''Show signal overview.'''
    config, df = load_run(args.run)
    sig_cols = get_signal_cols(df)
    fwd_cols = get_fwd_cols(df)
    
    print(f'\n=== SIGNALS: {args.run} ===')
    print(f'Config: {config.get("signal_tag", "unknown")} | {config.get("run_date", "unknown")} | {config.get("run_timestamp", "unknown")}')
    print(f'Rows: {len(df):,} | Signals: {len(sig_cols)} | Fwd returns: {len(fwd_cols)}')
    print(f'Date range: {df["date"].min()} to {df["date"].max()}')
    print(f'Tickers: {df["ticker"].n_unique()}')
    print(f'\nSignal columns: {sig_cols}')
    print(f'Fwd return columns: {fwd_cols}')
    
    print('\n--- Sample (first 10 rows) ---')
    print(df.head(10))
    
    print('\n--- Signal distributions ---')
    for sig in sig_cols:
        s = df[sig]
        print(f'{sig:25s} mean={s.mean():.4f} std={s.std():.4f} min={s.min():.4f} max={s.max():.4f} nulls={s.null_count():,} ({s.null_count()/len(s)*100:.2f}%)')


def cmd_stats(args):
    '''Show detailed statistics from config.'''
    config, df = load_run(args.run)
    
    print(f'\n=== DETAILED STATS: {args.run} ===')
    
    # Signal stats
    print('\n--- Signal Statistics ---')
    for sig, stats in config.get('signal_stats', {}).items():
        print(f'{sig:25s} mean={stats["mean"]:.4f} std={stats["std"]:.4f} '
              f'min={stats["min"]:.4f} max={stats["max"]:.4f} '
              f'nulls={stats["null_count"]} ({stats["null_pct"]:.2f}%)')
    
    # Cross-sectional stats
    print('\n--- Cross-Sectional Statistics ---')
    for sig, stats in config.get('cross_sectional_stats', {}).items():
        print(f'{sig:25s} avg_cs_mean={stats["avg_cs_mean"]:.4f} avg_cs_std={stats["avg_cs_std"]:.4f}')
    
    # IC stats
    print('\n--- IC Statistics (fwdret20) ---')
    for sig, stats in config.get('ic_stats', {}).items():
        if 'error' in stats:
            print(f'{sig:25s} ERROR: {stats["error"]}')
        else:
            print(f'{sig:25s} mean={stats["mean"]:.4f} std={stats["std"]:.4f} '
                  f'IR={stats["ir"]:.4f} hit_rate={stats["hit_rate"]:.2%} n_days={int(stats["n_days"])}')
    
    # Performance stats
    print('\n--- Performance Statistics (decile LS, fwdret20) ---')
    for sig, stats in config.get('performance_stats', {}).items():
        if 'error' in stats:
            print(f'{sig:25s} ERROR: {stats["error"]}')
        else:
            print(f'{sig:25s} Sharpe={stats["sharpe"]:.3f} MaxDD={stats["max_dd"]:.3f} '
                  f'Turnover={stats["turnover_mean"]:.4f}±{stats["turnover_std"]:.4f} '
                  f'SpreadMean={stats["spread_mean"]:.6f}')


def cmd_ic(args):
    '''Compute and show IC for signals.'''
    config, df = load_run(args.run)
    sig_cols = get_signal_cols(df)
    fwd_col = f'fwdret{args.horizon}'
    
    if fwd_col not in df.columns:
        print(f'Forward return column {fwd_col} not found. Available: {get_fwd_cols(df)}')
        return
    
    print(f'\n=== IC Analysis (horizon={args.horizon}d) ===')
    run_dir = find_run_dir(SIGNALS_BASE, args.run)
    if not run_dir:
        print(f'Run directory not found: {args.run}')
        return
    lf = pl.scan_parquet(run_dir / 'signals.parquet')
    
    for sig in sig_cols:
        ic_lf = compute_ic(lf, sig, forward_return_col=fwd_col)
        stats = summarize_ic(ic_lf)
        
        print(f'\n{sig}:')
        print(f'  Mean IC: {stats["mean"]:.4f}')
        print(f'  Std IC:  {stats["std"]:.4f}')
        print(f'  IR:      {stats["ir"]:.4f}')
        print(f'  t-stat:  {stats["t_stat"]:.2f}')
        print(f'  Hit rate: {stats["hit_rate"]:.2%}')
        print(f'  n_days:  {int(stats["n_days"])}')
        
        # Show IC time series stats
        ic_df = ic_lf.collect()
        print(f'  IC range: [{ic_df["ic"].min():.4f}, {ic_df["ic"].max():.4f}]')
        
        if args.plot:
            # Plot IC time series
            ic_ts = ic_df.to_pandas().set_index('date')['ic']
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            ic_ts.plot(ax=axes[0, 0], title=f'IC Time Series: {sig}')
            ic_ts.cumsum().plot(ax=axes[0, 1], title='Cumulative IC')
            ic_ts.rolling(60).mean().plot(ax=axes[1, 0], title='60-day Rolling Mean IC')
            ic_ts.rolling(60).apply(lambda x: x.mean()/x.std() if x.std() > 0 else 0).plot(
                ax=axes[1, 1], title='60-day Rolling IR')
            plt.tight_layout()
            plt.savefig(f'tmp/ic_{sig}.png', dpi=150)
            plt.close()
            print(f'  Plot saved to tmp/ic_{sig}.png')


def cmd_perf(args):
    '''Show decile long-short performance.'''
    config, df = load_run(args.run)
    sig_cols = get_signal_cols(df)
    fwd_col = f'fwdret{args.horizon}'
    
    if fwd_col not in df.columns:
        print(f'Forward return column {fwd_col} not found. Available: {get_fwd_cols(df)}')
        return
    
    print(f'\n=== Decile Long-Short Performance (horizon={args.horizon}d) ===')
    run_dir = find_run_dir(SIGNALS_BASE, args.run)
    if not run_dir:
        print(f'Run directory not found: {args.run}')
        return
    lf = pl.scan_parquet(run_dir / 'signals.parquet')
    
    for sig in sig_cols:
        ls_lf = decile_longshort_returns(lf, sig, fwd_ret_col=fwd_col)
        ls_df = ls_lf.collect().to_pandas().set_index('date')
        
        if len(ls_df) == 0:
            print(f'{sig}: No data')
            continue
        
        sharpe = sharpe_ratio(ls_df['spread_ret'].values)
        dd = max_drawdown(ls_df['spread_ret'].values)
        
        print(f'\n{sig}:')
        print(f'  Spread mean: {ls_df["spread_ret"].mean():.6f}')
        print(f'  Spread std:  {ls_df["spread_ret"].std():.6f}')
        print(f'  Sharpe:      {sharpe:.3f}')
        print(f'  Max DD:      {dd:.3f}')
        print(f'  n_days:      {len(ls_df)}')
        print(f'  Long mean:   {ls_df["long_ret"].mean():.6f}')
        print(f'  Short mean:  {ls_df["short_ret"].mean():.6f}')
        
        if args.plot:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            ls_df['spread_ret'].plot(ax=axes[0, 0], title=f'Daily Spread Return: {sig}')
            ls_df['spread_ret'].cumsum().plot(ax=axes[0, 1], title='Cumulative Return')
            ls_df[['long_ret', 'short_ret']].plot(ax=axes[1, 0], title='Long/Short Leg Returns')
            ls_df['spread_ret'].rolling(60).apply(lambda x: sharpe_ratio(x.values)).plot(
                ax=axes[1, 1], title='60-day Rolling Sharpe')
            plt.tight_layout()
            plt.savefig(f'tmp/perf_{sig}.png', dpi=150)
            plt.close()
            print(f'  Plot saved to tmp/perf_{sig}.png')


def cmd_turnover(args):
    '''Show signal turnover/stability.'''
    config, df = load_run(args.run)
    sig_cols = get_signal_cols(df)
    
    print(f'\n=== Signal Turnover (Stability) ===')
    run_dir = find_run_dir(SIGNALS_BASE, args.run)
    if not run_dir:
        print(f'Run directory not found: {args.run}')
        return
    lf = pl.scan_parquet(run_dir / 'signals.parquet')
    
    for sig in sig_cols:
        turnover_lf = compute_stability(lf, sig)
        turnover_df = turnover_lf.collect().to_pandas().set_index('date')
        
        if len(turnover_df) == 0:
            print(f'{sig}: No data')
            continue
        
        print(f'\n{sig}:')
        print(f'  Mean turnover:   {turnover_df["turnover"].mean():.4f}')
        print(f'  Std turnover:    {turnover_df["turnover"].std():.4f}')
        print(f'  Median turnover: {turnover_df["turnover"].median():.4f}')
        print(f'  Min turnover:    {turnover_df["turnover"].min():.4f}')
        print(f'  Max turnover:    {turnover_df["turnover"].max():.4f}')
        
        if args.plot:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            turnover_df['turnover'].plot(ax=axes[0], title=f'Daily Turnover: {sig}')
            turnover_df['turnover'].rolling(60).mean().plot(ax=axes[1], title='60-day Rolling Mean Turnover')
            plt.tight_layout()
            plt.savefig(f'tmp/turnover_{sig}.png', dpi=150)
            plt.close()
            print(f'  Plot saved to tmp/turnover_{sig}.png')


def cmd_compare(args):
    '''Compare multiple signal runs side by side.'''
    runs_data = []
    for run_id in args.runs:
        config, df = load_run(run_id)
        runs_data.append((run_id, config, df))
    
    print(f'\n=== COMPARISON: {len(runs_data)} runs ===')
    
    # Compare configs
    print('\n--- Config Comparison ---')
    for run_id, config, df in runs_data:
        sig_cols = get_signal_cols(df)
        print(f'{run_id}:')
        print(f'  Tag: {config.get("signal_tag")}, Date: {config.get("run_date")}')
        print(f'  Signals: {sig_cols}')
        print(f'  Rows: {len(df):,}, Tickers: {df["ticker"].n_unique()}')
    
    # Compare IC stats
    print('\n--- IC Comparison (fwdret20) ---')
    header = f'{"Signal":<25}'
    for run_id, _, _ in runs_data:
        header += f'  {run_id[:20]:<22}'
    print(header)
    print('-' * (25 + 24 * len(runs_data)))
    
    # Get all unique signals across runs
    all_signals = set()
    for _, config, df in runs_data:
        for spec in config.get('signal_specs', []):
            feat = spec[0] if isinstance(spec, list) else spec
            method = spec[1] if isinstance(spec, list) else 'zscore_tanh'
            all_signals.add(f'{feat}_{method}')
    
    for sig in sorted(all_signals):
        row = f'{sig:<25}'
        for run_id, config, _ in runs_data:
            ic = config.get('ic_stats', {}).get(sig, {})
            if 'error' in ic:
                row += f'  {"ERROR":<22}'
            elif ic:
                row += f'  IR={ic.get("ir", 0):.4f}  HR={ic.get("hit_rate", 0):.2%}'
            else:
                row += f'  {"N/A":<22}'
        print(row)
    
    # Compare Performance stats
    print('\n--- Performance Comparison (fwdret20) ---')
    header = f'{"Signal":<25}'
    for run_id, _, _ in runs_data:
        header += f'  {run_id[:20]:<22}'
    print(header)
    print('-' * (25 + 24 * len(runs_data)))
    
    for sig in sorted(all_signals):
        row = f'{sig:<25}'
        for run_id, config, _ in runs_data:
            perf = config.get('performance_stats', {}).get(sig, {})
            if 'error' in perf:
                row += f'  {"ERROR":<22}'
            elif perf:
                row += f'  Sh={perf.get("sharpe", 0):.3f}  TD={perf.get("turnover_mean", 0):.4f}'
            else:
                row += f'  {"N/A":<22}'
        print(row)


def cmd_all(args):
    '''Run all inspections for a run.'''
    # Set default horizon for sub-commands that need it
    if not hasattr(args, 'horizon'):
        args.horizon = 20
    cmd_signals(args)
    cmd_stats(args)
    cmd_ic(args)
    cmd_perf(args)
    cmd_turnover(args)


def main():
    parser = argparse.ArgumentParser(description='Inspect signal pipeline outputs')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # List runs
    subparsers.add_parser('list', help='List all available signal runs')
    
    # Single run inspections
    for cmd in ['signals', 'stats', 'ic', 'perf', 'turnover', 'all']:
        p = subparsers.add_parser(cmd, help=f'Inspect {cmd}')
        p.add_argument('--run', required=True, help='Run ID (e.g., 2026-10-29_default)')
        if cmd in ['ic', 'perf', 'all']:
            p.add_argument('--horizon', type=int, default=20, help='Forward return horizon')
        if cmd in ['ic', 'perf', 'turnover', 'all']:
            p.add_argument('--plot', action='store_true', help='Generate plots to /tmp/')
    
    # Compare runs
    p = subparsers.add_parser('compare', help='Compare multiple runs')
    p.add_argument('--runs', nargs='+', required=True, help='Run IDs to compare')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_runs()
    elif args.command == 'signals':
        cmd_signals(args)
    elif args.command == 'stats':
        cmd_stats(args)
    elif args.command == 'ic':
        cmd_ic(args)
    elif args.command == 'perf':
        cmd_perf(args)
    elif args.command == 'turnover':
        cmd_turnover(args)
    elif args.command == 'compare':
        cmd_compare(args)
    elif args.command == 'all':
        cmd_all(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()