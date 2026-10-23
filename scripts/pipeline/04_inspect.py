
'''
===============================================
Inspect Pipeline Outputs
===============================================

Quick look at any phase output.

Usage:
  python scripts/pipeline/04_inspect.py --phase signals --tag default
  python scripts/pipeline/04_inspect.py --phase eval --tag default
  python scripts/pipeline/04_inspect.py --phase risk --tag default
  python scripts/pipeline/04_inspect.py --phase weights --tag default
  python scripts/pipeline/04_inspect.py --phase backtest --tag default
  python scripts/pipeline/04_inspect.py --all --tag default
'''

import argparse
from pathlib import Path
import polars as pl
import numpy as np


def inspect_signals(tag):
    f = Path(f'data/processed/signals_{tag}.parquet')
    if not f.exists(): return print(f'Not found: {f}')
    df = pl.read_parquet(f)
    print(f'\n=== SIGNALS ({tag}) ===')
    print(f'Rows: {len(df):,}, Cols: {len(df.columns)}')
    print(f'Date range: {df["date"].min()} to {df["date"].max()}')
    sig_cols = [c for c in df.columns if c not in {'date', 'ticker'} and not c.startswith('fwd_ret')]
    fwd_cols = [c for c in df.columns if c.startswith('fwd_ret')]
    print(f'Signals: {sig_cols}')
    print(f'Forward returns: {fwd_cols}')


def inspect_eval(tag):
    d = Path(f'data/exports/signal_eval_{tag}')
    if not d.exists(): return print(f'Not found: {d}')

    # IC summary
    ic_f = d / 'ic_summary.parquet'
    if ic_f.exists():
        df = pl.read_parquet(ic_f)
        print(f'\n=== IC SUMMARY ({tag}) ===')
        print(df.sort('ir', descending=True).head(15).to_string())

    # FDR
    fdr_f = d / 'fdr_report.parquet'
    if fdr_f.exists():
        df = pl.read_parquet(fdr_f)
        print(f'\n=== FDR ({tag}) ===')
        print(f'BH: {df["survives_bh"].sum()}, BY: {df["survives_by"].sum()}, q-value: {df["survives_q"].sum()}')
        print(df.filter(pl.col('survives_q')).select(['signal', 'p_value', 'q_value']).to_string())

    # Selected
    sel_f = d / 'selected_signals.json'
    if sel_f.exists():
        import json
        with open(sel_f) as f:
            print(f'\n=== SELECTED FOR COMBINER ===')
            print(json.load(f)['signals'])


def inspect_risk(tag):
    d = Path(f'data/exports/risk_model_{tag}')
    if not d.exists(): return print(f'Not found: {d}')

    # Summary
    sum_f = d / 'summary.json'
    if sum_f.exists():
        import json
        with open(sum_f) as f:
            s = json.load(f)
        print(f'\n=== RISK MODEL ({tag}) ===')
        print(f'Rebalances: {len(s)}')
        print(f'Date range: {s[0]["date"]} to {s[-1]["date"]}')
        n_cl = [x['n_clusters'] for x in s]
        n_tk = [x['n_tickers'] for x in s]
        print(f'Avg tickers: {np.mean(n_tk):.0f}, Avg clusters: {np.mean(n_cl):.1f}')
        print(f'Last: {s[-1]["date"]} - {s[-1]["n_tickers"]} tickers, {s[-1]["n_clusters"]} clusters, sizes: {s[-1]["cluster_sizes"]}')

    # Eigenvalues
    eig_f = d / 'eigenvalues.parquet'
    if eig_f.exists():
        df = pl.read_parquet(eig_f)
        last = df['date'].max()
        last_eigs = df.filter(pl.col('date') == last).sort('component')
        print(f'\nEigenvalues ({last}):')
        for r in last_eigs.head(10).iter_rows(named=True):
            print(f'  PC{r["component"]:2d}: {r["eigenvalue"]:.4f}')


def inspect_weights(tag):
    f = Path(f'data/exports/weights_{tag}.parquet')
    if not f.exists(): return print(f'Not found: {f}')
    df = pl.read_parquet(f)
    print(f'\n=== WEIGHTS ({tag}) ===')
    print(f'Rows: {len(df):,}, Dates: {df["date"].n_unique()}')

    # Gross exposure over time
    gross = df.group_by('date').agg(
        pl.sum(pl.col('weight').abs()).alias('gross'),
        pl.sum(pl.col('weight') > 0).alias('long'),
        pl.sum(pl.col('weight') < 0).alias('short'),
    ).sort('date')
    print(f'Avg gross: {gross["gross"].mean():.4f} (std {gross["gross"].std():.4f})')
    print(f'Last 5:')
    for r in gross.tail(5).iter_rows(named=True):
        print(f'  {r["date"]}: gross={r["gross"]:.4f}, L={r["long"]}, S={r["short"]}')

    # Cluster exposure (last date)
    last = df['date'].max()
    last_df = df.filter(pl.col('date') == last)
    clus = last_df.group_by('cluster_id').agg(
        pl.sum(pl.col('weight').abs()).alias('gross'),
        pl.count().alias('n'),
    ).sort('cluster_id')
    print(f'\nCluster exposure ({last}):')
    for r in clus.iter_rows(named=True):
        print(f'  Cluster {r["cluster_id"]}: {r["n"]} names, gross={r["gross"]:.4f}')


def inspect_backtest(tag):
    d = Path(f'data/exports/backtest_{tag}')
    if not d.exists(): return print(f'Not found: {d}')

    met_f = d / 'metrics.json'
    if met_f.exists():
        import json
        with open(met_f) as f:
            m = json.load(f)
        print(f'\n=== BACKTEST METRICS ({tag}) ===')
        for k, v in m.items():
            print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')

    ret_f = d / 'returns.parquet'
    if ret_f.exists():
        df = pl.read_parquet(ret_f)
        print(f'\nReturns: {len(df):,} days')
        if 'net_return' in df.columns:
            net = df['net_return']
            print(f'  Net: mean={net.mean():.6f}, std={net.std():.6f}, Sharpe={net.mean()/net.std()*np.sqrt(252):.3f}')
        if 'gross_return' in df.columns:
            gross = df['gross_return']
            print(f'  Gross: mean={gross.mean():.6f}, std={gross.std():.6f}, Sharpe={gross.mean()/gross.std()*np.sqrt(252):.3f}')
        if 'cost' in df.columns:
            cost = df['cost']
            print(f'  Cost: total={cost.sum():.2f}, mean={cost.mean():.6f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['signals', 'eval', 'risk', 'weights', 'backtest', 'all'])
    parser.add_argument('--tag', type=str, default='default')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    if args.all or args.phase == 'signals':
        inspect_signals(args.tag)
    if args.all or args.phase == 'eval':
        inspect_eval(args.tag)
    if args.all or args.phase == 'risk':
        inspect_risk(args.tag)
    if args.all or args.phase == 'weights':
        inspect_weights(args.tag)
    if args.all or args.phase == 'backtest':
        inspect_backtest(args.tag)

    if not args.phase and not args.all:
        parser.print_help()


if __name__ == '__main__':
    main()