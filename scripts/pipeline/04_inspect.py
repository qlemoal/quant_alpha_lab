#!/usr/bin/env python
'''
Inspect Intermediate Results
=============================

Quick inspection scripts for each phase's outputs.
Run after each phase to verify results before proceeding.

Usage:
  python scripts/pipeline/04_inspect.py --phase 1  # Inspect risk model
  python scripts/pipeline/04_inspect.py --phase 2  # Inspect weights
  python scripts/pipeline/04_inspect.py --phase 3  # Inspect backtest
  python scripts/pipeline/04_inspect.py --all      # Inspect all
'''

import argparse
import polars as pl
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
EXPORTS_DIR = PROJECT_ROOT / 'data' / 'exports'
RISK_DIR = EXPORTS_DIR / 'risk_model'


def inspect_phase1():
    '''Inspect risk model outputs'''
    print('=' * 70)
    print('PHASE 1 INSPECTION: Risk Model')
    print('=' * 70)

    # Cluster summary
    summary_file = RISK_DIR / 'cluster_summary.json'
    if summary_file.exists():
        import json
        with open(summary_file) as f:
            summaries = json.load(f)
        print(f'\nCluster Summary: {len(summaries)} rebalance dates')
        print(f'Date range: {summaries[0]["date"]} to {summaries[-1]["date"]}')

        # Stats
        n_clusters = [s['n_clusters'] for s in summaries]
        n_tickers = [s['n_tickers'] for s in summaries]
        print(f'Avg tickers: {np.mean(n_tickers):.0f} (min={min(n_tickers)}, max={max(n_tickers)})')
        print(f'Avg clusters: {np.mean(n_clusters):.1f} (min={min(n_clusters)}, max={max(n_clusters)})')

        # Show last few
        print('\nLast 5 rebalances:')
        for s in summaries[-5:]:
            print(f'  {s["date"]}: {s["n_tickers"]} tickers, {s["n_clusters"]} clusters, '
                  f'q={s["q_ratio"]:.3f}, λ_max={s["lambda_max"]:.3f}')
            print(f'    Sizes: {s["cluster_sizes"]}')
    else:
        print('  cluster_summary.json not found')

    # Eigenvalues
    eigen_file = RISK_DIR / 'eigenvalues.parquet'
    if eigen_file.exists():
        df = pl.read_parquet(eigen_file)
        print(f'\nEigenvalues: {len(df):,} records, {df["date"].n_unique()} dates')

        # Last date eigenvalues
        last_date = df['date'].max()
        last_eigs = df.filter(pl.col('date') == last_date).sort('component')
        print(f'\nLast date ({last_date}) eigenvalues (top 10):')
        for row in last_eigs.head(10).iter_rows(named=True):
            print(f'  Component {row["component"]:2d}: {row["eigenvalue"]:.4f}')
    else:
        print('  eigenvalues.parquet not found')

    # Correlation matrices (sample)
    corr_file = RISK_DIR / 'correlation_clean.parquet'
    if corr_file.exists():
        df = pl.read_parquet(corr_file)
        print(f'\nCleaned correlation: {len(df):,} records, {df["date"].n_unique()} dates')

        # Sample: diagonal and off-diagonal stats for last date
        last_date = df['date'].max()
        last_corr = df.filter(pl.col('date') == last_date)
        diag = last_corr.filter(pl.col('ticker_i') == pl.col('ticker_j'))['corr']
        offdiag = last_corr.filter(pl.col('ticker_i') != pl.col('ticker_j'))['corr']
        print(f'  {last_date}: diag mean={diag.mean():.4f}, off-diag mean={offdiag.mean():.4f}, '
              f'off-diag std={offdiag.std():.4f}, range=[{offdiag.min():.4f}, {offdiag.max():.4f}]')
    else:
        print('  correlation_clean.parquet not found')

    # Covariance
    cov_file = RISK_DIR / 'covariance.parquet'
    if cov_file.exists():
        df = pl.read_parquet(cov_file)
        print(f'\nCovariance: {len(df):,} records, {df["date"].n_unique()} dates')
    else:
        print('  covariance.parquet not found')

    # Clusters
    clus_file = RISK_DIR / 'clusters.parquet'
    if clus_file.exists():
        df = pl.read_parquet(clus_file)
        print(f'\nClusters: {len(df):,} records, {df["date"].n_unique()} dates')

        # Last date cluster distribution
        last_date = df['date'].max()
        last_clus = df.filter(pl.col('date') == last_date)
        print(f'  {last_date}: {last_clus["cluster_id"].n_unique()} clusters')
        dist = last_clus.group_by('cluster_id').len().sort('cluster_id')
        print(f'  Distribution: {dict(zip(dist["cluster_id"].to_list(), dist["len"].to_list()))}')
    else:
        print('  clusters.parquet not found')


def inspect_phase2():
    '''Inspect portfolio weights'''
    print('=' * 70)
    print('PHASE 2 INSPECTION: Portfolio Weights')
    print('=' * 70)

    weights_file = EXPORTS_DIR / 'weights_pre_backtest.parquet'
    diag_file = EXPORTS_DIR / 'weights_diagnostics.parquet'

    if weights_file.exists():
        df = pl.read_parquet(weights_file)
        print(f'\nWeights: {len(df):,} records, {df["date"].n_unique()} dates, {df["ticker"].n_unique()} tickers')

        # Gross exposure over time
        gross = df.group_by('date').agg(
            pl.sum(pl.col('weight').abs()).alias('gross'),
            pl.sum(pl.col('weight') > 0).alias('n_long'),
            pl.sum(pl.col('weight') < 0).alias('n_short'),
        ).sort('date')

        print(f'\nGross exposure stats:')
        print(f'  Mean: {gross["gross"].mean():.4f}')
        print(f'  Std:  {gross["gross"].std():.4f}')
        print(f'  Min:  {gross["gross"].min():.4f}')
        print(f'  Max:  {gross["gross"].max():.4f}')

        print(f'\nLast 10 rebalances:')
        for row in gross.tail(10).iter_rows(named=True):
            print(f'  {row["date"]}: gross={row["gross"]:.4f}, long={row["n_long"]}, short={row["n_short"]}')

        # Cluster exposure
        print(f'\nCluster exposure (last date):')
        last_date = df['date'].max()
        last = df.filter(pl.col('date') == last_date)
        clus_exp = last.group_by('cluster_id').agg(
            pl.sum(pl.col('weight').abs()).alias('gross'),
            pl.count().alias('n_names'),
        ).sort('cluster_id')
        for row in clus_exp.iter_rows(named=True):
            print(f'  Cluster {row["cluster_id"]}: {row["n_names"]} names, gross={row["gross"]:.4f}')

        # Weight stats
        print(f'\nWeight statistics (all dates):')
        w = df['weight']
        print(f'  Mean: {w.mean():.6f}, Std: {w.std():.6f}')
        print(f'  Min: {w.min():.6f}, Max: {w.max():.6f}')
        print(f'  Non-zero: {(w.abs() > 1e-6).sum()} / {len(w)} ({(w.abs() > 1e-6).mean():.1%})')

    else:
        print('  weights_pre_backtest.parquet not found')

    if diag_file.exists():
        df = pl.read_parquet(diag_file)
        print(f'\nDiagnostics: {len(df)} records')
        print(f'  Avg max cluster exposure: {df["max_cluster_exposure"].mean():.2%}')
        print(f'  Max cluster exposure:     {df["max_cluster_exposure"].max():.2%}')
    else:
        print('  weights_diagnostics.parquet not found')


def inspect_phase3():
    '''Inspect backtest results'''
    print('=' * 70)
    print('PHASE 3 INSPECTION: Backtest Results')
    print('=' * 70)

    metrics_file = EXPORTS_DIR / 'metrics.json'
    returns_file = EXPORTS_DIR / 'returns.parquet'
    weights_file = EXPORTS_DIR / 'weights.parquet'

    if metrics_file.exists():
        import json
        with open(metrics_file) as f:
            metrics = json.load(f)
        print('\nMetrics:')
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f'  {k}: {v:.4f}')
            else:
                print(f'  {k}: {v}')

    if returns_file.exists():
        df = pl.read_parquet(returns_file)
        print(f'\nReturns: {len(df):,} daily records')
        print(f'  Date range: {df["date"].min()} to {df["date"].max()}')

        if 'net_return' in df.columns:
            net = df['net_return']
            gross = df.get_column('gross_return') if 'gross_return' in df.columns else None
            cost = df.get_column('cost') if 'cost' in df.columns else None

            print(f'\nReturn stats (daily):')
            print(f'  Net:   mean={net.mean():.6f}, std={net.std():.6f}, Sharpe={net.mean()/net.std()*np.sqrt(252):.3f}')
            if gross is not None:
                print(f'  Gross: mean={gross.mean():.6f}, std={gross.std():.6f}, Sharpe={gross.mean()/gross.std()*np.sqrt(252):.3f}')
            if cost is not None:
                print(f'  Cost:  mean={cost.mean():.6f}, total={cost.sum():.2f}')

    if weights_file.exists():
        df = pl.read_parquet(weights_file)
        print(f'\nFinal weights: {len(df):,} records, {df["date"].n_unique()} dates')

        # Turnover
        turnover = df.group_by('date').agg(
            pl.sum(pl.col('weight').abs()).alias('gross'),
        ).sort('date')
        print(f'  Avg gross: {turnover["gross"].mean():.4f}')


def main():
    parser = argparse.ArgumentParser(description='Inspect pipeline intermediate results')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3], help='Phase to inspect')
    parser.add_argument('--all', action='store_true', help='Inspect all phases')
    args = parser.parse_args()

    if args.all or args.phase == 1:
        inspect_phase1()

    if args.all or args.phase == 2:
        inspect_phase2()

    if args.all or args.phase == 3:
        inspect_phase3()

    if not args.all and args.phase is None:
        parser.print_help()


if __name__ == '__main__':
    main()