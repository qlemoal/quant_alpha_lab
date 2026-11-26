#!/usr/bin/env python
'''
===============================================
Stage 05: Evaluate Signals and Select
===============================================
QUESTION ANSWERED
    "Which candidate signals carry real information about future returns, and are not luck?"

THE STAGE IN ONE PICTURE
    signals.parquet (04)
        |
        |  [1] keep the SEARCH period only        the holdout (last HOLDOUT_DATES dates) and the gap before it are cut
        |  [2] daily rank IC of each signal       Spearman(signal, fwdret{h}) across the stocks of each date
        |  [3] bootstrap test, signal by signal   is the mean IC above 0, in the direction of the prior?
        |  [4] Romano-Wolf                        correct the p-values for the number of signals of this run
        |  [5] earlier trials                     p_all = p_RW x (K + n_trials_extra) / K
        |  [6] cross-checks (decide nothing)      Newey-West t, non-overlapping t, BH / BY / q, deflated Sharpe, IC by year
        |  [7] decision                           gates: prior -> sign -> p_all < 5%
        v
    selected_signals.json (06 reads it)

WHY THESE TESTS (short; the long version is in docs/selection.md)
    - The daily IC series has memory: labels of neighbour days share h - 1 days, and slow signals add more.
      A plain t-test, and even Newey-West, says "significant" too often. A studentized block bootstrap keeps the
      memory inside blocks of days; its false-positive rate is about 5% at 5% (scripts/research/calibrate_tests.py).
    - Many signals are tested at once. Romano-Wolf controls the chance of admitting ANY false signal, and it uses the
      real correlation between the signals (it resamples the same days for all of them).
    - Signals tried in earlier runs are invisible to Romano-Wolf. --n_trials_extra counts them: the p-value is
      scaled by the number of families (Bonferroni over runs).

NO HOLDOUT LEAKAGE
    Only dates <= train_end are used. train_end is computed with the SAME rule as 06:
        train_end = labelled_dates[-holdout_dates - embargo_w]       embargo_w = (horizon + 1) + EMBARGO_WINDOW

Admitting no signal is a valid result. It means the data show no robust signal.

OUTPUTS  {clean_base}/runs/{run_id}/signal_eval/
    ic_summary.parquet        per signal: IC statistics, Newey-West and non-overlapping t (search period)
    bootstrap_tests.parquet   per signal: bootstrap t, single p-value, Romano-Wolf adjusted p-value
    ic_by_year.parquet        mean IC per calendar year (stability, survivorship check)
    ic_decay.parquet          IC against the label horizon
    fdr_report.parquet        BH, BY, Storey q-value (cross-check; 06 reads the q-values in legacy mode)
    selection_gates.parquet   per signal: one pass/fail per gate, first failed gate
    signal_report.json        everything above in one file
    selected_signals.json     admitted signals + family and sign of each (06 reads it)

USAGE
    python scripts/pipeline/05_evaluate_signals.py --date 2026-11-07 --tag default
    python scripts/pipeline/05_evaluate_signals.py --date 2026-11-07 --tag default --mode legacy --alpha 0.2
'''

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import polars as pl

from config.paths import get_paths, run_id_from_args
from config.constants import EMBARGO_WINDOW, HOLDOUT_DATES, TRADING_DAYS
from src.evaluation.signals.ic.core import compute_ic, summarize_ic
from src.evaluation.signals.ic.metrics import ic_decay, newey_west_ic_tstat, ic_pvalue_nw, ic_tstat_non_overlapping
from src.evaluation.signals.fdr import fdr_report
from src.evaluation.signals.benchmarks import benchmark_gate_report
from src.evaluation.signals.bootstrap import bootstrap_test_table
from src.evaluation.signals.dsr import deflated_sharpe_table
from src.evaluation.signals.yearly import yearly_ic_table
from src.evaluation.signals.selection import (
    SelectionConfig, GATES, GATE_WHY, apply_gates, gate_summary, config_dict, earlier_trials_factor,
)
from src.signals.priors import priors_table
from src.utils import console as out
from src.utils.console import num

N_STEPS = 7
DECAY_HORIZONS = [1, 5, 10, 20, 60]


# =============================================================================
# STEPS (one function per step; main() only calls them in order)
# =============================================================================

def load_search_period(signals_file: Path, fwd_col: str, holdout_dates: int, embargo_w: int):
    '''[1] Read the signals, cut the gap and the holdout. Returns (df, signal_cols, split dict).'''
    df = pl.read_parquet(signals_file)
    signal_cols = [c for c in df.columns if c not in ('date', 'ticker') and not c.startswith('fwdret')]
    if fwd_col not in df.columns:
        raise ValueError(f'{fwd_col} not in signals.parquet. Available: {[c for c in df.columns if c.startswith("fwdret")]}')
    labelled = df.filter(pl.col(fwd_col).is_not_null())['date'].unique().sort().to_list()
    if len(labelled) <= holdout_dates + embargo_w + 100:
        raise ValueError(f'Only {len(labelled)} labelled dates: too few for a holdout of {holdout_dates} '
                         f'and a gap of {embargo_w}. Lower HOLDOUT_DATES in config/constants.py.')
    split = {
        'first_date': labelled[0], 'train_end': labelled[-holdout_dates - embargo_w],
        'holdout_start': labelled[-holdout_dates], 'last_date': labelled[-1],
        'n_tickers': df['ticker'].n_unique(),
    }
    df = df.filter(pl.col('date') <= split['train_end'])
    split['n_train_dates'] = df['date'].n_unique()
    return df, signal_cols, split


def daily_ic(df: pl.DataFrame, signal_cols: list, fwd_col: str, horizon: int):
    '''[2] Daily rank IC of every signal + the classic statistics. Returns (ic_df, ic_dated).'''
    rows, ic_dated = [], {}
    for col in signal_cols:
        ic = compute_ic(df.lazy(), col, fwd_col).collect()
        s = summarize_ic(ic)
        vals = ic['ic'].to_numpy()
        ic_dated[col] = (ic['date'].to_list(), vals)
        nw = newey_west_ic_tstat(vals, horizon=horizon)
        no = ic_tstat_non_overlapping(vals, horizon=horizon)
        rows.append({'signal': col, 'mean_ic': s['mean'], 'std_ic': s['std'], 'ir': s['ir'],
                     'hit_rate': s['hit_rate'], 'n_obs': s['n_days'],
                     'nw_tstat': nw.get('t_stat'), 'nw_lag': nw.get('max_lag'), 'p_value': ic_pvalue_nw(nw),
                     'nonoverlap_tstat': no.get('t_stat')})
    return pl.DataFrame(rows, infer_schema_length=None), ic_dated


def cross_checks(df: pl.DataFrame, ic_df: pl.DataFrame, signal_cols: list, alpha: float):
    '''[6] FDR table (from the Newey-West p-values) and IC decay. They decide nothing in strict mode.'''
    fdr = fdr_report(
        signal_names=ic_df['signal'].to_list(),
        tstats=[float('nan') if t is None else t for t in ic_df['nw_tstat'].to_list()],
        pvalues=[float('nan') if p is None else p for p in ic_df['p_value'].to_list()],
        alpha=alpha,
    )
    hs = [h for h in DECAY_HORIZONS if f'fwdret{h}' in df.columns]
    decay = pl.concat([ic_decay(df.lazy(), c, hs).with_columns(pl.lit(c).alias('signal')) for c in signal_cols])
    return fdr, decay


def gate_inputs(ic_df: pl.DataFrame, boot: dict, dsr: dict, fdr: pl.DataFrame, meta: dict) -> list:
    '''One dict per signal with what apply_gates needs.'''
    q = dict(zip(fdr['signal'].to_list(), fdr['qvalue'].to_list()))
    stats = []
    for row in ic_df.iter_rows(named=True):
        k, b = row['signal'], boot['signals'][row['signal']]
        stats.append({'signal': k, 'sign': (meta[k] or {}).get('sign'), 'family': (meta[k] or {}).get('family'),
                      'mean_ic': row['mean_ic'], 'p_single': b['p_single'], 'p_rw': b['p_rw'], 'dsr': dsr[k]['dsr'],
                      'nw_tstat': row['nw_tstat'], 'nonoverlap_tstat': row['nonoverlap_tstat'], 'qvalue': q.get(k)})
    return stats


# =============================================================================
# PRINTING (one function per block, so main() stays readable)
# =============================================================================

def print_split(split: dict, holdout_dates: int, embargo_w: int, n_signals: int) -> None:
    years_hold = holdout_dates / TRADING_DAYS
    out.info('signals to test', n_signals)
    out.info('stocks', split['n_tickers'])
    out.info('search period (used here)', f'{split["first_date"]} to {split["train_end"]}  ({split["n_train_dates"]} dates)')
    out.info('gap (not used)', f'{embargo_w} dates: no search label may reach the holdout')
    out.info('holdout (not used)', f'{split["holdout_start"]} to {split["last_date"]}  ({holdout_dates} dates, {years_hold:.1f} years)')
    out.note(f'a Sharpe ratio measured on {years_hold:.1f} years has a 95% interval of about '
             f'+-{1.96 / np.sqrt(years_hold):.1f} around its value')


def print_ic_table(ic_df: pl.DataFrame, meta: dict, top: int = 12) -> None:
    rows = []
    for r in ic_df.iter_rows(named=True):
        sign = (meta[r['signal']] or {}).get('sign')
        aligned = r['mean_ic'] * sign if (sign and r['mean_ic'] is not None) else r['mean_ic']
        rows.append((aligned if aligned is not None else -9, r, sign))
    rows.sort(key=lambda x: -x[0])
    out.table(['signal', 'prior', 'mean IC', 'IC x prior', 'hit rate', 'dates'],
              [[r['signal'][:30], {1: '+', -1: '-'}.get(sign, 'none'), num(r['mean_ic'], '+.4f'),
                num(a if a != -9 else None, '+.4f'), num(r['hit_rate'], '.0%'), r['n_obs']]
               for a, r, sign in rows[:top]],
              [-30, 5, 8, 10, 8, 6])
    if len(rows) > top:
        out.note(f'{len(rows) - top} more signals in ic_summary.parquet')
    out.note('"IC x prior" > 0 means the signal works in the direction its economic story predicts')


def print_bootstrap(boot: dict, alpha: float) -> "tuple[int, int]":
    sig = boot['signals']
    n_single = sum(v['p_single'] < alpha for v in sig.values())
    n_rw = sum(v['p_rw'] < alpha for v in sig.values())
    out.info('resamples', boot['n_boot'])
    out.info('mean block length', f'{boot["block_length"]:.0f} days (Politis-White, at least h)')
    out.info('standard error', f'batch means, batches of {boot["batch_length"]} days')
    return n_single, n_rw


def print_gate_table(gates: pl.DataFrame, top: int = 20) -> None:
    order = gates.sort(['selected', 'p_rw'], descending=[True, False], nulls_last=True)
    rows = []
    for r in order.head(top).iter_rows(named=True):
        flags = ''.join('Y' if r[f'pass_{g}'] else '.' for g in GATES)
        rows.append([r['signal'][:30], flags, num(r['p_single'], '.3f'), num(r['p_rw'], '.3f'), num(r['p_all'], '.3f'),
                     num(r['nw_tstat'], '+.1f'), num(r['qvalue'], '.3f'), num(r['dsr'], '.2f'),
                     'ADMITTED' if r['selected'] else f'stopped: {r["failed_gate"]}'])
    out.table(['signal', 'gates', 'p one', 'p RW', 'p all', 'NW t', 'q', 'DSR', 'decision'], rows,
              [-30, 5, 6, 6, 6, 5, 6, 5, -15])
    out.note('gates = ' + ' '.join(GATES) + ' (Y pass, . fail). p one: bootstrap, signal alone. p RW: Romano-Wolf. '
             'p all: p RW with earlier trials (the gate). NW t, q, DSR: cross-checks only.')


def print_benchmarks(bench: "dict | None") -> None:
    if not bench:
        return
    print()
    out.info('benchmark signals', 'known truth: cheats have a known IC > 0, placebos have IC = 0')
    if 'placebo' in bench:
        b = bench['placebo']
        out.info('placebos with p < 5% alone', f'{b["n_p_below"]} of {b["n"]} (luck gives about {b["expected_p_below"]:.1f})')
        out.verdict(not b['admitted'], 'no placebo admitted' if not b['admitted'] else f'placebos ADMITTED: {b["admitted"]}')
    if 'cheat' in bench:
        b = bench['cheat']
        out.info('cheats admitted', f'{len(b["admitted"])} of {b["n"]}: {b["admitted"] or "none"}')
        out.note('the weakest cheat that is admitted shows the smallest IC this pipeline can detect')


def print_by_year(yearly: pl.DataFrame, selected: list) -> None:
    if not selected:
        return
    ycols = [c for c in yearly.columns if c.startswith('ic_')]
    print()
    out.info('IC by year, admitted signals', '(raw IC; its sign should match the prior every year)')
    print(out.INDENT + f'{"signal":28s}' + ''.join(f'{c[3:]:>7s}' for c in ycols) + '  ok/yrs')
    for r in yearly.filter(pl.col('signal').is_in(selected)).iter_rows(named=True):
        print(out.INDENT + f'{r["signal"][:28]:28s}' + ''.join(f'{num(r[c], "7.3f", "    n/a")}' for c in ycols)
              + f'  {r["years_ok"]}/{r["n_years"]}')
    out.note('a signal that works only in old years, or fades in the second half, can be survivorship bias')


# =============================================================================
# MAIN
# =============================================================================

def main(date: str, tag: str = 'default', horizon: int = 20, alpha: float = 0.2,
         min_ir: float = 0.0, holdout_dates: int = HOLDOUT_DATES, mode: str = 'strict',
         selection: "SelectionConfig | None" = None):
    if mode not in ('strict', 'legacy'):
        raise ValueError(f"mode must be 'strict' or 'legacy', got {mode!r}")
    selection = selection or SelectionConfig()
    run_id = run_id_from_args(date, tag)
    cfg = get_paths(run_id=run_id)
    signals_file = cfg.stage_dir('signals') / 'signals.parquet'
    if not signals_file.exists():
        raise FileNotFoundError(f'Signals file not found: {signals_file}. Run 04_generate_signals.py first.')
    config = json.loads((cfg.stage_dir('signals') / 'config.json').read_text())
    OUT = cfg.stage_dir('signal_eval')
    OUT.mkdir(parents=True, exist_ok=True)
    fwd_col = f'fwdret{horizon}'
    embargo_w = (horizon + 1) + EMBARGO_WINDOW

    out.stage('05', 'Evaluate signals', 'Which signals carry real information about future returns, and are not luck?')
    out.info('run', run_id)
    out.info('label', f'{fwd_col}: log return from the open of t+1 to the open of t+1+{horizon}')
    out.info('selection mode', 'strict (gates)' if mode == 'strict' else 'LEGACY: q-value only, exploration, NOT evidence')

    # [1] ---------------------------------------------------------------------------------------------------------
    out.step(1, N_STEPS, 'Keep the search period only',
             'the holdout must never influence which signals we keep')
    df, signal_cols, split = load_search_period(signals_file, fwd_col, holdout_dates, embargo_w)
    print_split(split, holdout_dates, embargo_w, len(signal_cols))
    meta = priors_table(signal_cols)                     # family, sign, rationale, written before the test
    signs = {k: (v['sign'] if v else None) for k, v in meta.items()}

    # [2] ---------------------------------------------------------------------------------------------------------
    out.step(2, N_STEPS, 'Daily rank IC of each signal',
             'IC = rank correlation of the signal with the next return, across the stocks of one date: '
             'does the signal order the stocks well?')
    ic_df, ic_dated = daily_ic(df, signal_cols, fwd_col, horizon)
    ic_df.write_parquet(OUT / 'ic_summary.parquet')
    print_ic_table(ic_df, meta)

    # [3] + [4] ---------------------------------------------------------------------------------------------------
    out.step(3, N_STEPS, 'Bootstrap test, one signal at a time: is the mean IC above 0 in the prior direction?',
             'neighbour days share most of their label, so the IC series has memory; resampling BLOCKS of days keeps '
             'it, and the false-positive rate stays at 5%')
    boot = bootstrap_test_table(ic_dated, signs, horizon, n_boot=selection.n_boot, seed=selection.boot_seed)
    n_single, n_rw = print_bootstrap(boot, selection.fwer_alpha)
    out.info('signals with p < 5% alone', f'{n_single} of {len(signal_cols)}')
    out.note(f'with {len(signal_cols)} pure-noise signals, luck alone gives about {0.05 * len(signal_cols):.1f}')

    out.step(4, N_STEPS, 'Romano-Wolf: correct for the number of signals tested in this run',
             'the more signals you test, the more look good by luck; Romano-Wolf uses their real correlation')
    out.info(f'signals with adjusted p < {selection.fwer_alpha:.0%}', f'{n_rw} of {len(signal_cols)}')
    out.note(f'meaning: the chance that ANY of these {n_rw} is pure luck is below {selection.fwer_alpha:.0%}')
    pl.DataFrame([{'signal': k, **v} for k, v in boot['signals'].items()]).write_parquet(OUT / 'bootstrap_tests.parquet')

    # [5] ---------------------------------------------------------------------------------------------------------
    out.step(5, N_STEPS, 'Earlier trials: signals tried in earlier runs on the same data also count',
             'Romano-Wolf sees only this run; each earlier run is one more family of tests (Bonferroni over runs)')
    factor = earlier_trials_factor(len(signal_cols), selection.n_trials_extra)
    out.info('trials', f'{len(signal_cols)} in this run + {selection.n_trials_extra} earlier')
    out.info('p_all', f'= p_RW x {factor:.2f}')
    out.info(f'signals with p_all < {selection.fwer_alpha:.0%}',
             f'{sum(min(1, v["p_rw"] * factor) < selection.fwer_alpha for v in boot["signals"].values())} of {len(signal_cols)}')
    if selection.n_trials_extra == 0:
        out.warn('--n_trials_extra is 0: right only if no signal was tried before this run on these data')

    # [6] ---------------------------------------------------------------------------------------------------------
    out.step(6, N_STEPS, 'Cross-checks (printed, they decide nothing)',
             'classic tests; Newey-West over-rejects on slow signals and the deflated Sharpe drops 95% of the data. '
             'A big disagreement with the gates is a warning sign')
    dsr = deflated_sharpe_table({k: v for k, (_, v) in ic_dated.items()}, signs, step=horizon,
                                n_trials_extra=selection.n_trials_extra, var_mode=selection.dsr_variance)
    fdr, decay = cross_checks(df, ic_df, signal_cols, alpha)
    fdr.write_parquet(OUT / 'fdr_report.parquet')
    decay.write_parquet(OUT / 'ic_decay.parquet')
    out.info(f'Newey-West |t| >= 3', f'{int((ic_df["nw_tstat"].abs() >= 3).sum())} of {len(signal_cols)}')
    out.info(f'BH / BY / q < {alpha}', f'{int(fdr["survives_bh"].sum())} / {int(fdr["survives_by"].sum())} / '
             f'{int((fdr["qvalue"] < alpha).sum())}')
    out.info('deflated Sharpe >= 0.95', f'{sum(d["dsr"] >= 0.95 for d in dsr.values())} of {len(dsr)}')
    if len(signal_cols) < 20:
        out.note('fewer than 20 signals: Storey pi0 = 1, so the q-values equal the BH adjusted p-values')
    yearly = yearly_ic_table(ic_dated, signs)
    yearly.write_parquet(OUT / 'ic_by_year.parquet')

    # [7] ---------------------------------------------------------------------------------------------------------
    out.step(7, N_STEPS, 'Decision: a signal is admitted only if it passes every gate',
             'each gate stops a different kind of false discovery')
    for g in GATES:
        out.info(f'  {g}', GATE_WHY[g])
    gates = apply_gates(gate_inputs(ic_df, boot, dsr, fdr, meta), selection)
    gates.write_parquet(OUT / 'selection_gates.parquet')
    summary = gate_summary(gates)
    print()
    print_gate_table(gates)
    if mode == 'strict':
        selected = gates.filter(pl.col('selected'))['signal'].to_list()
        stopped = ', '.join(f'{g} {summary[g]}' for g in GATES if summary[g])
        out.info('stopped by each gate', stopped or 'none')
    else:
        selected = fdr.filter(pl.col('qvalue') < alpha)['signal'].to_list()
        if min_ir > 0:
            selected = [s for s in selected if abs(ic_df.filter(pl.col('signal') == s)['ir'][0]) >= min_ir]
        out.warn(f'LEGACY selection (q < {alpha}): exploration only, this is NOT evidence of a strategy')

    bench = benchmark_gate_report(gates)
    print_benchmarks(bench)
    print_by_year(yearly, selected)

    weak = ic_df.filter(pl.col('signal').is_in(selected) & (pl.col('nonoverlap_tstat').abs() < 1.0))['signal'].to_list()
    if weak:
        out.warn(f'admitted but the non-overlapping |t| < 1: {weak}. Look at them before you trust them.')
    print()
    if selected:
        out.ok(f'{len(selected)} signal(s) ADMITTED: {selected}')
    else:
        out.fail('NO SIGNAL ADMITTED. Stage 06 will stop. This is a valid result: no signal passed every gate.')
        out.note('to explore anyway (not evidence): --mode legacy, or relax a threshold on purpose and write it down')

    # --- save --------------------------------------------------------------------------------------------------------
    split_out = {k: str(v) for k, v in split.items()}
    report = {
        'run_id': run_id, 'fwd_ret_col': fwd_col, 'horizon': horizon, 'n_signals': len(signal_cols),
        'selection_mode': mode, 'selection_config': config_dict(selection), 'fdr_alpha': alpha, 'min_ic_ir': min_ir,
        'split': split_out, 'holdout_dates': holdout_dates, 'embargo_w': embargo_w,
        'bootstrap': {k: v for k, v in boot.items() if k != 'signals'},
        'gate_summary': summary, 'gates': gates.to_dicts(), 'selected_signals': selected,
        'ic_summary': ic_df.to_dicts(), 'benchmark_report': bench, 'config': config,
    }
    (OUT / 'signal_report.json').write_text(json.dumps(report, indent=2, default=str))
    (OUT / 'selected_signals.json').write_text(json.dumps({
        'signals': selected,
        'mode': mode,                                     # 06 uses the family combiner for 'strict'
        'signal_meta': {k: meta[k] for k in selected},
        'selection_config': config_dict(selection),
        'fwd_ret_col': fwd_col, 'horizon': horizon,
        'train_end': split_out['train_end'], 'holdout_start': split_out['holdout_start'],
        'holdout_dates': holdout_dates, 'embargo_w': embargo_w,     # 06 reads these: one source of truth for the split
    }, indent=2))
    out.done({'selected signals': OUT / 'selected_signals.json', 'full report': OUT / 'signal_report.json'},
             f'python scripts/pipeline/06_fit_combiner.py --date {date} --tag {tag}')


if __name__ == '__main__':
    d = SelectionConfig()
    ap = argparse.ArgumentParser(description='Evaluate signals and select the ones for the combiner')
    ap.add_argument('--date', required=True, help='run date, YYYY-MM-DD')
    ap.add_argument('--tag', default='default', help='run tag')
    ap.add_argument('--horizon', type=int, default=20, help='label horizon in days (fwdret{h})')
    ap.add_argument('--mode', choices=['strict', 'legacy'], default='strict',
                    help='strict = gates (default). legacy = q < --alpha only (exploration, not evidence)')
    ap.add_argument('--alpha', type=float, default=0.2, help='FDR level of the cross-check and of --mode legacy')
    ap.add_argument('--fwer_alpha', type=float, default=d.fwer_alpha, help='p_all must be below this')
    ap.add_argument('--n_trials_extra', type=int, default=0, help='signals tried in earlier runs on the same data')
    ap.add_argument('--dsr_variance', choices=['null', 'trials'], default='null',
                    help="DSR cross-check: variance of skill-less Sharpe ratios, null = 1/(T-1), trials = across signals")
    ap.add_argument('--n_boot', type=int, default=d.n_boot, help='bootstrap resamples')
    ap.add_argument('--boot_seed', type=int, default=d.boot_seed, help='bootstrap seed')
    ap.add_argument('--no_require_prior', action='store_true', help='signals without a prior skip gates 1 and 2 (exploration)')
    ap.add_argument('--min_ir', type=float, default=0.0, help='legacy mode: minimum |IC IR|')
    ap.add_argument('--holdout_dates', type=int, default=HOLDOUT_DATES, help='must match 06 (default from config/constants.py)')
    a = ap.parse_args()
    sel = SelectionConfig(fwer_alpha=a.fwer_alpha, require_prior=not a.no_require_prior,
                          n_trials_extra=a.n_trials_extra, dsr_variance=a.dsr_variance, n_boot=a.n_boot,
                          boot_seed=a.boot_seed)
    main(a.date, a.tag, a.horizon, a.alpha, a.min_ir, a.holdout_dates, a.mode, sel)
