#!/usr/bin/env python
'''
===============================================
Phase 06: Fit the Elastic Net Combiner
===============================================

QUESTION ANSWERED
    "How do the signals that survived 05 combine into ONE score per stock and date,
     and is that score honest?"

THE STAGE IN ONE PICTURE

    signals.parquet            selected_signals.json (05)         fdr_report.parquet (05)
    (signals + fwdret)         (signals, horizon, split dates)     (q-value of each signal)
            |                              |                               |
            v                              v                               v
     [1] load the selection, the split rule, the q-values
            |
            v
     [2] src/models/elastic_net_combiner.py  run_en_combiner()

         dates  ------------------------------------------------------------------------>
                |<------------- search ------------->|<-- gap -->|<----- holdout ----->|<- no label ->
                  CPCV folds choose alpha, l1_ratio     embargo      model never saw it      last h days
                  and give OUT-OF-FOLD scores                        (and the final model scores it)
            |
            v
     [3] checks: coefficients, per-fold stability, share of search rows scored out-of-fold
            |
            v
     [4] score EVERY row:  combined_score  +  period (search | gap | holdout)
            |
            v
     [5] report: IC on the search period (out-of-fold) and on the holdout, side by side
            |
            v
     [6] en_scores.parquet, model.json, stability.parquet, config.json

TWO MODES (--combine, default auto)
    family   (auto when 05 ran in strict mode)  LOW-DIMENSIONAL, SIGN-FIXED
             admitted signals -> multiply by the prior sign -> average inside each family -> ONE score per family
             -> non-negative Elastic Net with the one-standard-error rule on 2 to 5 family scores.
             Then a drop-one test: refit without each family, compare the out-of-fold IC.
             No (1 - q) weights: the strict gates of 05 already did that job, with a hard threshold.
    signal   (auto after 05 --mode legacy)       one coefficient per signal, as before. Exploration only.

WHERE IS THE COMPLICATED PART?
    src/models/family_combiner.py        families, aligned averages, drop-one test
    src/models/elastic_net_combiner.py   standardising, q-weights, demeaned target, CPCV, out-of-fold scores
    src/validation/cpcv.py               purged and embargoed folds
    This script only calls them, in order.

WHY THE SCORE IS HONEST (and where it is not)
    - search period : the score is OUT-OF-FOLD. The model that scored a row never trained on it
                      (labels overlapping the row are purged, see 05 header).
                      Weak spot: alpha and l1_ratio were picked with the same folds.
    - gap           : final model. Not trained on, but too close to the search period to be a clean test.
    - holdout       : final model, trained on dates that end before the gap. THE CLEAN TEST.
                      Judge every later result (08, 09) on the holdout.

UNITS
    Coefficients in model.json are given twice:
      coefficients_standardised   for comparing signals with each other
      coefficients_raw_units      to score a raw signal value by hand

INPUTS
    {clean_base}/signals/{run_id}/signals.parquet
    {clean_base}/signal_eval/{run_id}/selected_signals.json, fdr_report.parquet

OUTPUTS
    {clean_base}/en_scores/{run_id}/en_scores.parquet   [date, ticker, combined_score, period, <signals>]
    {clean_base}/en_scores/{run_id}/config.json
    {clean_base}/en_model/{run_id}/model.json           (includes train_end and holdout_start: 09 reads them)
    {clean_base}/en_model/{run_id}/stability.parquet    coefficients of each CV fold

USAGE
    python scripts/pipeline/06_fit_combiner.py --date 2026-11-07 --tag default
'''

import argparse
import json
import math
import sys
from pathlib import Path

# Make `import src...` and `import config...` work wherever the script is started from.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import polars as pl

from config.constants import EMBARGO_WINDOW
from config.paths import get_paths, run_id_from_args
from src.models.elastic_net_combiner import (
    run_en_combiner,
    coefficient_stability_by_fold,
    combined_score_column,
    raw_unit_coefficients,
    score_contributions,
)
from src.models.family_combiner import group_by_family, add_family_scores, family_column, drop_one_oof_ic
from src.evaluation.signals.report import signal_report
from src.signals.cheat import signal_group


# =============================================================================
# CONFIGURATION
# =============================================================================

EN_CONFIG = {
    'horizon': 20,                  # used only if selected_signals.json has no horizon
    'holdout_dates': 200,           # used only if selected_signals.json has no holdout_dates
    'n_blocks': 10,                 # CPCV: the search period is cut in 10 blocks ...
    'n_test_blocks': 2,             # ... and each fold tests on 2 of them (45 folds)
    'l1_ratio_grid': [.05, .1, .3, .5, .7, .9, .95, .99, 1],
    'q_value_weighting': False,     # legacy per-signal mode only. Off: the hard gates of 05 replace the (1 - q) weights
    'q_power': 1.0,                 # exponent on (1 - q)
    'family_l1_grid': [.1, .3, .5, .7, .9, 1],   # family mode: few inputs, so a short grid is enough
    'target': 'csdemean',           # return minus the same-date average: the book is dollar-neutral
}

MIN_OOF_SHARE = 0.95                # warn if fewer search rows than this have an out-of-fold score


def _clean(x):
    '''JSON-safe: NaN and inf become None, numpy numbers become Python numbers.'''
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _to_date(x):
    '''numpy datetime64 or Python date -> Python date.'''
    return np.datetime64(x, 'D').astype(object)


def _fmt(x, spec):
    return format(x, spec) if isinstance(x, (int, float)) else 'n/a'


def load_q_values(fdr_file: Path, signal_cols: list) -> dict:
    '''{signal: q-value} for the selected signals. A missing file gives {} (no weighting).'''
    if not fdr_file.exists():
        print(f'  No FDR report at {fdr_file}: q-weighting is off.')
        return {}
    fdr = pl.read_parquet(fdr_file)
    return {r['signal']: r['qvalue'] for r in fdr.iter_rows(named=True) if r['signal'] in signal_cols}


def print_report(name: str, rep: dict) -> None:
    print(f'  {name:<8} IC mean {_fmt(rep.get("ic_mean"), ".4f")}   IC IR {_fmt(rep.get("ic_ir"), ".3f")}'
          f'   t (Newey-West) {_fmt(rep.get("ic_tstat_nw"), ".2f")}   hit rate {_fmt(rep.get("hit_rate"), ".1%")}'
          f'   L/S Sharpe {_fmt(rep.get("long_short_sharpe"), ".2f")}   days {rep.get("n_days")}')


def main(date: str, tag: str = 'default', combine: str = 'auto'):
    run_id = run_id_from_args(date, tag)
    cfg = get_paths(run_id=run_id)

    print('=' * 70)
    print('PHASE 06: FIT ELASTIC NET COMBINER (CPCV search + holdout)')
    print('=' * 70)
    print(f'Run ID:  {run_id}')
    print(f'Config:  {EN_CONFIG}')

    # ------------------------------------------------------------------
    # [1/6] Load the selection, the split rule, the q-values
    # ------------------------------------------------------------------
    print('\n[1/6] Loading the selection from 05...')
    selected_file = cfg.stage_dir('signal_eval') / 'selected_signals.json'
    if not selected_file.exists():
        raise FileNotFoundError(f'Not found: {selected_file}. Run 05_evaluate_signals.py first.')
    with open(selected_file) as f:
        selected = json.load(f)

    signal_cols = selected['signals']
    if not signal_cols:
        raise ValueError('05 selected no signal. Nothing to combine.')
    fwd_ret_col = selected['fwd_ret_col']
    horizon = selected.get('horizon', EN_CONFIG['horizon'])
    # The split must be the one 05 used. 05 writes it; we only fall back to the config for old files.
    holdout_dates = selected.get('holdout_dates', EN_CONFIG['holdout_dates'])
    print(f'  {len(signal_cols)} signals: {signal_cols}')
    print(f'  label {fwd_ret_col}   horizon {horizon}   holdout {holdout_dates} dates')

    if combine not in ('auto', 'family', 'signal'):
        raise ValueError(f"combine must be auto, family or signal, got {combine!r}")
    family_mode = (combine == 'family') or (combine == 'auto' and selected.get('mode') == 'strict')
    families = group_by_family(selected['signal_meta']) if family_mode else {}
    print(f'  combination mode: {"FAMILY (sign-aligned, non-negative, one-SE)" if family_mode else "SIGNAL (one coefficient per signal, exploration)"}')
    if family_mode:
        for fam, members in families.items():
            print(f'    family {fam:<16} {[(m, "+" if sg > 0 else "-") for m, sg in members]}')

    q_values = {}
    if EN_CONFIG['q_value_weighting'] and not family_mode:
        q_values = load_q_values(cfg.stage_dir('signal_eval') / 'fdr_report.parquet', signal_cols)
        print(f'  q-values: { {k: round(v, 4) for k, v in q_values.items()} }')

    signals_file = cfg.stage_dir('signals') / 'signals.parquet'
    lf = pl.scan_parquet(signals_file)
    names = lf.collect_schema().names()
    missing = [c for c in signal_cols + [fwd_ret_col] if c not in names]
    if missing:
        raise ValueError(f'Columns missing in {signals_file}: {missing}')

    members = signal_cols                                    # the admitted signals
    if family_mode:
        lf = add_family_scores(lf, families)                 # fam_<family> columns
        signal_cols = [family_column(f) for f in families]   # from here on, the model inputs are the family scores
        print(f'  model inputs: {signal_cols}')

    # ------------------------------------------------------------------
    # [2/6] Fit (CPCV search, out-of-fold scores, holdout report)
    # ------------------------------------------------------------------
    print('\n[2/6] Fitting the Elastic Net...')
    result = run_en_combiner(
        lf=lf, signal_cols=signal_cols, horizon=horizon,
        embargo_window=EMBARGO_WINDOW,
        n_blocks=EN_CONFIG['n_blocks'], n_test_blocks=EN_CONFIG['n_test_blocks'],
        holdout_dates=holdout_dates,
        q_values=q_values or None,
        l1_ratio_grid=EN_CONFIG['family_l1_grid'] if family_mode else EN_CONFIG['l1_ratio_grid'],
        fwdret_col=fwd_ret_col, target=EN_CONFIG['target'], q_power=EN_CONFIG['q_power'],
        positive=family_mode, one_se=family_mode,
    )
    train_end = _to_date(result['train_end'])
    holdout_start = _to_date(result['holdout_start'])

    # Same cut as 05? If not, the signals were selected on dates the combiner may use differently.
    if selected.get('train_end') and str(train_end) != selected['train_end']:
        raise ValueError(f'train_end differs: 05 {selected["train_end"]} vs 06 {train_end}. '
                         f'Rerun 05 with the same holdout and horizon.')

    # ------------------------------------------------------------------
    # [3/6] Checks
    # ------------------------------------------------------------------
    print('\n[3/6] Checks...')
    model = result['model']
    coef_std = dict(zip(signal_cols, model.coef_))
    coef_raw, const_raw = raw_unit_coefficients(result)
    print(f'  alpha {model.alpha_:.6g}   l1_ratio {model.l1_ratio_:.3g}')
    nonzero = [k for k, v in coef_std.items() if abs(v) > 1e-9]
    print(f'  non-zero coefficients: {len(nonzero)} / {len(signal_cols)}')
    for k in sorted(nonzero, key=lambda k: -abs(coef_std[k])):
        print(f'    {k:<28} {coef_std[k]:+.6f}')
    if not nonzero:
        raise ValueError('All coefficients are zero: the combined score would be constant. '
                         'The signals carry no usable information, or alpha is too large.')

    # Role of each signal in the score: share of the score variance (the coefficient alone depends on units)
    share = score_contributions(result, result['search_panel'])
    group_share = {}
    for k, v in share.items():
        g = (k[len('fam_'):] if k.startswith('fam_') else None)
        g = g if g in ('cheat', 'placebo') else ('real' if g else signal_group(k))
        group_share[g] = group_share.get(g, 0.0) + v
    print('  share of the score variance, by signal:')
    for k in sorted(share, key=lambda k: -abs(share[k])):
        print(f'    {k:<28} {share[k]:+8.2%}')
    print('  by group: ' + '   '.join(f'{g} {v:.2%}' for g, v in sorted(group_share.items())))
    if group_share.get('placebo', 0.0) > 0.01:
        print(f'  WARNING: placebo signals (pure noise) carry {group_share["placebo"]:.1%} of the score variance. '
              f'The combiner fits noise. Use a stronger penalty or a stricter selection.')

    stability_df = coefficient_stability_by_fold(result, result['search_panel'])
    sign_flips = [k for k in signal_cols if (stability_df[k] > 1e-9).any() and (stability_df[k] < -1e-9).any()]
    zero_some = [k for k in nonzero if (stability_df[k].abs() <= 1e-9).any()]
    print(f'  folds: {stability_df.height}')
    print(f'  signals whose sign flips between folds: {sign_flips if sign_flips else "none"}')
    print(f'  signals that are zero in some folds:    {zero_some if zero_some else "none"}')

    drop_one = {}
    if family_mode and len(signal_cols) > 1:
        print('  drop-one test (out-of-fold IC with and without each family):')
        drop_one = drop_one_oof_ic(result['search_panel'], result)
        for k, d in drop_one.items():
            verdict = 'helps' if d['delta'] > 0 else 'adds nothing'
            print(f'    {k:<24} IC with {d["ic_full"]:.4f}  without {d["ic_without"]:.4f}  delta {d["delta"]:+.4f}  {verdict}')

    oof_share = result['oof'].height / max(result['search_panel'].height, 1)
    print(f'  search rows scored out-of-fold: {oof_share:.1%}')
    if oof_share < MIN_OOF_SHARE:
        print(f'  WARNING: below {MIN_OOF_SHARE:.0%}. The remaining search rows get the final model score (in-sample).')

    # ------------------------------------------------------------------
    # [4/6] Score every row, tag the period
    # ------------------------------------------------------------------
    print('\n[4/6] Scoring all rows...')
    scored = (
        combined_score_column(lf, result, out_col='combined_score')
        .with_columns(
            pl.when(pl.col('date') <= pl.lit(train_end)).then(pl.lit('search'))
              .when(pl.col('date') < pl.lit(holdout_start)).then(pl.lit('gap'))
              .otherwise(pl.lit('holdout'))
              .alias('period')
        )
        .collect()
    )
    s = scored['combined_score']
    print(f'  {scored.height:,} rows   score mean {s.mean():.6f}   std {s.std():.6f}   nulls {s.null_count():,}')
    counts = scored.group_by('period').agg(pl.col('date').n_unique().alias('dates')).sort('dates', descending=True)
    print('  dates per period: ' + ', '.join(f'{r["period"]} {r["dates"]}' for r in counts.iter_rows(named=True)))
    print(f'  search: up to {train_end}   gap: until {holdout_start}   holdout: from {holdout_start}')

    # ------------------------------------------------------------------
    # [5/6] Report: search (out-of-fold) vs holdout
    # ------------------------------------------------------------------
    print('\n[5/6] Quality of the combined score (labels with horizon overlap: Newey-West and sqrt(252/h) used)')
    reports = {}
    for name in ('search', 'holdout'):
        # Keep rows with a real label and a real score (the last h dates have no label yet; null AND NaN count as missing).
        part = scored.filter(
            (pl.col('period') == name)
            & pl.col(fwd_ret_col).is_not_null() & pl.col(fwd_ret_col).is_not_nan()
            & pl.col('combined_score').is_not_null() & pl.col('combined_score').is_not_nan()
        )
        reports[name] = signal_report(part.lazy(), 'combined_score', fwd_ret_col, horizon=horizon)
        print_report(name, reports[name])
    ic_s, ic_h = reports['search'].get('ic_mean'), reports['holdout'].get('ic_mean')
    if isinstance(ic_s, float) and isinstance(ic_h, float) and ic_s * ic_h < 0:
        print('  WARNING: the IC changes sign between search and holdout. Do not trust the combination.')
    elif isinstance(ic_s, float) and isinstance(ic_h, float) and abs(ic_h) < 0.5 * abs(ic_s):
        print('  NOTE: the holdout IC is less than half of the search IC. Expect a weaker strategy than the search suggests.')

    # ------------------------------------------------------------------
    # [6/6] Save
    # ------------------------------------------------------------------
    print('\n[6/6] Saving...')
    scores_dir, model_dir = cfg.stage_dir('en_scores'), cfg.stage_dir('en_model')
    scores_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    scored.select(['date', 'ticker', 'combined_score', 'period'] + signal_cols).write_parquet(scores_dir / 'en_scores.parquet')
    stability_df.write_parquet(model_dir / 'stability.parquet')

    with open(scores_dir / 'config.json', 'w') as f:
        json.dump(_clean({
            'run_id': run_id, 'signal_dir': str(cfg.stage_dir('signals')),
            'fwd_ret_col': fwd_ret_col, 'horizon': horizon, 'signal_cols': signal_cols, 'q_values': q_values,
            'train_end': train_end, 'holdout_start': holdout_start,
            'periods': 'search = out-of-fold score | gap = final model, close to search | holdout = final model, clean test',
        }), f, indent=2, default=str)

    with open(model_dir / 'model.json', 'w') as f:
        json.dump(_clean({
            'run_id': run_id,
            'signal_tag': tag,
            'signal_dir': str(cfg.stage_dir('signals')),
            'signals': signal_cols,
            'combine_mode': 'family' if family_mode else 'signal',
            'admitted_signals': members,
            'families': {f: [{'signal': m, 'sign': sg} for m, sg in ms] for f, ms in families.items()},
            'drop_one': drop_one,
            'positive_coefficients': family_mode,
            'one_se_rule': family_mode,
            'fwd_ret_col': fwd_ret_col,
            'horizon': horizon,
            'target': EN_CONFIG['target'],
            'alpha': model.alpha_,
            'l1_ratio': model.l1_ratio_,
            'coefficients_standardised': coef_std,
            'coefficients_raw_units': coef_raw,
            'score_variance_share': share,
            'score_variance_share_by_group': group_share,
            'intercept_raw_units': const_raw,
            'cv_method': 'cpcv',
            'n_blocks': EN_CONFIG['n_blocks'],
            'n_test_blocks': EN_CONFIG['n_test_blocks'],
            'n_folds': stability_df.height,
            'holdout_dates': holdout_dates,
            'embargo_window': EMBARGO_WINDOW,
            'purge_w': result['purge_w'],
            'embargo_w': result['embargo_w'],
            'train_end': train_end,
            'holdout_start': holdout_start,
            'q_weighting': EN_CONFIG['q_value_weighting'],
            'q_power': EN_CONFIG['q_power'],
            'q_values': q_values,
            'oof_share': oof_share,
            'sign_flips': sign_flips,
            'search_report': reports['search'],
            'holdout_report': reports['holdout'],
        }), f, indent=2, default=str)
    print(f'  scores: {scores_dir / "en_scores.parquet"}')
    print(f'  model:  {model_dir / "model.json"}')
    print(f'\nNext: python scripts/pipeline/07_estimate_risk_model.py --date {date} --tag {tag}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit the Elastic Net combiner (CPCV search + holdout)')
    parser.add_argument('--date', type=str, required=True, help='Run date, YYYY-MM-DD (used in the run_id)')
    parser.add_argument('--tag', type=str, default='default', help='Run tag (default: default)')
    parser.add_argument('--combine', choices=['auto', 'family', 'signal'], default='auto',
                        help='auto: family after a strict 05, signal after a legacy 05')
    args = parser.parse_args()
    main(args.date, args.tag, args.combine)