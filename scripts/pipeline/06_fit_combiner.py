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
    {clean_base}/runs/{run_id}/signals/signals.parquet
    {clean_base}/runs/{run_id}/signal_eval/selected_signals.json, fdr_report.parquet

OUTPUTS
    {clean_base}/runs/{run_id}/en_scores/en_scores.parquet   [date, ticker, combined_score, period, <signals>]
    {clean_base}/runs/{run_id}/en_scores/config.json
    {clean_base}/runs/{run_id}/en_model/model.json           (includes train_end and holdout_start: 09 reads them)
    {clean_base}/runs/{run_id}/en_model/stability.parquet    coefficients of each CV fold

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

from config.constants import EMBARGO_WINDOW, HOLDOUT_DATES
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
from src.utils import console as out
from src.utils.console import num


# =============================================================================
# CONFIGURATION
# =============================================================================

EN_CONFIG = {
    'horizon': 20,                  # used only if selected_signals.json has no horizon
    'holdout_dates': HOLDOUT_DATES,          # used only if selected_signals.json has no holdout_dates
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
        out.note(f'no FDR report at {fdr_file}: q-weighting is off')
        return {}
    fdr = pl.read_parquet(fdr_file)
    return {r['signal']: r['qvalue'] for r in fdr.iter_rows(named=True) if r['signal'] in signal_cols}


def report_row(name: str, rep: dict) -> list:
    return [name, num(rep.get('ic_mean'), '+.4f'), num(rep.get('ic_tstat_nw'), '+.2f'), num(rep.get('hit_rate'), '.0%'),
            num(rep.get('long_short_sharpe'), '+.2f'), rep.get('n_days')]


def main(date: str, tag: str = 'default', combine: str = 'auto'):
    run_id = run_id_from_args(date, tag)
    cfg = get_paths(run_id=run_id)

    out.stage('06', 'Combine the admitted signals', 'What is the best single score from the admitted signals, '
              'judged on dates the model never saw?')
    out.info('run', run_id)

    # ------------------------------------------------------------------
    # [1/6] Load the selection, the split rule, the q-values
    # ------------------------------------------------------------------
    out.step(1, 6, 'Read the admitted signals of stage 05 and group them by family',
             'signals of one family (for example mom5 and mom10) are the same bet: they are averaged with the sign of '
             'their prior, so the model has few inputs and little to overfit')
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
    out.info('admitted signals', f'{len(signal_cols)}: {", ".join(signal_cols)}')
    out.info('label', f'{fwd_ret_col}   (holdout: last {holdout_dates} dates, never used to fit)')

    if combine not in ('auto', 'family', 'signal'):
        raise ValueError(f"combine must be auto, family or signal, got {combine!r}")
    family_mode = (combine == 'family') or (combine == 'auto' and selected.get('mode') == 'strict')
    families = group_by_family(selected['signal_meta']) if family_mode else {}
    out.info('mode', 'FAMILY: sign-aligned averages, weights >= 0, one-standard-error rule' if family_mode
             else 'SIGNAL: one free weight per signal (exploration)')
    if family_mode:
        for fam, ms in families.items():
            out.info(f'  family {fam}', ', '.join(f'{"+" if sg > 0 else "-"}{m}' for m, sg in ms))

    q_values = {}
    if EN_CONFIG['q_value_weighting'] and not family_mode:
        q_values = load_q_values(cfg.stage_dir('signal_eval') / 'fdr_report.parquet', signal_cols)
        out.info('q-values', { k: round(v, 4) for k, v in q_values.items()})

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
        out.info('model inputs', ', '.join(signal_cols))

    # ------------------------------------------------------------------
    # [2/6] Fit (CPCV search, out-of-fold scores, holdout report)
    # ------------------------------------------------------------------
    out.step(2, 6, 'Fit an Elastic Net with combinatorial purged cross-validation (CPCV)',
             f'the search period is cut into {EN_CONFIG["n_blocks"]} blocks; each fold tests on '
             f'{EN_CONFIG["n_test_blocks"]} blocks and trains on the rest, minus a purge and an embargo around the test '
             'blocks, so no training label overlaps a test label. The penalty is chosen on these folds.')
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
    model = result['model']
    coef_std = dict(zip(signal_cols, model.coef_))
    coef_raw, const_raw = raw_unit_coefficients(result)
    out.info('folds', f'{len(result["cv_folds"])}   purge {result["purge_w"]} dates, embargo {result["embargo_w"]} dates')
    out.info('penalty chosen', f'alpha {model.alpha_:.3g}, l1_ratio {model.l1_ratio_:.2g}')
    nonzero = [k for k, v in coef_std.items() if abs(v) > 1e-9]

    out.step(3, 6, 'Check the model: which inputs matter, and are they stable across folds?',
             'a weight that changes sign between folds, or a placebo that carries weight, means the model fits noise')
    share = score_contributions(result, result['search_panel'])
    out.table(['input', 'weight (std units)', 'share of score variance'],
              [[k, f'{coef_std[k]:+.6f}', f'{share[k]:+.1%}'] for k in sorted(signal_cols, key=lambda k: -abs(coef_std[k]))],
              [-28, 18, 24])
    if not nonzero:
        raise ValueError('All coefficients are zero: the combined score would be constant. '
                         'The signals carry no usable information, or alpha is too large.')

    group_share = {}
    for k, v in share.items():
        g = (k[len('fam_'):] if k.startswith('fam_') else None)
        g = g if g in ('cheat', 'placebo') else ('real' if g else signal_group(k))
        group_share[g] = group_share.get(g, 0.0) + v
    if len(group_share) > 1:
        out.info('share by group', '   '.join(f'{g} {v:.1%}' for g, v in sorted(group_share.items())))
    if group_share.get('placebo', 0.0) > 0.01:
        out.warn(f'placebo signals (pure noise) carry {group_share["placebo"]:.1%} of the score variance: '
                 'the combiner fits noise. Use a stronger penalty or a stricter selection.')

    stability_df = coefficient_stability_by_fold(result, result['search_panel'])
    sign_flips = [k for k in signal_cols if (stability_df[k] > 1e-9).any() and (stability_df[k] < -1e-9).any()]
    zero_some = [k for k in nonzero if (stability_df[k].abs() <= 1e-9).any()]
    out.verdict(not sign_flips, 'no weight changes sign between folds' if not sign_flips
                else f'weights change sign between folds: {sign_flips}')
    if zero_some:
        out.note(f'zero in some folds (a weak input): {zero_some}')

    drop_one = {}
    if family_mode and len(signal_cols) > 1:
        drop_one = drop_one_oof_ic(result['search_panel'], result)
        out.info('drop-one test', 'out-of-fold IC of the score with and without each family')
        out.table(['family', 'IC with', 'IC without', 'change', ''],
                  [[k, f'{d["ic_full"]:.4f}', f'{d["ic_without"]:.4f}', f'{d["delta"]:+.4f}',
                    'helps' if d['delta'] > 0 else 'adds nothing'] for k, d in drop_one.items()],
                  [-24, 8, 10, 8, -12])

    oof_share = result['oof'].height / max(result['search_panel'].height, 1)
    out.verdict(oof_share >= MIN_OOF_SHARE, f'{oof_share:.1%} of the search rows are scored out-of-fold '
                '(by models that never trained on them)')

    # ------------------------------------------------------------------
    # [4/6] Score every row, tag the period
    # ------------------------------------------------------------------
    out.step(4, 6, 'Score every row and tag its period',
             'search = out-of-fold score; gap = final model, too close to the search to judge; holdout = final model '
             'on dates it never saw')
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
    counts = dict(scored.group_by('period').agg(pl.col('date').n_unique()).iter_rows())
    out.info('search', f'up to {train_end}   ({counts.get("search", 0)} dates)')
    out.info('gap', f'until {holdout_start}   ({counts.get("gap", 0)} dates)')
    out.info('holdout', f'from {holdout_start}   ({counts.get("holdout", 0)} dates)')
    out.info('rows scored', f'{scored.height:,} (missing scores: {s.null_count():,})')

    # ------------------------------------------------------------------
    # [5/6] Report: search (out-of-fold) vs holdout
    # ------------------------------------------------------------------
    out.step(5, 6, 'Quality of the combined score: search (out-of-fold) against holdout',
             'the search numbers are optimistic (the penalty was chosen on these folds); only the holdout is clean')
    reports = {}
    for name in ('search', 'holdout'):
        # Keep rows with a real label and a real score (the last h dates have no label yet; null AND NaN count as missing).
        part = scored.filter(
            (pl.col('period') == name)
            & pl.col(fwd_ret_col).is_not_null() & pl.col(fwd_ret_col).is_not_nan()
            & pl.col('combined_score').is_not_null() & pl.col('combined_score').is_not_nan()
        )
        reports[name] = signal_report(part.lazy(), 'combined_score', fwd_ret_col, horizon=horizon)
    out.table(['period', 'mean IC', 'NW t', 'hit rate', 'L/S Sharpe', 'dates'],
              [report_row(n, reports[n]) for n in ('search', 'holdout')], [-8, 8, 6, 8, 10, 6])
    out.note('NW t: Newey-West t of the daily IC (lag covers the label overlap). L/S Sharpe: top minus bottom '
             'quintile, annualised with sqrt(252 / h). No costs here.')
    ic_s, ic_h = reports['search'].get('ic_mean'), reports['holdout'].get('ic_mean')
    if isinstance(ic_s, float) and isinstance(ic_h, float) and ic_s * ic_h < 0:
        out.fail('the IC changes sign between search and holdout. Do not trust the combination.')
    elif isinstance(ic_s, float) and isinstance(ic_h, float) and abs(ic_h) < 0.5 * abs(ic_s):
        out.warn('the holdout IC is less than half of the search IC: expect a weaker strategy than the search suggests')
    elif isinstance(ic_s, float) and isinstance(ic_h, float):
        out.ok('the holdout IC has the sign of the search IC and at least half its size')

    # ------------------------------------------------------------------
    # [6/6] Save
    # ------------------------------------------------------------------
    out.step(6, 6, 'Save the scores and the model')
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
    out.done({'scores': scores_dir / 'en_scores.parquet', 'model': model_dir / 'model.json'},
             f'python scripts/pipeline/07_estimate_risk_model.py --date {date} --tag {tag}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit the Elastic Net combiner (CPCV search + holdout)')
    parser.add_argument('--date', type=str, required=True, help='Run date, YYYY-MM-DD (used in the run_id)')
    parser.add_argument('--tag', type=str, default='default', help='Run tag (default: default)')
    parser.add_argument('--combine', choices=['auto', 'family', 'signal'], default='auto',
                        help='auto: family after a strict 05, signal after a legacy 05')
    args = parser.parse_args()
    main(args.date, args.tag, args.combine)