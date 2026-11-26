'''
===============================================
Run One Experiment: stages 04 to 09
===============================================

WHAT IT DOES
    Stages 01 to 03 prepare the data. You run them once, by hand (01 downloads, 02 cleans, 03 builds the features).
    This script runs the EXPERIMENT: stages 04 to 09, from the signals to the backtest, under one run_id.
    Change a signal, a parameter or the tag, and run it again.
    It runs the stage scripts one after the other. Each phase is a normal script that you can also run by hand.
    Before a phase starts, the runner checks that the files this phase reads exist.
    If a phase fails, the runner stops. Later phases do not run on stale inputs.

THE PIPELINE

    04  generate signals        features.parquet, ohlcv.parquet  ->  signals/
     |
    05  evaluate signals        signals/                         ->  signal_eval/   (IC, FDR, selection)
     |
    06  fit the combiner        signals/, signal_eval/           ->  en_scores/, en_model/
     |
    07  estimate risk model     en_scores/, features.parquet     ->  risk_model/
     |
    08  construct weights       en_scores/, risk_model/          ->  weights/
     |
    09  run the backtest        weights/, ohlcv, features        ->  backtest/      (judge the HOLDOUT block)

    Every stage folder is {clean_base}/runs/{run_id}/{stage}/, with run_id = {date}_{tag}.

USAGE
    python scripts/pipeline/run_experiment.py --date 2026-11-07 --tag default                 # stages 04 to 09
    python scripts/pipeline/run_experiment.py --date 2026-11-07 --tag default --from 07      # from stage 07 to the end
    python scripts/pipeline/run_experiment.py --date 2026-11-07 --tag default --only 09      # one stage
    python scripts/pipeline/run_experiment.py --date 2026-11-07 --tag default --check         # inputs only, run nothing
    python scripts/pipeline/run_experiment.py --date 2026-11-07 --tag cheat --cheat           # with the cheating benchmark signals
    python scripts/pipeline/run_experiment.py --date 2026-11-07 --tag placebo --placebo       # with pure-noise placebo signals
    python scripts/pipeline/run_experiment.py --date 2026-11-07 --tag cheat_placebo --cheat --placebo

NOTES
    - --from re-uses the outputs of the earlier phases as they are on disk. If you changed an earlier phase
      (code or config), start from that phase, not later.
    - The phases of one run_id overwrite their own outputs. Use a new --tag to keep a previous result.
'''

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))          # `import config...` works wherever the script is started from

from config.paths import get_paths, run_id_from_args
from src.signals.cheat import check_benchmark_tag
from src.utils import console as out
from src.utils.console import num


# =============================================================================
# THE PHASES
# =============================================================================

PHASES = [
    ('04', '04_generate_signals.py',     'build the candidate signals and the labels'),
    ('05', '05_evaluate_signals.py',     'test the signals, keep the ones that are not luck'),
    ('06', '06_fit_combiner.py',         'combine the kept signals into one score'),
    ('07', '07_estimate_risk_model.py',  'estimate how the stocks move together'),
    ('08', '08_construct_weights.py',    'build the long-short books'),
    ('09', '09_run_backtest.py',         'backtest with trading costs'),
]
PHASE_IDS = [p[0] for p in PHASES]


def required_inputs(phase_id: str, cfg) -> list:
    '''
    Files that a phase reads. They must exist before the phase starts.
    Keep this list equal to what the phase script really opens.
    '''
    signals = cfg.stage_dir('signals')
    en_scores = cfg.stage_dir('en_scores') / 'en_scores.parquet'
    risk = cfg.stage_dir('risk_model')
    weights = cfg.stage_dir('weights')
    return {
        '04': [cfg.features_file, cfg.ohlcv_file],
        '05': [signals / 'signals.parquet', signals / 'config.json'],
        '06': [signals / 'signals.parquet', cfg.stage_dir('signal_eval') / 'selected_signals.json'],
        '07': [en_scores, cfg.features_file],
        '08': [en_scores, risk / 'cov.parquet', risk / 'clusters.parquet', risk / 'summary.json'],
        '09': [weights / 'weights.parquet', weights / 'weights_diag.parquet', cfg.ohlcv_file, cfg.features_file],
    }[phase_id]


def missing_inputs(phase_id: str, cfg) -> list:
    return [str(f) for f in required_inputs(phase_id, cfg) if not Path(f).exists()]


# =============================================================================
# RUNNING
# =============================================================================

def run_phase(phase_id: str, script: str, title: str, date: str, tag: str, cheat: bool = False,
              placebo: bool = False) -> bool:
    '''Check the inputs, run the script as a subprocess, return True if it succeeded.'''
    cfg = get_paths(run_id=run_id_from_args(date, tag))
    print(f'\n\n>>> stage {phase_id}: {title}\n')

    missing = missing_inputs(phase_id, cfg)
    if missing:
        out.fail('inputs missing. Run the earlier stages first:')
        for m in missing:
            out.info('  missing', out.short_path(m))
        return False

    cmd = [sys.executable, str(ROOT / 'scripts' / 'pipeline' / script), '--date', date, '--tag', tag]
    if cheat and phase_id == '04':
        cmd.append('--cheat')                       # only stage 04 builds the benchmark signals
    if placebo and phase_id == '04':
        cmd.append('--placebo')
    env = {**os.environ, 'PYTHONPATH': str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', '')}
    start = time.time()
    sys.stdout.flush()                              # our lines first, then the stage's own output
    result = subprocess.run(cmd, cwd=ROOT, env=env)
    minutes = (time.time() - start) / 60

    if result.returncode != 0:
        out.fail(f'stage {phase_id} stopped with code {result.returncode} after {_duration(minutes)}')
        return False
    TIMES[phase_id] = minutes
    return True


TIMES = {}


def _duration(minutes: float) -> str:
    return f'{minutes * 60:.0f} s' if minutes < 1 else f'{minutes:.1f} min'


def _read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def print_headline(cfg) -> None:
    '''The few numbers that matter, from the files of the stages, so that the last thing you read is the honest one.'''
    sel = _read_json(cfg.stage_dir('signal_eval') / 'selected_signals.json')
    if sel is not None:
        out.info('05 signals admitted', f'{len(sel["signals"])}: {", ".join(sel["signals"]) or "none"}')
    model = _read_json(cfg.stage_dir('en_model') / 'model.json')
    if model:
        s, h = (model.get('search_report') or {}).get('ic_mean'), (model.get('holdout_report') or {}).get('ic_mean')
        out.info('06 IC of the score', f'search {num(s, "+.4f")} (out-of-fold)   holdout {num(h, "+.4f")}')
    metrics = _read_json(cfg.stage_dir('backtest') / 'metrics.json')
    if metrics:
        block = metrics.get('holdout') or metrics.get('full')
        name = 'HOLDOUT' if metrics.get('holdout') else 'FULL PERIOD (no holdout reported)'
        if block and block.get('sharpe') is not None:
            lo, hi = block['sharpe_ci95']
            out.info(f'09 {name} net Sharpe', f'{block["sharpe"]:+.2f}   95% interval [{lo:+.2f}, {hi:+.2f}]   '
                     f'gross {num(block.get("sharpe_gross"), "+.2f")}')
            out.info('   return, drawdown', f'{block["ann_return"]:+.1%} a year, max drawdown {block["max_drawdown"]:.1%}')
            out.verdict(lo > 0, 'the holdout interval excludes 0' if lo > 0
                        else 'the holdout interval includes 0: no evidence of skill after costs')


def main():
    parser = argparse.ArgumentParser(description='Run the pipeline phases 04 to 09')
    parser.add_argument('--date', type=str, required=True, help='Run date, YYYY-MM-DD (used in the run_id)')
    parser.add_argument('--tag', type=str, default='default', help='Run tag (default: default)')
    parser.add_argument('--from', dest='from_phase', type=str, default='04', choices=PHASE_IDS,
                        help='First phase to run (default: 04)')
    parser.add_argument('--only', dest='only_phase', type=str, choices=PHASE_IDS, help='Run this phase only')
    parser.add_argument('--cheat', action='store_true',
                        help='Add the cheating benchmark signals in 04 (the tag must contain "cheat")')
    parser.add_argument('--placebo', action='store_true',
                        help='Add placebo signals (pure noise) in 04 (the tag must contain "placebo")')
    parser.add_argument('--check', action='store_true', help='Check the inputs of the phases and run nothing')
    args = parser.parse_args()

    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', args.date):
        parser.error(f'--date must be YYYY-MM-DD, got {args.date!r}')
    try:
        check_benchmark_tag(args.tag, args.cheat, args.placebo)
    except ValueError as e:
        parser.error(str(e))
    if args.only_phase and args.from_phase != '04':
        parser.error('use --only or --from, not both')

    run_id = run_id_from_args(args.date, args.tag)
    cfg = get_paths(run_id=run_id)
    phases = [args.only_phase] if args.only_phase else PHASE_IDS[PHASE_IDS.index(args.from_phase):]

    out.stage('04-09', 'Run one experiment', 'From candidate signals to a backtest after costs, one stage at a time.')
    out.info('run', run_id)
    for pid in phases:
        out.info(f'  stage {pid}', next(p[2] for p in PHASES if p[0] == pid))
    out.note('each stage stops the chain if a check fails. Read the [FAIL] and [WARN] lines.')

    # --check: report the inputs of every phase. A phase whose inputs are made by an earlier phase of
    # this same call is reported as "missing", because nothing has run yet.
    if args.check:
        out.step(1, 1, 'Input check (nothing is run)')
        for pid in phases:
            missing = missing_inputs(pid, cfg)
            out.verdict(not missing, f'stage {pid}: inputs ' + ('present' if not missing else 'MISSING'))
            for m in missing:
                out.info('  missing', out.short_path(m))
        return

    start = time.time()
    for pid in phases:
        _, script, title = next(p for p in PHASES if p[0] == pid)
        if not run_phase(pid, script, title, args.date, args.tag, args.cheat, args.placebo):
            out.fail(f'PIPELINE STOPPED at stage {pid}.')
            sys.exit(1)

    print('\n')
    out.stage('04-09', 'Summary of the experiment', f'{run_id}: all stages done in {_duration((time.time() - start) / 60)} '
              f'({", ".join(f"{k} {_duration(v)}" for k, v in TIMES.items())})')
    print_headline(cfg)


if __name__ == '__main__':
    main()