#!/usr/bin/env python
'''
Size and power of the signal tests of stage 05, on simulated IC series with a KNOWN truth.

QUESTION ANSWERED
    "When stage 05 says 'significant at 5%', is the false-positive rate really 5%? And how often does it find
     a real signal?"

WHAT IS SIMULATED (src/evaluation/signals/calibration.py)
    A daily IC series with the two memories of a real one: overlapping labels (h days) and a slow signal
    (persistence p days). True mean IC = 0 gives the SIZE. True mean IC > 0 gives the POWER.

USAGE
    python scripts/research/calibrate_tests.py                 # about 1 minute
    python scripts/research/calibrate_tests.py --sims 1000     # more precise, slower
'''

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.evaluation.signals.calibration import rejection_rates
from src.utils import console as out

N_DAYS = 2500           # about 10 years of daily IC
HORIZON = 20
PERSISTENCES = [1, 20, 60]
TRUE_ICS = [0.01, 0.02, 0.03]


def main(n_sims: int, seed: int) -> None:
    out.stage('--', 'Calibrate the signal tests', 'Is "significant at 5%" really a 5% false-positive rate?')
    out.info('simulated series', f'{n_sims} per cell, {N_DAYS} days, label horizon {HORIZON} days, IC std 0.15')
    se = np.sqrt(0.05 * 0.95 / n_sims)
    out.note(f'with {n_sims} simulations a rate has a standard error of about {se:.1%} at 5%')

    out.step(1, 2, 'SIZE: true IC = 0. Share of series called significant at 5% (should be about 5%)',
             'a test above 5% admits more junk than it promises')
    rows = []
    for p in PERSISTENCES:
        r = rejection_rates(n_sims, N_DAYS, HORIZON, 0.0, persistence=p, seed=seed + p)
        rows.append([f'{p} days'] + [f'{v:.1%}' for v in r.values()])
        names = list(r)
    out.table(['signal memory'] + names, rows, [-14, 24, 24, 22])

    out.step(2, 2, 'POWER: true IC > 0, signal memory 20 days. Share of series called significant',
             'higher is better, but only for a test whose size is right: an over-rejecting test looks powerful')
    rows = []
    for ic in TRUE_ICS:
        r = rejection_rates(n_sims, N_DAYS, HORIZON, ic, persistence=20, seed=seed + int(ic * 1000))
        rows.append([f'IC {ic:.2f}'] + [f'{v:.1%}' for v in r.values()])
    out.table(['true IC'] + names, rows, [-14, 24, 24, 22])
    out.note('the two-sided tests count only rejections with the right sign; the bootstrap is one-sided, '
             'as stage 05 uses it with a prior sign')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--sims', type=int, default=300, help='simulated series per cell')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    main(a.sims, a.seed)
