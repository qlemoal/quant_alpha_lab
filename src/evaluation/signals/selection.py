'''
GOAL: decide which signals are ADMITTED to the combiner (stage 06). This is the strict selection.

THE GATES (a signal must pass ALL, in this order; the first failed gate is reported)

    1 prior   the signal has an economic prior (src/signals/priors.py): a family, a sign, a rationale
    2 sign    the measured mean IC has the prior sign
    3 fwer    p_all < fwer_alpha (default 0.05), where
                  p_all = min(1, p_RW x (K + n_trials_extra) / K)
              p_RW: Romano-Wolf adjusted p-value over the K signals of this run (src/evaluation/signals/bootstrap.py)
              n_trials_extra: signals tried in EARLIER runs on the same data

    signal --> [prior?] --> [sign ok?] --> [p_all < 0.05?] --> ADMITTED
                  | no          | no             | no
                  v             v                v
              rejected      rejected         rejected

WHAT EACH GATE PROTECTS AGAINST
    prior   data mining: a pattern with no economic story.
    sign    a pattern in the wrong direction is not the effect we believe in.
    fwer    luck among ALL the signals tried. A studentized block bootstrap gives each signal a p-value with the
            right false-positive rate (about 5% at 5%, checked by simulation), and Romano-Wolf adjusts it for the
            K signals of this run, using their real correlation.
            Earlier runs: Romano-Wolf cannot see them. If each earlier run was tested the same way, the runs are
            (K + n_trials_extra) / K families, and the Bonferroni bound over families gives the factor above.
            Admitting with p_all < 5% means: the chance that ANY admitted signal is pure luck is below 5%.

WHY NOT THE OLD GATES (Newey-West t >= 3, non-overlapping t >= 2, q < 0.05, deflated Sharpe >= 0.95)?
    - Newey-West and the FDR rules built on it: their false-positive rate is not known. In simulation Newey-West
      says "significant" 9% to 22% of the time at a nominal 5% when the signal is slow
      (scripts/research/calibrate_tests.py).
    - The deflated Sharpe ratio keeps one IC every h days (95% of the data are dropped at h = 20). On the synthetic
      market it stopped planted effects that Romano-Wolf found. It controls the same luck, with less power.
    They are still computed and printed as cross-checks. They decide nothing.

WHAT THIS FILE DOES NOT DO
    It computes nothing from prices. It receives the table of statistics and applies the rules.
'''

from dataclasses import dataclass, asdict

import numpy as np
import polars as pl

GATES = ('prior', 'sign', 'fwer')

# One line per gate, printed by stage 05 so that a human knows what was tested and why.
GATE_WHY = {
    'prior': 'has an economic story written before the test (src/signals/priors.py)',
    'sign': 'mean IC has the sign the story predicts',
    'fwer': 'p_all < alpha: not luck, among the signals of this run AND of earlier runs',
}


@dataclass(frozen=True)
class SelectionConfig:
    '''The thresholds. Each one has a command-line option in stage 05.'''
    fwer_alpha: float = 0.05           # p_all below this
    require_prior: bool = True         # False: a signal without prior skips gates 1 and 2 (exploration only)
    n_trials_extra: int = 0            # signals tried in earlier runs on the same data (gate 3 and the DSR count them)
    dsr_variance: str = 'null'         # DSR cross-check: 'null' (sampling variance) or 'trials' (paper). See dsr.py
    n_boot: int = 1000                 # bootstrap resamples (more = more precise p-values, slower)
    boot_seed: int = 0


def _num(x) -> float:
    '''None -> nan, so that every comparison below is False for a missing statistic (a missing value fails).'''
    return float('nan') if x is None else float(x)


def _gate_prior_and_sign(sign, mean_ic: float, require_prior: bool) -> "tuple[bool, bool]":
    '''Gates 1 and 2. Without a prior both pass only in exploration mode (require_prior=False).'''
    if sign in (1, -1):
        return True, bool(np.isfinite(mean_ic) and mean_ic * sign > 0)
    return (not require_prior), (not require_prior)


def earlier_trials_factor(n_signals: int, n_trials_extra: int) -> float:
    '''(K + n_trials_extra) / K: the number of families (runs) of size K that the trials make, at least 1.'''
    k = max(1, int(n_signals))
    return (k + max(0, int(n_trials_extra))) / k


def apply_gates(stats: list, cfg: SelectionConfig) -> pl.DataFrame:
    '''
    stats: one dict per signal with the keys
        signal, sign (+1/-1/None), family (str/None), mean_ic, p_single, p_rw
        and, as cross-checks (not gates): dsr, nw_tstat, nonoverlap_tstat, qvalue
    K = len(stats). p_all = min(1, p_rw x earlier_trials_factor(K, cfg.n_trials_extra)).
    Returns a DataFrame with one row per signal: the statistics, p_all, one bool per gate,
    'selected' (all gates passed) and 'failed_gate' (name of the FIRST failed gate, None if selected).
    A missing (None / nan) statistic fails its gate.
    '''
    factor = earlier_trials_factor(len(stats), cfg.n_trials_extra)
    rows = []
    for s in stats:
        sign = s.get('sign')
        mean_ic, p_rw, dsr = _num(s.get('mean_ic')), _num(s.get('p_rw')), _num(s.get('dsr'))
        p_all = min(1.0, p_rw * factor) if np.isfinite(p_rw) else float('nan')
        ok_prior, ok_sign = _gate_prior_and_sign(sign, mean_ic, cfg.require_prior)
        gates = {
            'prior': ok_prior,
            'sign': ok_sign,
            'fwer': bool(np.isfinite(p_all) and p_all < cfg.fwer_alpha),
        }
        failed = next((g for g in GATES if not gates[g]), None)
        rows.append({
            'signal': s['signal'], 'family': s.get('family'), 'sign': sign,
            'mean_ic': mean_ic, 'p_single': _num(s.get('p_single')), 'p_rw': p_rw, 'p_all': p_all, 'dsr': dsr,
            'nw_tstat': _num(s.get('nw_tstat')), 'nonoverlap_tstat': _num(s.get('nonoverlap_tstat')),
            'qvalue': _num(s.get('qvalue')),
            **{f'pass_{g}': v for g, v in gates.items()},
            'selected': failed is None, 'failed_gate': failed,
        })
    schema_overrides = {'family': pl.Utf8, 'sign': pl.Int64, 'failed_gate': pl.Utf8}
    return pl.DataFrame(rows, schema_overrides=schema_overrides, infer_schema_length=None)


def gate_summary(table: pl.DataFrame) -> dict:
    '''How many signals each gate stopped: {gate: count}. Plus the number admitted. Useful for the log.'''
    out = {g: int((table['failed_gate'] == g).sum()) for g in GATES}
    out['admitted'] = int(table['selected'].sum())
    out['tested'] = table.height
    return out


def config_dict(cfg: SelectionConfig) -> dict:
    return asdict(cfg)
