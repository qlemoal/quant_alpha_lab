'''
GOAL: decide which signals are ADMITTED to the combiner (stage 06). This is the strict selection.

THE GATES (a signal must pass ALL, in this order; the first failed gate is reported)

    1 prior     the signal has an economic prior (src/signals/priors.py): a family, a sign, a rationale
    2 sign      the measured sign of the IC equals the prior sign
    3 t_nw      sign-aligned Newey-West t-stat >= min_t              (default 3.0, Harvey-Liu-Zhu 2016)
    4 t_nonov   sign-aligned non-overlapping t-stat >= min_nonoverlap_t   (default 2.0)
    5 fdr       q-value < fdr_alpha                                    (default 0.05)
    6 dsr       deflated Sharpe ratio >= min_dsr                       (default 0.95)

    signal --> [prior?] --> [sign ok?] --> [t_nw >= 3?] --> [t_nonov >= 2?] --> [q < 0.05?] --> [DSR >= 0.95?] --> ADMITTED
                  | no          | no            | no              | no                | no             | no
                  v             v               v                 v                   v                v
                rejected      rejected        rejected          rejected            rejected         rejected

WHY SO MANY GATES?
    Each gate catches a different failure:
      prior      data mining: a pattern with no story.
      sign       a pattern that is significant in the wrong direction is not the effect we believe in.
      t_nw       multiple testing: with hundreds of published factors tried, t = 2 is not enough (HLZ 2016).
      t_nonov    a Newey-West t that is too large because the label overlap is not fully corrected.
                 The non-overlapping t uses one IC every h days. It is exact, but has little power.
      fdr        the share of false discoveries among the admitted signals.
      dsr        the best of N tries is biased upwards (see dsr.py).
    The gates are strict ON PURPOSE. Admitting nothing is a valid result: it says the data show no robust signal.

WHAT THIS FILE DOES NOT DO
    It computes nothing from prices. It receives the table of statistics and applies the rules.
    That keeps it fast to test.
'''

from dataclasses import dataclass, asdict

import numpy as np
import polars as pl

GATES = ('prior', 'sign', 't_nw', 't_nonov', 'fdr', 'dsr')


@dataclass(frozen=True)
class SelectionConfig:
    '''The thresholds. Defaults follow the literature; each one has a command-line option in stage 05.'''
    min_t: float = 3.0                 # Newey-West, sign-aligned
    min_nonoverlap_t: float = 2.0      # non-overlapping, sign-aligned
    fdr_alpha: float = 0.05            # q-value below this
    min_dsr: float = 0.95
    require_prior: bool = True         # False: a signal without prior skips gates 1 and 2 (exploration only)
    n_trials_extra: int = 0            # variants tried and removed earlier; they count in the DSR
    dsr_variance: str = 'null'         # 'null' (sampling variance, default) or 'trials' (paper, stricter). See dsr.py


def _num(x) -> float:
    '''None -> nan, so that every comparison below is False for a missing statistic (a missing value fails).'''
    return float('nan') if x is None else float(x)


def _ge(x: float, thr: float) -> bool:
    return bool(np.isfinite(x) and x >= thr)


def apply_gates(stats: list, cfg: SelectionConfig) -> pl.DataFrame:
    '''
    stats: one dict per signal with the keys
        signal, sign (+1/-1/None), family (str/None), mean_ic, nw_tstat, nonoverlap_tstat, qvalue, dsr
    Returns a DataFrame with one row per signal: the statistics, aligned t-stats, one bool per gate,
    'selected' (all gates passed) and 'failed_gate' (name of the FIRST failed gate, None if selected).
    A missing (None / nan) statistic fails its gate.
    '''
    rows = []
    for s in stats:
        sign = s.get('sign')
        has_prior = sign in (1, -1)
        mean_ic = _num(s.get('mean_ic'))
        sgn = sign if has_prior else 1                       # no prior: align with +1 (the raw t-stat)
        t_nw = _num(s.get('nw_tstat')) * sgn
        t_no = _num(s.get('nonoverlap_tstat')) * sgn
        q = _num(s.get('qvalue'))
        dsr = _num(s.get('dsr'))

        if has_prior:
            ok_prior = True
            ok_sign = bool(np.isfinite(mean_ic) and mean_ic * sign > 0)
        else:
            ok_prior = not cfg.require_prior
            ok_sign = not cfg.require_prior                   # nothing to compare with; skipped in exploration mode

        gates = {
            'prior': ok_prior,
            'sign': ok_sign,
            't_nw': _ge(t_nw, cfg.min_t),
            't_nonov': _ge(t_no, cfg.min_nonoverlap_t),
            'fdr': bool(np.isfinite(q) and q < cfg.fdr_alpha),
            'dsr': _ge(dsr, cfg.min_dsr),
        }
        failed = next((g for g in GATES if not gates[g]), None)
        rows.append({
            'signal': s['signal'], 'family': s.get('family'), 'sign': sign,
            'mean_ic': mean_ic, 'nw_tstat': _num(s.get('nw_tstat')), 'nonoverlap_tstat': _num(s.get('nonoverlap_tstat')),
            'aligned_nw_t': t_nw, 'aligned_nonoverlap_t': t_no, 'qvalue': q, 'dsr': dsr,
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