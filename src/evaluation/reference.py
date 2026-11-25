'''
GOAL: compare the numbers of a reference run (synthetic market) with the stored golden numbers.

WHY
    Two machines (or two versions of the code) run the same synthetic experiment. The numbers must agree.
    They need not agree to the last digit: libraries differ slightly. So each metric has a rule:

        exact   the value must be identical (a list of signals, a count)
        abs     |measured - golden| <= tolerance
        max     measured <= bound  (a quantity that must be about 0, whatever the golden value)

    A FAIL is information: either the code changed behaviour, or a library behaves differently on this machine.
'''

import numpy as np

RULES = {
    # metric: (rule, parameter)
    'admitted_signals': ('exact', None),
    'families': ('exact', None),
    'holdout_ic': ('abs', 0.01),
    'avg_n_clusters': ('abs', 1.0),
    'cluster_ari': ('abs', 0.10),
    'market_mode_removed_share': ('abs', 0.05),
    'beta_exposure_max_abs': ('max', 1e-6),
    'realised_beta_full': ('max', 0.05),
    'sharpe_gross_full': ('abs', 0.15),
    'sharpe_net_full': ('abs', 0.15),
    'annual_cost_bps_full': ('abs', 40.0),
    'alpha_t_full': ('abs', 0.4),
}


def compare(measured: dict, golden: dict, rules: dict = RULES) -> list:
    '''
    One row per rule: {metric, rule, measured, golden, ok}. A metric missing from `measured` fails.
    For 'max' the absolute value of the measurement is tested against the bound.
    '''
    rows = []
    for name, (rule, param) in rules.items():
        m, g = measured.get(name), golden.get(name)
        if m is None:
            ok = False
        elif rule == 'exact':
            ok = (m == g)
        elif rule == 'abs':
            ok = g is not None and abs(m - g) <= param
        elif rule == 'max':
            ok = abs(m) <= param
        else:
            raise ValueError(f'unknown rule {rule}')
        rows.append({'metric': name, 'rule': rule if param is None else f'{rule} {param:g}',
                     'measured': m, 'golden': g, 'ok': bool(ok)})
    return rows


def adjusted_rand(labels_a, labels_b) -> float:
    '''Adjusted Rand index of two labelings (1 = same partition, 0 = chance). No sklearn needed here.'''
    a, b = np.asarray(labels_a), np.asarray(labels_b)
    _, ia = np.unique(a, return_inverse=True)
    _, ib = np.unique(b, return_inverse=True)
    table = np.zeros((ia.max() + 1, ib.max() + 1))
    for i, j in zip(ia, ib):
        table[i, j] += 1
    c2 = lambda x: x * (x - 1) / 2.0
    sum_ij, sum_a, sum_b = c2(table).sum(), c2(table.sum(1)).sum(), c2(table.sum(0)).sum()
    expected = sum_a * sum_b / c2(len(a))
    maximum = 0.5 * (sum_a + sum_b)
    return float((sum_ij - expected) / (maximum - expected)) if maximum != expected else 1.0
