'''Tests of the benchmark report (src/evaluation/signals/benchmarks.py): how the selection treats cheat and placebo signals.'''

import numpy as np
import polars as pl

from src.evaluation.signals.benchmarks import benchmark_report, format_benchmark_report
from src.evaluation.signals.fdr import fdr_report


def table(names, pvalues, alpha=0.2):
    return fdr_report(names, np.zeros(len(pvalues)), pvalues, alpha=alpha)


def test_no_benchmark_gives_none():
    full = table(['mom5_zscore_tanh', 'vol20_zscore_clip'], [0.001, 0.5])
    assert benchmark_report(full, None, 0.2) is None


def test_placebo_counts_and_chance_level():
    names = ['mom5_zscore_tanh'] + [f'placebo_{i:02d}' for i in range(1, 21)]
    p = [1e-6] + list(np.linspace(0.02, 0.98, 20))                    # 20 placebos, uniform p-values like a null: 1 below 0.05
    rep = benchmark_report(table(names, p), None, 0.2)
    assert rep['placebo']['n'] == 20 and rep['real']['n'] == 1
    assert rep['placebo']['n_p_below_005'] == 1
    assert rep['placebo']['expected_p_below_005'] == 1.0
    assert rep['placebo']['selected_q'] == []
    assert rep['real']['selected_q'] == ['mom5_zscore_tanh']
    text = '\n'.join(format_benchmark_report(rep))
    assert 'no placebo selected' in text


def test_a_selected_placebo_is_flagged():
    names = ['placebo_01', 'placebo_02']
    rep = benchmark_report(table(names, [1e-9, 0.9]), None, 0.2)
    assert rep['placebo']['selected_q'] == ['placebo_01']
    assert 'WARNING' in '\n'.join(format_benchmark_report(rep))


def test_dilution_a_true_signal_lets_a_marginal_one_pass():
    '''
    The same marginal real signal (p = 0.04) among 4 real signals is NOT selected alone,
    but IS selected when 4 sure signals (p ~ 0) sit in the family: the BH threshold k * alpha / m grows with k.
    '''
    real = ['r1', 'r2', 'r3', 'r4']
    p_real = [0.04, 0.5, 0.6, 0.7]
    cheats = [f'cheat_h20_ic{i:03d}' for i in (10, 20, 30, 50)]
    full = table(real + cheats, p_real + [1e-12] * 4, alpha=0.1)
    alone = table(real, p_real, alpha=0.1)
    rep = benchmark_report(full, alone, 0.1)
    assert 'r1' not in rep['real']['selected_alone']
    assert 'r1' in rep['real']['selected_with_benchmarks']
    assert rep['real']['only_thanks_to_benchmarks'] == ['r1']


def test_more_tests_can_cost_a_real_signal():
    '''20 placebos make the family larger: the threshold of a borderline real signal gets stricter.'''
    real = ['r1']
    placebos = [f'placebo_{i:02d}' for i in range(1, 21)]
    p_pl = list(np.linspace(0.2, 0.99, 20))
    full = table(real + placebos, [0.02] + p_pl, alpha=0.1)
    alone = table(real, [0.02], alpha=0.1)
    rep = benchmark_report(full, alone, 0.1)
    assert rep['real']['selected_alone'] == ['r1']
    assert rep['real']['lost_because_of_benchmarks'] == ['r1']