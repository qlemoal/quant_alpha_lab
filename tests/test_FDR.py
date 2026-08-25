'''
Tests for FDR control (src/evaluation/signals/FDR.py) against p-value edge cases in conftest.py. 
Run this file using "pytest tests/test_FDR.py -v"
'''

import numpy as np
import polars as pl
import pytest

from src.evaluation.signals.fdr import fdr_report





PVAL_FIXTURES_THAT_SHOULD_NEVER_CRASH = [
    'pvals_all_significant', 'pvals_all_null', 'pvals_single_candidate',
    'pvals_identical', 'pvals_at_threshold_boundary', 'pvals_mixed_with_nan',
    'pvals_empty', 'pvals_bh_by_disagree',
]

@pytest.mark.parametrize('pval_fixture', PVAL_FIXTURES_THAT_SHOULD_NEVER_CRASH)
def test_fdr_report_survives_every_panel(pval_fixture, request):
    pvals = request.getfixturevalue(pval_fixture)
    names = [f's{i}' for i in range(len(pvals))]
    df = fdr_report(names, pvals)
    assert df.height == len(pvals)

def test_all_significant_survives_both(pvals_all_significant):
    names = [f's{i}' for i in range(len(pvals_all_significant))]
    df = fdr_report(names, pvals_all_significant)
    assert df['survives_bh'].all() and df['survives_by'].all()

def test_all_null_survives_neither(pvals_all_null):
    names = [f's{i}' for i in range(len(pvals_all_null))]
    df = fdr_report(names, pvals_all_null)
    assert not df['survives_bh'].any() and not df['survives_by'].any()

def test_single_candidate_bh_by_agree(pvals_single_candidate):
    df = fdr_report(['s0'], pvals_single_candidate)
    assert df['survives_bh'][0] == df['survives_by'][0]

def test_by_never_more_permissive_than_bh(pvals_bh_by_disagree):
    # BY's threshold is always <= BH's (H_m >= 1 for m >= 1), so anything
    # surviving BY should also survive BH, never the reverse.
    names = [f's{i}' for i in range(len(pvals_bh_by_disagree))]
    df = fdr_report(names, pvals_bh_by_disagree)
    by_survivors = set(df.filter(pl.col('survives_by'))['signal'].to_list())
    bh_survivors = set(df.filter(pl.col('survives_bh'))['signal'].to_list())
    assert by_survivors.issubset(bh_survivors)

def test_nan_pvalue_fails_closed(pvals_mixed_with_nan):
    names = [f's{i}' for i in range(len(pvals_mixed_with_nan))]
    df = fdr_report(names, pvals_mixed_with_nan)
    for i, p in enumerate(pvals_mixed_with_nan):
        if np.isnan(p):
            row = df.filter(pl.col('signal') == names[i])
            assert not row['survives_bh'][0] and not row['survives_by'][0]

def test_empty_does_not_crash(pvals_empty):
    assert fdr_report([], pvals_empty).height == 0