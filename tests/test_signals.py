'''
Test signals on all the defined panels in conftest, to see if every method in make_signal is actually working before evaluating signals.
Run this file using pytest "tests/test_signals.py -v"
'''



import pytest
from src.signals.combine import make_signal



PANELS_THAT_SHOULD_NEVER_CRASH = [
    'panel_normal_multidate',
    'panel_unsorted',
    'panel_ragged',
    'panel_single_ticker_multidate',
    'panel_identical_values',
    'panel_with_nulls',
    'panel_uneven_group',
    'panel_with_ties',
]



@pytest.mark.parametrize('panel_fixture', PANELS_THAT_SHOULD_NEVER_CRASH)
@pytest.mark.parametrize('method', ['zscore_tanh', 'zscore_clip', 'rank', 'decile'])
def test_make_signal_survives_every_panel(panel_fixture, method, request):
    panel = request.getfixturevalue(panel_fixture)
    result = make_signal(panel.lazy(), 'feat', method=method).collect()
    assert result.height == panel.height


