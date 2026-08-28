'''
Quick standalone check for src/validation/cpcv.py, not a full pytest,
just enough to confirm the module does what the docstrings claim before trusting it further. 

Run using "python test_cpcv_quick.py" or "pytest test_cpcv_quick.py"
'''

from src.validation.cpcv import cpcv


def test_single_group_purge_embargo():
    for train_idx, test_idx, test_ids in cpcv(
        n_dates=20, n_blocks=5, n_test_blocks=1, purge_w=2, embargo_w=1
    ):
        if test_ids == (2,):
            assert sorted(test_idx.tolist()) == [8, 9, 10, 11]
            assert sorted(train_idx.tolist()) == [0, 1, 2, 3, 4, 5, 13, 14, 15, 16, 17, 18, 19]
            print('single-group case: OK')
            return
    raise AssertionError('test_ids (2,) never generated')


def test_adjacent_groups_merge_into_one_run():
    for train_idx, test_idx, test_ids in cpcv(
        n_dates=20, n_blocks=5, n_test_blocks=2, purge_w=2, embargo_w=1
    ):
        if test_ids == (1, 2):
            excluded = sorted(set(range(20)) - set(test_idx.tolist()) - set(train_idx.tolist()))
            # internal boundary between groups 1 and 2 (dates 7/8) should
            # NOT be purged or embargoed, only the outer edges (before 4,
            # after 11) should show up here
            assert excluded == [2, 3, 12], excluded
            print('adjacent-groups merge case: OK')
            return
    raise AssertionError('test_ids (1, 2) never generated')


def test_no_overlap_and_full_accounting():
    # every combination should partition [0, n_dates) into disjoint
    # train / test / excluded with no date double-counted or dropped
    n_dates = 20
    for train_idx, test_idx, test_ids in cpcv(
        n_dates=n_dates, n_blocks=5, n_test_blocks=2, purge_w=2, embargo_w=1
    ):
        train_set, test_set = set(train_idx.tolist()), set(test_idx.tolist())
        assert train_set.isdisjoint(test_set), f'{test_ids}: train/test overlap'
        # train ∪ test ∪ excluded must equal the full range, nothing lost,
        # nothing duplicated
        excluded = set(range(n_dates)) - train_set - test_set
        assert train_set | test_set | excluded == set(range(n_dates))
    print('no-overlap / full-accounting across all combinations: OK')


if __name__ == '__main__':
    test_single_group_purge_embargo()
    test_adjacent_groups_merge_into_one_run()
    test_no_overlap_and_full_accounting()
    print('all checks passed')