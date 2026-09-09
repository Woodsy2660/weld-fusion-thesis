"""No coupon_id may appear in both train and test of any fold."""

from __future__ import annotations

from weldfusion.evaluation.splits import nested_group_kfold
from weldfusion.registration.cells import all_cells


def test_no_coupon_in_both_train_and_test():
    cells = all_cells()  # 15 coupons × 24 cells
    groups = [c.coupon_id for c in cells]
    folds = nested_group_kfold(groups, n_outer=5, n_inner=3)
    assert len(folds) == 5
    for fold in folds:
        train_g = {groups[i] for i in fold["train"]}
        test_g = {groups[i] for i in fold["test"]}
        assert train_g.isdisjoint(test_g), train_g & test_g
        assert train_g and test_g
        for inner in fold["inner"]:
            inner_tr = {groups[i] for i in inner["train"]}
            inner_cal = {groups[i] for i in inner["calibrate"]}
            assert inner_tr.isdisjoint(inner_cal), inner_tr & inner_cal
            assert inner_tr.isdisjoint(test_g)
            assert inner_cal.isdisjoint(test_g)


def test_all_samples_assigned_once_per_outer():
    cells = all_cells()
    groups = [c.coupon_id for c in cells]
    n = len(groups)
    folds = nested_group_kfold(groups)
    for fold in folds:
        assigned = set(fold["train"]) | set(fold["test"])
        assert assigned == set(range(n))
        assert set(fold["train"]).isdisjoint(set(fold["test"]))
