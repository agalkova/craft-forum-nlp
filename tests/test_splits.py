import pandas as pd

from craft_forum_nlp.data import load_data
from craft_forum_nlp.splits import CV_RANDOM_STATE, OUTER_SPLITS, make_outer_folds


def test_outer_folds_are_grouped_disjoint_and_complete():
    frame = load_data("craft_forum_dataset.csv")
    folds = make_outer_folds(frame)
    held_out_ids = []
    held_out_threads = []

    assert len(folds) == OUTER_SPLITS == 5
    assert CV_RANDOM_STATE == 42
    for fold in folds:
        train_threads = set(fold.train["source_url"])
        test_threads = set(fold.test["source_url"])
        assert train_threads.isdisjoint(test_threads)
        held_out_ids.extend(fold.test["id"])
        held_out_threads.extend(test_threads)

    assert len(held_out_ids) == len(set(held_out_ids)) == len(frame)
    assert set(held_out_ids) == set(frame["id"])
    assert len(held_out_threads) == len(set(held_out_threads))
    assert set(held_out_threads) == set(frame["source_url"])


def test_fold_assignment_is_stable_when_input_order_changes():
    frame = load_data("craft_forum_dataset.csv")
    expected = make_outer_folds(frame)
    reversed_frame = frame.iloc[::-1].reset_index(drop=True)
    actual = make_outer_folds(reversed_frame)

    for expected_fold, actual_fold in zip(expected, actual, strict=True):
        assert expected_fold.number == actual_fold.number
        assert list(expected_fold.test["id"]) == list(actual_fold.test["id"])


def test_saved_fold_assignments_cover_each_thread_once_without_overlap():
    frame = load_data("craft_forum_dataset.csv")
    assignments = pd.read_csv("results/fold_assignments.csv")

    assert len(assignments) == len(frame)
    assert assignments["id"].is_unique
    assert set(assignments["id"]) == set(frame["id"])
    assert assignments.groupby("source_url")["fold"].nunique().eq(1).all()

    all_threads = set(frame["source_url"])
    for fold_number in sorted(assignments["fold"].unique()):
        test_threads = set(
            assignments.loc[assignments["fold"].eq(fold_number), "source_url"]
        )
        train_threads = all_threads.difference(test_threads)
        assert train_threads.isdisjoint(test_threads)
