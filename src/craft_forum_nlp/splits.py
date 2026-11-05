from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


CV_RANDOM_STATE = 42
OUTER_SPLITS = 5
INNER_SPLITS = 4


@dataclass(frozen=True)
class ThreadFold:
    number: int
    train: pd.DataFrame
    test: pd.DataFrame


def make_outer_folds(frame: pd.DataFrame) -> tuple[ThreadFold, ...]:
    """Create five deterministic evaluation folds without selecting among them."""
    return _make_thread_folds(frame, n_splits=OUTER_SPLITS)


def make_inner_folds(frame: pd.DataFrame) -> tuple[ThreadFold, ...]:
    """Create grouped folds for model selection inside an outer training set."""
    return _make_thread_folds(frame, n_splits=INNER_SPLITS)


def _make_thread_folds(
    frame: pd.DataFrame,
    *,
    n_splits: int,
) -> tuple[ThreadFold, ...]:
    frame = frame.sort_values("id").reset_index(drop=True)
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=CV_RANDOM_STATE,
    )
    folds = tuple(
        ThreadFold(
            number=fold_number,
            train=frame.iloc[train_idx].reset_index(drop=True),
            test=frame.iloc[test_idx].reset_index(drop=True),
        )
        for fold_number, (train_idx, test_idx) in enumerate(
            splitter.split(frame, frame["label"], groups=frame["source_url"]),
            start=1,
        )
    )
    _validate_folds(frame, folds)
    return folds


def _validate_folds(frame: pd.DataFrame, folds: tuple[ThreadFold, ...]) -> None:
    expected_ids = set(frame["id"])
    expected_threads = set(frame["source_url"])
    test_ids: list[int] = []
    test_threads: list[str] = []

    for fold in folds:
        train_ids = set(fold.train["id"])
        held_out_ids = set(fold.test["id"])
        train_threads = set(fold.train["source_url"])
        held_out_threads = set(fold.test["source_url"])
        if train_ids.intersection(held_out_ids):
            raise ValueError(f"Rows overlap in fold {fold.number}")
        if train_threads.intersection(held_out_threads):
            raise ValueError(f"Source threads overlap in fold {fold.number}")
        test_ids.extend(fold.test["id"].tolist())
        test_threads.extend(fold.test["source_url"].drop_duplicates().tolist())

    if len(test_ids) != len(set(test_ids)) or set(test_ids) != expected_ids:
        raise ValueError("Each row must appear in exactly one outer test fold")
    if len(test_threads) != len(set(test_threads)) or set(test_threads) != expected_threads:
        raise ValueError("Each source thread must appear in exactly one outer test fold")
