from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from craft_forum_nlp.splits import CV_RANDOM_STATE, INNER_SPLITS, make_inner_folds


TFIDF_C_VALUES = (0.25, 1.0, 4.0)


@dataclass(frozen=True)
class ModelResult:
    name: str
    predictions: np.ndarray
    accuracy: float
    macro_f1: float
    weighted_f1: float
    parameters: dict[str, object]


def score_predictions(
    name: str,
    truth: pd.Series,
    predictions: np.ndarray,
    parameters: dict[str, object],
) -> ModelResult:
    return ModelResult(
        name=name,
        predictions=predictions,
        accuracy=float(accuracy_score(truth, predictions)),
        macro_f1=float(f1_score(truth, predictions, average="macro", zero_division=0)),
        weighted_f1=float(
            f1_score(truth, predictions, average="weighted", zero_division=0)
        ),
        parameters=parameters,
    )


def evaluate_majority(train: pd.DataFrame, test: pd.DataFrame) -> ModelResult:
    majority_label = str(train["label"].mode().iloc[0])
    predictions = np.repeat(majority_label, len(test))
    return score_predictions(
        "majority",
        test["label"],
        predictions,
        {"majority_label": majority_label},
    )


def _tfidf_pipeline(c_value: float) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=CV_RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_tfidf(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[ModelResult, list[dict[str, object]]]:
    """Tune inside the outer training set, then evaluate its held-out threads."""
    candidates: list[tuple[float, float]] = []
    validation_results: list[dict[str, object]] = []
    inner_folds = make_inner_folds(train)

    for c_value in TFIDF_C_VALUES:
        truth_parts = []
        prediction_parts = []
        fold_metrics = []
        for fold in inner_folds:
            model = _tfidf_pipeline(c_value)
            model.fit(fold.train["text"], fold.train["label"])
            predictions = model.predict(fold.test["text"])
            truth_parts.append(fold.test["label"])
            prediction_parts.append(predictions)
            fold_metrics.append(
                {
                    "fold": fold.number,
                    "accuracy": float(accuracy_score(fold.test["label"], predictions)),
                    "macro_f1": float(
                        f1_score(
                            fold.test["label"],
                            predictions,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                }
            )

        truth = pd.concat(truth_parts, ignore_index=True)
        pooled_predictions = np.concatenate(prediction_parts)
        macro_f1 = float(
            f1_score(truth, pooled_predictions, average="macro", zero_division=0)
        )
        accuracy = float(accuracy_score(truth, pooled_predictions))
        validation_results.append(
            {
                "C": c_value,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "folds": fold_metrics,
            }
        )
        candidates.append((macro_f1, -c_value))

    _, negative_c = max(candidates)
    selected_c = float(-negative_c)
    selected_model = _tfidf_pipeline(selected_c)
    selected_model.fit(train["text"], train["label"])
    test_predictions = selected_model.predict(test["text"])
    result = score_predictions(
        "tfidf_logistic_regression",
        test["label"],
        test_predictions,
        {
            "C": selected_c,
            "class_weight": "balanced",
            "ngram_range": [1, 2],
            "selection_metric": "inner_grouped_oof_macro_f1",
            "inner_splits": INNER_SPLITS,
        },
    )
    return result, validation_results
