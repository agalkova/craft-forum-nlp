import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

from craft_forum_nlp.data import load_data
from craft_forum_nlp.models import evaluate_majority, evaluate_tfidf
from craft_forum_nlp.splits import make_outer_folds


def test_lightweight_benchmarks_return_one_prediction_per_outer_test_row():
    fold = make_outer_folds(load_data("craft_forum_dataset.csv"))[0]

    majority = evaluate_majority(fold.train, fold.test)
    tfidf, inner_results = evaluate_tfidf(fold.train, fold.test)

    assert len(majority.predictions) == len(fold.test)
    assert len(tfidf.predictions) == len(fold.test)
    assert len(inner_results) == 3
    assert all(len(candidate["folds"]) == 4 for candidate in inner_results)
    assert tfidf.parameters["C"] in {0.25, 1.0, 4.0}


def test_published_metrics_recompute_from_saved_oof_predictions():
    predictions = pd.read_csv("results/predictions.csv")
    published = pd.read_csv("results/metrics.csv").set_index("model")
    published_folds = pd.read_csv("results/fold_metrics.csv")

    assert len(predictions) == 212
    assert predictions["id"].is_unique
    for model in published.index:
        accuracy = accuracy_score(predictions["label"], predictions[model])
        macro_f1 = f1_score(
            predictions["label"],
            predictions[model],
            average="macro",
            zero_division=0,
        )
        assert published.loc[model, "accuracy"] == pytest.approx(accuracy, abs=1e-6)
        assert published.loc[model, "macro_f1"] == pytest.approx(macro_f1, abs=1e-6)

        model_folds = published_folds[published_folds["model"].eq(model)]
        for row in model_folds.itertuples(index=False):
            fold_predictions = predictions[predictions["fold"].eq(row.fold)]
            fold_accuracy = accuracy_score(
                fold_predictions["label"], fold_predictions[model]
            )
            fold_macro_f1 = f1_score(
                fold_predictions["label"],
                fold_predictions[model],
                average="macro",
                zero_division=0,
            )
            assert row.accuracy == pytest.approx(fold_accuracy, abs=1e-6)
            assert row.macro_f1 == pytest.approx(fold_macro_f1, abs=1e-6)

        assert published.loc[model, "fold_macro_f1_min"] == pytest.approx(
            model_folds["macro_f1"].min(), abs=1e-6
        )
        assert published.loc[model, "fold_macro_f1_max"] == pytest.approx(
            model_folds["macro_f1"].max(), abs=1e-6
        )
