from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from craft_forum_nlp.data import LABELS
from craft_forum_nlp.models import ModelResult
from craft_forum_nlp.splits import CV_RANDOM_STATE, OUTER_SPLITS, ThreadFold


DISPLAY_NAMES = {
    "majority": "Majority baseline",
    "tfidf_logistic_regression": "TF–IDF + logistic regression",
    "distilbert": "DistilBERT",
    "distilbert_weighted": "DistilBERT + class weights",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evaluation_frame(folds: tuple[ThreadFold, ...]) -> pd.DataFrame:
    return pd.concat(
        [fold.test.assign(fold=fold.number) for fold in folds],
        ignore_index=True,
    )


def write_results(
    output_dir: Path,
    dataset_path: Path,
    folds: tuple[ThreadFold, ...],
    results: list[ModelResult],
    per_fold_results: dict[str, list[ModelResult]],
    details: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation = _evaluation_frame(folds)

    metric_rows = []
    fold_metric_rows = []
    for result in results:
        fold_results = per_fold_results[result.name]
        fold_scores = [item.macro_f1 for item in fold_results]
        metric_rows.append(
            {
                "model": result.name,
                "accuracy": result.accuracy,
                "macro_f1": result.macro_f1,
                "weighted_f1": result.weighted_f1,
                "fold_macro_f1_min": min(fold_scores),
                "fold_macro_f1_max": max(fold_scores),
            }
        )
        for fold, fold_result in zip(folds, fold_results, strict=True):
            missing = sorted(set(LABELS).difference(fold.test["label"]))
            fold_metric_rows.append(
                {
                    "model": result.name,
                    "fold": fold.number,
                    "accuracy": fold_result.accuracy,
                    "macro_f1": fold_result.macro_f1,
                    "weighted_f1": fold_result.weighted_f1,
                    "test_excerpts": len(fold.test),
                    "test_threads": fold.test["source_url"].nunique(),
                    "missing_test_labels": "|".join(missing),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False, float_format="%.6f")
    fold_metrics.to_csv(
        output_dir / "fold_metrics.csv", index=False, float_format="%.6f"
    )

    predictions = evaluation[
        ["id", "fold", "source_site", "source_url", "source_context", "label"]
    ].copy()
    for result in results:
        predictions[result.name] = result.predictions
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    predictions[["id", "fold", "source_site", "source_url"]].to_csv(
        output_dir / "fold_assignments.csv", index=False
    )

    per_class = []
    for result in results:
        report = classification_report(
            evaluation["label"],
            result.predictions,
            labels=list(LABELS),
            output_dict=True,
            zero_division=0,
        )
        for label in LABELS:
            per_class.append({"model": result.name, "label": label, **report[label]})
    pd.DataFrame(per_class).to_csv(
        output_dir / "per_class_metrics.csv", index=False, float_format="%.6f"
    )

    run = {
        "dataset": {
            "path": str(dataset_path),
            "sha256": _sha256(dataset_path),
            "rows": len(evaluation),
        },
        "cross_validation": {
            "strategy": "StratifiedGroupKFold",
            "n_splits": OUTER_SPLITS,
            "shuffle": True,
            "random_state": CV_RANDOM_STATE,
            "evaluated_folds": [fold.number for fold in folds],
            "folds": [
                {
                    "fold": fold.number,
                    "train_excerpts": len(fold.train),
                    "train_threads": fold.train["source_url"].nunique(),
                    "test_excerpts": len(fold.test),
                    "test_threads": fold.test["source_url"].nunique(),
                    "test_label_counts": (
                        fold.test["label"].value_counts().sort_index().to_dict()
                    ),
                    "missing_test_labels": sorted(
                        set(LABELS).difference(fold.test["label"])
                    ),
                }
                for fold in folds
            ],
        },
        "models": {result.name: result.parameters for result in results},
        "details": details,
        "python": platform.python_version(),
        "packages": {
            package: importlib.metadata.version(package)
            for package in (
                "matplotlib",
                "numpy",
                "pandas",
                "scikit-learn",
                "torch",
                "transformers",
            )
            if _package_is_installed(package)
        },
    }
    (output_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")

    comparison = metrics.sort_values("macro_f1", ascending=True).copy()
    comparison["display_name"] = comparison["model"].map(DISPLAY_NAMES)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.barh(comparison["display_name"], comparison["macro_f1"], color="#4472C4")
    axis.set(
        xlabel="Pooled out-of-fold macro F1",
        xlim=(0, 1),
        title="Five-fold unseen-thread cross-validation",
    )
    for index, value in enumerate(comparison["macro_f1"]):
        axis.text(value + 0.015, index, f"{value:.3f}", va="center")
    figure.tight_layout()
    figure.savefig(output_dir / "model_comparison.png", dpi=160)
    plt.close(figure)

    for result in results:
        matrix = confusion_matrix(
            evaluation["label"], result.predictions, labels=list(LABELS)
        )
        figure, axis = plt.subplots(figsize=(7, 5.5))
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_xticks(range(len(LABELS)), LABELS, rotation=30, ha="right")
        axis.set_yticks(range(len(LABELS)), LABELS)
        axis.set(
            xlabel="Predicted",
            ylabel="Actual",
            title=DISPLAY_NAMES[result.name],
        )
        for row in range(len(LABELS)):
            for column in range(len(LABELS)):
                axis.text(column, row, matrix[row, column], ha="center", va="center")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.tight_layout()
        figure.savefig(
            output_dir / f"confusion_matrix_{result.name}.png", dpi=160
        )
        plt.close(figure)


def _package_is_installed(package: str) -> bool:
    try:
        importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True
