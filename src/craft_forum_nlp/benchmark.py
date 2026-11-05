from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from craft_forum_nlp.data import load_data
from craft_forum_nlp.models import (
    ModelResult,
    evaluate_majority,
    evaluate_tfidf,
    score_predictions,
)
from craft_forum_nlp.reporting import write_results
from craft_forum_nlp.splits import OUTER_SPLITS, ThreadFold, make_outer_folds


MODEL_NAMES = ("majority", "tfidf", "distilbert", "distilbert_weighted")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Craft Forum NLP benchmark")
    parser.add_argument("--data", type=Path, default=Path("craft_forum_dataset.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=MODEL_NAMES,
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        choices=range(1, OUTER_SPLITS + 1),
        help="Run only selected outer folds; intended for timing checks.",
    )
    return parser.parse_args()


def _pool_results(
    folds: tuple[ThreadFold, ...],
    per_fold: list[ModelResult],
) -> ModelResult:
    truth = pd.concat([fold.test["label"] for fold in folds], ignore_index=True)
    predictions = np.concatenate([result.predictions for result in per_fold])
    return score_predictions(
        per_fold[0].name,
        truth,
        predictions,
        {
            "outer_folds": [
                {"fold": fold.number, **result.parameters}
                for fold, result in zip(folds, per_fold, strict=True)
            ]
        },
    )


def main() -> None:
    args = parse_args()
    all_folds = make_outer_folds(load_data(args.data))
    selected = set(args.folds or range(1, OUTER_SPLITS + 1))
    folds = tuple(fold for fold in all_folds if fold.number in selected)
    per_fold_results: dict[str, list[ModelResult]] = defaultdict(list)
    details: dict[str, object] = {
        "tfidf_inner_validation": [],
        "distilbert_training": [],
        "distilbert_weighted_training": [],
    }

    for fold in folds:
        print(
            f"fold {fold.number}: train={len(fold.train)} excerpts, "
            f"test={len(fold.test)} excerpts"
        )
        if "majority" in args.models:
            per_fold_results["majority"].append(
                evaluate_majority(fold.train, fold.test)
            )
        if "tfidf" in args.models:
            result, inner_results = evaluate_tfidf(fold.train, fold.test)
            per_fold_results[result.name].append(result)
            details["tfidf_inner_validation"].append(
                {"outer_fold": fold.number, "candidates": inner_results}
            )
        if "distilbert" in args.models:
            from craft_forum_nlp.transformer import evaluate_distilbert

            result, history = evaluate_distilbert(fold.train, fold.test)
            per_fold_results[result.name].append(result)
            details["distilbert_training"].append(
                {"outer_fold": fold.number, "epochs": history}
            )
        if "distilbert_weighted" in args.models:
            from craft_forum_nlp.transformer import evaluate_distilbert

            result, history = evaluate_distilbert(
                fold.train,
                fold.test,
                class_weighted=True,
            )
            per_fold_results[result.name].append(result)
            details["distilbert_weighted_training"].append(
                {"outer_fold": fold.number, "epochs": history}
            )

    pooled = [
        _pool_results(folds, per_fold_results[name])
        for name in (
            "majority",
            "tfidf_logistic_regression",
            "distilbert",
            "distilbert_weighted",
        )
        if name in per_fold_results
    ]
    write_results(
        args.output_dir,
        args.data,
        folds,
        pooled,
        per_fold_results,
        details,
    )
    for result in pooled:
        fold_scores = [item.macro_f1 for item in per_fold_results[result.name]]
        print(
            f"{result.name}: accuracy={result.accuracy:.3f}, "
            f"macro_f1={result.macro_f1:.3f}, "
            f"fold_range={min(fold_scores):.3f}-{max(fold_scores):.3f}"
        )


if __name__ == "__main__":
    main()
