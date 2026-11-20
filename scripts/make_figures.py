"""Draw the evaluation figures from the committed results.

Reads results/fold_metrics.csv, results/metrics.csv and results/per_class_metrics.csv
and writes SVG figures to docs/figures/ (a wide and a narrow version of each).

    python scripts/make_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_style as fs
from figure_style import plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "figures"

MODELS = {
    "tfidf_logistic_regression": ("TF-IDF + logistic regression", fs.BLUE),
    "distilbert_weighted": ("DistilBERT, class-weighted", fs.ORANGE),
    "distilbert": ("DistilBERT", fs.GREY),
    "majority": ("Majority class", fs.GREY),
}
LABELS = {
    "technical_help": "Technical help",
    "showcase_reaction": "Showcase reaction",
    "critique_feedback": "Critique",
    "meta_opinion": "Opinion about the craft",
}


def fold_figure(narrow: bool) -> None:
    folds = pd.read_csv(RESULTS / "fold_metrics.csv")
    pooled = pd.read_csv(RESULTS / "metrics.csv").set_index("model")["macro_f1"]
    order = ["tfidf_logistic_regression", "distilbert_weighted", "distilbert", "majority"]
    fs.apply(narrow)
    width = fs.NARROW if narrow else fs.WIDE
    fig = plt.figure(figsize=(width, 4.6 if narrow else 3.9))
    left = 0.06 if narrow else 0.3
    axis = fig.add_axes([left, 0.17, 0.88 if narrow else 0.64, 0.62 if narrow else 0.6])
    axis.grid(axis="y", visible=False)
    axis.grid(axis="x", visible=True)
    for row, model in enumerate(order):
        name, color = MODELS[model]
        values = folds.loc[folds["model"] == model, "macro_f1"].to_numpy()
        y = len(order) - 1 - row
        axis.plot(
            [values.min(), values.max()],
            [y, y],
            color="#cfd5da",
            lw=5,
            solid_capstyle="round",
            zorder=1,
        )
        axis.scatter(
            values,
            np.full_like(values, y),
            s=26,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=2,
        )
        axis.plot([pooled[model]] * 2, [y - 0.24, y + 0.24], color=fs.INK, lw=2, zorder=3)
        if narrow:
            axis.text(
                0,
                y + 0.36,
                name,
                color=fs.INK,
                fontsize=plt.rcParams["font.size"] - 1.5,
                va="bottom",
            )
            axis.text(
                values.max() + 0.012,
                y,
                f"{pooled[model]:.3f}",
                va="center",
                fontsize=plt.rcParams["font.size"] - 1.5,
                color=fs.INK_SOFT,
            )
        else:
            axis.text(
                values.max() + 0.012,
                y,
                f"{pooled[model]:.3f} pooled",
                va="center",
                fontsize=plt.rcParams["font.size"] - 1,
                color=fs.INK_SOFT,
            )
    if not narrow:
        axis.set_yticks(range(len(order)), [MODELS[m][0] for m in reversed(order)])
        axis.tick_params(axis="y", colors=fs.INK, labelsize=plt.rcParams["font.size"])
    else:
        axis.set_yticks([])
        axis.spines["left"].set_visible(False)
    axis.set_ylim(-0.6, len(order) - 0.3 + (0.35 if narrow else 0))
    axis.set_xlim(0, 0.55)
    axis.set_xlabel("Macro F1")
    fs.header(
        fig,
        1,
        "Macro F1 in each of the five folds",
        None if narrow else "Dots: one fold each · black tick: pooled out-of-fold score",
        narrow,
    )
    fs.source(fig, "results/fold_metrics.csv", narrow)
    name = "fold-macro-f1" + ("-narrow" if narrow else "")
    fs.save(
        fig,
        OUT / f"{name}.svg",
        "Macro F1 in each of the five thread-grouped folds",
        "TF-IDF with logistic regression ranges from 0.303 to 0.424 across folds (pooled 0.363); class-weighted "
        "DistilBERT from 0.191 to 0.458 (pooled 0.318). Unweighted DistilBERT and the majority baseline both "
        "pool to 0.145.",
    )


def recall_figure(narrow: bool) -> None:
    per_class = pd.read_csv(RESULTS / "per_class_metrics.csv")
    models = ["tfidf_logistic_regression", "distilbert_weighted", "distilbert"]
    fs.apply(narrow)
    width = fs.NARROW if narrow else fs.WIDE
    fig = plt.figure(figsize=(width, 6.4 if narrow else 4.4))
    left = 0.06 if narrow else 0.25
    axis = fig.add_axes(
        [left, 0.12 if narrow else 0.14, 0.88 if narrow else 0.7, 0.62 if narrow else 0.56]
    )
    axis.grid(axis="y", visible=False)
    axis.grid(axis="x", visible=True)
    labels = list(LABELS)
    bar = 0.2 if narrow else 0.24
    gap = 1.35 if narrow else 1.0
    for k, model in enumerate(models):
        name, color = MODELS[model]
        recalls = (
            per_class[per_class["model"] == model]
            .set_index("label")
            .loc[labels, "recall"]
            .to_numpy()
        )
        ys = np.arange(len(labels))[::-1] * gap + (1 - k) * (bar + 0.03)
        axis.barh(ys, recalls, height=bar, color=color, label=name, zorder=2)
        for y, v in zip(ys, recalls):
            axis.text(
                v + 0.012,
                y,
                f"{v:.0%}",
                va="center",
                fontsize=plt.rcParams["font.size"] - 2.5,
                color=fs.INK_SOFT,
            )
    support = (
        per_class[per_class["model"] == models[0]]
        .set_index("label")
        .loc[labels, "support"]
        .astype(int)
    )
    tick_labels = [f"{LABELS[l]} ({support[l]})" for l in labels]
    if narrow:
        axis.set_yticks([])
        axis.spines["left"].set_visible(False)
        for i, text in enumerate(tick_labels):
            axis.text(
                0,
                (len(labels) - 1 - i) * gap + 0.42,
                text,
                fontsize=plt.rcParams["font.size"] - 1.5,
                color=fs.INK,
                va="bottom",
            )
        axis.set_ylim(-0.45, (len(labels) - 1) * gap + 0.75)
    else:
        axis.set_yticks(range(len(labels))[::-1], tick_labels)
        axis.tick_params(axis="y", colors=fs.INK, labelsize=plt.rcParams["font.size"])
    axis.set_xlim(0, 1.1)
    axis.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axis.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axis.set_xlabel(
        "Recall" if narrow else "Recall: share of each category the model labeled correctly"
    )
    axis.legend(
        loc="lower left",
        bbox_to_anchor=(-0.02 if narrow else 0, 1.0),
        ncol=1 if narrow else 3,
        fontsize=plt.rcParams["font.size"] - (2 if narrow else 1),
        handlelength=1.2,
        columnspacing=1.2,
    )
    fs.header(
        fig,
        2,
        "Recall by category",
        None
        if narrow
        else "Out-of-fold predictions, 212 excerpts (number in each category in brackets)",
        narrow,
    )
    fs.source(fig, "results/per_class_metrics.csv", narrow)
    name = "recall-by-category" + ("-narrow" if narrow else "")
    fs.save(
        fig,
        OUT / f"{name}.svg",
        "Recall by category for three models",
        "Unweighted DistilBERT labels every excerpt as technical help (100% recall there, 0% elsewhere). "
        "Class-weighted DistilBERT swings the other way: 14% on technical help, 83% on opinion. TF-IDF with "
        "logistic regression is more even, between 22% and 51% in every category.",
    )


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--plain",
        type=Path,
        metavar="DIR",
        help="write versions without the figure label, title and source line to DIR",
    )
    args = parser.parse_args()
    if args.plain:
        fs.PLAIN = True
        OUT = args.plain
    for narrow in (False, True):
        fold_figure(narrow)
        recall_figure(narrow)


if __name__ == "__main__":
    main()
