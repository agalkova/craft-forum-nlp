"""Shared plotting style for project figures.

Figures are saved as SVG with live text. Layout is measured with DejaVu Sans,
and the saved files ask the viewer for Tahoma/Verdana first, which are
narrower, so labels never grow past where they were placed.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#1d2b36"
INK_SOFT = "#43586a"
MUTED = "#6b7b87"
GRID = "#dcd8cc"
SPINE = "#8f9aa3"
BLUE = "#0b5dc4"
ORANGE = "#d9631e"
GREY = "#98a3ab"
SHADE = "#f2efe6"

FONT_STACK = "Tahoma, Verdana, 'DejaVu Sans', sans-serif"
MONO_STACK = "Consolas, Menlo, 'DejaVu Sans Mono', monospace"

# Plain figures leave out the figure label, title and source line, for pages
# that already show them as text.
PLAIN = False

# (width, height) in inches at 100 dpi -> SVG pixels
WIDE = 8.8
NARROW = 4.2


def apply(narrow: bool = False) -> None:
    base = 11.5 if narrow else 10.5
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": base,
            "axes.titlesize": base + 1,
            "axes.labelsize": base,
            "axes.labelcolor": INK_SOFT,
            "axes.edgecolor": SPINE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "axes.prop_cycle": plt.cycler(color=[BLUE, ORANGE, GREY]),
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": (0, (1, 2)),
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.5,
            "ytick.major.size": 0,
            "xtick.major.width": 0.8,
            "xtick.labelsize": base - 1,
            "ytick.labelsize": base - 1,
            "legend.frameon": False,
            "legend.fontsize": base - 0.5,
            "legend.handlelength": 1.8,
            "lines.solid_capstyle": "round",
            "figure.dpi": 100,
            "savefig.dpi": 100,
            "svg.fonttype": "none",
            "svg.hashsalt": "figures",
        }
    )


def header(fig, number: int, title: str, subtitle: str | None = None, narrow: bool = False) -> None:
    """Figure label and a plain, descriptive title, left aligned with the axes."""
    if PLAIN:
        return
    x = 0.035 if not narrow else 0.05
    h = fig.get_figheight() * 72
    size = plt.rcParams["font.size"]
    y = 1 - 12 / h
    fig.text(x, y, f"Fig. {number}", color=BLUE, fontsize=size - 0.5, fontweight="bold", va="top")
    y -= (size + 6) / h
    fig.text(x, y, title, color=INK, fontsize=size + 2, fontweight="bold", va="top")
    if subtitle:
        y -= (size + 9) / h
        fig.text(x, y, subtitle, color=MUTED, fontsize=size - 0.5, va="top")


def source(fig, text: str, narrow: bool = False) -> None:
    if PLAIN:
        return
    fig.text(
        0.035 if not narrow else 0.05,
        0.012,
        text,
        color=MUTED,
        fontsize=plt.rcParams["font.size"] - 2,
        family="DejaVu Sans Mono",
        va="bottom",
    )


def save(fig, path: Path, title: str, description: str) -> None:
    """Write an accessible SVG with deterministic ids and the site's font stack."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        format="svg",
        bbox_inches="tight" if PLAIN else None,
        pad_inches=0.06,
        metadata={"Date": None, "Creator": None, "Title": title, "Description": description},
    )
    plt.close(fig)
    svg = path.read_text()
    svg = svg.replace("'DejaVu Sans Mono'", MONO_STACK).replace("'DejaVu Sans'", FONT_STACK)
    svg = svg.replace('"DejaVu Sans Mono"', MONO_STACK).replace('"DejaVu Sans"', FONT_STACK)
    # accessible name for <img>-less contexts and screen readers opening the file directly
    svg = re.sub(
        r"<svg ",
        '<svg role="img" aria-label="' + _attr(title + ". " + description) + '" ',
        svg,
        count=1,
    )
    path.write_text(svg)


def _attr(text: str) -> str:
    return text.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
