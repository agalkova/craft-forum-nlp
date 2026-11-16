from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


LABELS = (
    "technical_help",
    "showcase_reaction",
    "critique_feedback",
    "meta_opinion",
)
REQUIRED_COLUMNS = {
    "id",
    "text",
    "label",
    "source_site",
    "source_url",
    "source_context",
}


def load_data(path: str | Path) -> pd.DataFrame:
    """Load and validate the labeled forum-post dataset."""
    frame = pd.read_csv(path)

    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Dataset contains no rows")
    if frame["text"].isna().any():
        raise ValueError("Every row must contain text")
    if frame["source_url"].isna().any():
        raise ValueError("Every row must contain a source URL")
    if not frame["text"].map(lambda value: isinstance(value, str)).all():
        raise ValueError("Every row must contain text as a string")
    if not frame["source_url"].map(lambda value: isinstance(value, str)).all():
        raise ValueError("Every row must contain a source URL as a string")
    if frame["text"].str.strip().eq("").any():
        raise ValueError("Every row must contain text")
    if frame["source_url"].str.strip().eq("").any():
        raise ValueError("Every row must contain a source URL")

    unknown = set(frame["label"]).difference(LABELS)
    if unknown:
        raise ValueError(f"Unknown labels: {sorted(unknown)}")
    if set(frame["label"]) != set(LABELS):
        raise ValueError("Dataset must contain every configured label")

    numeric_ids = pd.to_numeric(frame["id"], errors="raise")
    integer_valued = np.equal(numeric_ids, np.floor(numeric_ids)).all()
    if not np.isfinite(numeric_ids).all() or not integer_valued:
        raise ValueError("Dataset IDs must be integers")

    frame = frame.copy()
    frame["id"] = numeric_ids.astype(int)
    if frame["id"].duplicated().any():
        raise ValueError("Dataset IDs must be unique")
    return frame.sort_values("id").reset_index(drop=True)
