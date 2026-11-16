import pandas as pd
import pytest

from craft_forum_nlp.data import LABELS, load_data


def test_repository_dataset_satisfies_contract():
    frame = load_data("craft_forum_dataset.csv")

    assert len(frame) == 212
    assert frame["id"].is_unique
    assert set(frame["label"]) == set(LABELS)
    assert frame["source_url"].nunique() == 28


@pytest.mark.parametrize(
    ("ids", "message"),
    [
        ([1, 1, 3, 4], "IDs must be unique"),
        ([1, 2.5, 3, 4], "IDs must be integers"),
    ],
)
def test_invalid_ids_are_rejected(tmp_path, ids, message):
    frame = pd.DataFrame(
        {
            "id": ids,
            "text": ["text"] * 4,
            "label": list(LABELS),
            "source_site": ["site"] * 4,
            "source_url": [f"https://example.com/{i}" for i in range(4)],
            "source_context": ["thread"] * 4,
        }
    )
    path = tmp_path / "invalid-ids.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match=message):
        load_data(path)


@pytest.mark.parametrize(
    ("column", "message"),
    [
        ("text", "text as a string"),
        ("source_url", "source URL as a string"),
    ],
)
def test_non_string_fields_are_rejected(tmp_path, column, message):
    frame = pd.DataFrame(
        {
            "id": range(1, 5),
            "text": [101, 102, 103, 104] if column == "text" else ["text"] * 4,
            "label": list(LABELS),
            "source_site": ["site"] * 4,
            "source_url": (
                [101, 102, 103, 104]
                if column == "source_url"
                else [f"https://example.com/{i}" for i in range(4)]
            ),
            "source_context": ["thread"] * 4,
        }
    )
    path = tmp_path / "non-string.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match=message):
        load_data(path)
