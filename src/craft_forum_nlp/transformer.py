from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from craft_forum_nlp.data import LABELS
from craft_forum_nlp.models import ModelResult, score_predictions
from craft_forum_nlp.splits import CV_RANDOM_STATE


MODEL_NAME = "distilbert-base-uncased"
MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
DISTILBERT_EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 32
MAX_LENGTH = 256


@dataclass
class _EncodedDataset:
    encodings: dict[str, object]
    labels: list[int]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def evaluate_distilbert(
    development: pd.DataFrame,
    test: pd.DataFrame,
    *,
    class_weighted: bool = False,
) -> tuple[ModelResult, list[dict[str, float]]]:
    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            get_linear_schedule_with_warmup,
        )
    except ImportError as error:
        raise RuntimeError(
            "Install the transformer dependencies with pip install -e '.[transformer]'"
        ) from error

    random.seed(CV_RANDOM_STATE)
    np.random.seed(CV_RANDOM_STATE)
    torch.manual_seed(CV_RANDOM_STATE)
    torch.use_deterministic_algorithms(True)

    device = torch.device("cpu")
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)

    def encode(frame: pd.DataFrame) -> _EncodedDataset:
        encodings = tokenizer(
            frame["text"].tolist(),
            truncation=True,
            max_length=MAX_LENGTH,
        )
        return _EncodedDataset(
            encodings=encodings,
            labels=[label_to_id[label] for label in frame["label"]],
        )

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    train_loader = DataLoader(
        encode(development),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        encode(test), batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collator
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        num_labels=len(LABELS),
        id2label=id_to_label,
        label2id=label_to_id,
    ).to(device)
    loss_weights = None
    if class_weighted:
        counts = development["label"].value_counts()
        loss_weights = torch.tensor(
            [
                len(development) / (len(LABELS) * counts[label])
                for label in LABELS
            ],
            dtype=torch.float32,
            device=device,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    total_steps = DISTILBERT_EPOCHS * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(10, total_steps // 5),
        num_training_steps=total_steps,
    )

    def predict(loader: DataLoader) -> np.ndarray:
        model.eval()
        output: list[int] = []
        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                labels = batch.pop("labels")
                logits = model(**batch).logits
                output.extend(logits.argmax(dim=-1).cpu().tolist())
                del labels
        return np.array([id_to_label[index] for index in output])

    history = []
    for epoch in range(1, DISTILBERT_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            labels = batch.pop("labels")
            logits = model(**batch).logits
            loss = torch.nn.functional.cross_entropy(
                logits,
                labels,
                weight=loss_weights,
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.detach().cpu())

        history.append(
            {
                "epoch": float(epoch),
                "training_loss": running_loss / len(train_loader),
            }
        )

    test_predictions = predict(test_loader)
    return (
        score_predictions(
            "distilbert_weighted" if class_weighted else "distilbert",
            test["label"],
            test_predictions,
            {
                "model": MODEL_NAME,
                "revision": MODEL_REVISION,
                "epochs": DISTILBERT_EPOCHS,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "train_batch_size": TRAIN_BATCH_SIZE,
                "test_batch_size": TEST_BATCH_SIZE,
                "max_length": MAX_LENGTH,
                "seed": CV_RANDOM_STATE,
                "device": str(device),
                "class_weighted_loss": class_weighted,
                "training_rows": len(development),
            },
        ),
        history,
    )
