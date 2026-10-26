"""Train on the supplied reviews.csv and report held-out classification metrics."""
import json
import random
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, matthews_corrcoef, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import AttentionClassifier, build_vocab, encode

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "reviews.csv"
MAX_LENGTH = 180
EPOCHS = 3
BATCH_SIZE = 128
LABEL_TO_ID = {"negative": 0, "positive": 1}


def load_reviews() -> pd.DataFrame:
    """Read the user's CSV and validate its review/sentiment columns and labels."""
    frame = pd.read_csv(DATA_PATH, usecols=["review", "sentiment"])
    frame = frame.rename(columns={"review": "text", "sentiment": "label"})
    frame["label"] = frame["label"].astype(str).str.strip().str.lower()
    frame["text"] = frame["text"].fillna("").astype(str)
    frame = frame[frame["text"].str.strip().ne("")].copy()
    unexpected = sorted(set(frame["label"]) - set(LABEL_TO_ID))
    if unexpected:
        raise ValueError(f"Unexpected sentiment labels: {unexpected}. Expected positive and negative.")
    if frame["label"].nunique() != 2:
        raise ValueError("reviews.csv must contain both positive and negative labels.")
    return frame


def pad_rows(rows: list[list[int]], width: int) -> torch.Tensor:
    return torch.tensor([row[:width] + [0] * max(0, width - len(row)) for row in rows])


def main() -> None:
    random.seed(7)
    torch.manual_seed(7)
    frame = load_reviews()
    train_frame, test_frame = train_test_split(
        frame, test_size=0.20, random_state=7, stratify=frame["label"]
    )
    print(f"Loaded {len(frame):,} labeled reviews from {DATA_PATH.name}.")
    print(f"Training on {len(train_frame):,}; evaluating on {len(test_frame):,} held-out reviews.")

    vocab = build_vocab(train_frame.text.tolist(), min_freq=2)
    train_rows = [encode(text, vocab, MAX_LENGTH)[1] for text in train_frame.text]
    test_rows = [encode(text, vocab, MAX_LENGTH)[1] for text in test_frame.text]
    train_ids, test_ids = pad_rows(train_rows, MAX_LENGTH), pad_rows(test_rows, MAX_LENGTH)
    train_labels = torch.tensor([LABEL_TO_ID[label] for label in train_frame.label])
    test_labels = torch.tensor([LABEL_TO_ID[label] for label in test_frame.label])
    loader = DataLoader(TensorDataset(train_ids, train_labels), batch_size=BATCH_SIZE, shuffle=True)

    model = AttentionClassifier(len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch_ids, batch_labels in loader:
            optimizer.zero_grad()
            logits, _ = model(batch_ids)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{EPOCHS} - loss: {total_loss / len(loader):.4f}")

    model.eval()
    probabilities = []
    with torch.no_grad():
        for (batch_ids,) in DataLoader(TensorDataset(test_ids), batch_size=BATCH_SIZE):
            batch_logits, _ = model(batch_ids)
            probabilities.extend(torch.softmax(batch_logits, dim=1)[:, 1].tolist())
    predictions = [int(score >= 0.5) for score in probabilities]
    actual = test_labels.tolist()
    tn, fp, fn, tp = confusion_matrix(actual, predictions, labels=[0, 1]).ravel().tolist()
    report = classification_report(
        actual, predictions, labels=[0, 1], target_names=["negative", "positive"],
        output_dict=True, zero_division=0,
    )
    metrics = {
        "dataset_name": "User-provided reviews.csv",
        "dataset_total_labeled": len(frame),
        "train_size": len(train_frame),
        "test_size": len(test_frame),
        "accuracy": accuracy_score(actual, predictions),
        "balanced_accuracy": balanced_accuracy_score(actual, predictions),
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
        "f1_macro": report["macro avg"]["f1-score"],
        "precision_weighted": report["weighted avg"]["precision"],
        "recall_weighted": report["weighted avg"]["recall"],
        "f1_weighted": report["weighted avg"]["f1-score"],
        "specificity_negative": tn / (tn + fp) if tn + fp else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
        "negative_predictive_value": tn / (tn + fn) if tn + fn else 0.0,
        "sensitivity_positive": tp / (tp + fn) if tp + fn else 0.0,
        "roc_auc": roc_auc_score(actual, probabilities),
        "matthews_correlation": matthews_corrcoef(actual, predictions),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "matrix_labels": ["negative", "positive"],
        "true_negative": tn, "false_positive": fp,
        "false_negative": fn, "true_positive": tp,
        "classification_report": report,
    }
    torch.save(model.state_dict(), ROOT / "model.pt")
    (ROOT / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (ROOT / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nHeld-out test metrics")
    for name in ("accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc", "matthews_correlation"):
        print(f"{name}: {metrics[name]:.4f}")
    print("Confusion matrix (rows=actual [negative, positive], columns=predicted [negative, positive]):")
    print(metrics["confusion_matrix"])
    print("Saved model.pt, vocab.json, and eval_metrics.json")


if __name__ == "__main__":
    main()
