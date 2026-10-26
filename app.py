"""Streamlit interface for text classification and model evaluation."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from model import AttentionClassifier, encode

ROOT = Path(__file__).parent
st.set_page_config(page_title="Attention Explorer", page_icon="🔎", layout="centered")
st.title("🔎 Attention Visualization for Text Classification")
st.write("Classify a movie review and inspect the attention weight assigned to each token.")

MODEL_PATH, VOCAB_PATH = ROOT / "model.pt", ROOT / "vocab.json"
if not MODEL_PATH.exists() or not VOCAB_PATH.exists():
    st.info("Train the model first by running `python train.py` in this folder.")
    st.stop()

vocab = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
model = AttentionClassifier(len(vocab))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
model.eval()

examples = ["A delightful, funny story with wonderful characters", "A dull and confusing film with a terrible ending"]
choice = st.selectbox("Try an example", ["Write my own"] + examples)
text = st.text_area("Review text", value="" if choice == "Write my own" else choice,
                    placeholder="The story was charming and the acting was excellent…")

if st.button("Classify", type="primary", disabled=not text.strip()):
    tokens, ids = encode(text, vocab)
    if not tokens:
        st.warning("Please enter at least one word.")
    else:
        with torch.no_grad():
            logits, weights = model(torch.tensor([ids]))
            probabilities = torch.softmax(logits, dim=-1)[0]
        predicted_id = int(probabilities.argmax())
        st.subheader(f"Prediction: {'Positive' if predicted_id else 'Negative'}")
        st.metric("Model confidence", f"{probabilities[predicted_id].item():.1%}")
        st.caption("Attention shows how the model weights tokens. It is not a guaranteed causal explanation.")
        scores = weights[0].tolist()
        table = pd.DataFrame({"Token": tokens, "Attention weight": [f"{score:.4f}" for score in scores]})
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.markdown("**Token attention** (darker means more weight)")
        st.html("<div style='line-height:2.5'>" + " ".join(
            f"<span style='background:rgba(255, 166, 0, {min(.12 + score * 4, .85):.2f});padding:5px 7px;border-radius:5px;margin:2px'>{token}</span>"
            for token, score in zip(tokens, scores)
        ) + "</div>")

with st.expander("About the model"):
    st.write("A bidirectional GRU with additive attention, trained on the full user-provided reviews.csv dataset using a reproducible stratified 80/20 train/test split.")

st.divider()
st.header("Held-out evaluation metrics")
metrics_path = ROOT / "eval_metrics.json"
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    st.caption(
        f"Dataset: {metrics['dataset_name']} ({metrics['dataset_total_labeled']:,} labeled reviews). "
        f"This run trained on {metrics['train_size']:,} and evaluated on {metrics['test_size']:,} separate reviews."
    )
    primary = st.columns(4)
    for column, name, label in zip(primary,
        ("accuracy", "precision_macro", "recall_macro", "f1_macro"),
        ("Accuracy", "Macro precision", "Macro recall", "Macro F1")):
        column.metric(label, f"{metrics[name]:.1%}")
    secondary = st.columns(4)
    for column, name, label in zip(secondary,
        ("balanced_accuracy", "specificity_negative", "sensitivity_positive", "roc_auc"),
        ("Balanced accuracy", "Specificity", "Positive recall", "ROC AUC")):
        column.metric(label, f"{metrics[name]:.1%}")
    operational = st.columns(3)
    for column, name, label in zip(operational,
        ("false_positive_rate", "false_negative_rate", "negative_predictive_value"),
        ("False positive rate", "False negative rate", "Negative predictive value")):
        column.metric(label, f"{metrics[name]:.1%}")
    st.metric("Matthews correlation coefficient", f"{metrics['matthews_correlation']:.3f}",
              help="MCC ranges from -1 (wrong predictions), through 0 (random-like), to +1 (perfect predictions).")

    labels = metrics["matrix_labels"]
    matrix = pd.DataFrame(
        metrics["confusion_matrix"],
        index=pd.Index([f"Actual {label}" for label in labels], name="Actual \\ Predicted"),
        columns=[f"Predicted {label}" for label in labels],
    )
    st.subheader("Confusion matrix")
    st.dataframe(matrix, use_container_width=True)
    st.caption("Rows are actual labels; columns are predictions. Diagonal cells are correct; off-diagonal cells are errors.")
    report = metrics["classification_report"]
    report_table = pd.DataFrame({
        label: {key: report[label][key] for key in ("precision", "recall", "f1-score", "support")}
        for label in labels + ["macro avg", "weighted avg"]
    }).T
    st.subheader("Per-class and averaged scores")
    st.dataframe(report_table.style.format({
        "precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}", "support": "{:.0f}"
    }), use_container_width=True)
    st.download_button("Download metrics JSON", json.dumps(metrics, indent=2),
                       file_name="eval_metrics.json", mime="application/json")
else:
    st.info("Evaluation metrics will appear after `python train.py` finishes.")





