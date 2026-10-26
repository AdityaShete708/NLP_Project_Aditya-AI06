# Attention Visualization for Text Classification

A Python project that trains a sentiment classifier on the provided movie-review dataset, evaluates it on held-out reviews, and visualizes token-level attention.

## Dataset and labels

The bundled `data/reviews.csv` is the dataset supplied for this project. It contains 50,000 reviews with columns `review` and `sentiment`: 25,000 `positive` and 25,000 `negative`. Training uses the complete dataset with a reproducible stratified 80/20 split: 40,000 reviews for training and 10,000 for held-out testing. The data is included in this ZIP, so training does not need to download a dataset.

## Run on Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train.py
streamlit run app.py
```

`train.py` saves `model.pt`, `vocab.json`, and `eval_metrics.json`. It prints the test confusion matrix and key scores. The Streamlit app displays the confusion matrix, accuracy, precision, recall, F1, balanced accuracy, specificity, sensitivity, false-positive/false-negative rates, negative predictive value, ROC AUC, and Matthews correlation coefficient. It also shows per-class, macro-average, and weighted-average scores and offers a JSON download.

The confusion matrix uses rows for actual labels and columns for predicted labels, ordered negative then positive. The training run uses all 50,000 records, 180 tokens maximum per review, and three epochs. Runtime depends on your computer.

## Project files

- `app.py` — Streamlit interface, prediction, attention visualization, and evaluation metrics.
- `model.py` — tokenizer, vocabulary helpers, bidirectional GRU, and additive attention.
- `train.py` — loads the CSV, splits data, trains the model, and calculates metrics.
- `data/reviews.csv` — the supplied 50,000-row labeled dataset.
- `requirements.txt` — Python dependencies.

Attention weights show how this model distributes weight across tokens. They are not guaranteed to be a complete or causal explanation of a prediction.
