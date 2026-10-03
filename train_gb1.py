import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from features import FEATURE_COLUMNS


DATASET_PATH = Path("data") / "ml" / "features.parquet"
MODEL_DIR = Path("data") / "models"
MODEL_PATH = MODEL_DIR / "gb1.joblib"
METADATA_PATH = MODEL_DIR / "gb1_metadata.json"

MODEL_NAME = "GB1"
TARGET_COLUMN = "future_return_30m"
POSITIVE_RETURN_THRESHOLD = 0.003

# Gradient boosting needs numeric inputs. We keep the existing 31 numeric features
# and encode the stock symbol as five simple one-hot columns.
SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
SYMBOL_COLUMNS = [f"symbol_{symbol}" for symbol in SYMBOLS]
MODEL_INPUT_COLUMNS = list(FEATURE_COLUMNS) + SYMBOL_COLUMNS


def make_label(frame):
    return (frame[TARGET_COLUMN] >= POSITIVE_RETURN_THRESHOLD).astype(np.int8)


def add_symbol_columns(frame):
    frame = frame.copy()
    for symbol in SYMBOLS:
        frame[f"symbol_{symbol}"] = (frame["symbol"] == symbol).astype(np.int8)
    return frame


def load_splits():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing {DATASET_PATH}. Run python build_dataset.py first.")

    dataset = pd.read_parquet(
        DATASET_PATH,
        columns=["symbol", "split", *FEATURE_COLUMNS, TARGET_COLUMN],
    )
    train = dataset.loc[dataset["split"] == "train"].copy()
    validation = dataset.loc[dataset["split"] == "validation"].copy()

    if train.empty or validation.empty:
        raise ValueError("Training or validation split is empty.")

    return add_symbol_columns(train), add_symbol_columns(validation)


def balanced_sample_weights(labels):
    """Give the minority class equal total weight without changing the dataset."""
    positive_rate = float(labels.mean())
    if positive_rate <= 0.0 or positive_rate >= 1.0:
        return np.ones(len(labels), dtype=float)

    positive_weight = 0.5 / positive_rate
    negative_weight = 0.5 / (1.0 - positive_rate)
    return np.where(labels.to_numpy() == 1, positive_weight, negative_weight)


def build_model():
    # Conservative first boosting model: shallow trees reduce overfitting while
    # still allowing nonlinear interactions that logistic regression cannot learn.
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=80,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=20,
        random_state=42,
    )


def main():
    print("=" * 88)
    print("STAGE 4 - TRAIN GB1")
    print("=" * 88)
    print("Model:        Gradient Boosting #1")
    print("Train:        2023-2024")
    print("Validation:   2025")
    print("Target:       30m return >= 0.30%")
    print("Difference:   nonlinear tree boosting instead of logistic regression")

    train, validation = load_splits()
    y_train = make_label(train)
    y_validation = make_label(validation)

    print(f"\nTraining rows:   {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Training positive rate: {y_train.mean() * 100:.2f}%")

    model = build_model()
    weights = balanced_sample_weights(y_train)

    print("\nTraining GB1...")
    model.fit(train[MODEL_INPUT_COLUMNS], y_train, sample_weight=weights)

    probabilities = model.predict_proba(validation[MODEL_INPUT_COLUMNS])[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_validation, probabilities)),
        "average_precision": float(average_precision_score(y_validation, probabilities)),
        "log_loss": float(log_loss(y_validation, probabilities)),
        "iterations_used": int(model.n_iter_),
        "probability_percentiles": {
            "p50": float(np.percentile(probabilities, 50)),
            "p75": float(np.percentile(probabilities, 75)),
            "p90": float(np.percentile(probabilities, 90)),
            "p95": float(np.percentile(probabilities, 95)),
            "p99": float(np.percentile(probabilities, 99)),
        },
    }

    print("\n" + "=" * 88)
    print("2025 VALIDATION RESULTS")
    print("=" * 88)
    print(f"ROC AUC:           {metrics['roc_auc']:.4f}")
    print(f"Average precision: {metrics['average_precision']:.4f}")
    print(f"Log loss:          {metrics['log_loss']:.4f}")
    print(f"Boosting rounds:   {metrics['iterations_used']}")
    print("Probability percentiles:")
    for name, value in metrics["probability_percentiles"].items():
        print(f"  {name}: {value:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "model_name": MODEL_NAME,
        "model_type": "hist_gradient_boosting_classifier",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(DATASET_PATH),
        "model_path": str(MODEL_PATH),
        "target_column": TARGET_COLUMN,
        "positive_return_threshold": POSITIVE_RETURN_THRESHOLD,
        "train_period": "2023-2024",
        "validation_period": "2025",
        "numeric_features": FEATURE_COLUMNS,
        "symbol_features": SYMBOL_COLUMNS,
        "train_rows": int(len(train)),
        "train_positive_rate": float(y_train.mean()),
        "validation_metrics": metrics,
        "training_policy": "balanced sample weights; conservative shallow histogram gradient boosting",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")

    print("\n" + "=" * 88)
    print("GB1 SAVED")
    print("=" * 88)
    print(f"Model:    {MODEL_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    print("Next: inspect 2025 validation results before designing GB1 execution rules.")


if __name__ == "__main__":
    main()
