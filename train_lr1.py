import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import FEATURE_COLUMNS


DATASET_PATH = Path("data") / "ml" / "features.parquet"
MODEL_DIR = Path("data") / "models"
MODEL_PATH = MODEL_DIR / "lr1.joblib"
METADATA_PATH = MODEL_DIR / "lr1_metadata.json"

MODEL_NAME = "LR1"
TARGET_COLUMN = "future_return_30m"
# LR1 learns whether the close 30 minutes ahead is at least 0.30% higher.
# This is intentionally a fairly strong move because our current modeled
# round-trip trading friction is also 0.30%. The real profitability test comes
# later in the execution-aware backtest, not from this label alone.
POSITIVE_RETURN_THRESHOLD = 0.003
DEFAULT_PROBABILITY_THRESHOLD = 0.50

NUMERIC_COLUMNS = list(FEATURE_COLUMNS)
CATEGORICAL_COLUMNS = ["symbol"]
MODEL_INPUT_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


def make_label(frame):
    return (frame[TARGET_COLUMN] >= POSITIVE_RETURN_THRESHOLD).astype(int)


def load_splits():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATASET_PATH}. Run python build_dataset.py first."
        )

    dataset = pd.read_parquet(
        DATASET_PATH,
        columns=[
            "timestamp",
            "symbol",
            "split",
            *FEATURE_COLUMNS,
            TARGET_COLUMN,
        ],
    )

    train = dataset.loc[dataset["split"] == "train"].copy()
    validation = dataset.loc[dataset["split"] == "validation"].copy()

    # Deliberately do not load/evaluate the test split here. 2026 stays unseen
    # until LR1 is frozen and we run the execution-aware historical backtest.
    if train.empty or validation.empty:
        raise ValueError("Training or validation split is empty.")

    return train, validation


def build_model():
    """Create LR1: scaled numeric features + one-hot symbol + logistic regression."""
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_COLUMNS),
            ("symbol", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )

    classifier = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("classifier", classifier),
        ]
    )


def evaluate_validation(model, validation):
    x_validation = validation[MODEL_INPUT_COLUMNS]
    y_validation = make_label(validation)

    probabilities = model.predict_proba(x_validation)[:, 1]
    predictions = (probabilities >= DEFAULT_PROBABILITY_THRESHOLD).astype(int)

    metrics = {
        "rows": int(len(validation)),
        "positive_rate": float(y_validation.mean()),
        "roc_auc": float(roc_auc_score(y_validation, probabilities)),
        "average_precision": float(average_precision_score(y_validation, probabilities)),
        "log_loss": float(log_loss(y_validation, probabilities)),
        "accuracy_at_0_50": float(accuracy_score(y_validation, predictions)),
        "precision_at_0_50": float(precision_score(y_validation, predictions, zero_division=0)),
        "recall_at_0_50": float(recall_score(y_validation, predictions, zero_division=0)),
        "f1_at_0_50": float(f1_score(y_validation, predictions, zero_division=0)),
        "predicted_positive_rows_at_0_50": int(predictions.sum()),
        "confusion_matrix_at_0_50": confusion_matrix(y_validation, predictions).tolist(),
        "probability_percentiles": {
            "p50": float(np.percentile(probabilities, 50)),
            "p75": float(np.percentile(probabilities, 75)),
            "p90": float(np.percentile(probabilities, 90)),
            "p95": float(np.percentile(probabilities, 95)),
            "p99": float(np.percentile(probabilities, 99)),
        },
    }

    return metrics, y_validation, predictions


def strongest_coefficients(model, top_n=12):
    preprocess = model.named_steps["preprocess"]
    classifier = model.named_steps["classifier"]

    names = preprocess.get_feature_names_out()
    coefficients = classifier.coef_[0]

    pairs = sorted(
        zip(names, coefficients),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:top_n]

    return [
        {"feature": str(name), "coefficient": float(coefficient)}
        for name, coefficient in pairs
    ]


def main():
    print("=" * 88)
    print("STAGE 4 - TRAIN LR1")
    print("=" * 88)
    print("Model:        Logistic Regression #1")
    print("Train:        2023-2024")
    print("Validation:   2025")
    print("Final test:   2026 (NOT TOUCHED BY THIS SCRIPT)")
    print(f"Target:       30m return >= {POSITIVE_RETURN_THRESHOLD * 100:.2f}%")

    train, validation = load_splits()
    y_train = make_label(train)

    print(f"\nTraining rows:   {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Training positive rate: {y_train.mean() * 100:.2f}%")

    model = build_model()
    print("\nTraining LR1...")
    model.fit(train[MODEL_INPUT_COLUMNS], y_train)

    metrics, y_validation, predictions = evaluate_validation(model, validation)
    coefficients = strongest_coefficients(model)

    print("\n" + "=" * 88)
    print("2025 VALIDATION RESULTS")
    print("=" * 88)
    print(f"Actual positive rate: {metrics['positive_rate'] * 100:.2f}%")
    print(f"ROC AUC:              {metrics['roc_auc']:.4f}")
    print(f"Average precision:    {metrics['average_precision']:.4f}")
    print(f"Log loss:             {metrics['log_loss']:.4f}")
    print(f"Accuracy @ 0.50:      {metrics['accuracy_at_0_50'] * 100:.2f}%")
    print(f"Precision @ 0.50:     {metrics['precision_at_0_50'] * 100:.2f}%")
    print(f"Recall @ 0.50:        {metrics['recall_at_0_50'] * 100:.2f}%")
    print(f"F1 @ 0.50:            {metrics['f1_at_0_50']:.4f}")
    print(f"Signals @ 0.50:       {metrics['predicted_positive_rows_at_0_50']:,}")

    print("\nProbability percentiles:")
    for name, value in metrics["probability_percentiles"].items():
        print(f"  {name}: {value:.4f}")

    print("\nClassification report @ 0.50:")
    print(classification_report(y_validation, predictions, digits=4, zero_division=0))

    print("Strongest coefficients by absolute magnitude:")
    for item in coefficients:
        direction = "+" if item["coefficient"] >= 0 else "-"
        print(f"  {direction} {item['feature']:<38} {item['coefficient']:+.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "model_name": MODEL_NAME,
        "model_type": "logistic_regression",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(DATASET_PATH),
        "model_path": str(MODEL_PATH),
        "target_column": TARGET_COLUMN,
        "positive_return_threshold": POSITIVE_RETURN_THRESHOLD,
        "default_probability_threshold": DEFAULT_PROBABILITY_THRESHOLD,
        "train_period": "2023-2024",
        "validation_period": "2025",
        "final_test_period": "2026",
        "final_test_touched": False,
        "numeric_features": NUMERIC_COLUMNS,
        "categorical_features": CATEGORICAL_COLUMNS,
        "train_rows": int(len(train)),
        "train_positive_rate": float(y_train.mean()),
        "validation_metrics": metrics,
        "strongest_coefficients": coefficients,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")

    print("\n" + "=" * 88)
    print("LR1 SAVED")
    print("=" * 88)
    print(f"Model:    {MODEL_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    print("2026 remains untouched. Next step: tune/freeze execution rules on 2025, then backtest 2026.")


if __name__ == "__main__":
    main()
