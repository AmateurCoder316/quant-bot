import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import FEATURE_COLUMNS


DATASET_PATH = Path("data") / "ml" / "features.parquet"
MODEL_DIR = Path("data") / "models"
MODEL_PATH = MODEL_DIR / "lr2.joblib"
METADATA_PATH = MODEL_DIR / "lr2_metadata.json"

MODEL_NAME = "LR2"
TARGET_COLUMN = "future_return_30m"
POSITIVE_RETURN_THRESHOLD = 0.003
NUMERIC_COLUMNS = list(FEATURE_COLUMNS)
CATEGORICAL_COLUMNS = ["symbol"]
MODEL_INPUT_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


def make_label(frame):
    return (frame[TARGET_COLUMN] >= POSITIVE_RETURN_THRESHOLD).astype(int)


def load_splits():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing {DATASET_PATH}. Run python build_dataset.py first.")

    dataset = pd.read_parquet(
        DATASET_PATH,
        columns=["timestamp", "symbol", "split", *FEATURE_COLUMNS, TARGET_COLUMN],
    )
    train = dataset.loc[dataset["split"] == "train"].copy()
    validation = dataset.loc[dataset["split"] == "validation"].copy()
    if train.empty or validation.empty:
        raise ValueError("Training or validation split is empty.")
    return train, validation


def build_model():
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_COLUMNS),
            ("symbol", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )

    classifier = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced",
        C=1.0,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("classifier", classifier),
        ]
    )


def main():
    print("=" * 88)
    print("STAGE 4 - TRAIN LR2")
    print("=" * 88)
    print("Model:        Logistic Regression #2")
    print("Train:        2023-2024")
    print("Validation:   2025")
    print("Target:       30m return >= 0.30%")
    print("Difference:   class-balanced training; execution uses adaptive probability ranking")

    train, validation = load_splits()
    y_train = make_label(train)
    y_validation = make_label(validation)

    print(f"\nTraining rows:   {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Training positive rate: {y_train.mean() * 100:.2f}%")

    model = build_model()
    print("\nTraining LR2...")
    model.fit(train[MODEL_INPUT_COLUMNS], y_train)

    probabilities = model.predict_proba(validation[MODEL_INPUT_COLUMNS])[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_validation, probabilities)),
        "average_precision": float(average_precision_score(y_validation, probabilities)),
        "log_loss": float(log_loss(y_validation, probabilities)),
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
    print("Probability percentiles:")
    for name, value in metrics["probability_percentiles"].items():
        print(f"  {name}: {value:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "model_name": MODEL_NAME,
        "model_type": "logistic_regression_balanced",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_column": TARGET_COLUMN,
        "positive_return_threshold": POSITIVE_RETURN_THRESHOLD,
        "train_period": "2023-2024",
        "validation_period": "2025",
        "numeric_features": NUMERIC_COLUMNS,
        "categorical_features": CATEGORICAL_COLUMNS,
        "train_rows": int(len(train)),
        "train_positive_rate": float(y_train.mean()),
        "validation_metrics": metrics,
        "execution_design": "causal rolling probability percentile; tuned separately on 2025",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")

    print("\n" + "=" * 88)
    print("LR2 SAVED")
    print("=" * 88)
    print(f"Model:    {MODEL_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    print("Next: python tune_lr2.py")


if __name__ == "__main__":
    main()
