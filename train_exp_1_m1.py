import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from exp1_core import (
    HORIZON_MINUTES,
    MODEL_INPUT_COLUMNS,
    MODEL_NAME,
    TARGET_COLUMN,
    fit_dual_model,
    model_metrics,
    predict_dual,
)
from features import FEATURE_COLUMNS


DATASET_PATH = Path("data") / "ml" / "features.parquet"
OUTPUT_DIR = Path("data") / "experiments" / MODEL_NAME
MODEL_PATH = OUTPUT_DIR / "model.joblib"
METADATA_PATH = OUTPUT_DIR / "metadata.json"

TRAIN_YEARS = [2023, 2024]
VALIDATION_YEAR = 2025


def load_data():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing {DATASET_PATH}. Run python build_dataset.py first.")

    columns = [
        "timestamp",
        "symbol",
        "year",
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
    ]
    data = pd.read_parquet(DATASET_PATH, columns=columns)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data["year"] = data["year"].astype(int)
    data = data.dropna(subset=[*FEATURE_COLUMNS, TARGET_COLUMN]).copy()
    data = data.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    train = data.loc[data["year"].isin(TRAIN_YEARS)].copy().reset_index(drop=True)
    validation = data.loc[data["year"] == VALIDATION_YEAR].copy().reset_index(drop=True)

    if train.empty or validation.empty:
        raise ValueError("Training or validation data is empty.")

    return train, validation


def fmt_optional(value, digits=4):
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def main():
    print("=" * 96)
    print("EXP-1-M1 - TRAIN")
    print("=" * 96)
    print("Architecture: dual Extra Trees consensus")
    print("Head A:       classifier -> probability 60m trade is profitable after costs")
    print("Head B:       regressor  -> expected 60m net trade return")
    print("Uncertainty:  disagreement across a sample of regression trees")
    print(f"Train years:  {TRAIN_YEARS}")
    print(f"Validation:   {VALIDATION_YEAR}")
    print(f"Target:       {TARGET_COLUMN}")
    print(f"Hold horizon: {HORIZON_MINUTES} minutes")

    train, validation = load_data()
    print(f"\nTraining rows:   {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(
        f"Train positive-net rate: "
        f"{(train[TARGET_COLUMN] > 0.0).mean() * 100:.2f}%"
    )
    print(
        f"Validation positive-net rate: "
        f"{(validation[TARGET_COLUMN] > 0.0).mean() * 100:.2f}%"
    )

    print("\nTraining classifier and regressor...")
    bundle, _, _ = fit_dual_model(train)

    validation_predictions = predict_dual(bundle, validation)
    metrics = model_metrics(validation, validation_predictions)

    print("\n" + "=" * 96)
    print("2025 VALIDATION MODEL METRICS")
    print("=" * 96)
    print("Classifier:")
    print(f"  ROC AUC:            {fmt_optional(metrics['roc_auc'])}")
    print(f"  Average precision:  {fmt_optional(metrics['average_precision'])}")
    print(f"  Log loss:           {fmt_optional(metrics['log_loss'])}")
    print("Regressor:")
    print(f"  MAE:                {metrics['mae'] * 100:.3f}%")
    print(f"  RMSE:               {metrics['rmse'] * 100:.3f}%")
    print(f"  Pearson correlation:{metrics['pearson']:.4f}")
    print(f"  Spearman correlation:{metrics['spearman']:.4f}")
    print(f"  Sign accuracy:      {metrics['sign_accuracy'] * 100:.2f}%")

    print("\nPrediction diagnostics:")
    for percentile in (50, 75, 90, 95, 99):
        p = validation_predictions["win_probability"].quantile(percentile / 100.0)
        ev = validation_predictions["expected_net_return"].quantile(percentile / 100.0)
        u = validation_predictions["uncertainty"].quantile(percentile / 100.0)
        print(
            f"  p{percentile:02d}: win_prob={p:.4f} "
            f"expected_net={ev * 100:+.3f}% uncertainty={u * 100:.3f}%"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    metadata = {
        "model": MODEL_NAME,
        "family": "dual_extra_trees",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": {
            "classifier": "ExtraTreesClassifier",
            "regressor": "ExtraTreesRegressor",
            "decision_logic": (
                "Classifier estimates probability of positive net trade; regressor estimates "
                "net-return magnitude; execution later requires both to agree and may reject "
                "high-disagreement predictions."
            ),
        },
        "train_years": TRAIN_YEARS,
        "validation_year": VALIDATION_YEAR,
        "target_column": TARGET_COLUMN,
        "horizon_minutes": HORIZON_MINUTES,
        "feature_count": len(FEATURE_COLUMNS),
        "model_input_count": len(MODEL_INPUT_COLUMNS),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "validation_metrics": metrics,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")

    print("\n" + "=" * 96)
    print("EXP-1-M1 SAVED")
    print("=" * 96)
    print(f"Model:    {MODEL_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    print("Next: python tune_exp_1_m1.py")


if __name__ == "__main__":
    main()
