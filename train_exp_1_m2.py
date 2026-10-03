import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from exp1_m2_core import (
    HEAD_COLUMNS,
    HEAD_SPECS,
    HORIZON_MINUTES,
    MODEL_INPUT_COLUMNS,
    MODEL_NAME,
    TARGET_COLUMNS,
    fit_multihead_model,
    head_metrics,
    predict_multihead,
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
        *TARGET_COLUMNS,
    ]
    data = pd.read_parquet(DATASET_PATH, columns=columns)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data["year"] = data["year"].astype(int)
    data = data.dropna(subset=[*FEATURE_COLUMNS, *TARGET_COLUMNS]).copy()
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
    print("=" * 104)
    print("EXP-1-M2 - TRAIN")
    print("=" * 104)
    print("Architecture: multi-horizon Extra Trees classifier consensus")
    print("M1 lesson 1: exact-return regression removed")
    print("M1 lesson 2: regression-tree uncertainty filter removed")
    print("M1 lesson 3: later execution tuning uses one consensus percentile axis only")
    print(f"Train years:  {TRAIN_YEARS}")
    print(f"Validation:   {VALIDATION_YEAR}")
    print(f"Trade hold:   {HORIZON_MINUTES} minutes")
    print("Heads:")
    for head_name, spec in HEAD_SPECS.items():
        print(f"  {head_name:<14} -> {spec['description']}")

    train, validation = load_data()
    print(f"\nTraining rows:   {len(train):,}")
    print(f"Validation rows: {len(validation):,}")

    print("\nTraining three independent classifier heads...")
    bundle, train_labels = fit_multihead_model(train)
    validation_predictions = predict_multihead(bundle, validation)
    metrics = head_metrics(validation, validation_predictions)

    print("\n" + "=" * 104)
    print("2025 VALIDATION HEAD METRICS")
    print("=" * 104)
    for head_name in HEAD_COLUMNS:
        metric = metrics[head_name]
        train_rate = float(train_labels[head_name].mean())
        print(f"\n{head_name} | {metric['description']}")
        print(f"  train positive rate: {train_rate * 100:.2f}%")
        print(f"  valid positive rate: {metric['positive_rate'] * 100:.2f}%")
        print(f"  ROC AUC:             {fmt_optional(metric['roc_auc'])}")
        print(f"  Average precision:   {fmt_optional(metric['average_precision'])}")
        print(f"  Log loss:            {fmt_optional(metric['log_loss'])}")
        for percentile in (50, 90, 95, 99):
            probability = validation_predictions[head_name].quantile(percentile / 100.0)
            print(f"  p{percentile:02d} probability:     {probability:.4f}")

    print("\nConsensus diagnostics:")
    for percentile in (50, 75, 90, 95, 99):
        raw = validation_predictions["raw_consensus"].quantile(percentile / 100.0)
        weakest = validation_predictions["weakest_head"].quantile(percentile / 100.0)
        print(
            f"  p{percentile:02d}: geometric_consensus={raw:.4f} "
            f"weakest_head={weakest:.4f}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    metadata = {
        "model": MODEL_NAME,
        "family": "multi_horizon_extra_trees_classifiers",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": {
            "heads": HEAD_SPECS,
            "decision_logic": (
                "Three independent ExtraTreesClassifier heads predict 30m positive net, "
                "60m positive net, and 60m net >= +0.30%. Execution requires every head "
                "to be simultaneously in the same causal rolling probability tail."
            ),
            "m1_lessons": [
                "Remove exact-return regression because M1 regression ranking did not generalize.",
                "Remove uncertainty filtering because M1 selected the 100th-percentile cap.",
                "Reduce threshold-selection freedom and demand a larger stable trade sample.",
            ],
        },
        "train_years": TRAIN_YEARS,
        "validation_year": VALIDATION_YEAR,
        "hold_minutes": HORIZON_MINUTES,
        "feature_count": len(FEATURE_COLUMNS),
        "model_input_count": len(MODEL_INPUT_COLUMNS),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "validation_metrics": metrics,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")

    print("\n" + "=" * 104)
    print("EXP-1-M2 SAVED")
    print("=" * 104)
    print(f"Model:    {MODEL_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    print("Next: python tune_exp_1_m2.py")


if __name__ == "__main__":
    main()
