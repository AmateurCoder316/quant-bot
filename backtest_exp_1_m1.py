import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from exp1_core import (
    HORIZON_MINUTES,
    MODEL_NAME,
    TARGET_COLUMN,
    evaluate_config,
    execution_config_from_result,
    load_execution_frames,
    model_metrics,
    predict_dual,
)
from features import FEATURE_COLUMNS


DATASET_PATH = Path("data") / "ml" / "features.parquet"
OUTPUT_DIR = Path("data") / "experiments" / MODEL_NAME
MODEL_PATH = OUTPUT_DIR / "model.joblib"
CONFIG_PATH = OUTPUT_DIR / "execution.json"
TEST_YEAR = 2026


def load_test_data():
    data = pd.read_parquet(
        DATASET_PATH,
        columns=["timestamp", "symbol", "year", *FEATURE_COLUMNS, TARGET_COLUMN],
    )
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data["year"] = data["year"].astype(int)
    test = data.loc[data["year"] == TEST_YEAR].copy()
    test = test.dropna(subset=[*FEATURE_COLUMNS, TARGET_COLUMN])
    return test.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def print_metrics(title, metrics):
    print("\n" + "=" * 104)
    print(title)
    print("=" * 104)
    print(f"Total return:        {metrics['return']:+.2f}%")
    print(f"Completed trades:    {metrics['trades']}")
    print(f"Win rate:            {metrics['win_rate']:.2f}%")
    print(f"Profit factor:       {metrics['profit_factor']:.2f}")
    print(f"Average trade:       {metrics['avg_trade']:+.3f}%")
    print(f"Max drawdown:        {metrics['max_drawdown']:+.2f}%")
    print(f"Daily Sharpe:        {metrics['sharpe']:.2f}")
    print(f"Modeled friction:    ${metrics['friction']:,.2f}")


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing {MODEL_PATH}. Run python train_exp_1_m1.py first.")
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing {CONFIG_PATH}. Run python tune_exp_1_m1.py first.")

    config_payload = json.loads(CONFIG_PATH.read_text())
    if config_payload.get("model") != MODEL_NAME:
        raise ValueError("Execution config is not for exp-1-m1.")
    if config_payload.get("selection_period") != "2025":
        raise ValueError("exp-1-m1 execution config must be selected on 2025 only.")

    selected = config_payload["selected"]
    frozen_config = execution_config_from_result(selected)
    bundle = joblib.load(MODEL_PATH)

    print("=" * 104)
    print("EXP-1-M1 - FROZEN 2026 HISTORICAL BACKTEST")
    print("=" * 104)
    print("Simulation only. No real orders are placed.")
    print("Model family:          dual Extra Trees consensus")
    print("Train years:           2023-2024")
    print("Execution tuned on:    2025")
    print(f"Frozen test year:      {TEST_YEAR}")
    print(f"Target:                {TARGET_COLUMN}")
    print(f"Hold time:             {HORIZON_MINUTES} minutes")
    print(f"Probability threshold: {frozen_config.probability_threshold:.2f}")
    print(
        f"Expected-net threshold:{frozen_config.expected_return_threshold * 100:+.2f}%"
    )
    print(f"Uncertainty cap:       {frozen_config.uncertainty_cap * 100:.3f}%")
    print(
        "Deployment gate:       "
        + ("PASS" if selected.get("passes_deployment_gate") else "FAIL - SHADOW RESEARCH ONLY")
    )

    test = load_test_data()
    predictions = predict_dual(bundle, test)
    frames = load_execution_frames(TEST_YEAR)
    diagnostics = model_metrics(test, predictions)

    raw_signal_mask = (
        (predictions["win_probability"] >= frozen_config.probability_threshold)
        & (
            predictions["expected_net_return"]
            >= frozen_config.expected_return_threshold
        )
        & (predictions["uncertainty"] <= frozen_config.uncertainty_cap)
    )

    print("\nMODEL / SIGNAL DIAGNOSTICS")
    print(f"Prediction rows:        {len(predictions):,}")
    print(f"Rows passing all gates: {int(raw_signal_mask.sum()):,}")
    print(f"Max win probability:    {predictions['win_probability'].max():.4f}")
    print(
        f"Max predicted net:      "
        f"{predictions['expected_net_return'].max() * 100:+.3f}%"
    )
    print(f"Test ROC AUC:           {diagnostics['roc_auc']:.4f}")
    print(f"Test avg precision:     {diagnostics['average_precision']:.4f}")
    print(f"Test Spearman:          {diagnostics['spearman']:.4f}")
    print(f"Test sign accuracy:     {diagnostics['sign_accuracy'] * 100:.2f}%")

    result = evaluate_config(predictions, frames, frozen_config)
    cost = result["cost"]
    gross = result["gross"]

    print_metrics("2026 CONFIGURED COST MODEL", cost)
    print_metrics("2026 FRICTIONLESS DIAGNOSTIC", gross)

    print("\n" + "=" * 104)
    print("EXP-1-M1 BACKTEST SUMMARY")
    print("=" * 104)
    print(f"Return with costs:     {cost['return']:+.2f}%")
    print(f"Return without costs:  {gross['return']:+.2f}%")
    print(f"Cost-model drag:       {cost['return'] - gross['return']:+.2f} percentage points")
    print(f"Trades:                {cost['trades']}")
    print(f"Profit factor costs:   {cost['profit_factor']:.2f}")
    print(f"Profit factor gross:   {gross['profit_factor']:.2f}")
    print(f"Max drawdown:          {cost['max_drawdown']:+.2f}%")
    print(f"Daily Sharpe:          {cost['sharpe']:.2f}")


if __name__ == "__main__":
    main()
