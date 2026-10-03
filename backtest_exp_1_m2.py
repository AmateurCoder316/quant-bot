import json
from pathlib import Path

import joblib
import pandas as pd

from exp1_m2_core import (
    HEAD_COLUMNS,
    HORIZON_MINUTES,
    MODEL_NAME,
    TARGET_COLUMNS,
    evaluate_config,
    execution_config_from_result,
    head_metrics,
    load_execution_frames,
    predict_multihead,
    select_consensus_signals,
)
from features import FEATURE_COLUMNS


DATASET_PATH = Path("data") / "ml" / "features.parquet"
OUTPUT_DIR = Path("data") / "experiments" / MODEL_NAME
MODEL_PATH = OUTPUT_DIR / "model.joblib"
CONFIG_PATH = OUTPUT_DIR / "execution.json"
TUNE_YEAR = 2025
TEST_YEAR = 2026


def load_year(year):
    data = pd.read_parquet(
        DATASET_PATH,
        columns=["timestamp", "symbol", "year", *FEATURE_COLUMNS, *TARGET_COLUMNS],
    )
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data["year"] = data["year"].astype(int)
    result = data.loc[data["year"] == year].copy()
    result = result.dropna(subset=[*FEATURE_COLUMNS, *TARGET_COLUMNS])
    return result.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def print_metrics(title, metrics):
    print("\n" + "=" * 108)
    print(title)
    print("=" * 108)
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
        raise FileNotFoundError(f"Missing {MODEL_PATH}. Run python train_exp_1_m2.py first.")
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing {CONFIG_PATH}. Run python tune_exp_1_m2.py first.")

    config_payload = json.loads(CONFIG_PATH.read_text())
    if config_payload.get("model") != MODEL_NAME:
        raise ValueError("Execution config is not for exp-1-m2.")
    if config_payload.get("selection_period") != "2025":
        raise ValueError("exp-1-m2 execution config must be selected on 2025 only.")

    selected = config_payload["selected"]
    frozen_config = execution_config_from_result(selected)
    bundle = joblib.load(MODEL_PATH)

    print("=" * 108)
    print("EXP-1-M2 - FROZEN 2026 HISTORICAL BACKTEST")
    print("=" * 108)
    print("Simulation only. No real orders are placed.")
    print("Model family:          three-head multi-horizon Extra Trees consensus")
    print("Train years:           2023-2024")
    print("Execution tuned on:    2025")
    print(f"Frozen test year:      {TEST_YEAR}")
    print(f"Hold time:             {HORIZON_MINUTES} minutes")
    print(f"Consensus percentile:  {frozen_config.consensus_percentile:.3f}")
    print(
        "Deployment gate:       "
        + ("PASS" if selected.get("passes_deployment_gate") else "FAIL - SHADOW RESEARCH ONLY")
    )

    # 2025 predictions are used only as prior probability history for the causal
    # rolling thresholds. 2025 outcomes were already used by the tuner; no 2026
    # outcome or future probability is used to form a 2026 threshold.
    tune = load_year(TUNE_YEAR)
    test = load_year(TEST_YEAR)
    seed_predictions = predict_multihead(bundle, tune)
    test_predictions = predict_multihead(bundle, test)
    frames = load_execution_frames(TEST_YEAR)
    diagnostics = head_metrics(test, test_predictions)

    selected_rows = select_consensus_signals(
        test_predictions,
        frozen_config,
        seed_predictions=seed_predictions,
    )

    print("\nMODEL / SIGNAL DIAGNOSTICS")
    print(f"Prediction rows:        {len(test_predictions):,}")
    print(f"Rows passing consensus: {len(selected_rows):,}")
    print(f"Max raw consensus:      {test_predictions['raw_consensus'].max():.4f}")
    print(f"Max weakest head:       {test_predictions['weakest_head'].max():.4f}")
    for head_name in HEAD_COLUMNS:
        metric = diagnostics[head_name]
        print(
            f"{head_name:<22} AUC={metric['roc_auc']:.4f} "
            f"AP={metric['average_precision']:.4f} "
            f"positive={metric['positive_rate'] * 100:.2f}%"
        )

    result = evaluate_config(
        test_predictions,
        frames,
        frozen_config,
        seed_predictions=seed_predictions,
    )
    cost = result["cost"]
    gross = result["gross"]

    print_metrics("2026 CONFIGURED COST MODEL", cost)
    print_metrics("2026 FRICTIONLESS DIAGNOSTIC", gross)

    print("\n" + "=" * 108)
    print("EXP-1-M2 BACKTEST SUMMARY")
    print("=" * 108)
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
