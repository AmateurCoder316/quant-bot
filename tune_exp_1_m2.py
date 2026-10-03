import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from exp1_m2_core import (
    HEAD_COLUMNS,
    HORIZON_MINUTES,
    MIN_TRADES_FOR_SELECTION,
    MODEL_NAME,
    TARGET_COLUMNS,
    all_execution_configs,
    evaluate_config,
    head_metrics,
    load_execution_frames,
    predict_multihead,
    select_execution_config,
)
from features import FEATURE_COLUMNS


DATASET_PATH = Path("data") / "ml" / "features.parquet"
OUTPUT_DIR = Path("data") / "experiments" / MODEL_NAME
MODEL_PATH = OUTPUT_DIR / "model.joblib"
CONFIG_PATH = OUTPUT_DIR / "execution.json"
TUNE_YEAR = 2025


def load_tune_data():
    data = pd.read_parquet(
        DATASET_PATH,
        columns=["timestamp", "symbol", "year", *FEATURE_COLUMNS, *TARGET_COLUMNS],
    )
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data["year"] = data["year"].astype(int)
    tune = data.loc[data["year"] == TUNE_YEAR].copy()
    tune = tune.dropna(subset=[*FEATURE_COLUMNS, *TARGET_COLUMNS])
    return tune.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def json_clean(value):
    if isinstance(value, dict):
        return {key: json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing {MODEL_PATH}. Run python train_exp_1_m2.py first.")

    print("=" * 118)
    print("EXP-1-M2 - 2025 EXECUTION TUNING")
    print("=" * 118)
    print("Architecture is frozen. 2026 is not used for selection.")
    print(f"Hold: {HORIZON_MINUTES} minutes")
    print("Execution rule: every head must be in the same causal rolling probability tail.")
    print("Only five predefined consensus percentiles are tested.")
    print(f"Minimum trades for deployment support: {MIN_TRADES_FOR_SELECTION}")

    bundle = joblib.load(MODEL_PATH)
    tune = load_tune_data()
    predictions = predict_multihead(bundle, tune)
    frames = load_execution_frames(TUNE_YEAR)
    metrics = head_metrics(tune, predictions)

    print("\n2025 head diagnostics:")
    for head_name in HEAD_COLUMNS:
        metric = metrics[head_name]
        print(
            f"  {head_name:<14} AUC={metric['roc_auc']:.4f} "
            f"AP={metric['average_precision']:.4f} "
            f"positive={metric['positive_rate'] * 100:.2f}%"
        )

    results = []
    configs = all_execution_configs()
    print(f"\nTesting {len(configs)} consensus configurations...\n")

    for config in configs:
        result = evaluate_config(predictions, frames, config)
        results.append(result)
        cost = result["cost"]
        gross = result["gross"]
        print(
            f"{config.label:<23} | cost={cost['return']:+7.2f}% "
            f"PF={cost['profit_factor']:.2f} trades={cost['trades']:4d} "
            f"avg={cost['avg_trade']:+.3f}% DD={cost['max_drawdown']:+.2f}% "
            f"gross={gross['return']:+7.2f}%"
        )

    selected, scored = select_execution_config(results)

    print("\n" + "=" * 118)
    print("FROZEN EXP-1-M2 EXECUTION SETUP")
    print("=" * 118)
    print(f"Selected:                {selected['label']}")
    print(f"Consensus percentile:    {selected['consensus_percentile']:.3f}")
    print(f"2025 return w/costs:     {selected['cost']['return']:+.2f}%")
    print(f"2025 profit factor:      {selected['cost']['profit_factor']:.2f}")
    print(f"2025 average trade:      {selected['cost']['avg_trade']:+.3f}%")
    print(f"2025 trades:             {selected['cost']['trades']}")
    print(f"Local median return:     {selected['stability_return']:+.2f}%")
    print(f"Local worst return:      {selected['stability_min_return']:+.2f}%")
    print(f"Local median PF:         {selected['stability_profit_factor']:.2f}")
    print(
        "Deployment gate:         "
        + ("PASS" if selected["passes_deployment_gate"] else "FAIL - SHADOW RESEARCH ONLY")
    )

    payload = {
        "model": MODEL_NAME,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_period": str(TUNE_YEAR),
        "test_period": "2026",
        "architecture": "three-head multi-horizon Extra Trees classifier consensus",
        "hold_minutes": HORIZON_MINUTES,
        "selection_policy": (
            "Choose among five predefined causal rolling consensus percentiles only. Every head "
            "must independently exceed its own trailing probability quantile. Deployment requires "
            ">=75 trades, positive after-cost performance, PF/Sharpe support, and a positive "
            "three-point neighboring threshold plateau including the worst neighbor."
        ),
        "selected": selected,
        "all_results": scored,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(json_clean(payload), indent=2) + "\n")

    print(f"Saved: {CONFIG_PATH}")
    print("Next: python backtest_exp_1_m2.py")


if __name__ == "__main__":
    main()
