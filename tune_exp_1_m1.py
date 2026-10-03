import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from exp1_core import (
    HORIZON_MINUTES,
    MODEL_NAME,
    TARGET_COLUMN,
    all_execution_configs,
    evaluate_config,
    load_execution_frames,
    model_metrics,
    predict_dual,
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
        columns=["timestamp", "symbol", "year", *FEATURE_COLUMNS, TARGET_COLUMN],
    )
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data["year"] = data["year"].astype(int)
    tune = data.loc[data["year"] == TUNE_YEAR].copy()
    tune = tune.dropna(subset=[*FEATURE_COLUMNS, TARGET_COLUMN])
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
        raise FileNotFoundError(f"Missing {MODEL_PATH}. Run python train_exp_1_m1.py first.")

    print("=" * 118)
    print("EXP-1-M1 - 2025 EXECUTION TUNING")
    print("=" * 118)
    print("Architecture is frozen. This script only chooses the execution consensus gates.")
    print("2026 is not used for selection.")
    print(f"Hold: {HORIZON_MINUTES} minutes")
    print("Grid axes: win probability x expected net return x uncertainty cap")

    bundle = joblib.load(MODEL_PATH)
    tune = load_tune_data()
    predictions = predict_dual(bundle, tune)
    frames = load_execution_frames(TUNE_YEAR)
    metrics = model_metrics(tune, predictions)

    print(
        f"\n2025 model diagnostics: AUC={metrics['roc_auc']:.4f} "
        f"AP={metrics['average_precision']:.4f} "
        f"Spearman={metrics['spearman']:.4f} sign={metrics['sign_accuracy'] * 100:.1f}%"
    )

    results = []
    configs = all_execution_configs(predictions)
    print(f"Testing {len(configs)} execution configurations...\n")

    for config in configs:
        result = evaluate_config(predictions, frames, config)
        results.append(result)
        cost = result["cost"]
        print(
            f"{config.label:<31} | cost={cost['return']:+7.2f}% "
            f"PF={cost['profit_factor']:.2f} trades={cost['trades']:4d} "
            f"avg={cost['avg_trade']:+.3f}% DD={cost['max_drawdown']:+.2f}%"
        )

    selected, scored = select_execution_config(results)

    print("\n" + "=" * 118)
    print("FROZEN EXP-1-M1 EXECUTION SETUP")
    print("=" * 118)
    print(f"Selected:                {selected['label']}")
    print(f"Probability threshold:   {selected['probability_threshold']:.2f}")
    print(f"Expected-net threshold:  {selected['expected_return_threshold'] * 100:+.2f}%")
    print(f"Uncertainty percentile:  {selected['uncertainty_quantile']:.0%}")
    print(f"Frozen uncertainty cap:  {selected['uncertainty_cap'] * 100:.3f}%")
    print(f"2025 return w/costs:     {selected['cost']['return']:+.2f}%")
    print(f"2025 profit factor:      {selected['cost']['profit_factor']:.2f}")
    print(f"2025 average trade:      {selected['cost']['avg_trade']:+.3f}%")
    print(f"2025 trades:             {selected['cost']['trades']}")
    print(f"Local stability return:  {selected['stability_return']:+.2f}%")
    print(
        "Deployment gate:         "
        + ("PASS" if selected["passes_deployment_gate"] else "FAIL - SHADOW RESEARCH ONLY")
    )

    payload = {
        "model": MODEL_NAME,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_period": str(TUNE_YEAR),
        "test_period": "2026",
        "architecture": "dual Extra Trees classifier + regressor consensus",
        "target_column": TARGET_COLUMN,
        "hold_minutes": HORIZON_MINUTES,
        "selection_policy": (
            "Require classifier and regressor agreement; uncertainty cap is frozen from 2025. "
            "Rank supported configs by deployment gate, local 2-D threshold stability, after-cost "
            "return, PF and Sharpe."
        ),
        "selected": selected,
        "all_results": scored,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(json_clean(payload), indent=2) + "\n")

    print(f"Saved: {CONFIG_PATH}")
    print("Next: python backtest_exp_1_m1.py")


if __name__ == "__main__":
    main()
