import json
from pathlib import Path

import joblib
import pandas as pd

from features import FEATURE_COLUMNS
from market_data import LOCAL_PROVIDER
from strategy_core import COMMISSION_RATE, SLIPPAGE_RATE, SYMBOLS
from train_gb2 import MODEL_INPUT_COLUMNS, add_symbol_columns
from tune_lr1 import build_trade_candidates as build_fixed_candidates
from tune_lr1 import metrics_from_trades, run_portfolio
from tune_lr2 import add_adaptive_threshold, build_trade_candidates as build_adaptive_candidates


DATASET_PATH = Path("data") / "ml" / "features.parquet"
MODEL_PATH = Path("data") / "models" / "gb2.joblib"
CONFIG_PATH = Path("data") / "models" / "gb2_execution.json"

# GB2 targets larger moves, so test stricter confidence cutoffs than GB1.
FIXED_THRESHOLDS = [0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90]
ADAPTIVE_PERCENTILES = [0.95, 0.975, 0.99, 0.995]
HOLD_BARS = 6
MIN_TRADES_FOR_SELECTION = 30


def load_validation_predictions():
    dataset = pd.read_parquet(
        DATASET_PATH,
        columns=["timestamp", "symbol", "split", *FEATURE_COLUMNS],
    )
    validation = dataset.loc[dataset["split"] == "validation"].copy()
    validation["timestamp"] = pd.to_datetime(validation["timestamp"], utc=True)

    validation_for_model = add_symbol_columns(validation)
    model = joblib.load(MODEL_PATH)
    validation["probability"] = model.predict_proba(
        validation_for_model[MODEL_INPUT_COLUMNS]
    )[:, 1]

    return validation[["timestamp", "symbol", "probability"]]


def load_execution_data(year=2025):
    frames = {}
    for symbol in SYMBOLS:
        frame = LOCAL_PROVIDER.history(symbol).copy()
        frames[symbol] = frame.loc[frame.index.year == year]
    return frames


def evaluate_candidates(candidates):
    trades, final_value = run_portfolio(candidates, COMMISSION_RATE, SLIPPAGE_RATE)
    gross_trades, gross_final = run_portfolio(candidates, 0.0, 0.0)
    return {
        "cost": metrics_from_trades(trades, final_value),
        "gross": metrics_from_trades(gross_trades, gross_final),
    }


def evaluate_fixed(threshold, predictions, frames):
    candidates = build_fixed_candidates(predictions, frames, threshold)
    result = evaluate_candidates(candidates)
    return {"mode": "fixed", "value": threshold, **result}


def evaluate_adaptive(percentile, predictions, frames):
    ordered = predictions.sort_values(["symbol", "timestamp"])
    adaptive = add_adaptive_threshold(ordered, percentile)
    candidates = build_adaptive_candidates(adaptive, frames)
    result = evaluate_candidates(candidates)
    return {"mode": "adaptive", "value": percentile, **result}


def choose_result(results):
    eligible = [r for r in results if r["cost"]["trades"] >= MIN_TRADES_FOR_SELECTION]
    if not eligible:
        eligible = results
    return max(eligible, key=lambda r: (r["cost"]["return"], r["cost"]["profit_factor"]))


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing {MODEL_PATH}. Run python train_gb2.py first.")

    print("=" * 96)
    print("GB2 - 2025 EXECUTION TUNING")
    print("=" * 96)
    print(f"Hold: {HOLD_BARS * 5} minutes")
    print("Target model: 30m move >= +0.50%")
    print("Testing stricter fixed thresholds and causal adaptive per-symbol ranking.")

    predictions = load_validation_predictions()
    frames = load_execution_data(2025)
    results = []

    print("\nFIXED THRESHOLDS")
    for threshold in FIXED_THRESHOLDS:
        result = evaluate_fixed(threshold, predictions, frames)
        results.append(result)
        cost = result["cost"]
        gross = result["gross"]
        print(
            f"p>={threshold:.2f} | cost={cost['return']:+7.2f}% PF={cost['profit_factor']:.2f} "
            f"trades={cost['trades']:4d} win={cost['win_rate']:5.1f}% avg={cost['avg_trade']:+.3f}% "
            f"| gross={gross['return']:+7.2f}% PF={gross['profit_factor']:.2f}"
        )

    print("\nADAPTIVE RANKING")
    for percentile in ADAPTIVE_PERCENTILES:
        result = evaluate_adaptive(percentile, predictions, frames)
        results.append(result)
        cost = result["cost"]
        gross = result["gross"]
        print(
            f"top {(1-percentile)*100:>4.1f}% | cost={cost['return']:+7.2f}% PF={cost['profit_factor']:.2f} "
            f"trades={cost['trades']:4d} win={cost['win_rate']:5.1f}% avg={cost['avg_trade']:+.3f}% "
            f"| gross={gross['return']:+7.2f}% PF={gross['profit_factor']:.2f}"
        )

    selected = choose_result(results)

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "GB2",
        "selection_period": "2025",
        "target": "future_return_30m >= 0.005",
        "hold_bars": HOLD_BARS,
        "hold_minutes": HOLD_BARS * 5,
        "commission_rate": COMMISSION_RATE,
        "slippage_rate": SLIPPAGE_RATE,
        "selection_rule": "highest 2025 return after costs among configs with >=30 trades; PF tie-break",
        "selected_mode": selected["mode"],
        "selected_value": selected["value"],
        "all_results": results,
        "selected_result": selected,
    }
    CONFIG_PATH.write_text(json.dumps(payload, indent=2) + "\n")

    print("\n" + "=" * 96)
    print("FROZEN GB2 EXECUTION SETUP")
    print("=" * 96)
    print(f"Selected mode:         {selected['mode']}")
    if selected["mode"] == "fixed":
        print(f"Selected threshold:    {selected['value']:.2f}")
    else:
        print(f"Selected percentile:   top {(1-selected['value'])*100:.1f}%")
    print(f"2025 return w/costs:   {selected['cost']['return']:+.2f}%")
    print(f"2025 profit factor:    {selected['cost']['profit_factor']:.2f}")
    print(f"2025 trades:           {selected['cost']['trades']}")
    print(f"Saved execution config: {CONFIG_PATH}")
    print("Next: build/run the frozen GB2 2026 backtest.")


if __name__ == "__main__":
    main()
