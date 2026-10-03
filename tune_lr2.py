import json
from pathlib import Path

import joblib
import pandas as pd

from features import FEATURE_COLUMNS
from market_data import LOCAL_PROVIDER
from strategy_core import COMMISSION_RATE, SLIPPAGE_RATE, SYMBOLS
from tune_lr1 import metrics_from_trades, run_portfolio


DATASET_PATH = Path("data") / "ml" / "features.parquet"
MODEL_PATH = Path("data") / "models" / "lr2.joblib"
CONFIG_PATH = Path("data") / "models" / "lr2_execution.json"
MODEL_INPUT_COLUMNS = list(FEATURE_COLUMNS) + ["symbol"]

ROLLING_WINDOW = 1000
MIN_HISTORY = 500
PERCENTILES = [0.95, 0.975, 0.99]
HOLD_BARS = 6


def load_validation_predictions():
    dataset = pd.read_parquet(
        DATASET_PATH,
        columns=["timestamp", "symbol", "split", *FEATURE_COLUMNS],
    )
    validation = dataset.loc[dataset["split"] == "validation"].copy()
    validation["timestamp"] = pd.to_datetime(validation["timestamp"], utc=True)

    model = joblib.load(MODEL_PATH)
    validation["probability"] = model.predict_proba(validation[MODEL_INPUT_COLUMNS])[:, 1]
    return validation[["timestamp", "symbol", "probability"]].sort_values(["symbol", "timestamp"])


def add_adaptive_threshold(predictions, percentile):
    pieces = []
    for symbol, group in predictions.groupby("symbol", sort=False):
        group = group.sort_values("timestamp").copy()
        # shift(1) guarantees the current prediction is NOT used to define its own threshold.
        history = group["probability"].shift(1)
        group["adaptive_threshold"] = history.rolling(
            ROLLING_WINDOW,
            min_periods=MIN_HISTORY,
        ).quantile(percentile)
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def load_execution_data(year=2025):
    frames = {}
    for symbol in SYMBOLS:
        frame = LOCAL_PROVIDER.history(symbol).copy()
        frames[symbol] = frame.loc[frame.index.year == year]
    return frames


def build_trade_candidates(predictions, frames):
    candidates = []
    selected = predictions.loc[
        predictions["adaptive_threshold"].notna()
        & (predictions["probability"] >= predictions["adaptive_threshold"])
    ]

    for symbol in SYMBOLS:
        frame = frames[symbol]
        positions = {timestamp: i for i, timestamp in enumerate(frame.index)}
        rows = selected.loc[selected["symbol"] == symbol]

        for row in rows.itertuples(index=False):
            signal_pos = positions.get(row.timestamp)
            if signal_pos is None:
                continue
            entry_pos = signal_pos + 1
            exit_pos = entry_pos + HOLD_BARS
            if exit_pos >= len(frame):
                continue

            entry_time = frame.index[entry_pos]
            exit_time = frame.index[exit_pos]
            if entry_time.date() != exit_time.date():
                continue

            candidates.append({
                "symbol": symbol,
                "signal_time": row.timestamp,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "probability": float(row.probability),
                "entry_raw": float(frame.iloc[entry_pos]["Open"]),
                "exit_raw": float(frame.iloc[exit_pos]["Open"]),
            })

    return sorted(candidates, key=lambda item: (item["entry_time"], -item["probability"]))


def evaluate(percentile, predictions, frames):
    adaptive = add_adaptive_threshold(predictions, percentile)
    candidates = build_trade_candidates(adaptive, frames)
    trades, final_value = run_portfolio(candidates, COMMISSION_RATE, SLIPPAGE_RATE)
    gross_trades, gross_final = run_portfolio(candidates, 0.0, 0.0)
    return {
        "percentile": percentile,
        "cost": metrics_from_trades(trades, final_value),
        "gross": metrics_from_trades(gross_trades, gross_final),
    }


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing {MODEL_PATH}. Run python train_lr2.py first.")

    print("=" * 92)
    print("LR2 - 2025 ADAPTIVE EXECUTION TUNING")
    print("=" * 92)
    print(f"Rolling history: {ROLLING_WINDOW} bars per symbol (minimum {MIN_HISTORY})")
    print(f"Hold:            {HOLD_BARS * 5} minutes")
    print("Rule: trade when current probability is in the selected tail of that symbol's recent history.")
    print("The current bar is excluded from its own threshold calculation.")

    predictions = load_validation_predictions()
    frames = load_execution_data(2025)

    results = []
    for percentile in PERCENTILES:
        result = evaluate(percentile, predictions, frames)
        results.append(result)
        cost = result["cost"]
        gross = result["gross"]
        print(
            f"top {(1-percentile)*100:>4.1f}% | cost={cost['return']:+7.2f}% PF={cost['profit_factor']:.2f} "
            f"trades={cost['trades']:4d} win={cost['win_rate']:5.1f}% avg={cost['avg_trade']:+.3f}% "
            f"| gross={gross['return']:+7.2f}% PF={gross['profit_factor']:.2f}"
        )

    selected = max(results, key=lambda item: (item["cost"]["return"], item["cost"]["profit_factor"]))

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "LR2",
        "selection_period": "2025",
        "selection_method": "causal rolling probability percentile",
        "rolling_window": ROLLING_WINDOW,
        "min_history": MIN_HISTORY,
        "percentile": selected["percentile"],
        "hold_bars": HOLD_BARS,
        "hold_minutes": HOLD_BARS * 5,
        "commission_rate": COMMISSION_RATE,
        "slippage_rate": SLIPPAGE_RATE,
        "all_results": results,
        "selected_result": selected,
    }
    CONFIG_PATH.write_text(json.dumps(payload, indent=2) + "\n")

    print("\n" + "=" * 92)
    print("FROZEN LR2 EXECUTION SETUP")
    print("=" * 92)
    print(f"Selected percentile: top {(1-selected['percentile'])*100:.1f}%")
    print(f"2025 return w/costs: {selected['cost']['return']:+.2f}%")
    print(f"2025 profit factor:  {selected['cost']['profit_factor']:.2f}")
    print(f"2025 trades:         {selected['cost']['trades']}")
    print(f"Saved:               {CONFIG_PATH}")
    print("Next: python backtest_lr2.py")


if __name__ == "__main__":
    main()
