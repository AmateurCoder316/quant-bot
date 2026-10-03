import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import FEATURE_COLUMNS
from market_data import LOCAL_PROVIDER
from strategy_core import (
    COMMISSION_RATE,
    MAX_OPEN_POSITIONS,
    MAX_POSITION_PERCENT,
    SLIPPAGE_RATE,
    STARTING_CASH,
    SYMBOLS,
)


DATASET_PATH = Path("data") / "ml" / "features.parquet"
MODEL_PATH = Path("data") / "models" / "lr1.joblib"
CONFIG_PATH = Path("data") / "models" / "lr1_execution.json"

THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
HOLD_BARS = 6  # 6 x 5-minute bars = 30 minutes
MIN_TRADES_FOR_SELECTION = 30
MODEL_INPUT_COLUMNS = list(FEATURE_COLUMNS) + ["symbol"]


def load_validation_predictions():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing {DATASET_PATH}. Run python build_dataset.py first.")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing {MODEL_PATH}. Run python train_lr1.py first.")

    validation = pd.read_parquet(
        DATASET_PATH,
        columns=["timestamp", "symbol", "split", *FEATURE_COLUMNS],
    )
    validation = validation.loc[validation["split"] == "validation"].copy()
    validation["timestamp"] = pd.to_datetime(validation["timestamp"], utc=True)

    model = joblib.load(MODEL_PATH)
    validation["probability"] = model.predict_proba(validation[MODEL_INPUT_COLUMNS])[:, 1]
    return validation[["timestamp", "symbol", "probability"]]


def load_execution_data():
    frames = {}
    for symbol in SYMBOLS:
        frame = LOCAL_PROVIDER.history(symbol).copy()
        frames[symbol] = frame.loc[frame.index.year == 2025]
    return frames


def build_trade_candidates(predictions, frames, threshold):
    candidates = []

    for symbol in SYMBOLS:
        frame = frames[symbol]
        index = frame.index
        index_positions = {timestamp: i for i, timestamp in enumerate(index)}

        rows = predictions.loc[
            (predictions["symbol"] == symbol)
            & (predictions["probability"] >= threshold)
        ]

        for row in rows.itertuples(index=False):
            signal_time = row.timestamp
            signal_pos = index_positions.get(signal_time)
            if signal_pos is None:
                continue

            entry_pos = signal_pos + 1
            exit_pos = entry_pos + HOLD_BARS
            if exit_pos >= len(index):
                continue

            entry_time = index[entry_pos]
            exit_time = index[exit_pos]

            # Never turn this intraday experiment into an overnight trade.
            if entry_time.date() != exit_time.date():
                continue

            candidates.append(
                {
                    "symbol": symbol,
                    "signal_time": signal_time,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "probability": float(row.probability),
                    "entry_raw": float(frame.iloc[entry_pos]["Open"]),
                    "exit_raw": float(frame.iloc[exit_pos]["Open"]),
                }
            )

    return sorted(candidates, key=lambda item: (item["entry_time"], -item["probability"]))


def run_portfolio(candidates, commission_rate, slippage_rate):
    cash = STARTING_CASH
    open_positions = {}
    trades = []
    equity_points = []

    candidates_by_time = {}
    exits_by_time = {}
    all_times = set()

    for candidate in candidates:
        candidates_by_time.setdefault(candidate["entry_time"], []).append(candidate)
        all_times.add(candidate["entry_time"])
        all_times.add(candidate["exit_time"])

    for timestamp in sorted(all_times):
        # Exit first so capital can be reused at the same bar open.
        due = exits_by_time.pop(timestamp, [])
        for symbol in due:
            position = open_positions.pop(symbol, None)
            if position is None:
                continue

            raw_exit = position["exit_raw"]
            exit_price = raw_exit * (1.0 - slippage_rate)
            gross = position["shares"] * exit_price
            exit_commission = gross * commission_rate
            net = gross - exit_commission
            cash += net

            pnl = net - position["entry_cost"]
            trades.append(
                {
                    "symbol": symbol,
                    "entry_time": position["entry_time"],
                    "exit_time": timestamp,
                    "probability": position["probability"],
                    "pnl": pnl,
                    "return_percent": pnl / position["entry_cost"] * 100.0,
                    "commission": position["entry_commission"] + exit_commission,
                    "slippage_cost": position["shares"]
                    * ((position["entry_price"] - position["entry_raw"]) + (raw_exit - exit_price)),
                }
            )

        entries = candidates_by_time.get(timestamp, [])
        entries.sort(key=lambda item: item["probability"], reverse=True)

        for candidate in entries:
            symbol = candidate["symbol"]
            if symbol in open_positions:
                continue
            if len(open_positions) >= MAX_OPEN_POSITIONS:
                break

            marked_value = cash + sum(
                position["shares"] * position["entry_raw"]
                for position in open_positions.values()
            )
            allocation = min(cash, marked_value * MAX_POSITION_PERCENT)
            if allocation <= 0:
                continue

            raw_entry = candidate["entry_raw"]
            entry_price = raw_entry * (1.0 + slippage_rate)
            trade_value = allocation / (1.0 + commission_rate)
            entry_commission = trade_value * commission_rate
            shares = trade_value / entry_price
            entry_cost = trade_value + entry_commission

            cash -= entry_cost
            open_positions[symbol] = {
                **candidate,
                "shares": shares,
                "entry_price": entry_price,
                "entry_commission": entry_commission,
                "entry_cost": entry_cost,
            }
            exits_by_time.setdefault(candidate["exit_time"], []).append(symbol)

        equity_points.append((timestamp, cash))

    trades = pd.DataFrame(trades)
    final_value = cash
    return trades, final_value


def metrics_from_trades(trades, final_value):
    total_return = (final_value / STARTING_CASH - 1.0) * 100.0
    if trades.empty:
        return {
            "return": total_return,
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_trade": 0.0,
            "friction": 0.0,
        }

    wins = trades.loc[trades["pnl"] > 0]
    losses = trades.loc[trades["pnl"] < 0]
    gross_profit = wins["pnl"].sum()
    gross_loss = abs(losses["pnl"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "return": total_return,
        "trades": int(len(trades)),
        "win_rate": float((trades["pnl"] > 0).mean() * 100.0),
        "profit_factor": float(profit_factor),
        "avg_trade": float(trades["return_percent"].mean()),
        "friction": float(trades["commission"].sum() + trades["slippage_cost"].sum()),
    }


def evaluate_threshold(threshold, predictions, frames):
    candidates = build_trade_candidates(predictions, frames, threshold)
    trades, final_value = run_portfolio(candidates, COMMISSION_RATE, SLIPPAGE_RATE)
    gross_trades, gross_final = run_portfolio(candidates, 0.0, 0.0)

    cost = metrics_from_trades(trades, final_value)
    gross = metrics_from_trades(gross_trades, gross_final)
    return {
        "threshold": threshold,
        "cost": cost,
        "gross": gross,
    }


def choose_threshold(results):
    eligible = [item for item in results if item["cost"]["trades"] >= MIN_TRADES_FOR_SELECTION]
    if not eligible:
        eligible = results

    # Selection uses 2025 only. Prefer realized return after costs; PF breaks ties.
    return max(
        eligible,
        key=lambda item: (item["cost"]["return"], item["cost"]["profit_factor"]),
    )


def main():
    print("=" * 92)
    print("LR1 - 2025 EXECUTION THRESHOLD TUNING")
    print("=" * 92)
    print("2026 is NOT loaded or evaluated by this script.")
    print(f"Hold: {HOLD_BARS * 5} minutes | Costs: {COMMISSION_RATE:.3%} commission/side + {SLIPPAGE_RATE:.3%} slippage/side")
    print(f"Thresholds: {', '.join(f'{value:.2f}' for value in THRESHOLDS)}")

    predictions = load_validation_predictions()
    frames = load_execution_data()

    results = []
    for threshold in THRESHOLDS:
        result = evaluate_threshold(threshold, predictions, frames)
        results.append(result)
        cost = result["cost"]
        gross = result["gross"]
        print(
            f"p>={threshold:.2f} | cost={cost['return']:+7.2f}% PF={cost['profit_factor']:.2f} "
            f"trades={cost['trades']:4d} win={cost['win_rate']:5.1f}% avg={cost['avg_trade']:+.3f}% "
            f"| gross={gross['return']:+7.2f}% PF={gross['profit_factor']:.2f}"
        )

    selected = choose_threshold(results)

    print("\n" + "=" * 92)
    print("FROZEN LR1 EXECUTION SETUP")
    print("=" * 92)
    print(f"Probability threshold: {selected['threshold']:.2f}")
    print(f"Hold time:             {HOLD_BARS * 5} minutes")
    print(f"2025 return w/costs:   {selected['cost']['return']:+.2f}%")
    print(f"2025 profit factor:    {selected['cost']['profit_factor']:.2f}")
    print(f"2025 trades:           {selected['cost']['trades']}")
    print("2026 remains untouched.")

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "LR1",
        "selection_period": "2025",
        "final_test_period": "2026",
        "probability_threshold": selected["threshold"],
        "hold_bars": HOLD_BARS,
        "hold_minutes": HOLD_BARS * 5,
        "commission_rate": COMMISSION_RATE,
        "slippage_rate": SLIPPAGE_RATE,
        "starting_cash": STARTING_CASH,
        "max_open_positions": MAX_OPEN_POSITIONS,
        "max_position_percent": MAX_POSITION_PERCENT,
        "selection_rule": "highest 2025 return after costs among thresholds with >=30 trades; PF tie-break",
        "all_results": results,
        "selected_result": selected,
    }
    CONFIG_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved execution config: {CONFIG_PATH}")
    print("\nNext: run the frozen setup once on untouched 2026.")


if __name__ == "__main__":
    main()
