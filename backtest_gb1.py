import json
from pathlib import Path

import joblib
import pandas as pd

from features import FEATURE_COLUMNS
from market_data import LOCAL_PROVIDER
from strategy_core import COMMISSION_RATE, SLIPPAGE_RATE, STARTING_CASH, SYMBOLS
from train_gb1 import add_symbol_columns, MODEL_INPUT_COLUMNS
from tune_lr1 import build_trade_candidates as build_fixed_candidates
from tune_lr1 import metrics_from_trades, run_portfolio
from tune_lr2 import add_adaptive_threshold, build_trade_candidates as build_adaptive_candidates


DATASET_PATH = Path("data") / "ml" / "features.parquet"
MODEL_PATH = Path("data") / "models" / "gb1.joblib"
CONFIG_PATH = Path("data") / "models" / "gb1_execution.json"
TEST_YEAR = 2026


def load_frozen_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing {CONFIG_PATH}. Run python tune_gb1.py first.")

    config = json.loads(CONFIG_PATH.read_text())
    if config.get("model") != "GB1":
        raise ValueError("Execution config is not for GB1.")
    if config.get("selection_period") != "2025":
        raise ValueError("GB1 execution config must have been selected on 2025 only.")
    return config


def load_test_predictions():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing {DATASET_PATH}. Run python build_dataset.py first.")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing {MODEL_PATH}. Run python train_gb1.py first.")

    dataset = pd.read_parquet(
        DATASET_PATH,
        columns=["timestamp", "symbol", "split", *FEATURE_COLUMNS],
    )
    test = dataset.loc[dataset["split"] == "test"].copy()
    test["timestamp"] = pd.to_datetime(test["timestamp"], utc=True)

    if test.empty:
        raise ValueError("2026 test split is empty.")

    test = add_symbol_columns(test)
    model = joblib.load(MODEL_PATH)
    test["probability"] = model.predict_proba(test[MODEL_INPUT_COLUMNS])[:, 1]
    return test[["timestamp", "symbol", "probability"]]


def load_execution_data():
    frames = {}
    for symbol in SYMBOLS:
        frame = LOCAL_PROVIDER.history(symbol).copy()
        frames[symbol] = frame.loc[frame.index.year == TEST_YEAR]
        if frames[symbol].empty:
            raise ValueError(f"No {TEST_YEAR} execution data for {symbol}.")
    return frames


def build_candidates(predictions, frames, config):
    mode = config["selected_mode"]
    value = float(config["selected_value"])

    if mode == "fixed":
        return build_fixed_candidates(predictions, frames, value)

    if mode == "adaptive":
        ordered = predictions.sort_values(["symbol", "timestamp"])
        adaptive = add_adaptive_threshold(ordered, value)
        return build_adaptive_candidates(adaptive, frames)

    raise ValueError(f"Unknown GB1 execution mode: {mode}")


def print_metrics(title, metrics, final_value):
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)
    print(f"Starting cash:       ${STARTING_CASH:,.2f}")
    print(f"Final value:         ${final_value:,.2f}")
    print(f"Total return:        {metrics['return']:+.2f}%")
    print(f"Completed trades:    {metrics['trades']}")
    print(f"Win rate:            {metrics['win_rate']:.2f}%")
    print(f"Profit factor:       {metrics['profit_factor']:.2f}")
    print(f"Average trade:       {metrics['avg_trade']:+.3f}%")
    print(f"Modeled friction:    ${metrics['friction']:,.2f}")


def main():
    config = load_frozen_config()
    hold_bars = int(config["hold_bars"])

    print("=" * 96)
    print("GB1 - 2026 HISTORICAL BACKTEST")
    print("=" * 96)
    print("Simulation only. No real orders are placed.")
    print("GB1 trained on 2023-2024; execution rule selected on 2025.")
    print(f"Frozen hold time: {hold_bars * 5} minutes")

    if config["selected_mode"] == "fixed":
        print(f"Frozen execution rule: probability >= {float(config['selected_value']):.2f}")
    else:
        print(f"Frozen execution rule: top {(1-float(config['selected_value']))*100:.1f}% of recent per-symbol scores")

    predictions = load_test_predictions()
    frames = load_execution_data()
    candidates = build_candidates(predictions, frames, config)

    print("\n2026 SIGNAL DIAGNOSTICS")
    print(f"Prediction rows:         {len(predictions):,}")
    print(f"Maximum probability:     {predictions['probability'].max():.4f}")
    print(f"99th percentile:         {predictions['probability'].quantile(0.99):.4f}")
    print(f"Executable candidates:   {len(candidates):,}")

    trades, final_value = run_portfolio(candidates, COMMISSION_RATE, SLIPPAGE_RATE)
    gross_trades, gross_final = run_portfolio(candidates, 0.0, 0.0)

    cost_metrics = metrics_from_trades(trades, final_value)
    gross_metrics = metrics_from_trades(gross_trades, gross_final)

    print_metrics("2026 CONFIGURED COST MODEL", cost_metrics, final_value)
    print_metrics("2026 FRICTIONLESS DIAGNOSTIC", gross_metrics, gross_final)

    print("\n" + "=" * 96)
    print("GB1 BACKTEST SUMMARY")
    print("=" * 96)
    print(f"Return with costs:    {cost_metrics['return']:+.2f}%")
    print(f"Return without costs: {gross_metrics['return']:+.2f}%")
    print(f"Cost-model drag:      {cost_metrics['return'] - gross_metrics['return']:+.2f} percentage points")
    print(f"Trades:               {cost_metrics['trades']}")
    print(f"Profit factor costs:  {cost_metrics['profit_factor']:.2f}")
    print(f"Profit factor gross:  {gross_metrics['profit_factor']:.2f}")


if __name__ == "__main__":
    main()
