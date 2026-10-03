import json
from pathlib import Path

import joblib
import pandas as pd

from features import FEATURE_COLUMNS
from market_data import LOCAL_PROVIDER
from strategy_core import COMMISSION_RATE, SLIPPAGE_RATE, STARTING_CASH, SYMBOLS
from tune_lr1 import build_trade_candidates, metrics_from_trades, run_portfolio


DATASET_PATH = Path("data") / "ml" / "features.parquet"
MODEL_PATH = Path("data") / "models" / "lr1.joblib"
CONFIG_PATH = Path("data") / "models" / "lr1_execution.json"
MODEL_INPUT_COLUMNS = list(FEATURE_COLUMNS) + ["symbol"]
TEST_YEAR = 2026


def load_frozen_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CONFIG_PATH}. Run python tune_lr1.py first so the 2025 setup is frozen."
        )

    config = json.loads(CONFIG_PATH.read_text())
    if config.get("model") != "LR1":
        raise ValueError("Execution config is not for LR1.")
    if config.get("selection_period") != "2025":
        raise ValueError("LR1 execution config must have been selected on 2025 only.")

    return config


def load_test_predictions():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing {DATASET_PATH}. Run python build_dataset.py first.")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing {MODEL_PATH}. Run python train_lr1.py first.")

    dataset = pd.read_parquet(
        DATASET_PATH,
        columns=["timestamp", "symbol", "split", *FEATURE_COLUMNS],
    )
    test = dataset.loc[dataset["split"] == "test"].copy()
    test["timestamp"] = pd.to_datetime(test["timestamp"], utc=True)

    if test.empty:
        raise ValueError("2026 test split is empty.")

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


def print_metrics(title, metrics):
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)
    print(f"Starting cash:       ${STARTING_CASH:,.2f}")
    print(f"Total return:        {metrics['return']:+.2f}%")
    print(f"Completed trades:    {metrics['trades']}")
    print(f"Win rate:            {metrics['win_rate']:.2f}%")
    print(f"Profit factor:       {metrics['profit_factor']:.2f}")
    print(f"Average trade:       {metrics['avg_trade']:+.3f}%")
    print(f"Modeled friction:    ${metrics['friction']:,.2f}")


def print_symbol_breakdown(trades):
    if trades.empty:
        return

    print("\nBY SYMBOL")
    for symbol, group in trades.groupby("symbol"):
        wins = group.loc[group["pnl"] > 0, "pnl"].sum()
        losses = abs(group.loc[group["pnl"] < 0, "pnl"].sum())
        pf = wins / losses if losses > 0 else float("inf")
        print(
            f"{symbol:<6} trades={len(group):>4} "
            f"P/L=${group['pnl'].sum():+9.2f} "
            f"win={((group['pnl'] > 0).mean() * 100):5.1f}% "
            f"PF={pf:.2f}"
        )


def main():
    config = load_frozen_config()
    threshold = float(config["probability_threshold"])
    hold_bars = int(config["hold_bars"])

    from tune_lr1 import HOLD_BARS
    if hold_bars != HOLD_BARS:
        raise ValueError(
            f"Frozen config says hold_bars={hold_bars}, but tuner code says {HOLD_BARS}."
        )

    print("=" * 92)
    print("LR1 - FINAL UNSEEN 2026 BACKTEST")
    print("=" * 92)
    print("This run uses the LR1 model trained on 2023-2024 and the execution threshold selected on 2025.")
    print("No 2026 parameter tuning is performed.")
    print(f"Frozen probability threshold: {threshold:.2f}")
    print(f"Frozen hold time:             {hold_bars * 5} minutes")
    print(
        f"Costs:                        {COMMISSION_RATE:.3%} commission/side + "
        f"{SLIPPAGE_RATE:.3%} slippage/side"
    )

    predictions = load_test_predictions()
    frames = load_execution_data()

    above_threshold = predictions.loc[predictions["probability"] >= threshold]
    print("\n2026 PREDICTION DIAGNOSTICS")
    print(f"Prediction rows:              {len(predictions):,}")
    print(f"Maximum probability:          {predictions['probability'].max():.4f}")
    print(f"Rows >= frozen threshold:     {len(above_threshold):,}")
    print(f"99th percentile probability:  {predictions['probability'].quantile(0.99):.4f}")

    candidates = build_trade_candidates(predictions, frames, threshold)
    print(f"Executable trade candidates:  {len(candidates):,}")

    trades, final_value = run_portfolio(candidates, COMMISSION_RATE, SLIPPAGE_RATE)
    gross_trades, gross_final_value = run_portfolio(candidates, 0.0, 0.0)

    cost_metrics = metrics_from_trades(trades, final_value)
    gross_metrics = metrics_from_trades(gross_trades, gross_final_value)

    print_metrics("2026 CONFIGURED COST MODEL", cost_metrics)
    print_symbol_breakdown(trades)
    print_metrics("2026 FRICTIONLESS DIAGNOSTIC", gross_metrics)

    print("\n" + "=" * 92)
    print("LR1 FINAL TEST SUMMARY")
    print("=" * 92)
    print(f"Return with costs:    {cost_metrics['return']:+.2f}%")
    print(f"Return without costs: {gross_metrics['return']:+.2f}%")
    print(f"Cost-model drag:      {cost_metrics['return'] - gross_metrics['return']:+.2f} percentage points")
    print(f"Trades:               {cost_metrics['trades']}")
    print(f"Profit factor costs:  {cost_metrics['profit_factor']:.2f}")
    print(f"Profit factor gross:  {gross_metrics['profit_factor']:.2f}")
    print("\n2026 has now been consumed as LR1's final unseen test. Do not tune LR1 against this result.")


if __name__ == "__main__":
    main()
