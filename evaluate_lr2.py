import json
from pathlib import Path

import joblib
import pandas as pd

from features import FEATURE_COLUMNS
from market_data import LOCAL_PROVIDER
from strategy_core import COMMISSION_RATE, SLIPPAGE_RATE, STARTING_CASH, SYMBOLS
from tune_lr1 import metrics_from_trades, run_portfolio
from tune_lr2 import add_adaptive_threshold, build_trade_candidates

DATASET_PATH = Path("data") / "ml" / "features.parquet"
MODEL_PATH = Path("data") / "models" / "lr2.joblib"
CONFIG_PATH = Path("data") / "models" / "lr2_execution.json"
MODEL_INPUT_COLUMNS = list(FEATURE_COLUMNS) + ["symbol"]
TEST_YEAR = 2026


def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing {CONFIG_PATH}. Run python tune_lr2.py first.")
    config = json.loads(CONFIG_PATH.read_text())
    if config.get("model") != "LR2":
        raise ValueError("Execution config is not for LR2.")
    return config


def load_predictions():
    dataset = pd.read_parquet(
        DATASET_PATH,
        columns=["timestamp", "symbol", "split", *FEATURE_COLUMNS],
    )
    data = dataset.loc[dataset["split"].isin(["validation", "test"])].copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    model = joblib.load(MODEL_PATH)
    data["probability"] = model.predict_proba(data[MODEL_INPUT_COLUMNS])[:, 1]
    return data[["timestamp", "symbol", "split", "probability"]].sort_values(["symbol", "timestamp"])


def load_execution_data():
    frames = {}
    for symbol in SYMBOLS:
        frame = LOCAL_PROVIDER.history(symbol).copy()
        frames[symbol] = frame.loc[frame.index.year == TEST_YEAR]
    return frames


def main():
    config = load_config()
    percentile = float(config["percentile"])

    print("=" * 92)
    print("LR2 - 2026 HISTORICAL EVALUATION")
    print("=" * 92)
    print("Simulation only. No real orders are placed.")
    print("LR2 trained on 2023-2024; adaptive percentile selected on 2025.")
    print("2026 is no longer pristine because LR1 already examined it, but LR2 was not tuned on 2026.")
    print(f"Adaptive rule: top {(1-percentile)*100:.1f}% of recent per-symbol model scores")

    predictions = load_predictions()
    adaptive = add_adaptive_threshold(predictions, percentile)
    test = adaptive.loc[adaptive["split"] == "test"].copy()
    frames = load_execution_data()
    candidates = build_trade_candidates(test, frames)

    selected = test.loc[
        test["adaptive_threshold"].notna()
        & (test["probability"] >= test["adaptive_threshold"])
    ]

    print("\n2026 SIGNAL DIAGNOSTICS")
    print(f"Prediction rows:         {len(test):,}")
    print(f"Adaptive-qualified rows: {len(selected):,}")
    print(f"Executable candidates:   {len(candidates):,}")
    print(f"Maximum probability:     {test['probability'].max():.4f}")

    trades, final_value = run_portfolio(candidates, COMMISSION_RATE, SLIPPAGE_RATE)
    gross_trades, gross_final = run_portfolio(candidates, 0.0, 0.0)
    cost = metrics_from_trades(trades, final_value)
    gross = metrics_from_trades(gross_trades, gross_final)

    print("\n" + "=" * 92)
    print("2026 CONFIGURED COST MODEL")
    print("=" * 92)
    print(f"Starting cash:       ${STARTING_CASH:,.2f}")
    print(f"Final value:         ${final_value:,.2f}")
    print(f"Total return:        {cost['return']:+.2f}%")
    print(f"Completed trades:    {cost['trades']}")
    print(f"Win rate:            {cost['win_rate']:.2f}%")
    print(f"Profit factor:       {cost['profit_factor']:.2f}")
    print(f"Average trade:       {cost['avg_trade']:+.3f}%")
    print(f"Modeled friction:    ${cost['friction']:,.2f}")

    print("\n" + "=" * 92)
    print("2026 FRICTIONLESS DIAGNOSTIC")
    print("=" * 92)
    print(f"Final value:         ${gross_final:,.2f}")
    print(f"Total return:        {gross['return']:+.2f}%")
    print(f"Completed trades:    {gross['trades']}")
    print(f"Win rate:            {gross['win_rate']:.2f}%")
    print(f"Profit factor:       {gross['profit_factor']:.2f}")
    print(f"Average trade:       {gross['avg_trade']:+.3f}%")


if __name__ == "__main__":
    main()
