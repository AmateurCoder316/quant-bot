import json
from pathlib import Path

import joblib

from ml_research_core import (
    ExecutionRule,
    ModelSpec,
    classification_metrics,
    evaluate_rule,
    load_execution_frames,
    make_label,
    predict_probabilities,
)
from walk_forward_ml import load_dataset, ordered


REPORT_PATH = Path("data") / "research" / "ml_walk_forward_v2.json"
FOLD_NAME = "WF2"


def load_gb3_config():
    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {REPORT_PATH}. Run python walk_forward_ml.py first."
        )

    report = json.loads(REPORT_PATH.read_text())
    fold = next(
        (item for item in report.get("fold_results", []) if item.get("fold") == FOLD_NAME),
        None,
    )
    if fold is None:
        raise ValueError(f"Could not find {FOLD_NAME} in {REPORT_PATH}.")

    spec_data = fold["selected_spec"]
    rule_data = fold["selected_rule"]

    spec = ModelSpec(
        name=spec_data["name"],
        target_column=spec_data["target_column"],
        positive_threshold=float(spec_data["positive_threshold"]),
        horizon_minutes=int(spec_data["horizon_minutes"]),
        hold_bars=int(spec_data["hold_bars"]),
    )
    rule = ExecutionRule(
        mode=rule_data["mode"],
        value=float(rule_data["value"]),
    )

    model_path = Path(fold["selected_model_path"])
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing frozen model {model_path}. Run python walk_forward_ml.py first."
        )

    return fold, spec, rule, model_path


def print_metrics(title, metrics):
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)
    print(f"Total return:        {metrics['return']:+.2f}%")
    print(f"Completed trades:    {metrics['trades']}")
    print(f"Win rate:            {metrics['win_rate']:.2f}%")
    print(f"Profit factor:       {metrics['profit_factor']:.2f}")
    print(f"Average trade:       {metrics['avg_trade']:+.3f}%")
    print(f"Max drawdown:        {metrics['max_drawdown']:.2f}%")
    print(f"Daily Sharpe:        {metrics['sharpe']:.2f}")
    print(f"Modeled friction:    ${metrics['friction']:,.2f}")


def main():
    fold, spec, rule, model_path = load_gb3_config()

    train_years = fold["train_years"]
    tune_year = int(fold["tune_year"])
    test_year = int(fold["test_year"])
    passes_gate = bool(fold["passes_gate"])

    print("=" * 96)
    print("GB3 - STANDALONE FROZEN HISTORICAL BACKTEST")
    print("=" * 96)
    print("Simulation only. No real orders are placed.")
    print(f"Model train years:     {train_years}")
    print(f"Execution tuned on:    {tune_year}")
    print(f"Frozen test year:      {test_year}")
    print(f"Target:                {spec.name} ({spec.target_column} >= {spec.positive_threshold * 100:+.2f}%)")
    print(f"Hold time:             {spec.horizon_minutes} minutes")
    print(f"Frozen execution rule: {rule.label}")
    print(f"Deployment gate:       {'PASS' if passes_gate else 'FAIL - SHADOW RESEARCH ONLY'}")

    dataset = load_dataset()
    test = ordered(dataset.loc[dataset["year"] == test_year].copy())
    if test.empty:
        raise ValueError(f"Dataset has no rows for {test_year}.")

    model = joblib.load(model_path)
    test_predictions = predict_probabilities(model, test)
    test_frames = load_execution_frames(test_year)

    seed_predictions = None
    if rule.mode == "adaptive":
        tune = ordered(dataset.loc[dataset["year"] == tune_year].copy())
        seed_predictions = predict_probabilities(model, tune)

    result = evaluate_rule(
        rule,
        test_predictions,
        test_frames,
        spec.hold_bars,
        seed_predictions=seed_predictions,
    )

    y_test = make_label(test, spec)
    cls = classification_metrics(y_test, test_predictions["probability"])
    cost = result["cost"]
    gross = result["gross"]

    print("\nSIGNAL DIAGNOSTICS")
    print(f"Prediction rows:       {len(test_predictions):,}")
    print(f"Maximum probability:   {test_predictions['probability'].max():.4f}")
    print(f"99th percentile:       {test_predictions['probability'].quantile(0.99):.4f}")
    if rule.mode == "fixed":
        qualified = int((test_predictions["probability"] >= rule.value).sum())
        print(f"Rows >= {rule.value:.3f}:       {qualified:,}")
    print(f"Test ROC AUC:          {cls['roc_auc']:.4f}")
    print(f"Test avg precision:    {cls['average_precision']:.4f}")

    print_metrics(f"{test_year} CONFIGURED COST MODEL", cost)
    print_metrics(f"{test_year} FRICTIONLESS DIAGNOSTIC", gross)

    print("\n" + "=" * 96)
    print("GB3 BACKTEST SUMMARY")
    print("=" * 96)
    print(f"Return with costs:     {cost['return']:+.2f}%")
    print(f"Return without costs:  {gross['return']:+.2f}%")
    print(f"Cost-model drag:       {cost['return'] - gross['return']:+.2f} percentage points")
    print(f"Trades:                {cost['trades']}")
    print(f"Profit factor costs:   {cost['profit_factor']:.2f}")
    print(f"Profit factor gross:   {gross['profit_factor']:.2f}")
    print(f"Max drawdown:          {cost['max_drawdown']:.2f}%")
    print(f"Daily Sharpe:          {cost['sharpe']:.2f}")

    if not passes_gate:
        print("\nNOTE: GB3 failed the tuning-year deployment gate, so this is a shadow backtest only.")
        print("The research policy would have placed no trades in this period.")


if __name__ == "__main__":
    main()
