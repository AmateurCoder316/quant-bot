import json

import joblib

from exp2_m2_config import (
    EXECUTION_PATH,
    MODEL_NAME,
    MODEL_PATH,
    TEST_YEAR,
    TUNE_YEAR,
)
from exp2_m2_execution import (
    evaluate_rule,
    execution_rule_from_result,
    selected_events,
)
from exp2_m2_training import (
    load_events,
    predict_bundle,
    prediction_metrics,
)


def print_metrics(title, metrics):
    print("\n" + "=" * 112)
    print(title)
    print("=" * 112)
    print(f"Total return:        {metrics['return']:+.2f}%")
    print(f"Completed trades:    {metrics['trades']}")
    print(f"Win rate:            {metrics['win_rate']:.2f}%")
    print(f"Profit factor:       {metrics['profit_factor']:.2f}")
    print(f"Average trade:       {metrics['avg_trade']:+.3f}%")
    print(f"Max drawdown:        {metrics['max_drawdown']:+.2f}%")
    print(f"Daily Sharpe:        {metrics['sharpe']:+.2f}")
    print(f"Positive months:     {metrics['positive_months']}/{metrics['months']} ({metrics['positive_month_fraction']*100:.1f}%)")
    print(f"Median month:        {metrics['median_month']:+.2f}%")
    print(f"Worst month:         {metrics['worst_month']:+.2f}%")
    print(f"Modeled friction:    ${metrics['friction']:,.2f}")


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing {MODEL_PATH}. Run python train_exp_2_m2.py first.")
    if not EXECUTION_PATH.exists():
        raise FileNotFoundError(f"Missing {EXECUTION_PATH}. Run python tune_exp_2_m2.py first.")

    execution = json.loads(EXECUTION_PATH.read_text())
    if execution.get("model") != MODEL_NAME:
        raise ValueError("Frozen execution file is not for exp-2-m2.")
    if execution.get("selection_period") != str(TUNE_YEAR):
        raise ValueError("EXP-2-M2 execution must be selected on 2025 only.")

    selected = execution["selected"]
    rule = execution_rule_from_result(selected)
    bundle = joblib.load(MODEL_PATH)
    if bundle.get("model_name") != MODEL_NAME:
        raise ValueError("Loaded model bundle is not exp-2-m2.")

    events = load_events()
    tune = events.loc[events["year"] == TUNE_YEAR].copy().reset_index(drop=True)
    test = events.loc[events["year"] == TEST_YEAR].copy().reset_index(drop=True)
    if tune.empty or test.empty:
        raise ValueError("Missing 2025 seed events or 2026 test events for EXP-2-M2.")

    tune_predictions = predict_bundle(bundle, tune)
    test_predictions = predict_bundle(bundle, test)
    diagnostics = prediction_metrics(test, test_predictions)
    chosen = selected_events(test_predictions, rule, seed_predictions=tune_predictions)

    print("=" * 112)
    print("EXP-2-M2 - FROZEN 2026 HISTORICAL BACKTEST")
    print("=" * 112)
    print("Simulation only. No real orders are placed.")
    print("Architecture: purged hyperparameter-optimized heterogeneous ensemble + OOF blender")
    print("Model search: 2023-2024 only")
    print("Execution tune: 2025 only")
    print(f"Frozen test:   {TEST_YEAR}")
    print(f"Rule:          {rule.label}")
    print(
        "Deployment gate from 2025: "
        + ("PASS" if selected.get("passes_deployment_gate") else "FAIL - SHADOW RESEARCH ONLY")
    )
    print("2026 was observed by older research trees, so this is not globally pristine evidence.")

    print("\n2026 FROZEN MODEL DIAGNOSTICS")
    print(f"Event predictions:        {len(test_predictions):,}")
    print(f"Events passing score rule:{len(chosen):,}")
    print(f"Economic-positive rate:   {diagnostics['positive_rate']*100:.2f}%")
    print(f"ROC AUC:                  {diagnostics['roc_auc']:.4f}")
    print(f"Average precision:        {diagnostics['average_precision']:.4f}")
    print(f"Spearman vs net return:   {diagnostics['spearman_net_return']:.4f}")
    print(
        f"Top 10% precision:        {diagnostics['top10']['precision']*100:.2f}% | "
        f"avg net={diagnostics['top10']['avg_net_return']*100:+.3f}%"
    )
    print(
        f"Top 5% precision:         {diagnostics['top05']['precision']*100:.2f}% | "
        f"avg net={diagnostics['top05']['avg_net_return']*100:+.3f}%"
    )
    print(f"Max model score:          {test_predictions['meta_score'].max():.5f}")

    result = evaluate_rule(
        test_predictions,
        rule,
        TEST_YEAR,
        seed_predictions=tune_predictions,
    )
    cost = result["cost"]
    gross = result["gross"]

    print_metrics("2026 CONFIGURED COST MODEL", cost)
    print_metrics("2026 FRICTIONLESS DIAGNOSTIC", gross)

    print("\n" + "=" * 112)
    print("EXP-2-M2 BACKTEST SUMMARY")
    print("=" * 112)
    print(f"Return with costs:     {cost['return']:+.2f}%")
    print(f"Return without costs:  {gross['return']:+.2f}%")
    print(f"Cost-model drag:       {cost['return'] - gross['return']:+.2f} percentage points")
    print(f"Trades:                {cost['trades']}")
    print(f"Profit factor costs:   {cost['profit_factor']:.2f}")
    print(f"Profit factor gross:   {gross['profit_factor']:.2f}")
    print(f"Average trade costs:   {cost['avg_trade']:+.3f}%")
    print(f"Average trade gross:   {gross['avg_trade']:+.3f}%")
    print(f"Max drawdown:          {cost['max_drawdown']:+.2f}%")
    print(f"Daily Sharpe:          {cost['sharpe']:+.2f}")


if __name__ == "__main__":
    main()
