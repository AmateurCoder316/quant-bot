import json

import joblib

from exp2_config import EXECUTION_PATH, MODEL_NAME, MODEL_PATH, TEST_YEAR, TUNE_YEAR
from exp2_core import (
    execution_rule_from_result,
    load_event_dataset,
    predict_stack,
    selected_events,
    stack_metrics,
)
from exp2_execution import evaluate_rule


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
    print(f"Modeled friction:    ${metrics['friction']:,.2f}")
    print(
        f"Positive months:     {metrics['positive_months']}/{metrics['months']} "
        f"({metrics['positive_month_fraction']*100:.1f}%)"
    )
    print(f"Median month:        {metrics['median_month']:+.2f}%")
    print(f"Worst month:         {metrics['worst_month']:+.2f}%")


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing {MODEL_PATH}. Run python train_exp_2_m1.py first.")
    if not EXECUTION_PATH.exists():
        raise FileNotFoundError(f"Missing {EXECUTION_PATH}. Run python tune_exp_2_m1.py first.")

    execution = json.loads(EXECUTION_PATH.read_text())
    if execution.get("model") != MODEL_NAME:
        raise ValueError("Frozen execution file is not for exp-2-m1.")
    if execution.get("selection_period") != str(TUNE_YEAR):
        raise ValueError("EXP-2-M1 execution must be selected on 2025 only.")

    selected = execution["selected"]
    rule = execution_rule_from_result(selected)
    bundle = joblib.load(MODEL_PATH)
    events = load_event_dataset()

    tune = events.loc[events["year"] == TUNE_YEAR].copy().reset_index(drop=True)
    test = events.loc[events["year"] == TEST_YEAR].copy().reset_index(drop=True)
    if tune.empty or test.empty:
        raise ValueError("Missing EXP-2 2025 seed events or 2026 test events.")

    # 2025 scores seed the causal 2026 rolling percentile. They are prior history,
    # not 2026 labels, and no 2026 outcome participates in threshold construction.
    tune_predictions = predict_stack(bundle, tune)
    test_predictions = predict_stack(bundle, test)
    diagnostics = stack_metrics(test, test_predictions)
    chosen = selected_events(test_predictions, rule, seed_predictions=tune_predictions)

    print("=" * 112)
    print("EXP-2-M1 - FROZEN 2026 HISTORICAL BACKTEST")
    print("=" * 112)
    print("Simulation only. No real orders are placed.")
    print("Architecture: stacked event-driven meta-label model")
    print("Train:        2023-2024")
    print("Meta OOF:     purged monthly 2024")
    print("Tune:         2025")
    print(f"Frozen test:  {TEST_YEAR}")
    print(f"Rule:         {rule.label}")
    print(
        "Deployment gate from 2025: "
        + ("PASS" if selected.get("passes_deployment_gate") else "FAIL - SHADOW RESEARCH ONLY")
    )
    print("2026 was already observed by older research trees, so this is not globally pristine.")

    print("\n2026 MODEL / SIGNAL DIAGNOSTICS")
    meta = diagnostics["meta_economic"]
    print(
        f"META economic: AUC={meta['roc_auc']:.4f} AP={meta['average_precision']:.4f} "
        f"logloss={meta['log_loss']:.4f} brier={meta['brier']:.4f} "
        f"positive={meta['positive_rate']*100:.2f}%"
    )
    for name in ("barrier_hgb", "barrier_et", "economic_hgb", "economic_lr"):
        item = diagnostics[name]
        print(
            f"{name:<14}: AUC={item['roc_auc']:.4f} AP={item['average_precision']:.4f} "
            f"positive={item['positive_rate']*100:.2f}%"
        )
    print(f"Event predictions:        {len(test_predictions):,}")
    print(f"Events passing meta rule: {len(chosen):,}")
    print(f"Max meta probability:     {test_predictions['meta_probability'].max():.4f}")
    print(f"Max meta score:           {test_predictions['meta_score'].max():.4f}")

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
    print("EXP-2-M1 BACKTEST SUMMARY")
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
