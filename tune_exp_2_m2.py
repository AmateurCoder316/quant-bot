from datetime import datetime, timezone

import joblib

from exp2_m2_config import (
    EXECUTION_PATH,
    MIN_TRADES_FOR_DEPLOYMENT,
    MODEL_NAME,
    MODEL_PATH,
    TUNE_YEAR,
)
from exp2_m2_execution import (
    all_execution_rules,
    evaluate_rule,
    select_execution_rule,
)
from exp2_m2_training import (
    load_events,
    predict_bundle,
    prediction_metrics,
    write_json,
)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing {MODEL_PATH}. Run python train_exp_2_m2.py first.")

    bundle = joblib.load(MODEL_PATH)
    if bundle.get("model_name") != MODEL_NAME:
        raise ValueError("Loaded model is not exp-2-m2.")

    events = load_events()
    tune = events.loc[events["year"] == TUNE_YEAR].copy().reset_index(drop=True)
    if tune.empty:
        raise ValueError("No 2025 events available for EXP-2-M2 tuning.")

    predictions = predict_bundle(bundle, tune)
    diagnostics = prediction_metrics(tune, predictions)

    print("=" * 124)
    print("EXP-2-M2 - 2025 EXECUTION TUNING")
    print("=" * 124)
    print("All model families, hyperparameters, finalists and blender are already frozen.")
    print("Hyperparameter search used 2023-2024 only. 2026 is not used for selection.")
    print("Only one execution axis is searched: causal rolling percentile of frozen M2 score.")
    print(f"Minimum completed trades for deployment support: {MIN_TRADES_FOR_DEPLOYMENT}")
    print("Boundary percentiles remain diagnostic only; deployment requires both neighbors.")

    print("\n2025 FROZEN MODEL DIAGNOSTICS")
    print(f"AUC:                    {diagnostics['roc_auc']:.4f}")
    print(f"Average precision:      {diagnostics['average_precision']:.4f}")
    print(f"Economic-positive rate: {diagnostics['positive_rate']*100:.2f}%")
    print(f"Spearman vs net return: {diagnostics['spearman_net_return']:.4f}")
    print(
        f"Top 10% precision:      {diagnostics['top10']['precision']*100:.2f}% | "
        f"avg net={diagnostics['top10']['avg_net_return']*100:+.3f}%"
    )
    print(
        f"Top 5% precision:       {diagnostics['top05']['precision']*100:.2f}% | "
        f"avg net={diagnostics['top05']['avg_net_return']*100:+.3f}%"
    )

    print("\nTesting five frozen execution percentiles...\n")
    results = []
    for rule in all_execution_rules():
        result = evaluate_rule(predictions, rule, TUNE_YEAR)
        results.append(result)
        cost = result["cost"]
        gross = result["gross"]
        print(
            f"{rule.label:<20} | candidates={result['candidate_events']:4d} "
            f"cost={cost['return']:+7.2f}% PF={cost['profit_factor']:.2f} "
            f"trades={cost['trades']:4d} avg={cost['avg_trade']:+.3f}% "
            f"DD={cost['max_drawdown']:+.2f}% Sharpe={cost['sharpe']:+.2f} "
            f"posmo={cost['positive_month_fraction']*100:4.0f}% | "
            f"gross={gross['return']:+7.2f}% PF={gross['profit_factor']:.2f}"
        )

    selected, scored = select_execution_rule(results)
    cost = selected["cost"]
    gross = selected["gross"]

    print("\n" + "=" * 124)
    print("FROZEN EXP-2-M2 EXECUTION SETUP")
    print("=" * 124)
    print(f"Selected:                 {selected['label']}")
    print(f"Causal percentile:        {selected['percentile']:.3f}")
    print(f"2025 return w/costs:      {cost['return']:+.2f}%")
    print(f"2025 profit factor:       {cost['profit_factor']:.2f}")
    print(f"2025 average trade:       {cost['avg_trade']:+.3f}%")
    print(f"2025 trades:              {cost['trades']}")
    print(f"2025 max drawdown:        {cost['max_drawdown']:+.2f}%")
    print(f"2025 daily Sharpe:        {cost['sharpe']:+.2f}")
    print(
        f"Positive months:          {cost['positive_months']}/{cost['months']} "
        f"({cost['positive_month_fraction']*100:.1f}%)"
    )
    print(f"Median month:             {cost['median_month']:+.2f}%")
    print(f"Worst month:              {cost['worst_month']:+.2f}%")
    print(f"Gross profit factor:      {gross['profit_factor']:.2f}")
    print(f"Local median return:      {selected['stability_median_return']:+.2f}%")
    print(f"Local WORST return:       {selected['stability_worst_return']:+.2f}%")
    print(f"Local median PF:          {selected['stability_median_pf']:.2f}")
    print(f"Neighbor count:           {selected['neighbor_count']}")
    print(
        "Deployment gate:          "
        + ("PASS" if selected["passes_deployment_gate"] else "FAIL - SHADOW RESEARCH ONLY")
    )

    payload = {
        "model": MODEL_NAME,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_period": str(TUNE_YEAR),
        "test_period": "2026",
        "model_diagnostics_2025": diagnostics,
        "selection_policy": (
            "Five predeclared causal percentiles only. Model/hyperparameters were selected entirely "
            "inside 2023-2024. Deployment requires >=75 trades, positive after-cost return/average "
            "trade/Sharpe, PF>=1.15, >55% positive months, positive median month, gross PF>=1.25, "
            "and a fully positive local three-threshold plateau."
        ),
        "selected": selected,
        "all_results": scored,
    }
    write_json(EXECUTION_PATH, payload)

    print(f"Saved: {EXECUTION_PATH}")
    print("Next: python backtest_exp_2_m2.py")


if __name__ == "__main__":
    main()
