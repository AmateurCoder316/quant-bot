import json
from datetime import datetime, timezone

import joblib
import numpy as np

from exp2_config import (
    EXECUTION_PATH,
    MIN_TRADES_FOR_DEPLOYMENT,
    MODEL_NAME,
    MODEL_PATH,
    OUTPUT_DIR,
    TUNE_YEAR,
)
from exp2_core import (
    all_execution_rules,
    evaluate_rule,
    load_event_dataset,
    predict_stack,
    select_execution_rule,
    stack_metrics,
)


def json_clean(value):
    if isinstance(value, dict):
        return {key: json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing {MODEL_PATH}. Run python train_exp_2_m1.py first.")

    print("=" * 124)
    print("EXP-2-M1 - 2025 EXECUTION TUNING")
    print("=" * 124)
    print("Model architecture, labels, barriers and meta-model are frozen.")
    print("2026 is not used for selection.")
    print("Only one execution axis is searched: causal rolling percentile of meta-score.")
    print(f"Minimum completed trades for deployment support: {MIN_TRADES_FOR_DEPLOYMENT}")
    print("Boundary percentiles are diagnostic only; deployment requires both neighbors.")

    bundle = joblib.load(MODEL_PATH)
    events = load_event_dataset()
    tune = events.loc[events["year"] == TUNE_YEAR].copy().reset_index(drop=True)
    if tune.empty:
        raise ValueError("No EXP-2 tuning events for 2025.")

    predictions = predict_stack(bundle, tune)
    metrics = stack_metrics(tune, predictions)

    print("\n2025 MODEL DIAGNOSTICS")
    meta = metrics["meta_economic"]
    print(
        f"META economic: AUC={meta['roc_auc']:.4f} AP={meta['average_precision']:.4f} "
        f"logloss={meta['log_loss']:.4f} brier={meta['brier']:.4f} "
        f"positive={meta['positive_rate']*100:.2f}%"
    )
    for name in ("barrier_hgb", "barrier_et", "economic_hgb", "economic_lr"):
        item = metrics[name]
        print(
            f"{name:<14}: AUC={item['roc_auc']:.4f} AP={item['average_precision']:.4f} "
            f"positive={item['positive_rate']*100:.2f}%"
        )

    print("\nTesting frozen percentile family...\n")
    results = []
    for rule in all_execution_rules():
        result = evaluate_rule(predictions, rule, TUNE_YEAR)
        results.append(result)
        cost = result["cost"]
        gross = result["gross"]
        print(
            f"{rule.label:<18} | candidates={result['candidate_events']:4d} "
            f"cost={cost['return']:+7.2f}% PF={cost['profit_factor']:.2f} "
            f"trades={cost['trades']:4d} avg={cost['avg_trade']:+.3f}% "
            f"DD={cost['max_drawdown']:+.2f}% Sharpe={cost['sharpe']:+.2f} | "
            f"gross={gross['return']:+7.2f}% PF={gross['profit_factor']:.2f}"
        )

    selected, scored = select_execution_rule(results)

    print("\n" + "=" * 124)
    print("FROZEN EXP-2-M1 EXECUTION SETUP")
    print("=" * 124)
    print(f"Selected:                 {selected['label']}")
    print(f"Causal percentile:        {selected['percentile']:.3f}")
    print(f"2025 return w/costs:      {selected['cost']['return']:+.2f}%")
    print(f"2025 profit factor:       {selected['cost']['profit_factor']:.2f}")
    print(f"2025 average trade:       {selected['cost']['avg_trade']:+.3f}%")
    print(f"2025 trades:              {selected['cost']['trades']}")
    print(f"2025 max drawdown:        {selected['cost']['max_drawdown']:+.2f}%")
    print(f"2025 daily Sharpe:        {selected['cost']['sharpe']:+.2f}")
    print(f"Gross profit factor:      {selected['gross']['profit_factor']:.2f}")
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
        "selection_policy": (
            "Choose among five predeclared causal meta-score percentiles. A deployable interior "
            "rule requires >=75 completed trades, positive after-cost return/average trade/Sharpe, "
            "PF>=1.15, gross PF>=1.25, and a fully positive local three-threshold plateau."
        ),
        "selected": selected,
        "all_results": scored,
        "model_metrics_2025": metrics,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EXECUTION_PATH.write_text(json.dumps(json_clean(payload), indent=2) + "\n")

    print(f"Saved: {EXECUTION_PATH}")
    print("Next: python backtest_exp_2_m1.py")


if __name__ == "__main__":
    main()
