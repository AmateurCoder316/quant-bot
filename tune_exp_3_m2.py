from __future__ import annotations

import json

import pandas as pd

from exp3_m2_config import (
    DATASET_PATH,
    EXECUTION_PATH,
    EXECUTION_PERCENTILES,
    METADATA_PATH,
    MIN_TRADES_FOR_DEPLOYMENT,
    MODEL_PATH,
    OUTPUT_DIR,
    SCALER_PATH,
    TARGET_2H,
    TUNE_YEAR,
)
from exp3_m2_execution import deployment_gate, simulate_execution
from exp3_m2_model import load_model_bundle, predict_dataframe, tail_metrics


def fmt_pct(value: float) -> str:
    return f"{value * 100:+.2f}%"


def main() -> None:
    required = [DATASET_PATH, MODEL_PATH, SCALER_PATH, METADATA_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(
            f"Missing frozen EXP-3-M2 artifacts: {missing}. Run train_exp_3_m2.py first."
        )

    model, feature_scaler, target_scaler, metadata, device = load_model_bundle(
        MODEL_PATH, SCALER_PATH, METADATA_PATH
    )
    data = pd.read_parquet(DATASET_PATH).sort_index()
    forward = data[data["year"] == TUNE_YEAR].copy()

    prediction_path = OUTPUT_DIR / "predictions_2025.parquet"
    if prediction_path.exists():
        predictions = pd.read_parquet(prediction_path).sort_index()
    else:
        predictions = predict_dataframe(model, feature_scaler, target_scaler, forward, device)
        predictions.to_parquet(prediction_path)

    diagnostics = tail_metrics(
        predictions["pred_net_2h"].to_numpy(),
        predictions[TARGET_2H].to_numpy(),
    )

    print("=" * 126)
    print("EXP-3-M2 - 2025 EXECUTION TUNING")
    print("=" * 126)
    print("Large residual neural network, loss, architecture and hyperparameters are frozen.")
    print("Model search used 2023-2024 only. 2026 is not used for selection.")
    print("Only one execution axis is searched: causal rolling percentile of frozen M2 score.")
    print(f"Minimum completed trades for deployment support: {MIN_TRADES_FOR_DEPLOYMENT}")
    print("Boundary percentiles remain diagnostic only; deployment requires both neighbors.")

    print("\n2025 FROZEN MODEL DIAGNOSTICS")
    print(f"Score-vs-return rank:    {diagnostics['spearman']:+.4f}")
    print(f"Mean absolute error:     {diagnostics['mae'] * 100:.3f}%")
    print(
        f"Top 10%: avg net={diagnostics['top10_mean'] * 100:+.3f}% "
        f"median={diagnostics['top10_median'] * 100:+.3f}% "
        f"positive={diagnostics['top10_positive'] * 100:.1f}%"
    )
    print(
        f"Top  5%: avg net={diagnostics['top05_mean'] * 100:+.3f}% "
        f"median={diagnostics['top05_median'] * 100:+.3f}% "
        f"positive={diagnostics['top05_positive'] * 100:.1f}%"
    )

    print("\nTesting five frozen causal score percentiles...\n")
    results = []
    for percentile in EXECUTION_PERCENTILES:
        cost_result, _, _ = simulate_execution(predictions, percentile, friction=True)
        gross_result, _, _ = simulate_execution(predictions, percentile, friction=False)
        row = dict(cost_result)
        row["gross"] = gross_result
        results.append(row)

        print(
            f"top {(1.0 - percentile) * 100:>4.1f}% | candidates={cost_result['candidate_signals']:>5} "
            f"cost={fmt_pct(cost_result['return']):>8} PF={cost_result['profit_factor']:.2f} "
            f"trades={cost_result['trades']:>4} avg={fmt_pct(cost_result['average_trade']):>8} "
            f"DD={fmt_pct(cost_result['max_drawdown']):>8} Sharpe={cost_result['daily_sharpe']:+.2f} "
            f"posmo={cost_result['positive_month_fraction'] * 100:4.0f}% | "
            f"gross={fmt_pct(gross_result['return']):>8} PF={gross_result['profit_factor']:.2f}"
        )

    supported = [row for row in results if int(row["trades"]) >= MIN_TRADES_FOR_DEPLOYMENT]
    selection_pool = supported if supported else results
    selected = max(selection_pool, key=lambda row: float(row["return"]))
    selected_index = next(
        index
        for index, row in enumerate(results)
        if float(row["percentile"]) == float(selected["percentile"])
    )
    neighbors = []
    if selected_index > 0:
        neighbors.append(results[selected_index - 1])
    if selected_index < len(results) - 1:
        neighbors.append(results[selected_index + 1])

    gate_pass, failures, local_summary = deployment_gate(selected, neighbors)
    if float(selected["gross"]["profit_factor"]) < 1.20:
        failures.append("gross profit factor < 1.20")
        gate_pass = False
    if not bool(metadata.get("training_gate_pass", False)):
        failures.append("2023-2024 neural-network economic training gate failed")
        gate_pass = False

    payload = {
        "percentile": float(selected["percentile"]),
        "selected": selected,
        "neighbors": neighbors,
        "local_summary": local_summary,
        "deployment_gate_pass": bool(gate_pass),
        "deployment_failures": failures,
        "all_results": results,
        "model_diagnostics_2025": diagnostics,
    }
    EXECUTION_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 126)
    print("FROZEN EXP-3-M2 EXECUTION SETUP")
    print("=" * 126)
    print(f"Selected causal percentile: p{selected['percentile'] * 100:.1f}")
    print(f"Approx selected tail:       top {(1.0 - selected['percentile']) * 100:.1f}%")
    print(f"2025 return w/costs:        {fmt_pct(selected['return'])}")
    print(f"2025 profit factor:         {selected['profit_factor']:.2f}")
    print(f"2025 average trade:         {fmt_pct(selected['average_trade'])}")
    print(f"2025 trades:                {selected['trades']}")
    print(f"2025 max drawdown:          {fmt_pct(selected['max_drawdown'])}")
    print(f"2025 daily Sharpe:          {selected['daily_sharpe']:+.2f}")
    print(
        f"Positive months:            {selected['positive_months']}/{selected['month_count']} "
        f"({selected['positive_month_fraction'] * 100:.1f}%)"
    )
    print(f"Median month:               {fmt_pct(selected['median_month'])}")
    print(f"Worst month:                {fmt_pct(selected['worst_month'])}")
    print(f"Gross profit factor:        {selected['gross']['profit_factor']:.2f}")
    print(f"Local median return:        {fmt_pct(local_summary['median_return'])}")
    print(f"Local WORST return:         {fmt_pct(local_summary['worst_return'])}")
    print(f"Local median PF:            {local_summary['median_profit_factor']:.2f}")
    print(f"Neighbor count:             {len(neighbors)}")
    if gate_pass:
        print("Deployment gate:            PASS - PAPER RESEARCH ELIGIBLE")
    else:
        print("Deployment gate:            FAIL - SHADOW RESEARCH ONLY")
        for failure in failures:
            print(f"  - {failure}")
    print(f"Saved: {EXECUTION_PATH}")
    print("Next: python backtest_exp_3_m2.py")


if __name__ == "__main__":
    main()
