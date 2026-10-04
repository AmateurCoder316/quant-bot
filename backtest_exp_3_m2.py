from __future__ import annotations

import json

import pandas as pd

from exp3_m2_config import (
    DATASET_PATH,
    EXECUTION_PATH,
    METADATA_PATH,
    MODEL_PATH,
    OUTPUT_DIR,
    SCALER_PATH,
    TARGET_2H,
    TEST_YEAR,
)
from exp3_m2_execution import simulate_execution
from exp3_m2_model import load_model_bundle, predict_dataframe, tail_metrics


def fmt_pct(value: float) -> str:
    return f"{value * 100:+.2f}%"


def main() -> None:
    required = [DATASET_PATH, MODEL_PATH, SCALER_PATH, METADATA_PATH, EXECUTION_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Missing frozen EXP-3-M2 artifacts: {missing}")

    execution = json.loads(EXECUTION_PATH.read_text(encoding="utf-8"))
    percentile = float(execution["percentile"])
    model, feature_scaler, target_scaler, metadata, device = load_model_bundle(
        MODEL_PATH, SCALER_PATH, METADATA_PATH
    )

    data = pd.read_parquet(DATASET_PATH).sort_index()
    test = data[data["year"] == TEST_YEAR].copy()
    predictions = predict_dataframe(model, feature_scaler, target_scaler, test, device)
    prediction_path = OUTPUT_DIR / "predictions_2026.parquet"
    predictions.to_parquet(prediction_path)

    diagnostics = tail_metrics(
        predictions["pred_net_2h"].to_numpy(),
        predictions[TARGET_2H].to_numpy(),
    )
    cost_result, cost_trades, _ = simulate_execution(predictions, percentile, friction=True)
    gross_result, _, _ = simulate_execution(predictions, percentile, friction=False)

    print("=" * 116)
    print("EXP-3-M2 - FROZEN 2026 HISTORICAL BACKTEST")
    print("=" * 116)
    print("Simulation only. No real orders are placed.")
    print("Architecture: large residual 30-minute neural network with ranking-aware loss")
    print("Model search: 2023-2024 only")
    print("Execution tune: 2025 only")
    print("Frozen test:   2026")
    print(f"Rule:          causal score percentile >= p{percentile * 100:.1f}")
    print(
        "Deployment gate from 2025: "
        + (
            "PASS - PAPER RESEARCH ELIGIBLE"
            if execution["deployment_gate_pass"]
            else "FAIL - SHADOW RESEARCH ONLY"
        )
    )
    print("2026 has been observed by older research, so this remains model-relative frozen evidence.")

    print("\n2026 FROZEN MODEL DIAGNOSTICS")
    print(f"Event predictions:          {len(predictions):,}")
    print(f"Selected candidate signals: {cost_result['candidate_signals']:,}")
    print(f"Score-vs-return rank:       {diagnostics['spearman']:+.4f}")
    print(f"Mean absolute error:        {diagnostics['mae'] * 100:.3f}%")
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
    print(f"Max predicted 2h net:       {predictions['pred_net_2h'].max() * 100:+.3f}%")

    print("\n" + "=" * 116)
    print("2026 CONFIGURED COST MODEL")
    print("=" * 116)
    print(f"Total return:        {fmt_pct(cost_result['return'])}")
    print(f"Completed trades:    {cost_result['trades']}")
    print(f"Win rate:            {cost_result['win_rate'] * 100:.2f}%")
    print(f"Profit factor:       {cost_result['profit_factor']:.2f}")
    print(f"Average trade:       {fmt_pct(cost_result['average_trade'])}")
    print(f"Max drawdown:        {fmt_pct(cost_result['max_drawdown'])}")
    print(f"Daily Sharpe:        {cost_result['daily_sharpe']:+.2f}")
    print(
        f"Positive months:     {cost_result['positive_months']}/{cost_result['month_count']} "
        f"({cost_result['positive_month_fraction'] * 100:.1f}%)"
    )
    print(f"Median month:        {fmt_pct(cost_result['median_month'])}")
    print(f"Worst month:         {fmt_pct(cost_result['worst_month'])}")
    print(f"Modeled friction:    ${cost_result['modeled_friction']:.2f}")

    print("\n" + "=" * 116)
    print("2026 FRICTIONLESS DIAGNOSTIC")
    print("=" * 116)
    print(f"Total return:        {fmt_pct(gross_result['return'])}")
    print(f"Completed trades:    {gross_result['trades']}")
    print(f"Win rate:            {gross_result['win_rate'] * 100:.2f}%")
    print(f"Profit factor:       {gross_result['profit_factor']:.2f}")
    print(f"Average trade:       {fmt_pct(gross_result['average_trade'])}")
    print(f"Max drawdown:        {fmt_pct(gross_result['max_drawdown'])}")
    print(f"Daily Sharpe:        {gross_result['daily_sharpe']:+.2f}")
    print(
        f"Positive months:     {gross_result['positive_months']}/{gross_result['month_count']} "
        f"({gross_result['positive_month_fraction'] * 100:.1f}%)"
    )
    print(f"Median month:        {fmt_pct(gross_result['median_month'])}")
    print(f"Worst month:         {fmt_pct(gross_result['worst_month'])}")

    summary = {
        "percentile": percentile,
        "deployment_gate_pass_2025": bool(execution["deployment_gate_pass"]),
        "model_diagnostics_2026": diagnostics,
        "cost_model": cost_result,
        "frictionless": gross_result,
        "cost_model_drag": float(cost_result["return"] - gross_result["return"]),
    }
    (OUTPUT_DIR / "backtest_2026.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if not cost_trades.empty:
        cost_trades.to_csv(OUTPUT_DIR / "trades_2026.csv", index=False)

    print("\n" + "=" * 116)
    print("EXP-3-M2 BACKTEST SUMMARY")
    print("=" * 116)
    print(f"Return with costs:     {fmt_pct(cost_result['return'])}")
    print(f"Return without costs:  {fmt_pct(gross_result['return'])}")
    print(f"Cost-model drag:       {fmt_pct(cost_result['return'] - gross_result['return'])}")
    print(f"Trades:                {cost_result['trades']}")
    print(f"Profit factor costs:   {cost_result['profit_factor']:.2f}")
    print(f"Profit factor gross:   {gross_result['profit_factor']:.2f}")
    print(f"Average trade costs:   {fmt_pct(cost_result['average_trade'])}")
    print(f"Average trade gross:   {fmt_pct(gross_result['average_trade'])}")
    print(f"Max drawdown:          {fmt_pct(cost_result['max_drawdown'])}")
    print(f"Daily Sharpe:          {cost_result['daily_sharpe']:+.2f}")


if __name__ == "__main__":
    main()
