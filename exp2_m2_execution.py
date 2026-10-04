from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from exp2_execution import (
    calendar_robustness,
    load_year_frames,
    simulate_event_portfolio,
)
from exp2_m2_config import (
    EXECUTION_PERCENTILES,
    MIN_ROLLING_EVENT_HISTORY,
    MIN_TRADES_FOR_DEPLOYMENT,
    ROLLING_EVENT_WINDOW,
)
from ml_research_core import portfolio_metrics
from strategy_core import COMMISSION_RATE, SLIPPAGE_RATE


@dataclass(frozen=True)
class ExecutionRule:
    percentile: float

    @property
    def label(self):
        return f"M2 score top {(1.0 - self.percentile) * 100:.1f}%"


def add_causal_percentile_threshold(predictions, percentile, seed_predictions=None):
    pieces = []

    for symbol, group in predictions.groupby("symbol", sort=False):
        group = group.sort_values("timestamp").copy()
        current = group["meta_score"].astype(float).reset_index(drop=True)

        if seed_predictions is not None:
            seed = seed_predictions.loc[seed_predictions["symbol"] == symbol]
            seed_values = (
                seed.sort_values("timestamp")["meta_score"]
                .astype(float)
                .tail(ROLLING_EVENT_WINDOW)
                .reset_index(drop=True)
            )
            combined = pd.concat([seed_values, current], ignore_index=True)
            threshold = combined.shift(1).rolling(
                ROLLING_EVENT_WINDOW,
                min_periods=MIN_ROLLING_EVENT_HISTORY,
            ).quantile(percentile)
            group["causal_threshold"] = threshold.iloc[-len(group):].to_numpy()
        else:
            group["causal_threshold"] = current.shift(1).rolling(
                ROLLING_EVENT_WINDOW,
                min_periods=MIN_ROLLING_EVENT_HISTORY,
            ).quantile(percentile).to_numpy()

        pieces.append(group)

    return pd.concat(pieces, ignore_index=True).sort_values(
        ["timestamp", "symbol"]
    ).reset_index(drop=True)


def selected_events(predictions, rule, seed_predictions=None):
    scored = add_causal_percentile_threshold(
        predictions,
        rule.percentile,
        seed_predictions=seed_predictions,
    )
    chosen = scored.loc[
        scored["causal_threshold"].notna()
        & (scored["meta_score"] >= scored["causal_threshold"])
    ].copy()
    chosen["score"] = chosen["meta_score"] - chosen["causal_threshold"]
    return chosen.sort_values(["entry_time", "score"], ascending=[True, False])


def evaluate_rule(predictions, rule, year, seed_predictions=None):
    chosen = selected_events(predictions, rule, seed_predictions=seed_predictions)
    frames = load_year_frames(year)

    trades, equity, final_value = simulate_event_portfolio(
        chosen,
        frames,
        COMMISSION_RATE,
        SLIPPAGE_RATE,
    )
    gross_trades, gross_equity, gross_final = simulate_event_portfolio(
        chosen,
        frames,
        0.0,
        0.0,
    )

    cost_metrics = portfolio_metrics(trades, equity, final_value, frames)
    gross_metrics = portfolio_metrics(gross_trades, gross_equity, gross_final, frames)
    cost_metrics.update(calendar_robustness(equity))
    gross_metrics.update(calendar_robustness(gross_equity))

    return {
        "percentile": float(rule.percentile),
        "label": rule.label,
        "candidate_events": int(len(chosen)),
        "cost": cost_metrics,
        "gross": gross_metrics,
    }


def all_execution_rules():
    return [ExecutionRule(value) for value in EXECUTION_PERCENTILES]


def add_stability(results):
    output = [dict(item) for item in results]
    output.sort(key=lambda item: item["percentile"])

    for index, item in enumerate(output):
        start = max(0, index - 1)
        stop = min(len(output), index + 2)
        neighbors = output[start:stop]
        returns = [neighbor["cost"]["return"] for neighbor in neighbors]
        pfs = [
            neighbor["cost"]["profit_factor"]
            for neighbor in neighbors
            if np.isfinite(neighbor["cost"]["profit_factor"])
        ]
        item["neighbor_count"] = int(len(neighbors))
        item["stability_median_return"] = float(np.median(returns))
        item["stability_worst_return"] = float(np.min(returns))
        item["stability_median_pf"] = float(np.median(pfs) if pfs else 0.0)

    return output


def deployment_gate(result):
    cost = result["cost"]
    gross = result["gross"]
    return bool(
        cost["trades"] >= MIN_TRADES_FOR_DEPLOYMENT
        and cost["return"] > 0.0
        and cost["profit_factor"] >= 1.15
        and cost["avg_trade"] > 0.0
        and cost["sharpe"] > 0.0
        and cost["positive_month_fraction"] >= 0.55
        and cost["median_month"] > 0.0
        and gross["profit_factor"] >= 1.25
        and result.get("neighbor_count", 0) >= 3
        and result.get("stability_median_return", -np.inf) > 0.0
        and result.get("stability_worst_return", -np.inf) > 0.0
        and result.get("stability_median_pf", 0.0) >= 1.05
    )


def select_execution_rule(results):
    scored = add_stability(results)
    for item in scored:
        item["passes_deployment_gate"] = deployment_gate(item)

    supported = [
        item for item in scored if item["cost"]["trades"] >= MIN_TRADES_FOR_DEPLOYMENT
    ]
    pool = supported if supported else scored

    selected = max(
        pool,
        key=lambda item: (
            1 if item["passes_deployment_gate"] else 0,
            item["stability_worst_return"],
            item["stability_median_return"],
            item["cost"]["positive_month_fraction"],
            item["cost"]["return"],
            item["cost"]["profit_factor"],
            item["cost"]["trades"],
        ),
    )
    return dict(selected), scored


def execution_rule_from_result(result):
    return ExecutionRule(float(result["percentile"]))


__all__ = [
    "ExecutionRule",
    "all_execution_rules",
    "selected_events",
    "evaluate_rule",
    "select_execution_rule",
    "execution_rule_from_result",
]
