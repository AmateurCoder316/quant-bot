from __future__ import annotations

import numpy as np
import pandas as pd

from exp2_config import (
    MAX_NEW_ENTRIES_PER_TIMESTAMP,
    MIN_TRADES_FOR_DEPLOYMENT,
)
from exp2_core import ExecutionRule, selected_events
from market_data import LOCAL_PROVIDER
from ml_research_core import portfolio_metrics
from strategy_core import (
    COMMISSION_RATE,
    MARKET_TIMEZONE,
    MAX_OPEN_POSITIONS,
    MAX_POSITION_PERCENT,
    SLIPPAGE_RATE,
    STARTING_CASH,
    SYMBOLS,
)


def load_year_frames(year):
    frames = {}
    for symbol in SYMBOLS:
        raw = LOCAL_PROVIDER.history(symbol)
        local = raw.index.tz_convert(MARKET_TIMEZONE)
        minutes = local.hour * 60 + local.minute
        mask = (
            (local.year == year)
            & (minutes >= 9 * 60 + 30)
            & (minutes < 16 * 60)
        )
        frame = raw.loc[mask].copy()
        if frame.empty:
            raise ValueError(f"No regular-session EXP-2 data for {symbol} in {year}.")
        frames[symbol] = frame
    return frames


def _mark_positions(open_positions, timestamp, open_maps):
    value = 0.0
    for symbol, position in open_positions.items():
        mark = open_maps[symbol].get(timestamp, position["entry_raw"])
        value += position["shares"] * float(mark)
    return value


def _close_position(position, timestamp, commission_rate, slippage_rate):
    raw_exit = float(position["exit_raw"])
    exit_price = raw_exit * (1.0 - slippage_rate)
    gross_value = position["shares"] * exit_price
    exit_commission = gross_value * commission_rate
    net_value = gross_value - exit_commission
    pnl = net_value - position["entry_cost"]

    trade = {
        "symbol": position["symbol"],
        "entry_time": position["entry_time"],
        "exit_time": timestamp,
        "exit_reason": position["exit_reason"],
        "meta_probability": position["meta_probability"],
        "meta_score": position["meta_score"],
        "pnl": pnl,
        "return_percent": pnl / position["entry_cost"] * 100.0,
        "commission": position["entry_commission"] + exit_commission,
        "slippage_cost": position["shares"]
        * (
            (position["entry_price"] - position["entry_raw"])
            + (raw_exit - exit_price)
        ),
    }
    return net_value, trade


def simulate_event_portfolio(chosen, frames, commission_rate, slippage_rate):
    candidates = chosen.to_dict("records")
    entries_by_time = {}
    for candidate in candidates:
        entries_by_time.setdefault(candidate["entry_time"], []).append(candidate)

    open_maps = {
        symbol: frame["Open"].astype(float).to_dict()
        for symbol, frame in frames.items()
    }
    all_times = sorted({timestamp for frame in frames.values() for timestamp in frame.index})

    cash = STARTING_CASH
    open_positions = {}
    exits_by_time = {}
    trades = []
    equity_points = []

    for timestamp in all_times:
        # Exit positions whose barrier/time event occurs on this bar.
        due = exits_by_time.pop(timestamp, [])
        for symbol in due:
            position = open_positions.pop(symbol, None)
            if position is None:
                continue
            net_value, trade = _close_position(
                position, timestamp, commission_rate, slippage_rate
            )
            cash += net_value
            trades.append(trade)

        entries = entries_by_time.get(timestamp, [])
        entries.sort(key=lambda item: item["score"], reverse=True)
        entries = entries[:MAX_NEW_ENTRIES_PER_TIMESTAMP]

        for candidate in entries:
            symbol = candidate["symbol"]
            if symbol in open_positions:
                continue
            if len(open_positions) >= MAX_OPEN_POSITIONS:
                break

            marked_value = cash + _mark_positions(open_positions, timestamp, open_maps)
            allocation = min(cash, marked_value * MAX_POSITION_PERCENT)
            if allocation <= 0.0:
                continue

            raw_entry = float(candidate["entry_raw"])
            entry_price = raw_entry * (1.0 + slippage_rate)
            trade_value = allocation / (1.0 + commission_rate)
            entry_commission = trade_value * commission_rate
            shares = trade_value / entry_price
            entry_cost = trade_value + entry_commission
            cash -= entry_cost

            position = {
                **candidate,
                "shares": shares,
                "entry_price": entry_price,
                "entry_commission": entry_commission,
                "entry_cost": entry_cost,
            }

            # A horizontal barrier can be touched in the same 5-minute candle as
            # the next-open entry. Handle that immediately rather than scheduling
            # an exit that has already passed in this timestamp's event loop.
            if candidate["exit_time"] == timestamp:
                net_value, trade = _close_position(
                    position, timestamp, commission_rate, slippage_rate
                )
                cash += net_value
                trades.append(trade)
            else:
                open_positions[symbol] = position
                exits_by_time.setdefault(candidate["exit_time"], []).append(symbol)

        marked_equity = cash + _mark_positions(open_positions, timestamp, open_maps)
        equity_points.append((timestamp, marked_equity))

    if open_positions:
        raise RuntimeError("EXP-2 simulation ended with open positions.")

    trades_frame = pd.DataFrame(trades)
    equity_frame = pd.DataFrame(equity_points, columns=["timestamp", "equity"])
    return trades_frame, equity_frame, cash


def calendar_robustness(equity_frame):
    if equity_frame.empty:
        return {
            "months": 0,
            "positive_months": 0,
            "positive_month_fraction": 0.0,
            "worst_month": 0.0,
            "median_month": 0.0,
        }

    equity = equity_frame.copy()
    local = equity["timestamp"].dt.tz_convert(MARKET_TIMEZONE)
    equity["month"] = local.dt.to_period("M").astype(str)
    month_end = equity.groupby("month", sort=True)["equity"].last().astype(float)

    previous = STARTING_CASH
    returns = []
    for value in month_end:
        returns.append((float(value) / previous - 1.0) * 100.0)
        previous = float(value)

    values = np.asarray(returns, dtype=float)
    return {
        "months": int(len(values)),
        "positive_months": int((values > 0.0).sum()),
        "positive_month_fraction": float((values > 0.0).mean()) if len(values) else 0.0,
        "worst_month": float(values.min()) if len(values) else 0.0,
        "median_month": float(np.median(values)) if len(values) else 0.0,
    }


def evaluate_rule(predictions, rule, year, seed_predictions=None):
    chosen = selected_events(predictions, rule, seed_predictions=seed_predictions)
    frames = load_year_frames(year)

    trades, equity, final_value = simulate_event_portfolio(
        chosen, frames, COMMISSION_RATE, SLIPPAGE_RATE
    )
    gross_trades, gross_equity, gross_final = simulate_event_portfolio(
        chosen, frames, 0.0, 0.0
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


def add_stability(results):
    output = [dict(item) for item in results]
    output.sort(key=lambda item: item["percentile"])

    for i, item in enumerate(output):
        start = max(0, i - 1)
        stop = min(len(output), i + 2)
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


__all__ = [
    "evaluate_rule",
    "select_execution_rule",
    "deployment_gate",
    "calendar_robustness",
    "simulate_event_portfolio",
]
