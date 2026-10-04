from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from exp3_config import (
    COMMISSION_RATE,
    MARKET_TIMEZONE,
    MAX_OPEN_POSITIONS,
    MAX_POSITION_PERCENT,
    MIN_TRADES_FOR_DEPLOYMENT,
    SLIPPAGE_RATE,
    STARTING_CASH,
)


@dataclass
class OpenPosition:
    symbol: str
    shares: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    raw_entry_price: float
    raw_exit_price: float
    entry_fill: float
    total_cost: float
    predicted_net_return: float


def _profit_factor(pnl: pd.Series) -> float:
    wins = float(pnl[pnl > 0].sum())
    losses = float(-pnl[pnl < 0].sum())
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return wins / losses


def _finish_position(
    position: OpenPosition,
    cash: float,
    *,
    friction: bool,
) -> tuple[float, dict[str, Any], float]:
    exit_fill = position.raw_exit_price * (1.0 - SLIPPAGE_RATE if friction else 1.0)
    gross_value = position.shares * exit_fill
    exit_commission = gross_value * COMMISSION_RATE if friction else 0.0
    net_value = gross_value - exit_commission
    pnl = net_value - position.total_cost
    trade_return = pnl / position.total_cost if position.total_cost else 0.0

    entry_slippage = (
        position.shares * (position.entry_fill - position.raw_entry_price)
        if friction
        else 0.0
    )
    exit_slippage = (
        position.shares * (position.raw_exit_price - exit_fill)
        if friction
        else 0.0
    )
    implied_entry_commission = max(
        0.0,
        position.total_cost - position.shares * position.entry_fill,
    )
    friction_cost = entry_slippage + exit_slippage + implied_entry_commission + exit_commission

    trade = {
        "symbol": position.symbol,
        "entry_time": position.entry_time,
        "exit_time": position.exit_time,
        "entry_price": position.entry_fill,
        "exit_price": exit_fill,
        "shares": position.shares,
        "predicted_net_return": position.predicted_net_return,
        "pnl": pnl,
        "return": trade_return,
        "friction_cost": friction_cost,
    }
    return cash + net_value, trade, friction_cost


def simulate_execution(
    predictions: pd.DataFrame,
    threshold: float,
    *,
    friction: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    data = predictions.copy()
    data["entry_time"] = pd.to_datetime(data["entry_time"], utc=True)
    data["exit_time_2h"] = pd.to_datetime(data["exit_time_2h"], utc=True)
    data = data.dropna(
        subset=[
            "pred_net_2h",
            "entry_time",
            "exit_time_2h",
            "entry_open",
            "exit_open_2h",
        ]
    )
    candidates = data[data["pred_net_2h"] >= float(threshold)].copy()

    all_sessions = sorted(
        set(pd.DatetimeIndex(data.index).tz_convert(MARKET_TIMEZONE).date)
    )
    candidate_sessions = pd.DatetimeIndex(candidates.index).tz_convert(MARKET_TIMEZONE).date
    candidates["_session_date"] = candidate_sessions

    cash = float(STARTING_CASH)
    trades: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    total_friction = 0.0

    grouped_candidates = {
        day: group.copy()
        for day, group in candidates.groupby("_session_date", sort=True)
    }

    for session_date in all_sessions:
        day_signals = grouped_candidates.get(session_date)
        open_positions: list[OpenPosition] = []

        if day_signals is not None and not day_signals.empty:
            for entry_time, entry_group in day_signals.groupby("entry_time", sort=True):
                due = [position for position in open_positions if position.exit_time <= entry_time]
                for position in sorted(due, key=lambda item: item.exit_time):
                    cash, trade, friction_cost = _finish_position(
                        position, cash, friction=friction
                    )
                    total_friction += friction_cost
                    trades.append(trade)
                    open_positions.remove(position)

                ranked = entry_group.sort_values("pred_net_2h", ascending=False)
                for _, row in ranked.iterrows():
                    symbol = str(row["symbol"])
                    if len(open_positions) >= MAX_OPEN_POSITIONS:
                        break
                    if any(position.symbol == symbol for position in open_positions):
                        continue

                    book_equity = cash + sum(
                        position.total_cost for position in open_positions
                    )
                    allocation = min(cash, book_equity * MAX_POSITION_PERCENT)
                    if allocation <= 0:
                        continue

                    raw_entry = float(row["entry_open"])
                    raw_exit = float(row["exit_open_2h"])
                    entry_fill = raw_entry * (1.0 + SLIPPAGE_RATE if friction else 1.0)
                    if entry_fill <= 0 or raw_exit <= 0:
                        continue

                    if friction:
                        trade_value = allocation / (1.0 + COMMISSION_RATE)
                        entry_commission = trade_value * COMMISSION_RATE
                    else:
                        trade_value = allocation
                        entry_commission = 0.0

                    shares = trade_value / entry_fill
                    total_cost = trade_value + entry_commission
                    cash -= total_cost
                    open_positions.append(
                        OpenPosition(
                            symbol=symbol,
                            shares=shares,
                            entry_time=pd.Timestamp(entry_time),
                            exit_time=pd.Timestamp(row["exit_time_2h"]),
                            raw_entry_price=raw_entry,
                            raw_exit_price=raw_exit,
                            entry_fill=entry_fill,
                            total_cost=total_cost,
                            predicted_net_return=float(row["pred_net_2h"]),
                        )
                    )

        for position in sorted(open_positions, key=lambda item: item.exit_time):
            cash, trade, friction_cost = _finish_position(
                position, cash, friction=friction
            )
            total_friction += friction_cost
            trades.append(trade)

        daily_rows.append(
            {
                "date": pd.Timestamp(session_date),
                "equity": cash,
            }
        )

    trades_frame = pd.DataFrame(trades)
    equity = pd.DataFrame(daily_rows).set_index("date") if daily_rows else pd.DataFrame(columns=["equity"])

    if equity.empty:
        daily_returns = pd.Series(dtype=float)
        max_drawdown = 0.0
        sharpe = 0.0
        month_returns = pd.Series(dtype=float)
    else:
        values = pd.concat(
            [pd.Series([STARTING_CASH], index=[equity.index.min() - pd.Timedelta(days=1)]), equity["equity"]]
        ).sort_index()
        returns = values.pct_change().fillna(0.0)
        daily_returns = returns.iloc[1:]
        running_max = values.cummax()
        drawdown = values / running_max - 1.0
        max_drawdown = float(drawdown.min())
        std = float(daily_returns.std(ddof=0))
        sharpe = (
            float(daily_returns.mean()) / std * math.sqrt(252.0)
            if std > 0
            else 0.0
        )
        month_key = equity.index.to_period("M")
        month_returns = (1.0 + daily_returns).groupby(month_key).prod() - 1.0

    if trades_frame.empty:
        win_rate = 0.0
        profit_factor = 0.0
        average_trade = 0.0
    else:
        win_rate = float((trades_frame["pnl"] > 0).mean())
        profit_factor = _profit_factor(trades_frame["pnl"])
        average_trade = float(trades_frame["return"].mean())

    positive_months = int((month_returns > 0).sum()) if len(month_returns) else 0
    result = {
        "threshold": float(threshold),
        "candidate_signals": int(len(candidates)),
        "return": float(cash / STARTING_CASH - 1.0),
        "final_value": float(cash),
        "trades": int(len(trades_frame)),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "average_trade": average_trade,
        "max_drawdown": max_drawdown,
        "daily_sharpe": sharpe,
        "positive_months": positive_months,
        "month_count": int(len(month_returns)),
        "positive_month_fraction": float(positive_months / len(month_returns)) if len(month_returns) else 0.0,
        "median_month": float(month_returns.median()) if len(month_returns) else 0.0,
        "worst_month": float(month_returns.min()) if len(month_returns) else 0.0,
        "modeled_friction": float(total_friction),
    }
    return result, trades_frame, equity


def deployment_gate(
    selected: dict[str, Any],
    neighbors: list[dict[str, Any]],
) -> tuple[bool, list[str], dict[str, float]]:
    local = [selected] + list(neighbors)
    local_returns = [float(item["return"]) for item in local]
    local_pfs = [float(item["profit_factor"]) for item in local]
    local_summary = {
        "median_return": float(np.median(local_returns)),
        "worst_return": float(np.min(local_returns)),
        "median_profit_factor": float(np.median(local_pfs)),
    }

    failures = []
    if int(selected["trades"]) < MIN_TRADES_FOR_DEPLOYMENT:
        failures.append(f"fewer than {MIN_TRADES_FOR_DEPLOYMENT} completed trades")
    if float(selected["return"]) <= 0:
        failures.append("after-cost return <= 0")
    if float(selected["profit_factor"]) < 1.15:
        failures.append("profit factor < 1.15")
    if float(selected["average_trade"]) <= 0:
        failures.append("average trade <= 0")
    if float(selected["daily_sharpe"]) <= 0:
        failures.append("daily Sharpe <= 0")
    if float(selected["positive_month_fraction"]) <= 0.55:
        failures.append("positive months <= 55%")
    if len(neighbors) < 2:
        failures.append("selected threshold is a boundary / lacks two neighbors")
    if local_summary["median_return"] <= 0:
        failures.append("local median return <= 0")
    if local_summary["worst_return"] <= 0:
        failures.append("local worst return <= 0")
    if local_summary["median_profit_factor"] < 1.05:
        failures.append("local median profit factor < 1.05")

    return len(failures) == 0, failures, local_summary
