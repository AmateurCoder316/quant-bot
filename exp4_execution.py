from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from exp4_config import (
    COMMISSION_RATE,
    MARKET_TIMEZONE,
    MAX_OPEN_POSITIONS,
    MAX_POSITION_PERCENT,
    MIN_ROLLING_EVENT_HISTORY,
    MIN_TRADES_FOR_DEPLOYMENT,
    ROLLING_EVENT_WINDOW,
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
    score: float


def apply_causal_percentile(predictions: pd.DataFrame, percentile: float) -> pd.DataFrame:
    data = predictions.sort_index().copy()
    data["selected"] = False
    data["causal_threshold"] = np.nan
    history: deque[float] = deque(maxlen=ROLLING_EVENT_WINDOW)

    for timestamp, group in data.groupby(level=0, sort=True):
        if len(history) >= MIN_ROLLING_EVENT_HISTORY:
            threshold = float(np.quantile(np.fromiter(history, dtype=float), percentile))
            idx = group.index
            mask = group["pred_net_4h"].to_numpy(dtype=float) >= threshold
            data.loc[idx, "causal_threshold"] = threshold
            # duplicate timestamp index across symbols requires row-position-safe assignment
            timestamp_rows = data.index == timestamp
            row_positions = np.flatnonzero(timestamp_rows)
            group_positions = row_positions[: len(group)]
            selected_col = data.columns.get_loc("selected")
            threshold_col = data.columns.get_loc("causal_threshold")
            data.iloc[group_positions, threshold_col] = threshold
            data.iloc[group_positions, selected_col] = mask
        for score in group["pred_net_4h"].to_numpy(dtype=float):
            if np.isfinite(score):
                history.append(float(score))
    return data


def _profit_factor(pnl: pd.Series) -> float:
    wins = float(pnl[pnl > 0].sum())
    losses = float(-pnl[pnl < 0].sum())
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return wins / losses


def _finish(position: OpenPosition, cash: float, friction: bool):
    exit_fill = position.raw_exit_price * (1 - SLIPPAGE_RATE if friction else 1)
    gross = position.shares * exit_fill
    exit_commission = gross * COMMISSION_RATE if friction else 0.0
    net = gross - exit_commission
    pnl = net - position.total_cost
    trade_return = pnl / position.total_cost if position.total_cost else 0.0
    entry_slippage = position.shares * (position.entry_fill - position.raw_entry_price) if friction else 0.0
    exit_slippage = position.shares * (position.raw_exit_price - exit_fill) if friction else 0.0
    entry_commission = max(0.0, position.total_cost - position.shares * position.entry_fill)
    friction_cost = entry_slippage + exit_slippage + entry_commission + exit_commission
    trade = {
        "symbol": position.symbol, "entry_time": position.entry_time, "exit_time": position.exit_time,
        "entry_price": position.entry_fill, "exit_price": exit_fill, "shares": position.shares,
        "predicted_net_4h": position.score, "pnl": pnl, "return": trade_return,
        "friction_cost": friction_cost,
    }
    return cash + net, trade, friction_cost


def simulate_execution(predictions: pd.DataFrame, percentile: float, *, friction: bool = True):
    data = predictions.copy()
    data["entry_time"] = pd.to_datetime(data["entry_time"], utc=True)
    data["exit_time_4h"] = pd.to_datetime(data["exit_time_4h"], utc=True)
    data = data.dropna(subset=["pred_net_4h", "entry_time", "exit_time_4h", "entry_open", "exit_open_4h"])
    scored = apply_causal_percentile(data, percentile)
    candidates = scored[scored["selected"]].copy()
    candidates["_session_date"] = pd.DatetimeIndex(candidates.index).tz_convert(MARKET_TIMEZONE).date
    all_sessions = sorted(set(pd.DatetimeIndex(data.index).tz_convert(MARKET_TIMEZONE).date))
    grouped = {day: group for day, group in candidates.groupby("_session_date", sort=True)}

    cash = float(STARTING_CASH)
    trades, daily = [], []
    total_friction = 0.0
    for session_date in all_sessions:
        open_positions: list[OpenPosition] = []
        signals = grouped.get(session_date)
        if signals is not None and not signals.empty:
            for entry_time, entry_group in signals.groupby("entry_time", sort=True):
                due = [p for p in open_positions if p.exit_time <= entry_time]
                for position in sorted(due, key=lambda p: p.exit_time):
                    cash, trade, friction_cost = _finish(position, cash, friction)
                    trades.append(trade); total_friction += friction_cost; open_positions.remove(position)

                ranked = entry_group.sort_values("pred_net_4h", ascending=False)
                for _, row in ranked.iterrows():
                    if len(open_positions) >= MAX_OPEN_POSITIONS:
                        break
                    symbol = str(row["symbol"])
                    if any(p.symbol == symbol for p in open_positions):
                        continue
                    book_equity = cash + sum(p.total_cost for p in open_positions)
                    allocation = min(cash, book_equity * MAX_POSITION_PERCENT)
                    if allocation <= 0:
                        continue
                    raw_entry, raw_exit = float(row["entry_open"]), float(row["exit_open_4h"])
                    entry_fill = raw_entry * (1 + SLIPPAGE_RATE if friction else 1)
                    if entry_fill <= 0 or raw_exit <= 0:
                        continue
                    if friction:
                        trade_value = allocation / (1 + COMMISSION_RATE)
                        entry_commission = trade_value * COMMISSION_RATE
                    else:
                        trade_value, entry_commission = allocation, 0.0
                    shares = trade_value / entry_fill
                    total_cost = trade_value + entry_commission
                    cash -= total_cost
                    open_positions.append(OpenPosition(
                        symbol=symbol, shares=shares, entry_time=pd.Timestamp(entry_time),
                        exit_time=pd.Timestamp(row["exit_time_4h"]), raw_entry_price=raw_entry,
                        raw_exit_price=raw_exit, entry_fill=entry_fill, total_cost=total_cost,
                        score=float(row["pred_net_4h"]),
                    ))

        for position in sorted(open_positions, key=lambda p: p.exit_time):
            cash, trade, friction_cost = _finish(position, cash, friction)
            trades.append(trade); total_friction += friction_cost
        daily.append({"date": pd.Timestamp(session_date), "equity": cash})

    trades_frame = pd.DataFrame(trades)
    equity = pd.DataFrame(daily).set_index("date") if daily else pd.DataFrame(columns=["equity"])
    if equity.empty:
        max_dd = sharpe = 0.0
        month_returns = pd.Series(dtype=float)
    else:
        values = pd.concat([pd.Series([STARTING_CASH], index=[equity.index.min() - pd.Timedelta(days=1)]), equity["equity"]]).sort_index()
        daily_returns = values.pct_change().fillna(0.0).iloc[1:]
        max_dd = float((values / values.cummax() - 1).min())
        std = float(daily_returns.std(ddof=0))
        sharpe = float(daily_returns.mean() / std * math.sqrt(252)) if std > 0 else 0.0
        month_returns = (1 + daily_returns).groupby(equity.index.to_period("M")).prod() - 1

    if trades_frame.empty:
        win_rate = pf = avg_trade = 0.0
    else:
        win_rate = float((trades_frame["pnl"] > 0).mean())
        pf = _profit_factor(trades_frame["pnl"])
        avg_trade = float(trades_frame["return"].mean())
    positive_months = int((month_returns > 0).sum()) if len(month_returns) else 0
    result = {
        "percentile": float(percentile), "candidate_signals": int(len(candidates)),
        "return": float(cash / STARTING_CASH - 1), "final_value": float(cash),
        "trades": int(len(trades_frame)), "win_rate": win_rate, "profit_factor": pf,
        "average_trade": avg_trade, "max_drawdown": max_dd, "daily_sharpe": sharpe,
        "positive_months": positive_months, "month_count": int(len(month_returns)),
        "positive_month_fraction": float(positive_months / len(month_returns)) if len(month_returns) else 0.0,
        "median_month": float(month_returns.median()) if len(month_returns) else 0.0,
        "worst_month": float(month_returns.min()) if len(month_returns) else 0.0,
        "modeled_friction": float(total_friction),
    }
    return result, trades_frame, equity


def deployment_gate(selected: dict[str, Any], neighbors: list[dict[str, Any]]):
    local = [selected] + neighbors
    returns = [float(x["return"]) for x in local]
    pfs = [float(x["profit_factor"]) for x in local]
    summary = {"median_return": float(np.median(returns)), "worst_return": float(np.min(returns)), "median_profit_factor": float(np.median(pfs))}
    failures = []
    if int(selected["trades"]) < MIN_TRADES_FOR_DEPLOYMENT: failures.append(f"fewer than {MIN_TRADES_FOR_DEPLOYMENT} completed trades")
    if float(selected["return"]) <= 0: failures.append("after-cost return <= 0")
    if float(selected["profit_factor"]) < 1.15: failures.append("profit factor < 1.15")
    if float(selected["average_trade"]) <= 0: failures.append("average trade <= 0")
    if float(selected["daily_sharpe"]) <= 0: failures.append("daily Sharpe <= 0")
    if float(selected["positive_month_fraction"]) <= 0.55: failures.append("positive months <= 55%")
    if len(neighbors) < 2: failures.append("selected percentile is a boundary / lacks two neighbors")
    if summary["median_return"] <= 0: failures.append("local median return <= 0")
    if summary["worst_return"] <= 0: failures.append("local worst return <= 0")
    if summary["median_profit_factor"] < 1.05: failures.append("local median profit factor < 1.05")
    return not failures, failures, summary
