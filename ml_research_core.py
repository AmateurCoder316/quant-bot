from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from features import FEATURE_COLUMNS
from market_data import LOCAL_PROVIDER
from strategy_core import (
    COMMISSION_RATE,
    MARKET_TIMEZONE,
    MAX_OPEN_POSITIONS,
    MAX_POSITION_PERCENT,
    SLIPPAGE_RATE,
    STARTING_CASH,
    SYMBOLS,
)


SYMBOL_COLUMNS = [f"symbol_{symbol}" for symbol in SYMBOLS]
MODEL_INPUT_COLUMNS = list(FEATURE_COLUMNS) + SYMBOL_COLUMNS

ROLLING_WINDOW = 1000
MIN_ROLLING_HISTORY = 500
MIN_TRADES_FOR_SELECTION = 75

FIXED_THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.925]
ADAPTIVE_PERCENTILES = [0.95, 0.975, 0.99]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    target_column: str
    positive_threshold: float
    horizon_minutes: int
    hold_bars: int


@dataclass(frozen=True)
class ExecutionRule:
    mode: str
    value: float

    @property
    def label(self):
        if self.mode == "fixed":
            return f"p>={self.value:.3f}"
        return f"top {(1.0 - self.value) * 100:.1f}%"


def add_symbol_columns(frame):
    result = frame.copy()
    for symbol in SYMBOLS:
        result[f"symbol_{symbol}"] = (result["symbol"] == symbol).astype(np.int8)
    return result


def make_label(frame, spec):
    return (frame[spec.target_column] >= spec.positive_threshold).astype(np.int8)


def balanced_sample_weights(labels):
    positive_rate = float(labels.mean())
    if positive_rate <= 0.0 or positive_rate >= 1.0:
        return np.ones(len(labels), dtype=float)

    positive_weight = 0.5 / positive_rate
    negative_weight = 0.5 / (1.0 - positive_rate)
    return np.where(labels.to_numpy() == 1, positive_weight, negative_weight)


def build_model(random_state=42):
    """One deliberately fixed model family for fair walk-forward comparisons.

    Model complexity is NOT retuned per year. The walk-forward search focuses on
    economically meaningful targets/horizons and execution rules instead of a
    huge hyperparameter grid that would be easy to overfit on four years.
    """
    return HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=300,
        max_leaf_nodes=15,
        min_samples_leaf=100,
        l2_regularization=1.5,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=25,
        random_state=random_state,
    )


def fit_model(train, spec, random_state=42):
    prepared = add_symbol_columns(train)
    labels = make_label(prepared, spec)
    model = build_model(random_state=random_state)
    model.fit(
        prepared[MODEL_INPUT_COLUMNS],
        labels,
        sample_weight=balanced_sample_weights(labels),
    )
    return model, labels


def predict_probabilities(model, frame):
    prepared = add_symbol_columns(frame)
    result = frame[["timestamp", "symbol"]].copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    result["probability"] = model.predict_proba(prepared[MODEL_INPUT_COLUMNS])[:, 1]
    return result.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def classification_metrics(labels, probabilities):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(np.unique(labels)) < 2:
        return {
            "roc_auc": None,
            "average_precision": None,
            "log_loss": None,
            "positive_rate": float(labels.mean()) if len(labels) else 0.0,
        }

    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities)),
        "positive_rate": float(labels.mean()),
    }


def regular_session_frame(frame, year):
    result = frame.copy().sort_index()
    local = result.index.tz_convert(MARKET_TIMEZONE)
    minutes = local.hour * 60 + local.minute
    mask = (
        (local.year == year)
        & (minutes >= 9 * 60 + 30)
        & (minutes < 16 * 60)
    )
    return result.loc[mask].copy()


def load_execution_frames(year):
    frames = {}
    for symbol in SYMBOLS:
        frame = regular_session_frame(LOCAL_PROVIDER.history(symbol), year)
        if frame.empty:
            raise ValueError(f"No regular-session execution data for {symbol} in {year}.")
        frames[symbol] = frame
    return frames


def add_adaptive_threshold(predictions, percentile, seed_predictions=None):
    pieces = []

    for symbol, group in predictions.groupby("symbol", sort=False):
        group = group.sort_values("timestamp").copy()

        if seed_predictions is not None:
            seed = seed_predictions.loc[seed_predictions["symbol"] == symbol].copy()
            seed = seed.sort_values("timestamp").tail(ROLLING_WINDOW)
            history_values = pd.concat(
                [seed[["probability"]], group[["probability"]]],
                ignore_index=True,
            )["probability"]
            thresholds = history_values.shift(1).rolling(
                ROLLING_WINDOW,
                min_periods=MIN_ROLLING_HISTORY,
            ).quantile(percentile)
            group["adaptive_threshold"] = thresholds.iloc[-len(group):].to_numpy()
        else:
            history = group["probability"].shift(1)
            group["adaptive_threshold"] = history.rolling(
                ROLLING_WINDOW,
                min_periods=MIN_ROLLING_HISTORY,
            ).quantile(percentile)

        pieces.append(group)

    return pd.concat(pieces, ignore_index=True)


def _same_market_date(left, right):
    return (
        left.tz_convert(MARKET_TIMEZONE).date()
        == right.tz_convert(MARKET_TIMEZONE).date()
    )


def build_trade_candidates(predictions, frames, hold_bars, rule, seed_predictions=None):
    if rule.mode == "fixed":
        selected = predictions.loc[predictions["probability"] >= rule.value].copy()
        selected["score"] = selected["probability"]
    elif rule.mode == "adaptive":
        adaptive = add_adaptive_threshold(
            predictions,
            rule.value,
            seed_predictions=seed_predictions,
        )
        selected = adaptive.loc[
            adaptive["adaptive_threshold"].notna()
            & (adaptive["probability"] >= adaptive["adaptive_threshold"])
        ].copy()
        selected["score"] = selected["probability"] - selected["adaptive_threshold"]
    else:
        raise ValueError(f"Unknown execution rule mode: {rule.mode}")

    candidates = []

    for symbol in SYMBOLS:
        frame = frames[symbol]
        positions = {timestamp: i for i, timestamp in enumerate(frame.index)}
        rows = selected.loc[selected["symbol"] == symbol]

        for row in rows.itertuples(index=False):
            signal_pos = positions.get(row.timestamp)
            if signal_pos is None:
                continue

            entry_pos = signal_pos + 1
            exit_pos = entry_pos + hold_bars
            if exit_pos >= len(frame):
                continue

            entry_time = frame.index[entry_pos]
            exit_time = frame.index[exit_pos]
            if not _same_market_date(entry_time, exit_time):
                continue

            candidates.append(
                {
                    "symbol": symbol,
                    "signal_time": row.timestamp,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "probability": float(row.probability),
                    "score": float(row.score),
                    "entry_raw": float(frame.iloc[entry_pos]["Open"]),
                    "exit_raw": float(frame.iloc[exit_pos]["Open"]),
                }
            )

    return sorted(candidates, key=lambda item: (item["entry_time"], -item["score"]))


def _mark_open_positions(open_positions, timestamp, open_maps):
    value = 0.0
    for symbol, position in open_positions.items():
        mark = open_maps[symbol].get(timestamp, position["entry_raw"])
        value += position["shares"] * mark
    return value


def simulate_portfolio(candidates, frames, commission_rate, slippage_rate):
    """Event-driven long-only simulation matching the project execution policy."""
    cash = STARTING_CASH
    open_positions = {}
    trades = []
    equity_points = []

    candidates_by_time = {}
    exits_by_time = {}
    all_times = set()
    open_maps = {
        symbol: frame["Open"].astype(float).to_dict()
        for symbol, frame in frames.items()
    }

    for candidate in candidates:
        candidates_by_time.setdefault(candidate["entry_time"], []).append(candidate)
        all_times.add(candidate["entry_time"])
        all_times.add(candidate["exit_time"])

    for timestamp in sorted(all_times):
        due = exits_by_time.pop(timestamp, [])
        for symbol in due:
            position = open_positions.pop(symbol, None)
            if position is None:
                continue

            raw_exit = position["exit_raw"]
            exit_price = raw_exit * (1.0 - slippage_rate)
            gross = position["shares"] * exit_price
            exit_commission = gross * commission_rate
            net = gross - exit_commission
            cash += net

            pnl = net - position["entry_cost"]
            trades.append(
                {
                    "symbol": symbol,
                    "entry_time": position["entry_time"],
                    "exit_time": timestamp,
                    "probability": position["probability"],
                    "score": position["score"],
                    "pnl": pnl,
                    "return_percent": pnl / position["entry_cost"] * 100.0,
                    "commission": position["entry_commission"] + exit_commission,
                    "slippage_cost": position["shares"]
                    * (
                        (position["entry_price"] - position["entry_raw"])
                        + (raw_exit - exit_price)
                    ),
                }
            )

        entries = candidates_by_time.get(timestamp, [])
        entries.sort(key=lambda item: item["score"], reverse=True)

        for candidate in entries:
            symbol = candidate["symbol"]
            if symbol in open_positions:
                continue
            if len(open_positions) >= MAX_OPEN_POSITIONS:
                break

            marked_value = cash + _mark_open_positions(
                open_positions,
                timestamp,
                open_maps,
            )
            allocation = min(cash, marked_value * MAX_POSITION_PERCENT)
            if allocation <= 0.0:
                continue

            raw_entry = candidate["entry_raw"]
            entry_price = raw_entry * (1.0 + slippage_rate)
            trade_value = allocation / (1.0 + commission_rate)
            entry_commission = trade_value * commission_rate
            shares = trade_value / entry_price
            entry_cost = trade_value + entry_commission

            cash -= entry_cost
            open_positions[symbol] = {
                **candidate,
                "shares": shares,
                "entry_price": entry_price,
                "entry_commission": entry_commission,
                "entry_cost": entry_cost,
            }
            exits_by_time.setdefault(candidate["exit_time"], []).append(symbol)

        marked_equity = cash + _mark_open_positions(
            open_positions,
            timestamp,
            open_maps,
        )
        equity_points.append((timestamp, marked_equity))

    if open_positions:
        raise RuntimeError("Simulation ended with open positions; candidate scheduling is inconsistent.")

    trades_frame = pd.DataFrame(trades)
    equity_frame = pd.DataFrame(equity_points, columns=["timestamp", "equity"])
    return trades_frame, equity_frame, cash


def _daily_equity(equity_frame, frames):
    all_dates = sorted(
        {
            timestamp.tz_convert(MARKET_TIMEZONE).date()
            for frame in frames.values()
            for timestamp in frame.index
        }
    )
    if not all_dates:
        return pd.Series(dtype=float)

    if equity_frame.empty:
        return pd.Series(STARTING_CASH, index=pd.Index(all_dates, name="date"), dtype=float)

    equity = equity_frame.copy()
    equity["date"] = equity["timestamp"].map(
        lambda ts: ts.tz_convert(MARKET_TIMEZONE).date()
    )
    daily = equity.groupby("date", sort=True)["equity"].last()
    daily = daily.reindex(all_dates).ffill().fillna(STARTING_CASH)
    return daily.astype(float)


def portfolio_metrics(trades, equity_frame, final_value, frames):
    total_return = (final_value / STARTING_CASH - 1.0) * 100.0
    daily = _daily_equity(equity_frame, frames)

    if daily.empty:
        max_drawdown = 0.0
        sharpe = 0.0
    else:
        running_peak = daily.cummax()
        drawdown = daily / running_peak - 1.0
        max_drawdown = float(drawdown.min() * 100.0)

        daily_returns = daily.pct_change().dropna()
        if len(daily_returns) >= 2 and daily_returns.std(ddof=1) > 0:
            sharpe = float(
                daily_returns.mean() / daily_returns.std(ddof=1) * sqrt(252.0)
            )
        else:
            sharpe = 0.0

    if trades.empty:
        return {
            "return": float(total_return),
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_trade": 0.0,
            "friction": 0.0,
            "max_drawdown": max_drawdown,
            "sharpe": sharpe,
        }

    wins = trades.loc[trades["pnl"] > 0, "pnl"]
    losses = trades.loc[trades["pnl"] < 0, "pnl"]
    gross_profit = float(wins.sum())
    gross_loss = float(abs(losses.sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "return": float(total_return),
        "trades": int(len(trades)),
        "win_rate": float((trades["pnl"] > 0).mean() * 100.0),
        "profit_factor": float(profit_factor),
        "avg_trade": float(trades["return_percent"].mean()),
        "friction": float(trades["commission"].sum() + trades["slippage_cost"].sum()),
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
    }


def evaluate_rule(rule, predictions, frames, hold_bars, seed_predictions=None):
    candidates = build_trade_candidates(
        predictions,
        frames,
        hold_bars,
        rule,
        seed_predictions=seed_predictions,
    )

    trades, equity, final_value = simulate_portfolio(
        candidates,
        frames,
        COMMISSION_RATE,
        SLIPPAGE_RATE,
    )
    gross_trades, gross_equity, gross_final = simulate_portfolio(
        candidates,
        frames,
        0.0,
        0.0,
    )

    return {
        "mode": rule.mode,
        "value": float(rule.value),
        "label": rule.label,
        "candidate_rows": int(len(candidates)),
        "cost": portfolio_metrics(trades, equity, final_value, frames),
        "gross": portfolio_metrics(gross_trades, gross_equity, gross_final, frames),
    }


def all_execution_rules():
    return [
        *[ExecutionRule("fixed", value) for value in FIXED_THRESHOLDS],
        *[ExecutionRule("adaptive", value) for value in ADAPTIVE_PERCENTILES],
    ]


def add_stability_scores(results):
    """Penalize isolated threshold luck by scoring local parameter plateaus.

    A rule's stability return is the median after-cost return of itself and its
    immediate neighbors within the same rule family. A sharp one-point spike
    therefore cannot dominate a broad, consistently good region.
    """
    output = [dict(item) for item in results]

    for mode in ("fixed", "adaptive"):
        indexes = [i for i, item in enumerate(output) if item["mode"] == mode]
        indexes.sort(key=lambda i: output[i]["value"])

        for local_i, global_i in enumerate(indexes):
            start = max(0, local_i - 1)
            stop = min(len(indexes), local_i + 2)
            neighbors = [output[indexes[j]] for j in range(start, stop)]

            output[global_i]["stability_return"] = float(
                np.median([item["cost"]["return"] for item in neighbors])
            )
            finite_pfs = [
                item["cost"]["profit_factor"]
                for item in neighbors
                if np.isfinite(item["cost"]["profit_factor"])
            ]
            output[global_i]["stability_profit_factor"] = float(
                np.median(finite_pfs) if finite_pfs else 0.0
            )
            output[global_i]["neighbor_count"] = int(len(neighbors))

    return output


def deployment_gate(result):
    cost = result["cost"]
    gross = result["gross"]
    return bool(
        cost["trades"] >= MIN_TRADES_FOR_SELECTION
        and cost["return"] > 0.0
        and cost["profit_factor"] >= 1.05
        and result.get("stability_return", -np.inf) > 0.0
        and gross["profit_factor"] >= 1.10
        and cost["avg_trade"] > 0.0
    )


def select_execution_rule(results):
    scored = add_stability_scores(results)
    eligible = [
        item
        for item in scored
        if item["cost"]["trades"] >= MIN_TRADES_FOR_SELECTION
    ]
    pool = eligible if eligible else scored

    selected = max(
        pool,
        key=lambda item: (
            item["stability_return"],
            item["cost"]["return"],
            item["cost"]["profit_factor"],
            item["cost"]["trades"],
        ),
    )
    selected = dict(selected)
    selected["passes_deployment_gate"] = deployment_gate(selected)
    return selected, scored
