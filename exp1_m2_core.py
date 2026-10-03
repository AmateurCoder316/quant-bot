from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from features import FEATURE_COLUMNS
from ml_research_core import load_execution_frames, portfolio_metrics, simulate_portfolio
from strategy_core import COMMISSION_RATE, MARKET_TIMEZONE, SLIPPAGE_RATE, SYMBOLS


MODEL_NAME = "exp-1-m2"
HORIZON_MINUTES = 60
HOLD_BARS = 12
ROLLING_WINDOW = 1000
MIN_ROLLING_HISTORY = 500
MIN_TRADES_FOR_SELECTION = 75

# Three deliberately different classification questions. M2 does not attempt to
# predict an exact future return. It asks whether the move is immediately good,
# persists, and is large enough to matter after modeled trading costs.
HEAD_SPECS = {
    "p30_positive": {
        "target_column": "trade_net_return_30m",
        "threshold": 0.0,
        "description": "30m net return > 0",
        "random_state": 301,
    },
    "p60_positive": {
        "target_column": "trade_net_return_60m",
        "threshold": 0.0,
        "description": "60m net return > 0",
        "random_state": 302,
    },
    "p60_large": {
        "target_column": "trade_net_return_60m",
        "threshold": 0.003,
        "description": "60m net return >= +0.30%",
        "random_state": 303,
    },
}
HEAD_COLUMNS = list(HEAD_SPECS)
TARGET_COLUMNS = sorted({spec["target_column"] for spec in HEAD_SPECS.values()})

# One-dimensional execution search: every head must be in the same causal tail
# percentile of its own recent probability distribution. This deliberately
# avoids M1's 3-D threshold grid and makes isolated tuning luck harder.
CONSENSUS_PERCENTILES = [0.90, 0.925, 0.95, 0.975, 0.99]

SYMBOL_COLUMNS = [f"symbol_{symbol}" for symbol in SYMBOLS]
MODEL_INPUT_COLUMNS = list(FEATURE_COLUMNS) + SYMBOL_COLUMNS


@dataclass(frozen=True)
class ExecutionConfig:
    consensus_percentile: float

    @property
    def label(self):
        return f"all-heads top {(1.0 - self.consensus_percentile) * 100:.1f}%"


def add_symbol_columns(frame):
    result = frame.copy()
    for symbol in SYMBOLS:
        result[f"symbol_{symbol}"] = (result["symbol"] == symbol).astype(np.int8)
    return result


def _head_label(frame, spec):
    values = frame[spec["target_column"]].astype(float)
    if spec["threshold"] == 0.0:
        return (values > 0.0).astype(np.int8)
    return (values >= spec["threshold"]).astype(np.int8)


def build_head_model(random_state):
    return ExtraTreesClassifier(
        n_estimators=400,
        max_depth=16,
        min_samples_leaf=25,
        max_features=0.75,
        class_weight="balanced",
        bootstrap=False,
        n_jobs=-1,
        random_state=random_state,
    )


def fit_multihead_model(train):
    prepared = add_symbol_columns(train)
    x = prepared[MODEL_INPUT_COLUMNS]
    models = {}
    labels = {}

    for head_name, spec in HEAD_SPECS.items():
        y = _head_label(prepared, spec)
        model = build_head_model(spec["random_state"])
        model.fit(x, y)
        models[head_name] = model
        labels[head_name] = y

    bundle = {
        "model_name": MODEL_NAME,
        "models": models,
        "head_specs": HEAD_SPECS,
        "input_columns": MODEL_INPUT_COLUMNS,
        "hold_bars": HOLD_BARS,
        "horizon_minutes": HORIZON_MINUTES,
    }
    return bundle, labels


def predict_multihead(bundle, frame):
    prepared = add_symbol_columns(frame)
    x = prepared[MODEL_INPUT_COLUMNS]

    result = frame[["timestamp", "symbol"]].copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)

    for head_name in HEAD_COLUMNS:
        result[head_name] = bundle["models"][head_name].predict_proba(x)[:, 1]

    # Diagnostics/ranking only. Entry eligibility is based on the causal rolling
    # threshold for every head, not on this raw aggregate probability.
    probabilities = result[HEAD_COLUMNS].to_numpy(dtype=float)
    result["raw_consensus"] = np.exp(
        np.mean(np.log(np.clip(probabilities, 1e-9, 1.0)), axis=1)
    )
    result["weakest_head"] = probabilities.min(axis=1)

    return result.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def head_metrics(frame, predictions):
    output = {}
    for head_name, spec in HEAD_SPECS.items():
        labels = _head_label(frame, spec).to_numpy(dtype=int)
        probabilities = predictions[head_name].to_numpy(dtype=float)

        if len(np.unique(labels)) < 2:
            auc = None
            ap = None
            ll = None
        else:
            auc = float(roc_auc_score(labels, probabilities))
            ap = float(average_precision_score(labels, probabilities))
            ll = float(log_loss(labels, probabilities))

        output[head_name] = {
            "description": spec["description"],
            "positive_rate": float(labels.mean()),
            "roc_auc": auc,
            "average_precision": ap,
            "log_loss": ll,
        }
    return output


def _same_market_date(left, right):
    return (
        left.tz_convert(MARKET_TIMEZONE).date()
        == right.tz_convert(MARKET_TIMEZONE).date()
    )


def add_causal_consensus_thresholds(predictions, percentile, seed_predictions=None):
    pieces = []

    for symbol, group in predictions.groupby("symbol", sort=False):
        group = group.sort_values("timestamp").copy()

        if seed_predictions is not None:
            seed = seed_predictions.loc[seed_predictions["symbol"] == symbol].copy()
            seed = seed.sort_values("timestamp").tail(ROLLING_WINDOW)
        else:
            seed = None

        for head_name in HEAD_COLUMNS:
            if seed is not None and not seed.empty:
                history = pd.concat(
                    [seed[[head_name]], group[[head_name]]],
                    ignore_index=True,
                )[head_name]
                thresholds = history.shift(1).rolling(
                    ROLLING_WINDOW,
                    min_periods=MIN_ROLLING_HISTORY,
                ).quantile(percentile)
                group[f"{head_name}_threshold"] = thresholds.iloc[-len(group):].to_numpy()
            else:
                history = group[head_name].shift(1)
                group[f"{head_name}_threshold"] = history.rolling(
                    ROLLING_WINDOW,
                    min_periods=MIN_ROLLING_HISTORY,
                ).quantile(percentile)

        pieces.append(group)

    return pd.concat(pieces, ignore_index=True)


def select_consensus_signals(predictions, config, seed_predictions=None):
    adaptive = add_causal_consensus_thresholds(
        predictions,
        config.consensus_percentile,
        seed_predictions=seed_predictions,
    )

    valid = pd.Series(True, index=adaptive.index)
    normalized_margins = []

    for head_name in HEAD_COLUMNS:
        threshold_column = f"{head_name}_threshold"
        threshold = adaptive[threshold_column]
        probability = adaptive[head_name]
        valid &= threshold.notna() & (probability >= threshold)
        normalized_margins.append(
            (probability - threshold) / np.maximum(1.0 - threshold, 1e-6)
        )

    selected = adaptive.loc[valid].copy()
    if selected.empty:
        selected["score"] = pd.Series(dtype=float)
        return selected

    margin_frame = pd.concat(normalized_margins, axis=1)
    margin_frame.columns = HEAD_COLUMNS
    selected["score"] = margin_frame.loc[selected.index].min(axis=1).to_numpy()
    return selected.sort_values(["timestamp", "score"], ascending=[True, False])


def build_trade_candidates(predictions, frames, config, seed_predictions=None):
    selected = select_consensus_signals(
        predictions,
        config,
        seed_predictions=seed_predictions,
    )
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
            exit_pos = entry_pos + HOLD_BARS
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
                    "probability": float(row.raw_consensus),
                    "score": float(row.score),
                    "entry_raw": float(frame.iloc[entry_pos]["Open"]),
                    "exit_raw": float(frame.iloc[exit_pos]["Open"]),
                }
            )

    return sorted(candidates, key=lambda item: (item["entry_time"], -item["score"]))


def evaluate_config(predictions, frames, config, seed_predictions=None):
    candidates = build_trade_candidates(
        predictions,
        frames,
        config,
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
        "label": config.label,
        "consensus_percentile": float(config.consensus_percentile),
        "candidate_rows": int(len(candidates)),
        "cost": portfolio_metrics(trades, equity, final_value, frames),
        "gross": portfolio_metrics(gross_trades, gross_equity, gross_final, frames),
    }


def all_execution_configs():
    return [ExecutionConfig(value) for value in CONSENSUS_PERCENTILES]


def add_stability_scores(results):
    output = [dict(item) for item in results]
    output.sort(key=lambda item: item["consensus_percentile"])

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
        item["stability_return"] = float(np.median(returns))
        item["stability_min_return"] = float(min(returns))
        item["stability_profit_factor"] = float(np.median(pfs) if pfs else 0.0)

    return output


def deployment_gate(result):
    cost = result["cost"]
    gross = result["gross"]
    return bool(
        cost["trades"] >= MIN_TRADES_FOR_SELECTION
        and cost["return"] > 0.0
        and cost["profit_factor"] >= 1.10
        and cost["avg_trade"] > 0.0
        and cost["sharpe"] > 0.0
        and gross["profit_factor"] >= 1.20
        and result.get("neighbor_count", 0) >= 3
        and result.get("stability_return", -np.inf) > 0.0
        and result.get("stability_min_return", -np.inf) > 0.0
        and result.get("stability_profit_factor", 0.0) >= 1.05
    )


def select_execution_config(results):
    scored = add_stability_scores(results)
    for item in scored:
        item["passes_deployment_gate"] = deployment_gate(item)

    supported = [
        item for item in scored if item["cost"]["trades"] >= MIN_TRADES_FOR_SELECTION
    ]
    pool = supported if supported else scored

    selected = max(
        pool,
        key=lambda item: (
            1 if item["passes_deployment_gate"] else 0,
            item["stability_min_return"],
            item["stability_return"],
            item["cost"]["return"],
            item["cost"]["profit_factor"],
            item["cost"]["trades"],
        ),
    )
    return dict(selected), scored


def execution_config_from_result(result):
    return ExecutionConfig(
        consensus_percentile=float(result["consensus_percentile"]),
    )


__all__ = [
    "MODEL_NAME",
    "HORIZON_MINUTES",
    "HOLD_BARS",
    "HEAD_SPECS",
    "HEAD_COLUMNS",
    "TARGET_COLUMNS",
    "MODEL_INPUT_COLUMNS",
    "MIN_TRADES_FOR_SELECTION",
    "ExecutionConfig",
    "fit_multihead_model",
    "predict_multihead",
    "head_metrics",
    "load_execution_frames",
    "select_consensus_signals",
    "all_execution_configs",
    "evaluate_config",
    "select_execution_config",
    "execution_config_from_result",
]
