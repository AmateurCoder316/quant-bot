from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from features import FEATURE_COLUMNS
from ml_research_core import load_execution_frames, portfolio_metrics, simulate_portfolio
from strategy_core import COMMISSION_RATE, MARKET_TIMEZONE, SLIPPAGE_RATE, SYMBOLS


MODEL_NAME = "exp-1-m1"
TARGET_COLUMN = "trade_net_return_60m"
HORIZON_MINUTES = 60
HOLD_BARS = 12
CLASS_THRESHOLD = 0.0
REGRESSION_CLIP = 0.03
UNCERTAINTY_TREE_STRIDE = 5

SYMBOL_COLUMNS = [f"symbol_{symbol}" for symbol in SYMBOLS]
MODEL_INPUT_COLUMNS = list(FEATURE_COLUMNS) + SYMBOL_COLUMNS

# m1 deliberately uses a small, interpretable execution grid. The experiment is
# about the architecture, not a giant threshold search.
PROBABILITY_THRESHOLDS = [0.55, 0.60, 0.65, 0.70]
EXPECTED_RETURN_THRESHOLDS = [0.0010, 0.0020, 0.0030, 0.0040]
UNCERTAINTY_QUANTILES = [0.50, 0.75, 1.00]
MIN_TRADES_FOR_SELECTION = 40


@dataclass(frozen=True)
class ExecutionConfig:
    probability_threshold: float
    expected_return_threshold: float
    uncertainty_quantile: float
    uncertainty_cap: float

    @property
    def label(self):
        return (
            f"p>={self.probability_threshold:.2f} "
            f"ev>={self.expected_return_threshold * 100:+.2f}% "
            f"u<={self.uncertainty_quantile:.0%}q"
        )


def add_symbol_columns(frame):
    result = frame.copy()
    for symbol in SYMBOLS:
        result[f"symbol_{symbol}"] = (result["symbol"] == symbol).astype(np.int8)
    return result


def build_classifier(random_state=101):
    return ExtraTreesClassifier(
        n_estimators=300,
        max_depth=16,
        min_samples_leaf=25,
        max_features=0.75,
        class_weight="balanced",
        bootstrap=False,
        n_jobs=-1,
        random_state=random_state,
    )


def build_regressor(random_state=202):
    return ExtraTreesRegressor(
        n_estimators=300,
        max_depth=16,
        min_samples_leaf=25,
        max_features=0.75,
        bootstrap=False,
        n_jobs=-1,
        random_state=random_state,
    )


def fit_dual_model(train):
    prepared = add_symbol_columns(train)
    x = prepared[MODEL_INPUT_COLUMNS]
    y_return = prepared[TARGET_COLUMN].astype(float).clip(
        lower=-REGRESSION_CLIP,
        upper=REGRESSION_CLIP,
    )
    y_class = (prepared[TARGET_COLUMN].astype(float) > CLASS_THRESHOLD).astype(np.int8)

    classifier = build_classifier()
    regressor = build_regressor()

    classifier.fit(x, y_class)
    regressor.fit(x, y_return)

    return {
        "classifier": classifier,
        "regressor": regressor,
        "input_columns": MODEL_INPUT_COLUMNS,
        "target_column": TARGET_COLUMN,
        "horizon_minutes": HORIZON_MINUTES,
        "hold_bars": HOLD_BARS,
        "class_threshold": CLASS_THRESHOLD,
        "regression_clip": REGRESSION_CLIP,
    }, y_class, y_return


def _regression_uncertainty(regressor, x):
    """Estimate model disagreement without allocating a huge tree x row matrix.

    We sample every Nth tree and accumulate first/second moments. This is not a
    formal predictive interval; it is an ensemble-disagreement feature used only
    as an execution filter/ranking input.

    Individual ExtraTreeRegressor estimators are fitted internally on NumPy
    arrays by scikit-learn even when the parent ensemble receives a DataFrame.
    Passing a NumPy view here therefore avoids misleading feature-name warnings
    without changing the values or the model's predictions.
    """
    x_array = x.to_numpy(dtype=float, copy=False) if hasattr(x, "to_numpy") else np.asarray(x)

    total = np.zeros(len(x_array), dtype=float)
    total_sq = np.zeros(len(x_array), dtype=float)
    count = 0

    for tree in regressor.estimators_[::UNCERTAINTY_TREE_STRIDE]:
        prediction = tree.predict(x_array)
        total += prediction
        total_sq += prediction * prediction
        count += 1

    if count == 0:
        return np.zeros(len(x_array), dtype=float)

    mean = total / count
    variance = np.maximum(total_sq / count - mean * mean, 0.0)
    return np.sqrt(variance)


def predict_dual(bundle, frame):
    prepared = add_symbol_columns(frame)
    x = prepared[MODEL_INPUT_COLUMNS]

    classifier = bundle["classifier"]
    regressor = bundle["regressor"]

    probability = classifier.predict_proba(x)[:, 1]
    expected_return = regressor.predict(x)
    uncertainty = _regression_uncertainty(regressor, x)

    # Ranking score: magnitude of expected net edge, weighted by directional
    # conviction and penalized by tree disagreement. Thresholds are still the
    # actual entry gates, so this score only resolves simultaneous candidates.
    probability_edge = np.clip(2.0 * probability - 1.0, 0.0, None)
    score = expected_return * probability_edge / (uncertainty + 0.001)

    result = frame[["timestamp", "symbol"]].copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    result["win_probability"] = probability
    result["expected_net_return"] = expected_return
    result["uncertainty"] = uncertainty
    result["score"] = score
    return result.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def model_metrics(frame, predictions):
    actual = frame[TARGET_COLUMN].astype(float).to_numpy()
    labels = (actual > CLASS_THRESHOLD).astype(np.int8)
    probability = predictions["win_probability"].to_numpy(dtype=float)
    expected = predictions["expected_net_return"].to_numpy(dtype=float)

    if len(np.unique(labels)) >= 2:
        auc = float(roc_auc_score(labels, probability))
        ap = float(average_precision_score(labels, probability))
        ll = float(log_loss(labels, probability))
    else:
        auc = None
        ap = None
        ll = None

    mae = float(mean_absolute_error(actual, expected))
    rmse = float(sqrt(mean_squared_error(actual, expected)))
    pearson = float(pd.Series(actual).corr(pd.Series(expected), method="pearson"))
    spearman = float(pd.Series(actual).corr(pd.Series(expected), method="spearman"))
    sign_accuracy = float(((expected > 0.0) == (actual > 0.0)).mean())

    return {
        "positive_rate": float(labels.mean()),
        "roc_auc": auc,
        "average_precision": ap,
        "log_loss": ll,
        "mae": mae,
        "rmse": rmse,
        "pearson": pearson,
        "spearman": spearman,
        "sign_accuracy": sign_accuracy,
    }


def _same_market_date(left, right):
    return (
        left.tz_convert(MARKET_TIMEZONE).date()
        == right.tz_convert(MARKET_TIMEZONE).date()
    )


def build_trade_candidates(predictions, frames, config):
    selected = predictions.loc[
        (predictions["win_probability"] >= config.probability_threshold)
        & (predictions["expected_net_return"] >= config.expected_return_threshold)
        & (predictions["uncertainty"] <= config.uncertainty_cap)
    ].copy()

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
                    "probability": float(row.win_probability),
                    "score": float(row.score),
                    "expected_net_return": float(row.expected_net_return),
                    "uncertainty": float(row.uncertainty),
                    "entry_raw": float(frame.iloc[entry_pos]["Open"]),
                    "exit_raw": float(frame.iloc[exit_pos]["Open"]),
                }
            )

    return sorted(candidates, key=lambda item: (item["entry_time"], -item["score"]))


def evaluate_config(predictions, frames, config):
    candidates = build_trade_candidates(predictions, frames, config)

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
        "probability_threshold": float(config.probability_threshold),
        "expected_return_threshold": float(config.expected_return_threshold),
        "uncertainty_quantile": float(config.uncertainty_quantile),
        "uncertainty_cap": float(config.uncertainty_cap),
        "candidate_rows": int(len(candidates)),
        "cost": portfolio_metrics(trades, equity, final_value, frames),
        "gross": portfolio_metrics(gross_trades, gross_equity, gross_final, frames),
    }


def uncertainty_caps(predictions):
    uncertainty = predictions["uncertainty"].astype(float)
    return {
        quantile: float(uncertainty.quantile(quantile))
        for quantile in UNCERTAINTY_QUANTILES
    }


def all_execution_configs(predictions):
    caps = uncertainty_caps(predictions)
    configs = []
    for probability in PROBABILITY_THRESHOLDS:
        for expected_return in EXPECTED_RETURN_THRESHOLDS:
            for quantile in UNCERTAINTY_QUANTILES:
                configs.append(
                    ExecutionConfig(
                        probability_threshold=probability,
                        expected_return_threshold=expected_return,
                        uncertainty_quantile=quantile,
                        uncertainty_cap=caps[quantile],
                    )
                )
    return configs


def add_grid_stability(results):
    """Score a 2-D local plateau instead of rewarding one lucky threshold pair."""
    output = [dict(item) for item in results]
    p_index = {value: i for i, value in enumerate(PROBABILITY_THRESHOLDS)}
    ev_index = {value: i for i, value in enumerate(EXPECTED_RETURN_THRESHOLDS)}

    for item in output:
        pi = p_index[item["probability_threshold"]]
        ei = ev_index[item["expected_return_threshold"]]
        q = item["uncertainty_quantile"]

        neighbors = []
        for other in output:
            if other["uncertainty_quantile"] != q:
                continue
            opi = p_index[other["probability_threshold"]]
            oei = ev_index[other["expected_return_threshold"]]
            if abs(opi - pi) <= 1 and abs(oei - ei) <= 1:
                neighbors.append(other)

        item["neighbor_count"] = int(len(neighbors))
        item["stability_return"] = float(
            np.median([neighbor["cost"]["return"] for neighbor in neighbors])
        )
        finite_pf = [
            neighbor["cost"]["profit_factor"]
            for neighbor in neighbors
            if np.isfinite(neighbor["cost"]["profit_factor"])
        ]
        item["stability_profit_factor"] = float(
            np.median(finite_pf) if finite_pf else 0.0
        )

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
        and result.get("stability_return", -np.inf) > 0.0
        and result.get("neighbor_count", 0) >= 4
    )


def select_execution_config(results):
    scored = add_grid_stability(results)
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
            item["stability_return"],
            item["cost"]["return"],
            item["cost"]["profit_factor"],
            item["cost"]["sharpe"],
            item["cost"]["trades"],
        ),
    )
    return dict(selected), scored


def execution_config_from_result(result):
    return ExecutionConfig(
        probability_threshold=float(result["probability_threshold"]),
        expected_return_threshold=float(result["expected_return_threshold"]),
        uncertainty_quantile=float(result["uncertainty_quantile"]),
        uncertainty_cap=float(result["uncertainty_cap"]),
    )


__all__ = [
    "MODEL_NAME",
    "TARGET_COLUMN",
    "HORIZON_MINUTES",
    "HOLD_BARS",
    "MODEL_INPUT_COLUMNS",
    "ExecutionConfig",
    "fit_dual_model",
    "predict_dual",
    "model_metrics",
    "load_execution_frames",
    "all_execution_configs",
    "evaluate_config",
    "select_execution_config",
    "execution_config_from_result",
]
