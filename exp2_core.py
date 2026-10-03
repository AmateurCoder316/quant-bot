from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from exp2_config import (
    ECONOMIC_NET_THRESHOLD,
    EVENT_DATASET_PATH,
    EXECUTION_PERCENTILES,
    EXP2_FEATURE_COLUMNS,
    MAX_NEW_ENTRIES_PER_TIMESTAMP,
    META_CONTEXT_COLUMNS,
    META_OOF_YEAR,
    MIN_ROLLING_EVENT_HISTORY,
    MIN_TRADES_FOR_DEPLOYMENT,
    MODEL_NAME,
    OOF_MIN_TRAIN_ROWS,
    PURGE_GAP_MINUTES,
    ROLLING_EVENT_WINDOW,
)
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


BASE_MODEL_SPECS = [
    ("barrier_hgb", "barrier_success", "hgb", 301),
    ("barrier_et", "barrier_success", "extra", 302),
    ("economic_hgb", "economic_success", "hgb", 401),
    ("economic_lr", "economic_success", "logit", 402),
]
BASE_PROBABILITY_COLUMNS = [name for name, _, _, _ in BASE_MODEL_SPECS]
META_AGGREGATE_COLUMNS = [
    "base_mean",
    "base_min",
    "base_max",
    "base_std",
    "barrier_consensus",
    "economic_consensus",
]
META_INPUT_COLUMNS = BASE_PROBABILITY_COLUMNS + META_AGGREGATE_COLUMNS + META_CONTEXT_COLUMNS


@dataclass(frozen=True)
class ExecutionRule:
    percentile: float

    @property
    def label(self):
        return f"meta top {(1.0 - self.percentile) * 100:.1f}%"


def load_event_dataset():
    if not EVENT_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Missing {EVENT_DATASET_PATH}. Run python build_exp_2_dataset.py first."
        )
    data = pd.read_parquet(EVENT_DATASET_PATH)
    for column in ("timestamp", "entry_time", "exit_time"):
        data[column] = pd.to_datetime(data[column], utc=True)
    data["year"] = data["year"].astype(int)
    return data.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def add_symbol_columns(frame):
    result = frame.copy()
    for symbol in SYMBOLS:
        result[f"symbol_{symbol}"] = (result["symbol"] == symbol).astype(np.int8)
    return result


def model_input_columns():
    return list(EXP2_FEATURE_COLUMNS) + [f"symbol_{symbol}" for symbol in SYMBOLS]


def _balanced_weights(labels, base_weight):
    labels = np.asarray(labels, dtype=int)
    weights = np.asarray(base_weight, dtype=float).copy()
    positive_rate = float(labels.mean()) if len(labels) else 0.0
    if 0.0 < positive_rate < 1.0:
        weights *= np.where(
            labels == 1,
            0.5 / positive_rate,
            0.5 / (1.0 - positive_rate),
        )
    mean = float(weights.mean()) if len(weights) else 1.0
    if mean > 0.0:
        weights /= mean
    return weights


def _build_estimator(kind, random_state):
    if kind == "hgb":
        # early_stopping=False is intentional: sklearn's internal validation split
        # is not a purged chronological financial validation set.
        return HistGradientBoostingClassifier(
            learning_rate=0.035,
            max_iter=240,
            max_leaf_nodes=15,
            min_samples_leaf=80,
            l2_regularization=2.0,
            early_stopping=False,
            random_state=random_state,
        )

    if kind == "extra":
        return ExtraTreesClassifier(
            n_estimators=400,
            max_depth=15,
            min_samples_leaf=20,
            max_features=0.70,
            class_weight=None,
            bootstrap=False,
            n_jobs=-1,
            random_state=random_state,
        )

    if kind == "logit":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.5,
                        solver="lbfgs",
                        max_iter=1200,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    raise ValueError(f"Unknown estimator kind: {kind}")


def _fit_estimator(kind, estimator, x, labels, weights):
    if kind == "logit":
        estimator.fit(x, labels, model__sample_weight=weights)
    else:
        estimator.fit(x, labels, sample_weight=weights)
    return estimator


def fit_base_models(train):
    prepared = add_symbol_columns(train)
    x = prepared[model_input_columns()]
    base_weight = prepared["sample_weight"].to_numpy(dtype=float)

    models = {}
    for name, target, kind, seed in BASE_MODEL_SPECS:
        labels = prepared[target].astype(np.int8).to_numpy()
        weights = _balanced_weights(labels, base_weight)
        estimator = _build_estimator(kind, seed)
        models[name] = _fit_estimator(kind, estimator, x, labels, weights)
    return models


def predict_base(models, frame):
    prepared = add_symbol_columns(frame)
    x = prepared[model_input_columns()]
    result = frame[["timestamp", "symbol"]].copy().reset_index(drop=True)

    for name, _, _, _ in BASE_MODEL_SPECS:
        result[name] = models[name].predict_proba(x)[:, 1]

    values = result[BASE_PROBABILITY_COLUMNS].to_numpy(dtype=float)
    result["base_mean"] = values.mean(axis=1)
    result["base_min"] = values.min(axis=1)
    result["base_max"] = values.max(axis=1)
    result["base_std"] = values.std(axis=1)
    result["barrier_consensus"] = np.sqrt(
        np.clip(result["barrier_hgb"] * result["barrier_et"], 0.0, 1.0)
    )
    result["economic_consensus"] = np.sqrt(
        np.clip(result["economic_hgb"] * result["economic_lr"], 0.0, 1.0)
    )
    return result


def _meta_frame(original, base_predictions):
    if len(original) != len(base_predictions):
        raise ValueError("Meta feature alignment mismatch.")

    meta = base_predictions.copy()
    for column in META_CONTEXT_COLUMNS:
        meta[column] = original[column].to_numpy(dtype=float)
    return meta


def _month_start(timestamp):
    ts = pd.Timestamp(timestamp)
    return pd.Timestamp(year=ts.year, month=ts.month, day=1, tz="UTC")


def generate_purged_oof_predictions(train_data):
    """Expanding 2024 monthly OOF predictions with label-overlap purging.

    Each validation month is predicted by base models fitted strictly on events
    whose labels resolved before a fixed gap ahead of that month. No future month
    participates in that fold's training set.
    """
    data = train_data.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    validation_year = data.loc[data["year"] == META_OOF_YEAR]
    months = sorted(validation_year["timestamp"].dt.to_period("M").unique())

    pieces = []
    fold_rows = []

    for period in months:
        period_start = pd.Timestamp(period.start_time, tz="UTC")
        period_end = pd.Timestamp(period.end_time, tz="UTC") + pd.Timedelta(nanoseconds=1)
        purge_cutoff = period_start - pd.Timedelta(minutes=PURGE_GAP_MINUTES)

        validation_mask = (
            (data["timestamp"] >= period_start)
            & (data["timestamp"] < period_end)
            & (data["year"] == META_OOF_YEAR)
        )
        train_mask = (
            (data["timestamp"] < purge_cutoff)
            & (data["exit_time"] < period_start)
        )

        fold_train = data.loc[train_mask].copy().reset_index(drop=True)
        fold_validation = data.loc[validation_mask].copy().reset_index(drop=True)

        if len(fold_train) < OOF_MIN_TRAIN_ROWS or fold_validation.empty:
            continue

        models = fit_base_models(fold_train)
        predictions = predict_base(models, fold_validation)
        predictions["source_index"] = fold_validation.index.to_numpy()
        predictions["fold_period"] = str(period)

        payload = fold_validation[
            [
                "timestamp",
                "symbol",
                "barrier_success",
                "economic_success",
                "net_positive",
                "event_net_return",
                "sample_weight",
                *META_CONTEXT_COLUMNS,
            ]
        ].copy().reset_index(drop=True)
        for column in BASE_PROBABILITY_COLUMNS + META_AGGREGATE_COLUMNS:
            payload[column] = predictions[column].to_numpy()
        payload["fold_period"] = str(period)
        pieces.append(payload)
        fold_rows.append(
            {
                "period": str(period),
                "train_rows": int(len(fold_train)),
                "validation_rows": int(len(fold_validation)),
                "purge_cutoff": purge_cutoff.isoformat(),
            }
        )

    if not pieces:
        raise RuntimeError("No EXP-2 purged OOF folds were produced.")

    oof = pd.concat(pieces, ignore_index=True)
    oof = oof.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return oof, fold_rows


def build_meta_model():
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.35,
                    solver="lbfgs",
                    max_iter=1500,
                    random_state=501,
                ),
            ),
        ]
    )


def fit_meta_model(oof):
    labels = oof["economic_success"].astype(np.int8).to_numpy()
    weights = _balanced_weights(labels, oof["sample_weight"].to_numpy(dtype=float))
    model = build_meta_model()
    model.fit(oof[META_INPUT_COLUMNS], labels, model__sample_weight=weights)
    return model


def fit_stack(train_data):
    oof, fold_rows = generate_purged_oof_predictions(train_data)
    meta_model = fit_meta_model(oof)
    final_base_models = fit_base_models(train_data)

    bundle = {
        "model_name": MODEL_NAME,
        "base_models": final_base_models,
        "meta_model": meta_model,
        "base_probability_columns": BASE_PROBABILITY_COLUMNS,
        "meta_input_columns": META_INPUT_COLUMNS,
        "feature_columns": EXP2_FEATURE_COLUMNS,
        "oof_fold_rows": fold_rows,
    }
    return bundle, oof


def predict_stack(bundle, frame):
    base = predict_base(bundle["base_models"], frame)
    meta = _meta_frame(frame.reset_index(drop=True), base)
    meta_probability = bundle["meta_model"].predict_proba(meta[META_INPUT_COLUMNS])[:, 1]

    result = frame.copy().reset_index(drop=True)
    for column in BASE_PROBABILITY_COLUMNS + META_AGGREGATE_COLUMNS:
        result[column] = base[column].to_numpy(dtype=float)
    result["meta_probability"] = meta_probability

    # Ranking score rewards meta conviction while requiring broad base agreement.
    # It is used for ranking/percentiles only, not interpreted as a calibrated P&L.
    result["meta_score"] = (
        result["meta_probability"]
        * np.sqrt(np.clip(result["barrier_consensus"], 0.0, 1.0))
        * np.sqrt(np.clip(result["economic_consensus"], 0.0, 1.0))
    )
    return result.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def classification_metrics(labels, probabilities):
    y = np.asarray(labels, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-8, 1.0 - 1e-8)
    if len(np.unique(y)) < 2:
        return {
            "roc_auc": None,
            "average_precision": None,
            "log_loss": None,
            "brier": None,
            "positive_rate": float(y.mean()) if len(y) else 0.0,
        }
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "positive_rate": float(y.mean()),
    }


def stack_metrics(frame, predictions):
    result = {
        "meta_economic": classification_metrics(
            frame["economic_success"], predictions["meta_probability"]
        ),
        "barrier_hgb": classification_metrics(
            frame["barrier_success"], predictions["barrier_hgb"]
        ),
        "barrier_et": classification_metrics(
            frame["barrier_success"], predictions["barrier_et"]
        ),
        "economic_hgb": classification_metrics(
            frame["economic_success"], predictions["economic_hgb"]
        ),
        "economic_lr": classification_metrics(
            frame["economic_success"], predictions["economic_lr"]
        ),
    }
    return result


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
            group["causal_threshold"] = threshold.iloc[-len(group) :].to_numpy()
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


def _load_year_frames(year):
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
        frames[symbol] = raw.loc[mask].copy()
    return frames


def _mark_positions(open_positions, timestamp, open_maps):
    value = 0.0
    for symbol, position in open_positions.items():
        mark = open_maps[symbol].get(timestamp, position["entry_raw"])
        value += position["shares"] * float(mark)
    return value


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
        due = exits_by_time.pop(timestamp, [])
        for symbol in due:
            position = open_positions.pop(symbol, None)
            if position is None:
                continue

            raw_exit = float(position["exit_raw"])
            exit_price = raw_exit * (1.0 - slippage_rate)
            gross_value = position["shares"] * exit_price
            exit_commission = gross_value * commission_rate
            net_value = gross_value - exit_commission
            cash += net_value

            pnl = net_value - position["entry_cost"]
            trades.append(
                {
                    "symbol": symbol,
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
            )

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

            open_positions[symbol] = {
                **candidate,
                "shares": shares,
                "entry_price": entry_price,
                "entry_commission": entry_commission,
                "entry_cost": entry_cost,
            }
            exits_by_time.setdefault(candidate["exit_time"], []).append(symbol)

        marked_equity = cash + _mark_positions(open_positions, timestamp, open_maps)
        equity_points.append((timestamp, marked_equity))

    if open_positions:
        # Every event is same-session and frames contain the full regular session.
        raise RuntimeError("EXP-2 simulation ended with open positions.")

    trades_frame = pd.DataFrame(trades)
    equity_frame = pd.DataFrame(equity_points, columns=["timestamp", "equity"])
    return trades_frame, equity_frame, cash


def evaluate_rule(predictions, rule, year, seed_predictions=None):
    chosen = selected_events(predictions, rule, seed_predictions=seed_predictions)
    frames = _load_year_frames(year)

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

    return {
        "percentile": float(rule.percentile),
        "label": rule.label,
        "candidate_events": int(len(chosen)),
        "cost": portfolio_metrics(trades, equity, final_value, frames),
        "gross": portfolio_metrics(gross_trades, gross_equity, gross_final, frames),
    }


def all_execution_rules():
    return [ExecutionRule(value) for value in EXECUTION_PERCENTILES]


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
        item["neighbor_count"] = len(neighbors)
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
            item["cost"]["return"],
            item["cost"]["profit_factor"],
            item["cost"]["trades"],
        ),
    )
    return dict(selected), scored


def execution_rule_from_result(result):
    return ExecutionRule(float(result["percentile"]))


__all__ = [
    "BASE_MODEL_SPECS",
    "BASE_PROBABILITY_COLUMNS",
    "META_INPUT_COLUMNS",
    "ExecutionRule",
    "load_event_dataset",
    "fit_stack",
    "predict_stack",
    "stack_metrics",
    "classification_metrics",
    "all_execution_rules",
    "evaluate_rule",
    "select_execution_rule",
    "execution_rule_from_result",
    "selected_events",
]
