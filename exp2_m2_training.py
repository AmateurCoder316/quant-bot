from __future__ import annotations

import importlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import joblib
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

try:
    import optuna
except ImportError:  # handled explicitly by require_training_dependencies()
    optuna = None

from exp2_m2_config import (
    CV_BLOCK_MONTHS,
    CV_MIN_TRAIN_ROWS,
    CV_VALIDATION_YEAR,
    DEFAULT_TRIALS_PER_FAMILY,
    ECONOMIC_SCALE_RETURN,
    FEATURE_COLUMNS,
    MAX_FINALIST_FAMILIES,
    MIN_FINALIST_FAMILIES,
    MODEL_FAMILIES,
    MODEL_NAME,
    OUTPUT_DIR,
    PRIMARY_TAIL_FRACTION,
    PURGE_GAP_MINUTES,
    SEARCH_SEED,
    SECONDARY_TAIL_FRACTION,
    SOURCE_EVENT_DATASET_PATH,
    STUDY_DIR,
    TARGET_COLUMN,
)
from strategy_core import SYMBOLS


OPTIONAL_MODULES = {
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
}


@dataclass(frozen=True)
class Fold:
    name: str
    train_indices: np.ndarray
    validation_indices: np.ndarray


@dataclass(frozen=True)
class FamilyWinner:
    family: str
    score: float
    params: dict
    metrics: dict


def require_training_dependencies():
    missing = []
    if optuna is None:
        missing.append("optuna")
    for package in OPTIONAL_MODULES.values():
        try:
            importlib.import_module(package)
        except ImportError:
            missing.append(package)
    if missing:
        names = ", ".join(sorted(set(missing)))
        raise RuntimeError(
            f"EXP-2-M2 training dependencies are missing: {names}. "
            "Run: pip install -r requirements-exp2-m2.txt"
        )


def dependency_versions():
    result = {}
    for package in ["sklearn", "optuna", "xgboost", "lightgbm", "catboost"]:
        try:
            module = importlib.import_module(package)
            result[package] = getattr(module, "__version__", "unknown")
        except ImportError:
            result[package] = None
    return result


def load_events():
    if not SOURCE_EVENT_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SOURCE_EVENT_DATASET_PATH}. Run python build_exp_2_dataset.py first."
        )
    data = pd.read_parquet(SOURCE_EVENT_DATASET_PATH)
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
    return list(FEATURE_COLUMNS) + [f"symbol_{symbol}" for symbol in SYMBOLS]


def balanced_weights(labels, base_weight):
    labels = np.asarray(labels, dtype=np.int8)
    weights = np.asarray(base_weight, dtype=float).copy()
    prevalence = float(labels.mean()) if len(labels) else 0.0
    if 0.0 < prevalence < 1.0:
        weights *= np.where(
            labels == 1,
            0.5 / prevalence,
            0.5 / (1.0 - prevalence),
        )
    mean = float(weights.mean()) if len(weights) else 1.0
    if mean > 0.0:
        weights /= mean
    return weights


def build_purged_cv_folds(train):
    data = train.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    validation_year = data.loc[data["year"] == CV_VALIDATION_YEAR]
    if validation_year.empty:
        raise ValueError(f"No rows for CV validation year {CV_VALIDATION_YEAR}.")

    # Convert only for calendar grouping; actual timestamps remain timezone-aware.
    naive_months = (
        validation_year["timestamp"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .dt.to_period("M")
        .unique()
    )
    months = sorted(naive_months)

    folds = []
    for i in range(0, len(months), CV_BLOCK_MONTHS):
        block = months[i : i + CV_BLOCK_MONTHS]
        if len(block) < CV_BLOCK_MONTHS:
            continue

        start = pd.Timestamp(block[0].start_time, tz="UTC")
        end = pd.Timestamp(block[-1].end_time, tz="UTC") + pd.Timedelta(nanoseconds=1)
        purge_cutoff = start - pd.Timedelta(minutes=PURGE_GAP_MINUTES)

        train_mask = (data["timestamp"] < purge_cutoff) & (data["exit_time"] < start)
        valid_mask = (
            (data["year"] == CV_VALIDATION_YEAR)
            & (data["timestamp"] >= start)
            & (data["timestamp"] < end)
        )

        train_indices = np.flatnonzero(train_mask.to_numpy())
        validation_indices = np.flatnonzero(valid_mask.to_numpy())
        if len(train_indices) < CV_MIN_TRAIN_ROWS or len(validation_indices) == 0:
            continue

        folds.append(
            Fold(
                name=f"{block[0]}..{block[-1]}",
                train_indices=train_indices,
                validation_indices=validation_indices,
            )
        )

    if len(folds) < 4:
        raise ValueError(f"Expected at least 4 purged CV folds, got {len(folds)}.")
    return data, folds


def trials_per_family():
    raw = os.getenv("EXP2_M2_TRIALS", str(DEFAULT_TRIALS_PER_FAMILY))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("EXP2_M2_TRIALS must be an integer.") from exc
    if value < 1:
        raise ValueError("EXP2_M2_TRIALS must be >= 1.")
    return value


def suggest_params(family, trial):
    if family == "hist_gradient_boosting":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.14, log=True),
            "max_iter": trial.suggest_int("max_iter", 150, 650, step=50),
            "max_leaf_nodes": trial.suggest_categorical("max_leaf_nodes", [7, 15, 31, 63]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 220, log=True),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 30.0, log=True),
            "max_bins": trial.suggest_categorical("max_bins", [63, 127, 255]),
        }

    if family == "extra_trees":
        bootstrap = trial.suggest_categorical("bootstrap", [False, True])
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1000, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [8, 12, 16, 20, None]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 4, 100, log=True),
            "max_features": trial.suggest_float("max_features", 0.30, 1.0),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            "bootstrap": bootstrap,
        }
        if bootstrap:
            params["max_samples"] = trial.suggest_float("max_samples", 0.60, 1.0)
        return params

    if family == "logistic":
        return {
            "C": trial.suggest_float("C", 1e-3, 30.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
        }

    if family == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 250, 1100, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.16, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 30.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.55, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.40, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.05, 40.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 1.5),
        }

    if family == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 250, 1200, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.16, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 127, log=True),
            "max_depth": trial.suggest_categorical("max_depth", [-1, 4, 6, 8, 10]),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 220, log=True),
            "subsample": trial.suggest_float("subsample", 0.55, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.40, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 40.0, log=True),
        }

    if family == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 250, 1100, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.16, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 40.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 1e-4, 4.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 3.0),
            "border_count": trial.suggest_categorical("border_count", [64, 128, 254]),
        }

    raise ValueError(f"Unknown model family: {family}")


def baseline_params(family):
    if family == "hist_gradient_boosting":
        return {
            "learning_rate": 0.04,
            "max_iter": 300,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 80,
            "l2_regularization": 2.0,
            "max_bins": 255,
        }
    if family == "extra_trees":
        return {
            "n_estimators": 500,
            "max_depth": 16,
            "min_samples_leaf": 20,
            "max_features": 0.70,
            "criterion": "log_loss",
            "bootstrap": False,
        }
    if family == "logistic":
        return {"C": 0.5, "penalty": "l2"}
    if family == "xgboost":
        return {
            "n_estimators": 500,
            "learning_rate": 0.04,
            "max_depth": 4,
            "min_child_weight": 8.0,
            "subsample": 0.80,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.05,
            "reg_lambda": 3.0,
            "gamma": 0.0,
        }
    if family == "lightgbm":
        return {
            "n_estimators": 500,
            "learning_rate": 0.04,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 50,
            "subsample": 0.80,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.05,
            "reg_lambda": 3.0,
        }
    if family == "catboost":
        return {
            "iterations": 500,
            "learning_rate": 0.04,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "random_strength": 0.5,
            "bagging_temperature": 1.0,
            "border_count": 128,
        }
    raise ValueError(f"Unknown family: {family}")


def build_estimator(family, params, seed):
    if family == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            **params,
            early_stopping=False,
            random_state=seed,
        )

    if family == "extra_trees":
        return ExtraTreesClassifier(
            **params,
            class_weight=None,
            n_jobs=-1,
            random_state=seed,
        )

    if family == "logistic":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        **params,
                        solver="liblinear",
                        max_iter=2500,
                        random_state=seed,
                    ),
                ),
            ]
        )

    if family == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        )

    if family == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            **params,
            objective="binary",
            subsample_freq=1,
            verbosity=-1,
            n_jobs=-1,
            random_state=seed,
        )

    if family == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            **params,
            loss_function="Logloss",
            eval_metric="Logloss",
            verbose=False,
            allow_writing_files=False,
            use_best_model=False,
            thread_count=-1,
            random_seed=seed,
        )

    raise ValueError(f"Unknown family: {family}")


def fit_estimator(family, estimator, x, labels, weights):
    if family == "logistic":
        estimator.fit(x, labels, model__sample_weight=weights)
    else:
        estimator.fit(x, labels, sample_weight=weights)
    return estimator


def _safe_auc(labels, probability):
    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, probability))


def _safe_ap(labels, probability):
    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) < 2:
        return float(labels.mean())
    return float(average_precision_score(labels, probability))


def tail_diagnostics(validation, probability, fraction):
    count = max(1, int(math.ceil(len(validation) * fraction)))
    order = np.argsort(-np.asarray(probability, dtype=float))[:count]
    selected = validation.iloc[order]
    return {
        "count": int(count),
        "precision": float(selected[TARGET_COLUMN].mean()),
        "avg_net_return": float(selected["event_net_return"].mean()),
        "median_net_return": float(selected["event_net_return"].median()),
    }


def fold_diagnostics(validation, probability):
    labels = validation[TARGET_COLUMN].astype(np.int8).to_numpy()
    probability = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    prevalence = float(labels.mean())
    return {
        "rows": int(len(validation)),
        "prevalence": prevalence,
        "auc": _safe_auc(labels, probability),
        "average_precision": _safe_ap(labels, probability),
        "log_loss": float(log_loss(labels, probability)),
        "brier": float(brier_score_loss(labels, probability)),
        "top10": tail_diagnostics(validation, probability, PRIMARY_TAIL_FRACTION),
        "top05": tail_diagnostics(validation, probability, SECONDARY_TAIL_FRACTION),
    }


def summarize_fold_metrics(fold_metrics):
    auc = np.asarray([item["auc"] for item in fold_metrics], dtype=float)
    ap = np.asarray([item["average_precision"] for item in fold_metrics], dtype=float)
    prevalence = np.asarray([item["prevalence"] for item in fold_metrics], dtype=float)
    top_precision = np.asarray([item["top10"]["precision"] for item in fold_metrics], dtype=float)
    top_net = np.asarray([item["top10"]["avg_net_return"] for item in fold_metrics], dtype=float)
    top05_net = np.asarray([item["top05"]["avg_net_return"] for item in fold_metrics], dtype=float)
    brier = np.asarray([item["brier"] for item in fold_metrics], dtype=float)

    mean_prev = float(prevalence.mean())
    mean_auc = float(auc.mean())
    mean_ap = float(ap.mean())
    mean_top_precision = float(top_precision.mean())
    mean_top_net = float(top_net.mean())

    # The objective deliberately mixes ranking quality and economic tail quality.
    # Every term is derived only from purged 2024 folds; 2025/2026 are untouched.
    auc_edge = np.clip((mean_auc - 0.50) / 0.15, -2.0, 3.0)
    ap_lift = np.clip(mean_ap / max(mean_prev, 1e-6) - 1.0, -2.0, 3.0)
    tail_lift = np.clip(mean_top_precision / max(mean_prev, 1e-6) - 1.0, -2.0, 3.0)
    net_component = np.clip(mean_top_net / ECONOMIC_SCALE_RETURN, -2.0, 3.0)
    worst_net_component = np.clip(float(top_net.min()) / ECONOMIC_SCALE_RETURN, -2.0, 3.0)

    instability = (
        0.12 * np.clip(float(auc.std()) / 0.05, 0.0, 3.0)
        + 0.08 * np.clip(float(top_net.std()) / ECONOMIC_SCALE_RETURN, 0.0, 3.0)
    )
    below_random_penalty = 0.0
    min_auc = float(auc.min())
    if min_auc < 0.50:
        below_random_penalty = min(0.50, (0.50 - min_auc) * 3.0)

    score = float(
        0.35 * auc_edge
        + 0.20 * ap_lift
        + 0.20 * tail_lift
        + 0.15 * net_component
        + 0.10 * worst_net_component
        - instability
        - below_random_penalty
    )

    return {
        "objective": score,
        "mean_auc": mean_auc,
        "min_auc": min_auc,
        "std_auc": float(auc.std()),
        "mean_average_precision": mean_ap,
        "mean_prevalence": mean_prev,
        "mean_ap_lift": float(mean_ap / max(mean_prev, 1e-6)),
        "mean_top10_precision": mean_top_precision,
        "mean_top10_lift": float(mean_top_precision / max(mean_prev, 1e-6)),
        "mean_top10_net_return": mean_top_net,
        "worst_top10_net_return": float(top_net.min()),
        "mean_top05_net_return": float(top05_net.mean()),
        "mean_brier": float(brier.mean()),
        "folds": fold_metrics,
    }


def evaluate_params(data, folds, family, params, seed):
    prepared = add_symbol_columns(data)
    columns = model_input_columns()
    fold_metrics = []

    for fold_number, fold in enumerate(folds):
        fold_train = prepared.iloc[fold.train_indices]
        fold_validation = prepared.iloc[fold.validation_indices]
        labels = fold_train[TARGET_COLUMN].astype(np.int8).to_numpy()
        weights = balanced_weights(labels, fold_train["sample_weight"].to_numpy(dtype=float))

        estimator = build_estimator(family, params, seed + fold_number)
        fit_estimator(
            family,
            estimator,
            fold_train[columns],
            labels,
            weights,
        )
        probability = estimator.predict_proba(fold_validation[columns])[:, 1]
        metrics = fold_diagnostics(fold_validation, probability)
        metrics["fold"] = fold.name
        fold_metrics.append(metrics)

    return summarize_fold_metrics(fold_metrics)


def _family_seed(family):
    return SEARCH_SEED + sum((i + 1) * ord(char) for i, char in enumerate(family)) % 10_000


def run_family_study(data, folds, family, n_trials):
    require_training_dependencies()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STUDY_DIR.mkdir(parents=True, exist_ok=True)

    study_path = (STUDY_DIR / f"{family}.sqlite3").resolve()
    storage = f"sqlite:///{study_path}"
    sampler = optuna.samplers.TPESampler(
        seed=_family_seed(family),
        n_startup_trials=min(10, max(5, n_trials // 3)),
        multivariate=False,
    )
    study = optuna.create_study(
        study_name=f"{MODEL_NAME}-{family}",
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
    )

    completed = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed:
        study.enqueue_trial(baseline_params(family))

    remaining = max(0, n_trials - len(completed))

    def objective(trial):
        params = suggest_params(family, trial)
        metrics = evaluate_params(
            data,
            folds,
            family,
            params,
            _family_seed(family) + trial.number * 37,
        )
        for key, value in metrics.items():
            if key != "folds":
                trial.set_user_attr(key, float(value))
        trial.set_user_attr("folds", metrics["folds"])
        return metrics["objective"]

    if remaining:
        study.optimize(
            objective,
            n_trials=remaining,
            n_jobs=1,
            gc_after_trial=True,
            show_progress_bar=False,
        )

    best = study.best_trial
    metrics = {
        key: value for key, value in best.user_attrs.items()
    }
    return FamilyWinner(
        family=family,
        score=float(best.value),
        params=dict(best.params),
        metrics=metrics,
    )


def choose_finalists(winners):
    ordered = sorted(winners, key=lambda item: item.score, reverse=True)
    if len(ordered) < MIN_FINALIST_FAMILIES:
        raise ValueError(
            f"Need at least {MIN_FINALIST_FAMILIES} model-family winners, got {len(ordered)}."
        )
    return ordered[: min(MAX_FINALIST_FAMILIES, len(ordered))]


def _fit_family_on_frame(frame, family, params, seed):
    prepared = add_symbol_columns(frame)
    columns = model_input_columns()
    labels = prepared[TARGET_COLUMN].astype(np.int8).to_numpy()
    weights = balanced_weights(labels, prepared["sample_weight"].to_numpy(dtype=float))
    estimator = build_estimator(family, params, seed)
    fit_estimator(family, estimator, prepared[columns], labels, weights)
    return estimator


def generate_finalist_oof(data, folds, finalists):
    prepared = add_symbol_columns(data)
    columns = model_input_columns()
    pieces = []

    for fold_number, fold in enumerate(folds):
        fold_train = prepared.iloc[fold.train_indices]
        fold_validation = prepared.iloc[fold.validation_indices].copy()
        payload = fold_validation[
            [
                "timestamp",
                "symbol",
                TARGET_COLUMN,
                "event_net_return",
                "sample_weight",
            ]
        ].copy().reset_index(drop=True)
        payload["fold"] = fold.name

        for finalist_number, finalist in enumerate(finalists):
            labels = fold_train[TARGET_COLUMN].astype(np.int8).to_numpy()
            weights = balanced_weights(
                labels,
                fold_train["sample_weight"].to_numpy(dtype=float),
            )
            seed = _family_seed(finalist.family) + fold_number * 101 + finalist_number
            model = build_estimator(finalist.family, finalist.params, seed)
            fit_estimator(
                finalist.family,
                model,
                fold_train[columns],
                labels,
                weights,
            )
            payload[f"prob_{finalist.family}"] = model.predict_proba(
                fold_validation[columns]
            )[:, 1]

        pieces.append(payload)

    return pd.concat(pieces, ignore_index=True).sort_values(
        ["timestamp", "symbol"]
    ).reset_index(drop=True)


def blend_input_frame(probability_frame, probability_columns):
    result = probability_frame[probability_columns].copy()
    values = result.to_numpy(dtype=float)
    result["ensemble_mean"] = values.mean(axis=1)
    result["ensemble_min"] = values.min(axis=1)
    result["ensemble_max"] = values.max(axis=1)
    result["ensemble_std"] = values.std(axis=1)
    return result


def fit_blender(oof, probability_columns):
    labels = oof[TARGET_COLUMN].astype(np.int8).to_numpy()
    weights = balanced_weights(labels, oof["sample_weight"].to_numpy(dtype=float))
    x = blend_input_frame(oof, probability_columns)
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.25,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=SEARCH_SEED + 999,
                ),
            ),
        ]
    )
    model.fit(x, labels, model__sample_weight=weights)
    return model, list(x.columns)


def fit_final_bundle(train, finalists, oof):
    models = {}
    probability_columns = []
    for index, finalist in enumerate(finalists):
        models[finalist.family] = _fit_family_on_frame(
            train,
            finalist.family,
            finalist.params,
            _family_seed(finalist.family) + 50_000 + index,
        )
        probability_columns.append(f"prob_{finalist.family}")

    blender, blender_columns = fit_blender(oof, probability_columns)
    return {
        "model_name": MODEL_NAME,
        "target": TARGET_COLUMN,
        "feature_columns": model_input_columns(),
        "finalists": [
            {
                "family": item.family,
                "search_score": item.score,
                "params": item.params,
                "search_metrics": item.metrics,
            }
            for item in finalists
        ],
        "base_models": models,
        "probability_columns": probability_columns,
        "blender": blender,
        "blender_columns": blender_columns,
    }


def predict_bundle(bundle, frame):
    prepared = add_symbol_columns(frame)
    x = prepared[bundle["feature_columns"]]
    result = frame.copy().reset_index(drop=True)

    for finalist in bundle["finalists"]:
        family = finalist["family"]
        column = f"prob_{family}"
        result[column] = bundle["base_models"][family].predict_proba(x)[:, 1]

    blend_x = blend_input_frame(result, bundle["probability_columns"])
    blend_x = blend_x[bundle["blender_columns"]]
    result["meta_probability"] = bundle["blender"].predict_proba(blend_x)[:, 1]
    # Percentile execution uses ranking, so the blend probability itself is the
    # score. We avoid inventing another hand-tuned confidence formula.
    result["meta_score"] = result["meta_probability"]
    return result.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def prediction_metrics(frame, predictions):
    labels = frame[TARGET_COLUMN].astype(np.int8).to_numpy()
    probability = np.clip(predictions["meta_probability"].to_numpy(dtype=float), 1e-7, 1 - 1e-7)
    result = {
        "rows": int(len(frame)),
        "positive_rate": float(labels.mean()),
        "roc_auc": _safe_auc(labels, probability),
        "average_precision": _safe_ap(labels, probability),
        "log_loss": float(log_loss(labels, probability)),
        "brier": float(brier_score_loss(labels, probability)),
        "top10": tail_diagnostics(frame, probability, PRIMARY_TAIL_FRACTION),
        "top05": tail_diagnostics(frame, probability, SECONDARY_TAIL_FRACTION),
    }
    if len(frame) > 2:
        result["spearman_net_return"] = float(
            pd.Series(probability).corr(
                frame["event_net_return"].reset_index(drop=True),
                method="spearman",
            )
        )
    else:
        result["spearman_net_return"] = 0.0
    return result


def save_bundle(bundle, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def winner_to_dict(winner):
    return {
        "family": winner.family,
        "score": winner.score,
        "params": winner.params,
        "metrics": winner.metrics,
    }


def json_clean(value):
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(json_clean(payload), indent=2) + "\n")


__all__ = [
    "FamilyWinner",
    "require_training_dependencies",
    "dependency_versions",
    "load_events",
    "build_purged_cv_folds",
    "trials_per_family",
    "run_family_study",
    "choose_finalists",
    "generate_finalist_oof",
    "fit_final_bundle",
    "predict_bundle",
    "prediction_metrics",
    "save_bundle",
    "winner_to_dict",
    "write_json",
]
