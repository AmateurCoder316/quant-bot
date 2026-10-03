import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import FEATURE_COLUMNS
from ml_research_core import (
    ExecutionRule,
    ModelSpec,
    classification_metrics,
    evaluate_rule,
    fit_model,
    load_execution_frames,
    make_label,
    predict_probabilities,
    select_execution_rule,
)


DATASET_PATH = Path("data") / "ml" / "features.parquet"
OUTPUT_DIR = Path("data") / "research"
RESULTS_PATH = OUTPUT_DIR / "ml_walk_forward_v2.json"
MODEL_OUTPUT_DIR = OUTPUT_DIR / "walk_forward_models"

# Proper nested chronology:
#   1) fit model on past years
#   2) choose target/execution rule on a later tuning year
#   3) freeze everything
#   4) evaluate once on the following test year
FOLDS = [
    {
        "name": "WF1",
        "train_years": [2023],
        "tune_year": 2024,
        "test_year": 2025,
    },
    {
        "name": "WF2",
        "train_years": [2023, 2024],
        "tune_year": 2025,
        "test_year": 2026,
    },
]

# Research-v2 deliberately searches economic questions rather than a giant tree
# hyperparameter grid. Every target is the EXACT next-open -> later-open simulated
# trade return AFTER the configured commission/slippage assumptions.
MODEL_SPECS = [
    ModelSpec("N30_BE", "trade_net_return_30m", 0.0000, 30, 6),
    ModelSpec("N30_15BP", "trade_net_return_30m", 0.0015, 30, 6),
    ModelSpec("N30_30BP", "trade_net_return_30m", 0.0030, 30, 6),
    ModelSpec("N60_BE", "trade_net_return_60m", 0.0000, 60, 12),
    ModelSpec("N60_20BP", "trade_net_return_60m", 0.0020, 60, 12),
    ModelSpec("N60_40BP", "trade_net_return_60m", 0.0040, 60, 12),
]

# Wider than the old GB1/GB2 grids. Boundary points are diagnostic only: the
# deployment gate below requires the chosen rule to have BOTH neighbors so a
# single extreme threshold cannot be mistaken for a robust plateau.
RESEARCH_RULES = [
    *[
        ExecutionRule("fixed", value)
        for value in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95)
    ],
    *[
        ExecutionRule("adaptive", value)
        for value in (0.95, 0.975, 0.99, 0.995)
    ],
]

MIN_CLASS_EXAMPLES = 500


def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATASET_PATH}. Rebuild it with python build_dataset.py first."
        )

    target_columns = sorted({spec.target_column for spec in MODEL_SPECS})
    required = [
        "timestamp",
        "symbol",
        "year",
        *FEATURE_COLUMNS,
        *target_columns,
    ]

    try:
        dataset = pd.read_parquet(DATASET_PATH, columns=required)
    except Exception as exc:
        raise RuntimeError(
            "The ML dataset does not contain the research-v2 execution targets. "
            "Run python build_dataset.py after pulling the latest code."
        ) from exc

    dataset["timestamp"] = pd.to_datetime(dataset["timestamp"], utc=True)
    dataset["year"] = dataset["year"].astype(int)
    dataset = dataset.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    years = sorted(dataset["year"].unique().tolist())
    for required_year in (2023, 2024, 2025, 2026):
        if required_year not in years:
            raise ValueError(f"Dataset is missing required year {required_year}.")

    return dataset


def ordered(frame):
    """Stable ordering keeps labels and predict_proba output perfectly aligned."""
    return frame.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def spec_payload(spec):
    return {
        "name": spec.name,
        "target_column": spec.target_column,
        "positive_threshold": spec.positive_threshold,
        "horizon_minutes": spec.horizon_minutes,
        "hold_bars": spec.hold_bars,
    }


def safe_metric(value):
    if value is None:
        return -np.inf
    if not np.isfinite(value):
        return -np.inf
    return float(value)


def robust_gate(selected_rule):
    """A deployable rule must be good AND live inside a local performance plateau."""
    return bool(
        selected_rule.get("passes_deployment_gate", False)
        and selected_rule.get("neighbor_count", 0) >= 3
    )


def candidate_rank(candidate):
    rule = candidate["selected_rule"]
    auc = candidate["tune_classification"].get("roc_auc")
    return (
        1 if candidate["passes_gate"] else 0,
        safe_metric(rule.get("stability_return")),
        safe_metric(rule["cost"].get("return")),
        safe_metric(rule["cost"].get("profit_factor")),
        safe_metric(auc),
        rule["cost"].get("trades", 0),
    )


def json_clean(value):
    if isinstance(value, dict):
        return {key: json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def print_rule_line(prefix, rule):
    cost = rule["cost"]
    gross = rule["gross"]
    gate = "PASS" if robust_gate(rule) else "FAIL"
    print(
        f"{prefix:<10} {rule['label']:<11} "
        f"cost={cost['return']:+7.2f}% PF={cost['profit_factor']:.2f} "
        f"trades={cost['trades']:4d} avg={cost['avg_trade']:+.3f}% "
        f"stable={rule['stability_return']:+6.2f}% "
        f"gross={gross['return']:+7.2f}% | gate={gate}"
    )


def run_fold(dataset, fold):
    name = fold["name"]
    train_years = fold["train_years"]
    tune_year = fold["tune_year"]
    test_year = fold["test_year"]

    train = ordered(dataset.loc[dataset["year"].isin(train_years)].copy())
    tune = ordered(dataset.loc[dataset["year"] == tune_year].copy())
    test = ordered(dataset.loc[dataset["year"] == test_year].copy())

    tune_frames = load_execution_frames(tune_year)
    test_frames = load_execution_frames(test_year)

    print("\n" + "=" * 108)
    print(f"{name} | MODEL TRAIN {train_years} -> TUNE {tune_year} -> FROZEN TEST {test_year}")
    print("=" * 108)
    print(f"Train rows: {len(train):,} | Tune rows: {len(tune):,} | Test rows: {len(test):,}")

    candidates = []

    for spec_index, spec in enumerate(MODEL_SPECS, start=1):
        print(
            f"\n[{spec_index}/{len(MODEL_SPECS)}] {spec.name}: "
            f"{spec.horizon_minutes}m net trade return >= {spec.positive_threshold * 100:+.2f}%"
        )

        y_train_preview = make_label(train, spec)
        positives = int(y_train_preview.sum())
        negatives = int(len(y_train_preview) - positives)
        if positives < MIN_CLASS_EXAMPLES or negatives < MIN_CLASS_EXAMPLES:
            print(
                f"  SKIP: insufficient class support "
                f"({positives:,} positive / {negatives:,} negative)."
            )
            continue

        model, y_train = fit_model(train, spec, random_state=42)

        tune_predictions = predict_probabilities(model, tune)
        y_tune = make_label(tune, spec)
        tune_classification = classification_metrics(
            y_tune,
            tune_predictions["probability"],
        )

        print(
            f"  class rates: train={y_train.mean() * 100:5.2f}% "
            f"tune={y_tune.mean() * 100:5.2f}% | "
            f"tune AUC={tune_classification['roc_auc']:.4f} "
            f"AP={tune_classification['average_precision']:.4f}"
        )

        rule_results = []
        for rule in RESEARCH_RULES:
            rule_results.append(
                evaluate_rule(
                    rule,
                    tune_predictions,
                    tune_frames,
                    spec.hold_bars,
                )
            )

        selected_rule, scored_rules = select_execution_rule(rule_results)
        selected_rule["passes_deployment_gate"] = robust_gate(selected_rule)

        print_rule_line("  selected", selected_rule)

        # Predict the test year now, but DO NOT inspect test labels or trading
        # results until after every model/target candidate has been selected on
        # the tuning year. Prediction itself does not influence selection.
        test_predictions = predict_probabilities(model, test)

        candidates.append(
            {
                "spec": spec,
                "model": model,
                "train_positive_rate": float(y_train.mean()),
                "tune_classification": tune_classification,
                "selected_rule": selected_rule,
                "all_tune_rules": scored_rules,
                "passes_gate": bool(selected_rule["passes_deployment_gate"]),
                "tune_predictions": tune_predictions,
                "test_predictions": test_predictions,
            }
        )

    if not candidates:
        raise RuntimeError(f"{name}: every model specification was skipped.")

    selected = max(candidates, key=candidate_rank)
    spec = selected["spec"]
    rule_data = selected["selected_rule"]
    frozen_rule = ExecutionRule(rule_data["mode"], float(rule_data["value"]))

    print("\n" + "-" * 108)
    print(f"{name} FROZEN SELECTION FROM {tune_year} ONLY")
    print("-" * 108)
    print(
        f"Model target: {spec.name} | {spec.horizon_minutes}m net >= "
        f"{spec.positive_threshold * 100:+.2f}%"
    )
    print(f"Execution:    {frozen_rule.label} | hold {spec.horizon_minutes} minutes")
    print(
        f"Tune result:  {rule_data['cost']['return']:+.2f}% after costs | "
        f"PF {rule_data['cost']['profit_factor']:.2f} | "
        f"{rule_data['cost']['trades']} trades | "
        f"local stability {rule_data['stability_return']:+.2f}%"
    )
    print(
        "Process decision: "
        + ("DEPLOY" if selected["passes_gate"] else "NO-TRADE (research shadow test only)")
    )

    seed_predictions = (
        selected["tune_predictions"] if frozen_rule.mode == "adaptive" else None
    )
    test_result = evaluate_rule(
        frozen_rule,
        selected["test_predictions"],
        test_frames,
        spec.hold_bars,
        seed_predictions=seed_predictions,
    )

    y_test = make_label(test, spec)
    test_classification = classification_metrics(
        y_test,
        selected["test_predictions"]["probability"],
    )

    cost = test_result["cost"]
    gross = test_result["gross"]
    deployment_return = cost["return"] if selected["passes_gate"] else 0.0

    print("\nFROZEN TEST RESULT")
    print(
        f"{test_year}: cost={cost['return']:+.2f}% PF={cost['profit_factor']:.2f} "
        f"trades={cost['trades']} win={cost['win_rate']:.1f}% "
        f"avg={cost['avg_trade']:+.3f}% DD={cost['max_drawdown']:.2f}% "
        f"Sharpe={cost['sharpe']:.2f}"
    )
    print(
        f"gross={gross['return']:+.2f}% PF={gross['profit_factor']:.2f} | "
        f"test AUC={test_classification['roc_auc']:.4f} "
        f"AP={test_classification['average_precision']:.4f}"
    )
    if not selected["passes_gate"]:
        print(
            f"Deployment-policy return for {test_year}: +0.00% because the "
            f"{tune_year} evidence did not pass the robustness gate."
        )

    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_OUTPUT_DIR / f"{name.lower()}_selected.joblib"
    joblib.dump(selected["model"], model_path)

    candidate_summaries = []
    for candidate in candidates:
        candidate_summaries.append(
            {
                "spec": spec_payload(candidate["spec"]),
                "train_positive_rate": candidate["train_positive_rate"],
                "tune_classification": candidate["tune_classification"],
                "selected_rule": candidate["selected_rule"],
                "passes_gate": candidate["passes_gate"],
                "all_tune_rules": candidate["all_tune_rules"],
            }
        )

    return {
        "fold": name,
        "train_years": train_years,
        "tune_year": tune_year,
        "test_year": test_year,
        "selected_spec": spec_payload(spec),
        "selected_rule": rule_data,
        "passes_gate": selected["passes_gate"],
        "test_classification": test_classification,
        "test_result": test_result,
        "deployment_policy_return": float(deployment_return),
        "selected_model_path": str(model_path),
        "candidate_summaries": candidate_summaries,
    }


def main():
    print("=" * 108)
    print("ML WALK-FORWARD RESEARCH V2")
    print("=" * 108)
    print("Nested chronology, execution-aligned net-return targets, regular-session execution,")
    print("local threshold-stability checks, minimum trade support, and an explicit no-trade gate.")
    print("\nIMPORTANT: 2025 and 2026 have already been examined in earlier experiments.")
    print("This is retrospective PROCESS validation, not a new pristine final test.")
    print("Future/paper data must provide the next truly untouched confirmation.")

    dataset = load_dataset()
    fold_results = [run_fold(dataset, fold) for fold in FOLDS]

    deployment_growth = 1.0
    shadow_growth = 1.0
    deployed_folds = 0
    profitable_deployed_folds = 0

    print("\n" + "=" * 108)
    print("WALK-FORWARD PROCESS SUMMARY")
    print("=" * 108)

    for result in fold_results:
        test_cost = result["test_result"]["cost"]
        shadow_growth *= 1.0 + test_cost["return"] / 100.0
        deployment_growth *= 1.0 + result["deployment_policy_return"] / 100.0

        if result["passes_gate"]:
            deployed_folds += 1
            if test_cost["return"] > 0:
                profitable_deployed_folds += 1

        spec = result["selected_spec"]
        rule = result["selected_rule"]
        decision = "DEPLOY" if result["passes_gate"] else "NO-TRADE"
        print(
            f"{result['fold']} test {result['test_year']} | {decision:<8} | "
            f"{spec['name']:<9} {rule['label']:<11} | "
            f"shadow={test_cost['return']:+6.2f}% "
            f"PF={test_cost['profit_factor']:.2f} trades={test_cost['trades']:3d} | "
            f"policy={result['deployment_policy_return']:+6.2f}%"
        )

    shadow_compounded = (shadow_growth - 1.0) * 100.0
    deployment_compounded = (deployment_growth - 1.0) * 100.0

    print("\nPROCESS METRICS")
    print(f"Folds:                         {len(fold_results)}")
    print(f"Folds passing deploy gate:     {deployed_folds}")
    print(f"Profitable deployed test folds:{profitable_deployed_folds}")
    print(f"Compounded shadow return:      {shadow_compounded:+.2f}%")
    print(f"Compounded policy return:      {deployment_compounded:+.2f}%")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "research_version": "ML-WF-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(DATASET_PATH),
        "methodology": {
            "folds": FOLDS,
            "model_specs": [spec_payload(spec) for spec in MODEL_SPECS],
            "rule_selection": (
                "Select on tuning year only; require >=75 completed trades; rank by median "
                "after-cost return across neighboring thresholds before own return/PF; deploy "
                "only if return>0, PF>=1.05, gross PF>=1.10, avg trade>0, and selected rule "
                "has both local neighbors."
            ),
            "execution": (
                "Signal at current 5m bar; enter next regular-session bar open; exit after "
                "6 bars (30m) or 12 bars (60m); max 3 positions; max 20% portfolio each."
            ),
            "targets": (
                "Model labels use the same next-open to later-open trade path as the simulator "
                "and include configured commission/slippage in trade_net_return targets."
            ),
            "pristine_test_status": (
                "2025/2026 were previously observed; results validate the process retrospectively. "
                "New future/paper data is required for pristine confirmation."
            ),
        },
        "fold_results": fold_results,
        "summary": {
            "folds": len(fold_results),
            "deployed_folds": deployed_folds,
            "profitable_deployed_folds": profitable_deployed_folds,
            "compounded_shadow_return": shadow_compounded,
            "compounded_deployment_policy_return": deployment_compounded,
        },
    }
    RESULTS_PATH.write_text(json.dumps(json_clean(payload), indent=2) + "\n")

    print(f"\nSaved full research report: {RESULTS_PATH}")
    print("Selected fold models saved under: data/research/walk_forward_models/")
    print("\nNext decision should be based on these fold results, not on a single lucky year.")


if __name__ == "__main__":
    main()
