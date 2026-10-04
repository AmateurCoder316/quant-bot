from __future__ import annotations

import json
import os
import statistics

import numpy as np
import optuna
import pandas as pd

from exp3_config import (
    ACTIVATIONS,
    BATCH_SIZES,
    CV_BLOCK_MONTHS,
    CV_MIN_TRAIN_ROWS,
    CV_VALIDATION_YEAR,
    DATASET_PATH,
    DEFAULT_TRIALS,
    FROZEN_MARKER_PATH,
    HIDDEN_LAYOUTS,
    METADATA_PATH,
    MODEL_NAME,
    MODEL_PATH,
    OUTPUT_DIR,
    SCALER_PATH,
    SEARCH_RESULTS_PATH,
    SEARCH_SEED,
    STUDY_PATH,
    TARGET_2H,
    TRAIN_YEARS,
    TUNE_YEAR,
)
from exp3_model import (
    choose_device,
    fit_full_model,
    make_cv_folds,
    predict_dataframe,
    save_model_bundle,
    summarize_fold_metrics,
    tail_metrics,
    train_one_model,
)


LAYOUT_MAP = {"-".join(map(str, layout)): layout for layout in HIDDEN_LAYOUTS}


def trial_params(trial: optuna.Trial) -> dict:
    layout_name = trial.suggest_categorical("hidden_layout", list(LAYOUT_MAP))
    return {
        "hidden_layout": list(LAYOUT_MAP[layout_name]),
        "activation": trial.suggest_categorical("activation", ACTIVATIONS),
        "dropout": trial.suggest_float("dropout", 0.05, 0.35),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", BATCH_SIZES),
    }


def plain_params(optuna_params: dict) -> dict:
    result = dict(optuna_params)
    result["hidden_layout"] = list(LAYOUT_MAP[str(result["hidden_layout"])])
    return result


def completed_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]


def choose_winner(study: optuna.Study) -> tuple[optuna.trial.FrozenTrial, bool]:
    complete = completed_trials(study)
    eligible = [trial for trial in complete if bool(trial.user_attrs.get("eligible", False))]
    if eligible:
        return max(eligible, key=lambda trial: float(trial.value)), True
    if not complete:
        raise RuntimeError("No completed EXP-3 trials are available.")
    return max(complete, key=lambda trial: float(trial.value)), False


def main() -> None:
    if FROZEN_MARKER_PATH.exists():
        raise SystemExit(
            "EXP-3-M1 is already frozen after its first 2025 reveal. "
            "Do not add trials or retrain this model name. Any model change is EXP-3-M2."
        )

    if not DATASET_PATH.exists():
        raise SystemExit(
            f"Missing {DATASET_PATH}. Run: python build_exp_3_dataset.py"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(DATASET_PATH).sort_index()
    train_search = data[data["year"].isin(TRAIN_YEARS)].copy()
    forward_2025 = data[data["year"] == TUNE_YEAR].copy()

    folds = make_cv_folds(
        train_search,
        validation_year=CV_VALIDATION_YEAR,
        block_months=CV_BLOCK_MONTHS,
        min_train_rows=CV_MIN_TRAIN_ROWS,
    )
    if len(folds) < 3:
        raise RuntimeError(f"Only {len(folds)} CV folds were built; expected at least 3.")

    device = choose_device()
    requested_trials = int(os.getenv("EXP3_M1_TRIALS", DEFAULT_TRIALS))

    print("=" * 118)
    print("EXP-3-M1 - 30-MINUTE NEURAL NETWORK TRAINING")
    print("=" * 118)
    print("Goal: predict actual future after-cost return, not a yes/no success label.")
    print(f"Train/search years: {TRAIN_YEARS}")
    print(f"First forward year: {TUNE_YEAR} (not used by model search)")
    print(f"Training/search rows: {len(train_search):,}")
    print(f"2025 held-out rows:   {len(forward_2025):,}")
    print(f"Purged CV folds:      {len(folds)}")
    print(f"Device:               {device}")
    print(f"Search budget:        {requested_trials} total trials")
    print()
    for fold in folds:
        print(
            f"  {fold['name']:<18} train={len(fold['train_indices']):>7,} "
            f"validation={len(fold['validation_indices']):>6,}"
        )

    storage = f"sqlite:///{STUDY_PATH.resolve()}"
    study = optuna.create_study(
        study_name=MODEL_NAME,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEARCH_SEED),
    )

    def objective(trial: optuna.Trial) -> float:
        params = trial_params(trial)
        fold_metrics = []
        best_epochs = []

        for fold_number, fold in enumerate(folds, start=1):
            train_fold = train_search.iloc[fold["train_indices"]]
            valid_fold = train_search.iloc[fold["validation_indices"]]
            _, _, _, best_epoch, metrics = train_one_model(
                train_fold,
                valid_fold,
                params,
                seed=SEARCH_SEED + trial.number * 100 + fold_number,
                device=device,
            )
            metrics["fold"] = fold["name"]
            fold_metrics.append(metrics)
            best_epochs.append(int(best_epoch))

        summary = summarize_fold_metrics(fold_metrics)
        trial.set_user_attr("eligible", bool(summary["eligible"]))
        trial.set_user_attr("summary", summary)
        trial.set_user_attr("fold_metrics", fold_metrics)
        trial.set_user_attr("best_epochs", best_epochs)
        return float(summary["objective"])

    already_complete = len(completed_trials(study))
    remaining = max(0, requested_trials - already_complete)
    print("\n" + "=" * 118)
    print("HYPERPARAMETER SEARCH - 2023/2024 ONLY")
    print("=" * 118)
    print(f"Completed previously: {already_complete} | remaining now: {remaining}")
    if remaining:
        study.optimize(objective, n_trials=remaining, n_jobs=1, gc_after_trial=True)

    winner, training_gate_pass = choose_winner(study)
    params = plain_params(winner.params)
    summary = dict(winner.user_attrs["summary"])
    best_epochs = [int(value) for value in winner.user_attrs["best_epochs"]]
    final_epochs = max(1, int(round(statistics.median(best_epochs))))

    print("\n" + "=" * 118)
    print("TRAINING SEARCH WINNER")
    print("=" * 118)
    print(f"Trial:                  {winner.number}")
    print(f"Training economic gate: {'PASS' if training_gate_pass else 'FAIL - best diagnostic only'}")
    print(f"Objective:              {float(winner.value):+.4f}")
    print(f"Mean score-vs-return:   {summary['mean_spearman']:+.4f}")
    print(f"Top 10% avg net:        {summary['mean_top10'] * 100:+.3f}%")
    print(f"Top  5% avg net:        {summary['mean_top05'] * 100:+.3f}%")
    print(f"Worst fold top 10%:     {summary['worst_top10'] * 100:+.3f}%")
    print(f"Final full-data epochs: {final_epochs}")
    print(f"Architecture:           {params['hidden_layout']} {params['activation']} dropout={params['dropout']:.3f}")
    print(f"Learning rate:          {params['learning_rate']:.6g}")
    print(f"Weight decay:           {params['weight_decay']:.6g}")
    print(f"Batch size:             {params['batch_size']}")

    model, feature_scaler, target_scaler = fit_full_model(
        train_search,
        params,
        final_epochs,
        seed=SEARCH_SEED + 99_999,
        device=device,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    metadata = {
        "model_name": MODEL_NAME,
        "train_years": TRAIN_YEARS,
        "tune_year": TUNE_YEAR,
        "winner_trial": int(winner.number),
        "training_gate_pass": bool(training_gate_pass),
        "training_summary": summary,
        "final_epochs": final_epochs,
        "parameter_count": int(parameter_count),
        "device_used_for_training": str(device),
    }
    save_model_bundle(
        model,
        feature_scaler,
        target_scaler,
        params,
        MODEL_PATH,
        SCALER_PATH,
        METADATA_PATH,
        metadata,
    )

    search_export = {
        "requested_trials": requested_trials,
        "completed_trials": len(completed_trials(study)),
        "winner_trial": int(winner.number),
        "training_gate_pass": bool(training_gate_pass),
        "winner_params": params,
        "winner_summary": summary,
        "winner_fold_metrics": winner.user_attrs["fold_metrics"],
        "winner_best_epochs": best_epochs,
    }
    SEARCH_RESULTS_PATH.write_text(json.dumps(search_export, indent=2), encoding="utf-8")

    print(f"\nFrozen model parameters: {parameter_count:,}")
    print(f"Saved model:             {MODEL_PATH}")
    print(f"Saved metadata:          {METADATA_PATH}")
    print(f"Saved search results:    {SEARCH_RESULTS_PATH}")

    # Model and hyperparameters are frozen before this first 2025 look.
    prediction_2025 = predict_dataframe(
        model,
        feature_scaler,
        target_scaler,
        forward_2025,
        device,
    )
    prediction_path = OUTPUT_DIR / "predictions_2025.parquet"
    prediction_2025.to_parquet(prediction_path)
    metrics_2025 = tail_metrics(
        prediction_2025["pred_net_2h"].to_numpy(),
        prediction_2025[TARGET_2H].to_numpy(),
    )

    FROZEN_MARKER_PATH.write_text(
        "EXP-3-M1 architecture, features, targets and hyperparameters froze when 2025 was first revealed.\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 118)
    print("FIRST FORWARD MODEL EVALUATION - 2025")
    print("=" * 118)
    print(f"Rows:                    {metrics_2025['rows']:,}")
    print(f"Score-vs-return rank:    {metrics_2025['spearman']:+.4f}")
    print(f"Mean absolute error:     {metrics_2025['mae'] * 100:.3f}%")
    print(
        f"Top 10%: avg_net={metrics_2025['top10_mean'] * 100:+.3f}% "
        f"median_net={metrics_2025['top10_median'] * 100:+.3f}% "
        f"positive={metrics_2025['top10_positive'] * 100:.1f}%"
    )
    print(
        f"Top  5%: avg_net={metrics_2025['top05_mean'] * 100:+.3f}% "
        f"median_net={metrics_2025['top05_median'] * 100:+.3f}% "
        f"positive={metrics_2025['top05_positive'] * 100:.1f}%"
    )
    print("\n2025 predicted 2h net-return distribution:")
    for percentile in (50, 75, 90, 95, 97.5, 99):
        value = np.percentile(prediction_2025["pred_net_2h"], percentile)
        print(f"  p{percentile:>4}: {value * 100:+.3f}%")

    print("\n" + "=" * 118)
    print("EXP-3-M1 MODEL FROZEN")
    print("=" * 118)
    print("Do not change M1 architecture or hyperparameters based on 2025 results.")
    print("Next: python tune_exp_3_m1.py")


if __name__ == "__main__":
    main()
