from __future__ import annotations

import json
import os
import statistics

import numpy as np
import optuna
import pandas as pd

from exp4_config import (
    BATCH_SIZES, BLOCKS, CV_BLOCK_MONTHS, CV_MIN_TRAIN_ROWS, CV_VALIDATION_YEAR,
    DATASET_PATH, DEFAULT_TRIALS, FROZEN_MARKER_PATH, METADATA_PATH, MODEL_NAME,
    MODEL_PATH, OUTPUT_DIR, SCALER_PATH, SEARCH_RESULTS_PATH, SEARCH_SEED, STUDY_PATH,
    TARGET_4H, TRAIN_YEARS, TUNE_YEAR, WIDTHS,
)
from exp4_model import (
    choose_device, fit_full_model, make_cv_folds, predict_dataframe, save_model_bundle,
    summarize_fold_metrics, tail_metrics, train_one_model,
)


def completed_trials(study):
    return [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]


def choose_winner(study):
    complete = completed_trials(study)
    eligible = [t for t in complete if bool(t.user_attrs.get("eligible", False))]
    pool = eligible or complete
    if not pool:
        raise RuntimeError("No completed EXP-4 trials.")
    return max(pool, key=lambda t: float(t.value)), bool(eligible)


def params_for_trial(trial: optuna.Trial) -> dict:
    return {
        "width": trial.suggest_categorical("width", WIDTHS),
        "blocks": trial.suggest_categorical("blocks", BLOCKS),
        "dropout": trial.suggest_float("dropout", 0.08, 0.32),
        "learning_rate": trial.suggest_float("learning_rate", 5e-5, 1.2e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", BATCH_SIZES),
        "rank_weight": trial.suggest_float("rank_weight", 0.20, 0.50),
    }


def main() -> None:
    if FROZEN_MARKER_PATH.exists():
        raise SystemExit(
            f"{MODEL_NAME} already exposed 2025 and is frozen. Do not retrain this model name."
        )
    if not DATASET_PATH.exists():
        raise SystemExit(f"Missing {DATASET_PATH}. Run python build_exp_4_dataset.py first.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(DATASET_PATH).sort_index()
    train_search = data[data["year"].isin(TRAIN_YEARS)].copy()
    forward_2025 = data[data["year"] == TUNE_YEAR].copy()
    folds = make_cv_folds(train_search, CV_VALIDATION_YEAR, CV_BLOCK_MONTHS, CV_MIN_TRAIN_ROWS)
    if len(folds) < 4:
        raise RuntimeError(f"Only {len(folds)} CV folds built; expected at least 4.")

    device = choose_device()
    trials_requested = int(os.getenv("EXP4_M1_TRIALS", DEFAULT_TRIALS))
    print("=" * 126)
    print("EXP-4-M1 - LARGE-UNIVERSE REGIME-AWARE 4-HOUR NEURAL NETWORK")
    print("=" * 126)
    print("Core changes from EXP-3:")
    print("  - 30 target stocks instead of five")
    print("  - SPY/QQQ/IWM + sector ETF context")
    print("  - cross-sectional breadth, ranks and relative strength")
    print("  - 4-hour main after-cost return target")
    print("  - ranking-aware residual neural network")
    print(f"Train/search rows: {len(train_search):,} | held-out 2025: {len(forward_2025):,}")
    print(f"Purged CV folds: {len(folds)} | device: {device} | trials: {trials_requested}")
    for fold in folds:
        print(f"  {fold['name']:<18} train={len(fold['train_indices']):>7,} validation={len(fold['validation_indices']):>7,}")

    study = optuna.create_study(
        study_name=MODEL_NAME,
        storage=f"sqlite:///{STUDY_PATH.resolve()}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEARCH_SEED),
    )

    def objective(trial: optuna.Trial) -> float:
        params = params_for_trial(trial)
        metrics_by_fold, epochs = [], []
        for number, fold in enumerate(folds, start=1):
            train_fold = train_search.iloc[fold["train_indices"]]
            valid_fold = train_search.iloc[fold["validation_indices"]]
            _, _, _, best_epoch, metrics = train_one_model(
                train_fold, valid_fold, params,
                seed=SEARCH_SEED + trial.number * 100 + number,
                device=device,
            )
            metrics["fold"] = fold["name"]
            metrics_by_fold.append(metrics)
            epochs.append(int(best_epoch))
        summary = summarize_fold_metrics(metrics_by_fold)
        trial.set_user_attr("eligible", bool(summary["eligible"]))
        trial.set_user_attr("summary", summary)
        trial.set_user_attr("fold_metrics", metrics_by_fold)
        trial.set_user_attr("best_epochs", epochs)
        return float(summary["objective"])

    done = len(completed_trials(study))
    remaining = max(0, trials_requested - done)
    print("\n" + "=" * 126)
    print("HYPERPARAMETER SEARCH - 2023/2024 ONLY")
    print("=" * 126)
    print(f"Completed previously: {done} | remaining now: {remaining}")
    if remaining:
        study.optimize(objective, n_trials=remaining, n_jobs=1, gc_after_trial=True)

    winner, training_gate_pass = choose_winner(study)
    params = dict(winner.params)
    summary = dict(winner.user_attrs["summary"])
    epochs = [int(x) for x in winner.user_attrs["best_epochs"]]
    requested_final_epochs = int(round(statistics.median(epochs)))

    print("\n" + "=" * 126)
    print("EXP-4-M1 TRAINING SEARCH WINNER")
    print("=" * 126)
    print(f"Trial:                    {winner.number}")
    print(f"Training economic gate:   {'PASS' if training_gate_pass else 'FAIL - best diagnostic only'}")
    print(f"Objective:                {float(winner.value):+.4f}")
    print(f"Mean score-vs-return:     {summary['mean_spearman']:+.4f}")
    print(f"Top 5% avg net:           {summary['mean_top_primary']*100:+.3f}%")
    print(f"Top 2% avg net:           {summary['mean_top_secondary']*100:+.3f}%")
    print(f"Worst fold top 5%:        {summary['worst_top_primary']*100:+.3f}%")
    print(f"Positive top-5% folds:    {summary['positive_fold_fraction']*100:.1f}%")
    print(f"Best epochs by fold:      {epochs}")
    print(f"Requested final epochs:   {requested_final_epochs}")
    print(f"Width / residual blocks:  {params['width']} / {params['blocks']}")
    print(f"Dropout:                  {params['dropout']:.3f}")
    print(f"Learning rate:            {params['learning_rate']:.7g}")
    print(f"Weight decay:             {params['weight_decay']:.7g}")
    print(f"Batch size:               {params['batch_size']}")
    print(f"Ranking-loss weight:      {params['rank_weight']:.3f}")

    model, scaler, target_scaler, final_epochs = fit_full_model(
        train_search, params, requested_final_epochs,
        seed=SEARCH_SEED + 99999, device=device,
    )
    parameter_count = sum(p.numel() for p in model.parameters())
    metadata = {
        "model_name": MODEL_NAME, "train_years": TRAIN_YEARS, "tune_year": TUNE_YEAR,
        "winner_trial": int(winner.number), "training_gate_pass": bool(training_gate_pass),
        "training_summary": summary, "final_epochs": int(final_epochs),
        "parameter_count": int(parameter_count), "device_used_for_training": str(device),
    }
    save_model_bundle(model, scaler, target_scaler, params, MODEL_PATH, SCALER_PATH, METADATA_PATH, metadata)
    SEARCH_RESULTS_PATH.write_text(json.dumps({
        "requested_trials": trials_requested, "completed_trials": len(completed_trials(study)),
        "winner_trial": int(winner.number), "training_gate_pass": bool(training_gate_pass),
        "winner_params": params, "winner_summary": summary,
        "winner_fold_metrics": winner.user_attrs["fold_metrics"], "winner_best_epochs": epochs,
    }, indent=2), encoding="utf-8")

    print(f"\nFrozen model parameters: {parameter_count:,}")
    print(f"Final full-data epochs:  {final_epochs}")

    predictions = predict_dataframe(model, scaler, target_scaler, forward_2025, device)
    predictions.to_parquet(OUTPUT_DIR / "predictions_2025.parquet")
    metrics = tail_metrics(predictions["pred_net_4h"].to_numpy(), predictions[TARGET_4H].to_numpy())
    print("\n" + "=" * 126)
    print("FIRST FORWARD MODEL EVALUATION - 2025")
    print("=" * 126)
    print(f"Rows:                    {metrics['rows']:,}")
    print(f"Score-vs-return rank:    {metrics['spearman']:+.4f}")
    print(f"Mean absolute error:     {metrics['mae']*100:.3f}%")
    print(f"Top 5%: avg_net={metrics['top_primary_mean']*100:+.3f}% median={metrics['top_primary_median']*100:+.3f}% positive={metrics['top_primary_positive']*100:.1f}%")
    print(f"Top 2%: avg_net={metrics['top_secondary_mean']*100:+.3f}% median={metrics['top_secondary_median']*100:+.3f}% positive={metrics['top_secondary_positive']*100:.1f}%")
    print("\n2025 predicted 4h net-return distribution:")
    for percentile in (50, 75, 90, 95, 97.5, 99):
        print(f"  p{percentile:>4}: {np.percentile(predictions['pred_net_4h'], percentile)*100:+.3f}%")

    FROZEN_MARKER_PATH.write_text(
        "EXP-4-M1 exposed 2025. Architecture, features, loss and hyperparameters are frozen.\n",
        encoding="utf-8",
    )
    print("\n" + "=" * 126)
    print("EXP-4-M1 MODEL FROZEN")
    print("=" * 126)
    print("Next: python tune_exp_4_m1.py")


if __name__ == "__main__":
    main()
