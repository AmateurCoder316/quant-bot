import json
from datetime import datetime, timezone

from exp2_m2_config import (
    METADATA_PATH,
    MODEL_FAMILIES,
    MODEL_NAME,
    MODEL_PATH,
    OUTPUT_DIR,
    SEARCH_RESULTS_PATH,
    TARGET_COLUMN,
    TRAIN_YEARS,
    TUNE_YEAR,
)
from exp2_m2_training import (
    build_purged_cv_folds,
    choose_finalists,
    dependency_versions,
    fit_final_bundle,
    generate_finalist_oof,
    load_events,
    predict_bundle,
    prediction_metrics,
    require_training_dependencies,
    run_family_study,
    save_bundle,
    trials_per_family,
    winner_to_dict,
    write_json,
)


def pct(value):
    return f"{value * 100:+.3f}%"


def print_prediction_metrics(title, metrics):
    print("\n" + "=" * 122)
    print(title)
    print("=" * 122)
    print(f"Rows:                   {metrics['rows']:,}")
    print(f"Economic-positive rate: {metrics['positive_rate'] * 100:.2f}%")
    print(f"ROC AUC:                {metrics['roc_auc']:.4f}")
    print(f"Average precision:      {metrics['average_precision']:.4f}")
    print(f"Log loss:               {metrics['log_loss']:.4f}")
    print(f"Brier:                  {metrics['brier']:.4f}")
    print(f"Spearman vs net return: {metrics['spearman_net_return']:.4f}")
    print(
        f"Top 10%: precision={metrics['top10']['precision']*100:.2f}% "
        f"avg_net={pct(metrics['top10']['avg_net_return'])} "
        f"median_net={pct(metrics['top10']['median_net_return'])}"
    )
    print(
        f"Top  5%: precision={metrics['top05']['precision']*100:.2f}% "
        f"avg_net={pct(metrics['top05']['avg_net_return'])} "
        f"median_net={pct(metrics['top05']['median_net_return'])}"
    )


def main():
    require_training_dependencies()
    trial_budget = trials_per_family()
    versions = dependency_versions()

    print("=" * 122)
    print("EXP-2-M2 - PURGED MODEL TRAINING LAB")
    print("=" * 122)
    print("Goal: actually optimize the predictive model INSIDE 2023-2024 before 2025 is revealed.")
    print(f"Target: {TARGET_COLUMN} (EXP-2 economic-success event label)")
    print(f"Train/search years: {TRAIN_YEARS}")
    print("CV: expanding 2024 two-month validation blocks with 120-minute label purge")
    print(f"Search budget: {trial_budget} trials per family x {len(MODEL_FAMILIES)} families")
    print("Selection objective: ranking + AP lift + top-tail economic lift + net-return quality - instability")
    print("2025 is NOT used by hyperparameter search. 2026 is not used anywhere in model selection.")
    print("\nDependencies:")
    for name, version in versions.items():
        print(f"  {name:<10} {version}")

    events = load_events()
    train = events.loc[events["year"].isin(TRAIN_YEARS)].copy().reset_index(drop=True)
    validation = events.loc[events["year"] == TUNE_YEAR].copy().reset_index(drop=True)
    if train.empty or validation.empty:
        raise ValueError("EXP-2-M2 training or 2025 validation events are empty.")

    cv_data, folds = build_purged_cv_folds(train)
    print("\nDATA")
    print(f"Training/search events: {len(train):,}")
    print(f"2025 held-out events:   {len(validation):,}")
    print(f"Train econ+ rate:       {train[TARGET_COLUMN].mean()*100:.2f}%")
    print(f"2025 econ+ rate:        {validation[TARGET_COLUMN].mean()*100:.2f}%")
    print(f"Purged CV folds:        {len(folds)}")
    for fold in folds:
        print(
            f"  {fold.name:<18} train={len(fold.train_indices):6,d} "
            f"validation={len(fold.validation_indices):5,d}"
        )

    print("\n" + "=" * 122)
    print("HYPERPARAMETER SEARCH - 2023/2024 ONLY")
    print("=" * 122)

    winners = []
    for family in MODEL_FAMILIES:
        print(f"\n[{family}] optimizing up to {trial_budget} total trials...")
        winner = run_family_study(cv_data, folds, family, trial_budget)
        winners.append(winner)
        m = winner.metrics
        print(
            f"  BEST objective={winner.score:+.4f} "
            f"AUC={m.get('mean_auc', 0):.4f} minAUC={m.get('min_auc', 0):.4f} "
            f"AP={m.get('mean_average_precision', 0):.4f} "
            f"top10_lift={m.get('mean_top10_lift', 0):.2f}x "
            f"top10_net={pct(m.get('mean_top10_net_return', 0))} "
            f"worst_fold_net={pct(m.get('worst_top10_net_return', 0))}"
        )

    winners = sorted(winners, key=lambda item: item.score, reverse=True)
    finalists = choose_finalists(winners)

    print("\n" + "=" * 122)
    print("MODEL-FAMILY LEADERBOARD")
    print("=" * 122)
    for rank, winner in enumerate(winners, 1):
        marker = "FINALIST" if winner in finalists else ""
        m = winner.metrics
        print(
            f"{rank:>2}. {winner.family:<24} objective={winner.score:+.4f} "
            f"AUC={m.get('mean_auc', 0):.4f} min={m.get('min_auc', 0):.4f} "
            f"AP={m.get('mean_average_precision', 0):.4f} "
            f"tail_net={pct(m.get('mean_top10_net_return', 0))} {marker}"
        )

    print("\nRebuilding strictly OOF predictions for finalists...")
    oof = generate_finalist_oof(cv_data, folds, finalists)
    print(f"OOF blender rows: {len(oof):,}")
    print("Fitting final base models on ALL 2023-2024 and OOF logistic blender...")
    bundle = fit_final_bundle(train, finalists, oof)
    save_bundle(bundle, MODEL_PATH)

    # This is the first time the fully selected M2 model sees 2025. No search or
    # refit is performed from these diagnostics; execution selection is separate.
    validation_predictions = predict_bundle(bundle, validation)
    validation_metrics = prediction_metrics(validation, validation_predictions)
    print_prediction_metrics("FIRST FORWARD MODEL EVALUATION - 2025", validation_metrics)

    print("\n2025 score distribution:")
    for percentile in (50, 75, 90, 95, 97.5, 99):
        value = validation_predictions["meta_score"].quantile(percentile / 100.0)
        print(f"  p{percentile:>4}: {value:.5f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    search_payload = {
        "model": MODEL_NAME,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "trial_budget_per_family": trial_budget,
        "families": [winner_to_dict(item) for item in winners],
        "finalists": [item.family for item in finalists],
        "folds": [
            {
                "name": fold.name,
                "train_rows": int(len(fold.train_indices)),
                "validation_rows": int(len(fold.validation_indices)),
            }
            for fold in folds
        ],
    }
    write_json(SEARCH_RESULTS_PATH, search_payload)

    metadata = {
        "model": MODEL_NAME,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": "purged hyperparameter-optimized heterogeneous ensemble + OOF logistic blender",
        "target": TARGET_COLUMN,
        "train_years": TRAIN_YEARS,
        "first_forward_validation_year": TUNE_YEAR,
        "dependencies": versions,
        "trial_budget_per_family": trial_budget,
        "family_winners": [winner_to_dict(item) for item in winners],
        "finalists": [item.family for item in finalists],
        "validation_2025": validation_metrics,
        "research_policy": {
            "hyperparameter_search": "2023-2024 only using purged expanding chronological CV",
            "family_selection": "robust composite objective; one winner per family",
            "blending": "logistic blender fitted only to finalist out-of-fold probabilities",
            "final_refit": "finalists refit on all 2023-2024 after search",
            "execution_tuning": "2025 only, one causal percentile axis",
            "test": "2026 frozen historical shadow test",
        },
    }
    write_json(METADATA_PATH, metadata)

    print("\n" + "=" * 122)
    print("EXP-2-M2 MODEL FROZEN")
    print("=" * 122)
    print(f"Model:          {MODEL_PATH}")
    print(f"Metadata:       {METADATA_PATH}")
    print(f"Search results: {SEARCH_RESULTS_PATH}")
    print("Do not change M2 hyperparameters based on 2025 results.")
    print("Next: python tune_exp_2_m2.py")


if __name__ == "__main__":
    main()
