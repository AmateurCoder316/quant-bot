import json
from datetime import datetime, timezone

import joblib
import numpy as np

from exp2_config import (
    ECONOMIC_NET_THRESHOLD,
    HORIZON_MINUTES,
    META_OOF_YEAR,
    METADATA_PATH,
    MODEL_NAME,
    MODEL_PATH,
    OUTPUT_DIR,
    TRAIN_YEARS,
    TUNE_YEAR,
)
from exp2_core import (
    BASE_MODEL_SPECS,
    fit_stack,
    load_event_dataset,
    predict_stack,
    stack_metrics,
)


def fmt(value, digits=4):
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def print_metric_block(name, metrics):
    print(
        f"{name:<18} AUC={fmt(metrics['roc_auc'])} "
        f"AP={fmt(metrics['average_precision'])} "
        f"logloss={fmt(metrics['log_loss'])} "
        f"brier={fmt(metrics['brier'])} "
        f"positive={metrics['positive_rate']*100:5.2f}%"
    )


def main():
    print("=" * 118)
    print("EXP-2-M1 - TRAIN STACKED EVENT MODEL")
    print("=" * 118)
    print("Architecture:")
    print("  1) CUSUM information-event sampling")
    print("  2) volatility-scaled 120m triple-barrier outcomes")
    print("  3) overlap/uniqueness-aware sample weights")
    print("  4) diverse base stack: HGB + ExtraTrees + logistic + economic HGB")
    print("  5) expanding monthly PURGED 2024 OOF predictions")
    print("  6) logistic meta-label model trained only on OOF base predictions")
    print("  7) final base heads refit on 2023-2024")
    print(f"Economic meta-label: net trade return >= +{ECONOMIC_NET_THRESHOLD*100:.2f}%")
    print(f"Trade horizon:       up to {HORIZON_MINUTES} minutes")
    print(f"Train years:         {TRAIN_YEARS}")
    print(f"Meta OOF year:       {META_OOF_YEAR}")
    print(f"Execution tune year: {TUNE_YEAR}")
    print("2026 remains excluded from training and execution selection.")

    events = load_event_dataset()
    train = events.loc[events["year"].isin(TRAIN_YEARS)].copy().reset_index(drop=True)
    validation = events.loc[events["year"] == TUNE_YEAR].copy().reset_index(drop=True)

    if train.empty or validation.empty:
        raise ValueError("EXP-2 train or validation events are empty.")

    print("\nDATA")
    print(f"Training events:      {len(train):,}")
    print(f"2025 validation:      {len(validation):,}")
    print(f"Train barrier+ rate:  {train['barrier_success'].mean()*100:.2f}%")
    print(f"Train economic+ rate: {train['economic_success'].mean()*100:.2f}%")
    print(f"Mean uniqueness:      {train['uniqueness'].mean():.3f}")

    print("\nBuilding purged OOF stack and fitting final base models...")
    bundle, oof = fit_stack(train)

    # OOF meta probability must be produced after fitting the meta model. The base
    # columns in `oof` themselves are strictly OOF by construction.
    meta_model = bundle["meta_model"]
    oof_meta_probability = meta_model.predict_proba(
        oof[bundle["meta_input_columns"]]
    )[:, 1]
    oof_for_metrics = oof.copy()
    oof_for_metrics["meta_probability"] = oof_meta_probability

    # stack_metrics expects the final prediction schema; fill only the columns it
    # consumes. Base probabilities already came from purged OOF folds.
    oof_metrics = {
        "meta_economic": None,
        "base": {},
    }
    from exp2_core import classification_metrics

    oof_metrics["meta_economic"] = classification_metrics(
        oof["economic_success"], oof_meta_probability
    )
    for name, target, _, _ in BASE_MODEL_SPECS:
        oof_metrics["base"][name] = classification_metrics(oof[target], oof[name])

    validation_predictions = predict_stack(bundle, validation)
    validation_metrics = stack_metrics(validation, validation_predictions)

    print("\n" + "=" * 118)
    print("PURGED 2024 OOF DIAGNOSTICS")
    print("=" * 118)
    print_metric_block("META economic", oof_metrics["meta_economic"])
    for name, _, _, _ in BASE_MODEL_SPECS:
        print_metric_block(name, oof_metrics["base"][name])

    print("\nOOF folds:")
    for fold in bundle["oof_fold_rows"]:
        print(
            f"  {fold['period']}: train={fold['train_rows']:6,d} "
            f"validation={fold['validation_rows']:5,d}"
        )

    print("\n" + "=" * 118)
    print("2025 FORWARD VALIDATION MODEL DIAGNOSTICS")
    print("=" * 118)
    print_metric_block("META economic", validation_metrics["meta_economic"])
    for name, _, _, _ in BASE_MODEL_SPECS:
        print_metric_block(name, validation_metrics[name])

    print("\nMeta-score distribution:")
    for percentile in (50, 75, 90, 95, 97.5, 99):
        q = validation_predictions["meta_score"].quantile(percentile / 100.0)
        p = validation_predictions["meta_probability"].quantile(percentile / 100.0)
        print(
            f"  p{percentile:>4}: meta_score={q:.4f} meta_probability={p:.4f}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    payload = {
        "model": MODEL_NAME,
        "family": "stacked_event_meta_labeling",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "train_years": TRAIN_YEARS,
        "meta_oof_year": META_OOF_YEAR,
        "execution_tune_year": TUNE_YEAR,
        "horizon_minutes": HORIZON_MINUTES,
        "economic_net_threshold": ECONOMIC_NET_THRESHOLD,
        "train_events": int(len(train)),
        "validation_events": int(len(validation)),
        "base_models": [
            {"name": name, "target": target, "kind": kind, "seed": seed}
            for name, target, kind, seed in BASE_MODEL_SPECS
        ],
        "oof_folds": bundle["oof_fold_rows"],
        "oof_metrics": oof_metrics,
        "validation_metrics": validation_metrics,
        "methodology": {
            "base_training": (
                "Unique/economic sample weights plus class balancing; boosted heads use fixed "
                "iterations with no internal random early-stopping split."
            ),
            "stacking": (
                "2024 expanding monthly out-of-fold base probabilities; training events whose "
                "labels could overlap the next validation month are purged before each fold."
            ),
            "meta_model": (
                "Regularized logistic meta-label classifier trained on OOF base probabilities "
                "plus compact regime/context features."
            ),
        },
    }
    METADATA_PATH.write_text(json.dumps(payload, indent=2) + "\n")

    print("\n" + "=" * 118)
    print("EXP-2-M1 MODEL SAVED")
    print("=" * 118)
    print(f"Model:    {MODEL_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    print("Next: python tune_exp_2_m1.py")


if __name__ == "__main__":
    main()
