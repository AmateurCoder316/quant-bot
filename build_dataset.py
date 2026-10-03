import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from features import FEATURE_COLUMNS, TARGET_COLUMNS, build_feature_frame
from market_data import LOCAL_PROVIDER
from strategy_core import MARKET_TIMEZONE, SYMBOLS


OUTPUT_DIR = Path("data") / "ml"
DATASET_PATH = OUTPUT_DIR / "features.parquet"
METADATA_PATH = OUTPUT_DIR / "dataset_metadata.json"


def split_for_year(year):
    """Fixed chronological split. Never randomize time-series rows."""
    if year <= 2024:
        return "train"
    if year == 2025:
        return "validation"
    return "test"


def regular_session_only(data):
    """Keep 09:30-15:55 America/New_York bars before feature calculation."""
    local_index = data.index.tz_convert(MARKET_TIMEZONE)
    local_minutes = local_index.hour * 60 + local_index.minute
    mask = (local_minutes >= 9 * 60 + 30) & (local_minutes < 16 * 60)
    return data.loc[mask].copy()


def build_symbol_dataset(symbol):
    data = regular_session_only(LOCAL_PROVIDER.history(symbol))
    frame = build_feature_frame(data)

    local_index = frame.index.tz_convert(MARKET_TIMEZONE)
    frame["timestamp"] = frame.index
    frame["symbol"] = symbol
    frame["year"] = local_index.year
    frame["split"] = [split_for_year(year) for year in frame["year"]]

    keep = [
        "timestamp",
        "symbol",
        "year",
        "split",
        *FEATURE_COLUMNS,
        *TARGET_COLUMNS,
    ]
    return frame[keep]


def validate_dataset(dataset):
    if dataset.empty:
        raise ValueError("ML dataset is empty.")

    duplicate_keys = dataset.duplicated(subset=["timestamp", "symbol"]).sum()
    if duplicate_keys:
        raise ValueError(f"Found {duplicate_keys} duplicate timestamp/symbol rows.")

    values = dataset[FEATURE_COLUMNS + TARGET_COLUMNS].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Dataset contains infinite feature/target values.")

    split_counts = dataset["split"].value_counts()
    for required in ("train", "validation", "test"):
        if split_counts.get(required, 0) == 0:
            raise ValueError(f"Chronological split {required!r} has no rows.")


def main():
    print("=" * 88)
    print("STAGE 3 - BUILD MACHINE-LEARNING DATASET")
    print("=" * 88)

    datasets = []
    raw_rows = 0

    for symbol in SYMBOLS:
        print(f"Building {symbol}...")
        frame = build_symbol_dataset(symbol)
        raw_rows += len(frame)

        # MA warmup and final future-horizon bars naturally become NaN.
        # Dropping them here leaves only complete training examples.
        before = len(frame)
        frame = frame.dropna(subset=FEATURE_COLUMNS + TARGET_COLUMNS).copy()
        print(f"  usable rows: {len(frame):,} / {before:,}")
        datasets.append(frame)

    dataset = pd.concat(datasets, ignore_index=True)
    dataset = dataset.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    validate_dataset(dataset)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(DATASET_PATH, index=False, compression="zstd")

    split_counts = dataset["split"].value_counts().to_dict()
    symbol_counts = dataset["symbol"].value_counts().to_dict()

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(DATASET_PATH),
        "symbols": SYMBOLS,
        "rows": int(len(dataset)),
        "raw_regular_session_rows": int(raw_rows),
        "features": FEATURE_COLUMNS,
        "targets": TARGET_COLUMNS,
        "splits": {
            "train": "2023-2024",
            "validation": "2025",
            "test": "2026+",
        },
        "split_rows": {key: int(value) for key, value in split_counts.items()},
        "symbol_rows": {key: int(value) for key, value in symbol_counts.items()},
        "target_policy": "Future targets stay within the same America/New_York trading date.",
        "feature_policy": "Feature columns use current/past regular-session bars only; target columns are not model inputs.",
    }

    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")

    print("\n" + "=" * 88)
    print("DATASET COMPLETE")
    print("=" * 88)
    print(f"Rows:       {len(dataset):,}")
    print(f"Features:   {len(FEATURE_COLUMNS)}")
    print(f"Targets:    {len(TARGET_COLUMNS)}")
    print(f"Train:      {split_counts.get('train', 0):,}")
    print(f"Validation: {split_counts.get('validation', 0):,}")
    print(f"Test:       {split_counts.get('test', 0):,}")
    print(f"Saved:      {DATASET_PATH}")
    print(f"Metadata:   {METADATA_PATH}")
    print("\nStage 3 dataset is ready for model training.")


if __name__ == "__main__":
    main()
