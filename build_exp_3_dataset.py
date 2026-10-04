from __future__ import annotations

import json

import pandas as pd

from exp3_config import (
    DATASET_PATH,
    FEATURE_COLUMNS,
    MODEL_NAME,
    OUTPUT_DIR,
    SYMBOLS,
    TARGET_1H,
    TARGET_2H,
    TARGET_4H,
)
from exp3_data import build_dataset


def pct(value: float) -> str:
    return f"{value * 100:+.3f}%"


def main() -> None:
    print("=" * 104)
    print("EXP-3-M1 - BUILD 30-MINUTE NEURAL-NETWORK DATASET")
    print("=" * 104)
    print("Source: six local 5-minute bars -> one complete 30-minute regular-session bar")
    print("Entry:  next 30-minute bar open")
    print("Main target: 2-hour net return after commission + slippage")
    print("Auxiliary targets: 1-hour and 4-hour net returns")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = build_dataset()
    data.to_parquet(DATASET_PATH)

    print(f"Rows:       {len(data):,}")
    print(f"Features:   {len(FEATURE_COLUMNS)} numeric + {len(SYMBOLS)} symbol indicators")
    print(f"Range:      {data.index.min()} -> {data.index.max()}")
    print(f"Saved:      {DATASET_PATH}")
    print()

    print("BY YEAR")
    for year, group in data.groupby("year"):
        print(
            f"  {int(year)} rows={len(group):>7,} "
            f"2h mean={pct(group[TARGET_2H].mean())} "
            f"2h median={pct(group[TARGET_2H].median())} "
            f"2h positive={(group[TARGET_2H] > 0).mean() * 100:5.1f}%"
        )

    print("\nBY SYMBOL")
    for symbol in SYMBOLS:
        group = data[data["symbol"] == symbol]
        print(
            f"  {symbol:<6} rows={len(group):>7,} "
            f"2h mean={pct(group[TARGET_2H].mean())} "
            f"positive={(group[TARGET_2H] > 0).mean() * 100:5.1f}%"
        )

    aux_1h = data[TARGET_1H].notna().mean() * 100
    aux_4h = data[TARGET_4H].notna().mean() * 100
    print(f"\nAux target coverage: 1h={aux_1h:.1f}% | 4h={aux_4h:.1f}%")

    metadata = {
        "model": MODEL_NAME,
        "rows": int(len(data)),
        "first_timestamp": str(data.index.min()),
        "last_timestamp": str(data.index.max()),
        "feature_count": len(FEATURE_COLUMNS) + len(SYMBOLS),
        "symbols": SYMBOLS,
        "year_rows": {str(int(k)): int(v) for k, v in data.groupby("year").size().items()},
    }
    (OUTPUT_DIR / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
