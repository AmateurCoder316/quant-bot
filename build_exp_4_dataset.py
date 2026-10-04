from __future__ import annotations

import pandas as pd

from exp4_config import DATASET_PATH, FEATURE_COLUMNS, TARGET_4H, TARGET_SYMBOLS
from exp4_data import build_dataset


def main() -> None:
    print("=" * 110)
    print("EXP-4-M1 - BUILD LARGE-UNIVERSE REGIME DATASET")
    print("=" * 110)
    print("30 target stocks + SPY/QQQ/IWM + sector ETF context")
    print("Input bars: 30 minutes | main target: 4-hour after-cost return | no overnight labels")
    print()

    data = build_dataset()
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(DATASET_PATH)

    print(f"Rows:       {len(data):,}")
    print(f"Features:   {len(FEATURE_COLUMNS)} numeric/regime + {len(TARGET_SYMBOLS)} symbol indicators")
    print(f"Range:      {data.index.min()} -> {data.index.max()}")
    print(f"Saved:      {DATASET_PATH}")
    print("\nBY YEAR")
    for year, group in data.groupby("year"):
        target = group[TARGET_4H]
        print(
            f"  {year} rows={len(group):>7,} 4h mean={target.mean()*100:+.3f}% "
            f"median={target.median()*100:+.3f}% positive={(target>0).mean()*100:5.1f}%"
        )
    print("\nBY SYMBOL")
    for symbol in TARGET_SYMBOLS:
        group = data[data["symbol"] == symbol]
        if group.empty:
            continue
        target = group[TARGET_4H]
        print(
            f"  {symbol:<5} rows={len(group):>6,} mean={target.mean()*100:+.3f}% "
            f"positive={(target>0).mean()*100:5.1f}%"
        )


if __name__ == "__main__":
    main()
