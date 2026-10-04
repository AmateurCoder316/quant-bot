from __future__ import annotations

import pandas as pd

from exp4_config import DATASET_PATH, FEATURE_COLUMNS, TARGET_4H, TARGET_SYMBOLS
from exp4_data import build_dataset


def validate_dataset(data: pd.DataFrame) -> None:
    if data.empty:
        raise RuntimeError("EXP-4 dataset is empty")

    reset = data.reset_index().rename(columns={"index": "Timestamp"})
    if reset.duplicated(subset=["Timestamp", "symbol"]).any():
        raise RuntimeError("Duplicate (Timestamp, symbol) rows found in EXP-4 dataset")

    per_session = data.groupby(["symbol", "session_date"]).size()
    max_per_session = int(per_session.max())
    if max_per_session > 4:
        raise RuntimeError(
            f"4-hour same-session target invariant failed: found {max_per_session} valid rows "
            "for one symbol/session; maximum is 4"
        )

    signal_time = pd.DatetimeIndex(data.index)
    entry_time = pd.DatetimeIndex(pd.to_datetime(data["entry_time"], utc=True))
    exit_time = pd.DatetimeIndex(pd.to_datetime(data["exit_time_4h"], utc=True))

    entry_delta = entry_time - signal_time
    if not (entry_delta == pd.Timedelta(minutes=30)).all():
        bad = int((entry_delta != pd.Timedelta(minutes=30)).sum())
        raise RuntimeError(f"Entry-time invariant failed for {bad:,} rows; expected exactly +30 minutes")

    hold_delta = exit_time - entry_time
    if not (hold_delta == pd.Timedelta(hours=4)).all():
        bad = int((hold_delta != pd.Timedelta(hours=4)).sum())
        raise RuntimeError(f"4-hour hold invariant failed for {bad:,} rows; expected exactly +4 hours")

    print(
        "Integrity checks: PASS | unique symbol/timestamps | <=4 signals/session | "
        "entry +30m | exit +4h"
    )


def main() -> None:
    print("=" * 110)
    print("EXP-4-M1 - BUILD LARGE-UNIVERSE REGIME DATASET")
    print("=" * 110)
    print("30 target stocks + SPY/QQQ/IWM + sector ETF context")
    print("Input bars: 30 minutes | main target: 4-hour after-cost return | no overnight labels")
    print()

    data = build_dataset()
    validate_dataset(data)
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
