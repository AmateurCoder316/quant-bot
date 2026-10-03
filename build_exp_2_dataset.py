from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from exp2_config import (
    BASE_DATASET_PATH,
    CUSUM_MIN_THRESHOLD,
    CUSUM_VOL_MULTIPLIER,
    ECONOMIC_NET_THRESHOLD,
    EVENT_DATASET_PATH,
    EXP2_DERIVED_FEATURE_COLUMNS,
    EXP2_FEATURE_COLUMNS,
    HORIZON_BARS,
    HORIZON_MINUTES,
    MODEL_NAME,
    OUTPUT_DIR,
    PROFIT_ATR_MULTIPLIER,
    PROFIT_CAP,
    PROFIT_FLOOR,
    STOP_ATR_MULTIPLIER,
    STOP_CAP,
    STOP_FLOOR,
)
from market_data import LOCAL_PROVIDER
from strategy_core import COMMISSION_RATE, MARKET_TIMEZONE, SLIPPAGE_RATE, SYMBOLS


EPS = 1e-8


def regular_session_only(frame):
    result = frame.copy().sort_index()
    local = result.index.tz_convert(MARKET_TIMEZONE)
    minutes = local.hour * 60 + local.minute
    mask = (minutes >= 9 * 60 + 30) & (minutes < 16 * 60)
    return result.loc[mask].copy()


def load_base_dataset():
    if not BASE_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Missing {BASE_DATASET_PATH}. Run python build_dataset.py first."
        )

    data = pd.read_parquet(BASE_DATASET_PATH)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data["year"] = data["year"].astype(int)
    return data.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def add_exp2_features(data):
    frame = data.copy()

    vol = frame["rolling_vol_24"].abs().clip(lower=EPS)
    frame["ret_1_vol_scaled"] = frame["ret_1"] / vol
    frame["ret_3_vol_scaled"] = frame["ret_3"] / (vol * np.sqrt(3.0))
    frame["ret_6_vol_scaled"] = frame["ret_6"] / (vol * np.sqrt(6.0))
    frame["ret_12_vol_scaled"] = frame["ret_12"] / (vol * np.sqrt(12.0))

    frame["vol_ratio_12_24"] = frame["rolling_vol_12"] / vol
    atr = frame["atr14_pct"].abs().clip(lower=EPS)
    frame["trend_strength_atr"] = frame["ma50_gap_pct"] / atr
    frame["ma_stack_strength_atr"] = (
        frame["ma20_ma50_pct"] + frame["ma50_ma200_pct"]
    ) / atr
    frame["volume_ratio_spread"] = (
        frame["volume_ratio_20"] / frame["volume_ratio_50"].replace(0.0, np.nan)
        - 1.0
    )
    frame["breakout_pressure"] = (
        frame["dist_high20_atr"]
        + frame["dist_high50_atr"]
        + frame["dist_high80_atr"]
    ) / 3.0
    frame["session_progress"] = (
        frame["minutes_from_open"].clip(lower=0.0, upper=390.0) / 390.0
    )

    grouped = frame.groupby("timestamp", sort=False)
    frame["market_ret_1_mean"] = grouped["ret_1"].transform("mean")
    frame["market_ret_6_mean"] = grouped["ret_6"].transform("mean")
    frame["market_ret_12_mean"] = grouped["ret_12"].transform("mean")
    frame["market_ret_12_dispersion"] = grouped["ret_12"].transform("std").fillna(0.0)

    frame["relative_ret_1"] = frame["ret_1"] - frame["market_ret_1_mean"]
    frame["relative_ret_6"] = frame["ret_6"] - frame["market_ret_6_mean"]
    frame["relative_ret_12"] = frame["ret_12"] - frame["market_ret_12_mean"]

    frame["ret_12_cross_rank"] = grouped["ret_12"].rank(pct=True, method="average")
    frame["trend_cross_rank"] = grouped["ma50_gap_pct"].rank(pct=True, method="average")
    frame["volume_cross_rank"] = grouped["volume_ratio_20"].rank(pct=True, method="average")
    frame["atr_cross_rank"] = grouped["atr14_pct"].rank(pct=True, method="average")

    frame["breadth_ret_1_positive"] = grouped["ret_1"].transform(
        lambda series: float((series > 0.0).mean())
    )
    frame["breadth_above_ma50"] = grouped["ma50_gap_pct"].transform(
        lambda series: float((series > 0.0).mean())
    )
    frame["breadth_above_ma200"] = grouped["ma200_gap_pct"].transform(
        lambda series: float((series > 0.0).mean())
    )

    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    return frame


def _cusum_flags_one_session(group):
    group = group.sort_values("timestamp")
    positive = 0.0
    negative = 0.0
    flags = np.zeros(len(group), dtype=bool)

    returns = group["ret_1"].fillna(0.0).to_numpy(dtype=float)
    volatility = group["rolling_vol_24"].abs().fillna(0.0).to_numpy(dtype=float)

    for i, (ret, sigma) in enumerate(zip(returns, volatility)):
        threshold = max(CUSUM_MIN_THRESHOLD, CUSUM_VOL_MULTIPLIER * sigma)
        positive = max(0.0, positive + ret)
        negative = min(0.0, negative + ret)

        if positive >= threshold or negative <= -threshold:
            flags[i] = True
            positive = 0.0
            negative = 0.0

    return pd.Series(flags, index=group.index)


def add_cusum_events(data):
    frame = data.copy()
    local = frame["timestamp"].dt.tz_convert(MARKET_TIMEZONE)
    frame["session_date"] = local.dt.date
    frame["is_event"] = False

    for _, indexes in frame.groupby(["symbol", "session_date"], sort=False).groups.items():
        indexes = list(indexes)
        flags = _cusum_flags_one_session(frame.loc[indexes])
        frame.loc[flags.index, "is_event"] = flags.astype(bool)

    return frame


def _net_return(entry_raw, exit_raw):
    numerator = exit_raw * (1.0 - SLIPPAGE_RATE) * (1.0 - COMMISSION_RATE)
    denominator = entry_raw * (1.0 + SLIPPAGE_RATE) * (1.0 + COMMISSION_RATE)
    return numerator / denominator - 1.0


def build_symbol_events(symbol, candidates, raw_frame):
    raw = regular_session_only(raw_frame)
    positions = {timestamp: i for i, timestamp in enumerate(raw.index)}
    rows = []

    for row in candidates.itertuples(index=False):
        signal_time = row.timestamp
        signal_pos = positions.get(signal_time)
        if signal_pos is None:
            continue

        entry_pos = signal_pos + 1
        end_pos = entry_pos + HORIZON_BARS - 1
        if entry_pos >= len(raw) or end_pos >= len(raw):
            continue

        entry_time = raw.index[entry_pos]
        end_time = raw.index[end_pos]
        if entry_time.tz_convert(MARKET_TIMEZONE).date() != end_time.tz_convert(MARKET_TIMEZONE).date():
            continue

        entry_raw = float(raw.iloc[entry_pos]["Open"])
        atr_pct = float(row.atr14_pct)
        if not np.isfinite(entry_raw) or entry_raw <= 0.0 or not np.isfinite(atr_pct) or atr_pct <= 0.0:
            continue

        profit_return = float(
            np.clip(PROFIT_ATR_MULTIPLIER * atr_pct, PROFIT_FLOOR, PROFIT_CAP)
        )
        stop_return = float(
            np.clip(STOP_ATR_MULTIPLIER * atr_pct, STOP_FLOOR, STOP_CAP)
        )

        profit_price = entry_raw * (1.0 + profit_return)
        stop_price = entry_raw * (1.0 - stop_return)

        exit_reason = "time"
        exit_pos = end_pos
        exit_raw = float(raw.iloc[end_pos]["Close"])

        # Conservative OHLC convention: if both horizontal barriers are touched
        # inside the same five-minute candle, assume the stop happened first.
        for pos in range(entry_pos, end_pos + 1):
            high = float(raw.iloc[pos]["High"])
            low = float(raw.iloc[pos]["Low"])
            stop_hit = low <= stop_price
            profit_hit = high >= profit_price

            if stop_hit:
                exit_reason = "stop"
                exit_pos = pos
                exit_raw = stop_price
                break
            if profit_hit:
                exit_reason = "profit"
                exit_pos = pos
                exit_raw = profit_price
                break

        exit_time = raw.index[exit_pos]
        gross_return = exit_raw / entry_raw - 1.0
        net_return = _net_return(entry_raw, exit_raw)

        payload = row._asdict()
        payload.update(
            {
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_raw": entry_raw,
                "exit_raw": float(exit_raw),
                "profit_barrier_return": profit_return,
                "stop_barrier_return": stop_return,
                "exit_reason": exit_reason,
                "entry_pos": int(entry_pos),
                "exit_pos": int(exit_pos),
                "event_duration_bars": int(exit_pos - entry_pos + 1),
                "event_gross_return": float(gross_return),
                "event_net_return": float(net_return),
                "barrier_success": int(exit_reason == "profit"),
                "net_positive": int(net_return > 0.0),
                "economic_success": int(net_return >= ECONOMIC_NET_THRESHOLD),
            }
        )
        rows.append(payload)

    events = pd.DataFrame(rows)
    if events.empty:
        return events

    # Average uniqueness: down-weight events that share most of their outcome path
    # with neighboring labels. This corrects redundancy inside the training set;
    # validation purging is handled separately by the training pipeline.
    concurrency_diff = np.zeros(len(raw) + 1, dtype=np.int32)
    for event in events.itertuples(index=False):
        concurrency_diff[event.entry_pos] += 1
        concurrency_diff[event.exit_pos + 1] -= 1
    concurrency = np.cumsum(concurrency_diff[:-1])

    uniqueness = []
    for event in events.itertuples(index=False):
        active = concurrency[event.entry_pos : event.exit_pos + 1]
        uniqueness.append(float(np.mean(1.0 / np.maximum(active, 1))))
    events["uniqueness"] = uniqueness

    # A mild return-attribution component gives economically meaningful events
    # more influence without allowing a few extreme observations to dominate.
    magnitude = np.sqrt(
        np.clip(np.abs(events["event_net_return"].to_numpy()) / 0.003, 0.50, 3.00)
    )
    events["sample_weight"] = events["uniqueness"].to_numpy() * magnitude
    mean_weight = float(events["sample_weight"].mean())
    if mean_weight > 0.0:
        events["sample_weight"] /= mean_weight

    return events


def validate_events(events):
    if events.empty:
        raise ValueError("EXP-2 event dataset is empty.")

    if events.duplicated(subset=["timestamp", "symbol"]).any():
        raise ValueError("Duplicate EXP-2 timestamp/symbol events found.")

    required_numeric = [
        *EXP2_FEATURE_COLUMNS,
        "event_gross_return",
        "event_net_return",
        "uniqueness",
        "sample_weight",
    ]
    values = events[required_numeric].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("EXP-2 event dataset contains NaN/inf numeric values.")

    for year in (2023, 2024, 2025, 2026):
        if not (events["year"] == year).any():
            raise ValueError(f"EXP-2 event dataset is missing year {year}.")


def main():
    print("=" * 112)
    print("EXP-2-M1 - BUILD EVENT DATASET")
    print("=" * 112)
    print("Event sampling: symmetric volatility-scaled CUSUM")
    print(
        f"Triple barrier: next-open entry | {HORIZON_MINUTES}m max hold | "
        f"profit={PROFIT_ATR_MULTIPLIER:.1f} ATR (floor {PROFIT_FLOOR*100:.2f}%) | "
        f"stop={STOP_ATR_MULTIPLIER:.1f} ATR (floor {STOP_FLOOR*100:.2f}%)"
    )
    print("Same-bar profit+stop ambiguity: STOP FIRST (conservative)")
    print("Training weights: label uniqueness x clipped economic magnitude")

    base = load_base_dataset()
    print(f"\nBase rows: {len(base):,}")

    enriched = add_exp2_features(base)
    before = len(enriched)
    enriched = enriched.dropna(subset=EXP2_FEATURE_COLUMNS).copy()
    print(f"Rows after EXP-2 feature completeness: {len(enriched):,} / {before:,}")

    sampled = add_cusum_events(enriched)
    candidates = sampled.loc[sampled["is_event"]].copy()
    print(f"CUSUM event candidates: {len(candidates):,}")

    event_frames = []
    for symbol in SYMBOLS:
        symbol_candidates = candidates.loc[candidates["symbol"] == symbol].copy()
        print(f"Labeling {symbol}: {len(symbol_candidates):,} candidates...")
        raw = LOCAL_PROVIDER.history(symbol)
        symbol_events = build_symbol_events(symbol, symbol_candidates, raw)
        print(f"  usable path-complete events: {len(symbol_events):,}")
        event_frames.append(symbol_events)

    events = pd.concat(event_frames, ignore_index=True)
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)
    events["entry_time"] = pd.to_datetime(events["entry_time"], utc=True)
    events["exit_time"] = pd.to_datetime(events["exit_time"], utc=True)
    events["year"] = events["year"].astype(int)
    events = events.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    validate_events(events)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    events.to_parquet(EVENT_DATASET_PATH, index=False, compression="zstd")

    print("\n" + "=" * 112)
    print("EXP-2 EVENT DATASET COMPLETE")
    print("=" * 112)
    print(f"Events:              {len(events):,}")
    print(f"Features:            {len(EXP2_FEATURE_COLUMNS)}")
    print(f"Barrier success:     {events['barrier_success'].mean()*100:.2f}%")
    print(f"Net-positive:        {events['net_positive'].mean()*100:.2f}%")
    print(
        f"Economic success >= {ECONOMIC_NET_THRESHOLD*100:.2f}%: "
        f"{events['economic_success'].mean()*100:.2f}%"
    )
    print(f"Average uniqueness:  {events['uniqueness'].mean():.3f}")
    for year in (2023, 2024, 2025, 2026):
        year_rows = events.loc[events['year'] == year]
        print(
            f"{year}: {len(year_rows):6,d} events | "
            f"barrier+ {year_rows['barrier_success'].mean()*100:5.2f}% | "
            f"econ+ {year_rows['economic_success'].mean()*100:5.2f}%"
        )
    print(f"Saved: {EVENT_DATASET_PATH}")

    metadata = {
        "model": MODEL_NAME,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "event_dataset": str(EVENT_DATASET_PATH),
        "events": int(len(events)),
        "features": EXP2_FEATURE_COLUMNS,
        "derived_features": EXP2_DERIVED_FEATURE_COLUMNS,
        "event_sampling": {
            "method": "symmetric CUSUM",
            "vol_multiplier": CUSUM_VOL_MULTIPLIER,
            "minimum_threshold": CUSUM_MIN_THRESHOLD,
        },
        "triple_barrier": {
            "horizon_bars": HORIZON_BARS,
            "horizon_minutes": HORIZON_MINUTES,
            "profit_atr_multiplier": PROFIT_ATR_MULTIPLIER,
            "profit_floor": PROFIT_FLOOR,
            "profit_cap": PROFIT_CAP,
            "stop_atr_multiplier": STOP_ATR_MULTIPLIER,
            "stop_floor": STOP_FLOOR,
            "stop_cap": STOP_CAP,
            "same_bar_policy": "stop_first",
        },
        "economic_net_threshold": ECONOMIC_NET_THRESHOLD,
        "commission_rate_per_side": COMMISSION_RATE,
        "slippage_rate_per_side": SLIPPAGE_RATE,
    }
    (OUTPUT_DIR / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
