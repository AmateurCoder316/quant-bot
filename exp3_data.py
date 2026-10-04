from __future__ import annotations

from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

from exp3_config import (
    AUX_1H_BARS,
    AUX_4H_BARS,
    COMMISSION_RATE,
    FEATURE_COLUMNS,
    INPUT_BARS_PER_OUTPUT_BAR,
    MAIN_HORIZON_BARS,
    MARKET_TIMEZONE,
    MODEL_FEATURE_COLUMNS,
    SLIPPAGE_RATE,
    SOURCE_5M_DIR,
    SYMBOLS,
    SYMBOL_FEATURE_COLUMNS,
    TARGET_1H,
    TARGET_2H,
    TARGET_4H,
)


OHLCV = ["Open", "High", "Low", "Close", "Volume"]
SESSION_OPEN_MINUTE = 9 * 60 + 30
SESSION_BARS = 13


def load_5m(symbol: str, root: Path = SOURCE_5M_DIR) -> pd.DataFrame:
    path = Path(root) / f"{symbol.upper()}.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing 5-minute data for {symbol}: {path}")

    frame = pd.read_csv(path, parse_dates=["Timestamp"]).set_index("Timestamp")
    missing = [column for column in OHLCV if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    frame = frame[OHLCV].copy()
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame = frame[~frame.index.duplicated(keep="last")].sort_index().dropna()
    return frame


def resample_30m(frame_5m: pd.DataFrame) -> pd.DataFrame:
    """Build complete regular-session 30-minute candles from six 5-minute bars."""
    local = frame_5m.copy()
    local.index = local.index.tz_convert(MARKET_TIMEZONE)

    minutes = local.index.hour * 60 + local.index.minute
    in_session = (
        (local.index.weekday < 5)
        & (minutes >= SESSION_OPEN_MINUTE)
        & (minutes < 16 * 60)
    )
    local = local.loc[in_session].copy()
    if local.empty:
        return pd.DataFrame(columns=OHLCV)

    local["session_date"] = pd.Index(local.index.date)
    local["minute_of_day"] = local.index.hour * 60 + local.index.minute
    local["bar_in_session"] = (
        (local["minute_of_day"] - SESSION_OPEN_MINUTE) // 30
    ).astype(int)

    rows: list[dict] = []
    for (session_date, bar_number), group in local.groupby(
        ["session_date", "bar_in_session"], sort=True
    ):
        group = group.sort_index()
        if bar_number < 0 or bar_number >= SESSION_BARS:
            continue
        if len(group) != INPUT_BARS_PER_OUTPUT_BAR:
            continue

        rows.append(
            {
                "Timestamp": group.index[0].tz_convert("UTC"),
                "Open": float(group["Open"].iloc[0]),
                "High": float(group["High"].max()),
                "Low": float(group["Low"].min()),
                "Close": float(group["Close"].iloc[-1]),
                "Volume": float(group["Volume"].sum()),
                "session_date": str(session_date),
                "bar_in_session": int(bar_number),
            }
        )

    if not rows:
        return pd.DataFrame(columns=OHLCV)

    frame = pd.DataFrame(rows).set_index("Timestamp").sort_index()
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame


def _atr(data: pd.DataFrame, window: int) -> pd.Series:
    previous_close = data["Close"].shift(1)
    true_range = pd.concat(
        [
            data["High"] - data["Low"],
            (data["High"] - previous_close).abs(),
            (data["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def _rolling_location(close: pd.Series, low: pd.Series, high: pd.Series, window: int) -> pd.Series:
    rolling_low = low.rolling(window).min()
    rolling_high = high.rolling(window).max()
    width = (rolling_high - rolling_low).replace(0, np.nan)
    return (close - rolling_low) / width


def add_symbol_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    close = data["Close"]
    open_ = data["Open"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    for bars in (1, 2, 4, 8, 13, 26):
        data[f"ret_{bars}"] = close.pct_change(bars)

    previous_close = close.shift(1)
    safe_close = close.replace(0, np.nan)
    candle_high_body = pd.concat([open_, close], axis=1).max(axis=1)
    candle_low_body = pd.concat([open_, close], axis=1).min(axis=1)

    data["range_pct"] = (high - low) / safe_close
    data["body_pct"] = (close - open_) / open_.replace(0, np.nan)
    data["upper_wick_pct"] = (high - candle_high_body) / safe_close
    data["lower_wick_pct"] = (candle_low_body - low) / safe_close
    data["gap_pct"] = open_ / previous_close.replace(0, np.nan) - 1.0

    emas = {}
    for span in (4, 8, 13, 26, 52):
        emas[span] = close.ewm(span=span, adjust=False, min_periods=span).mean()
        data[f"ema{span}_gap"] = close / emas[span] - 1.0

    data["ema4_13_spread"] = emas[4] / emas[13] - 1.0
    data["ema13_26_spread"] = emas[13] / emas[26] - 1.0
    data["ema26_52_spread"] = emas[26] / emas[52] - 1.0
    for span in (8, 13, 26):
        data[f"ema{span}_slope4"] = emas[span] / emas[span].shift(4) - 1.0

    atr14 = _atr(data, 14)
    atr26 = _atr(data, 26)
    data["atr14_pct"] = atr14 / safe_close
    data["atr26_pct"] = atr26 / safe_close

    ret1 = data["ret_1"]
    for window in (4, 8, 13, 26):
        data[f"rv{window}"] = ret1.rolling(window).std(ddof=0)

    candle_range = (high - low).replace(0, np.nan)
    data["range_ratio13"] = candle_range / candle_range.rolling(13).mean()
    data["range_ratio26"] = candle_range / candle_range.rolling(26).mean()

    for window in (4, 13, 26):
        data[f"volume_ratio{window}"] = volume / volume.rolling(window).mean()
    data["volume_change"] = volume.pct_change()
    volume_mean26 = volume.rolling(26).mean()
    volume_std26 = volume.rolling(26).std(ddof=0).replace(0, np.nan)
    data["volume_z26"] = (volume - volume_mean26) / volume_std26

    for window in (13, 26):
        rolling_high = high.rolling(window).max()
        rolling_low = low.rolling(window).min()
        data[f"dist_high{window}_atr"] = (close - rolling_high) / atr14.replace(0, np.nan)
        data[f"dist_low{window}_atr"] = (close - rolling_low) / atr14.replace(0, np.nan)
        data[f"close_location{window}"] = _rolling_location(close, low, high, window)

    session_key = data["session_date"]
    session_open = data.groupby(session_key)["Open"].transform("first")
    data["session_return"] = close / session_open - 1.0

    session_closes = data.groupby("session_date")["Close"].last()
    previous_session_close = session_closes.shift(1)
    session_gaps = (
        data.groupby("session_date")["Open"].first() / previous_session_close - 1.0
    )
    data["day_gap"] = session_key.map(session_gaps)

    progress = data["bar_in_session"].astype(float) / (SESSION_BARS - 1)
    data["session_progress"] = progress
    data["session_sin"] = np.sin(2.0 * np.pi * progress)
    data["session_cos"] = np.cos(2.0 * np.pi * progress)
    data["bars_to_close"] = (SESSION_BARS - 1 - data["bar_in_session"]).astype(float)

    return data


def add_cross_symbol_features(combined: pd.DataFrame) -> pd.DataFrame:
    data = combined.reset_index().rename(columns={"index": "Timestamp"})
    grouped = data.groupby("Timestamp", sort=False)

    for bars in (1, 2, 4, 8):
        data[f"basket_ret{bars}"] = grouped[f"ret_{bars}"].transform("mean")

    data["breadth_up1"] = grouped["ret_1"].transform(lambda s: (s > 0).mean())
    data["breadth_up4"] = grouped["ret_4"].transform(lambda s: (s > 0).mean())
    data["basket_volatility"] = grouped["ret_1"].transform(lambda s: s.std(ddof=0))

    data["relative_ret1"] = data["ret_1"] - data["basket_ret1"]
    data["relative_ret4"] = data["ret_4"] - data["basket_ret4"]
    data["relative_ret8"] = data["ret_8"] - data["basket_ret8"]

    data["rank_ret1"] = grouped["ret_1"].rank(pct=True)
    data["rank_ret4"] = grouped["ret_4"].rank(pct=True)
    data["rank_volume13"] = grouped["volume_ratio13"].rank(pct=True)

    return data.set_index("Timestamp").sort_index()


def _net_return(entry_open: pd.Series, exit_open: pd.Series) -> pd.Series:
    entry_cash = entry_open * (1.0 + SLIPPAGE_RATE) * (1.0 + COMMISSION_RATE)
    exit_cash = exit_open * (1.0 - SLIPPAGE_RATE) * (1.0 - COMMISSION_RATE)
    return exit_cash / entry_cash - 1.0


def add_targets(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy().sort_index()

    entry_open = data["Open"].shift(-1)
    entry_time = pd.Series(data.index, index=data.index).shift(-1)
    entry_session = data["session_date"].shift(-1)

    data["entry_open"] = entry_open
    data["entry_time"] = entry_time

    horizon_specs = [
        (AUX_1H_BARS, TARGET_1H, "1h"),
        (MAIN_HORIZON_BARS, TARGET_2H, "2h"),
        (AUX_4H_BARS, TARGET_4H, "4h"),
    ]

    for horizon, target_name, suffix in horizon_specs:
        exit_shift = -(1 + horizon)
        exit_open = data["Open"].shift(exit_shift)
        exit_time = pd.Series(data.index, index=data.index).shift(exit_shift)
        exit_session = data["session_date"].shift(exit_shift)
        valid = entry_session.eq(exit_session) & entry_open.notna() & exit_open.notna()

        data[target_name] = _net_return(entry_open, exit_open).where(valid)
        data[f"target_gross_{suffix}"] = (exit_open / entry_open - 1.0).where(valid)
        data[f"exit_open_{suffix}"] = exit_open.where(valid)
        data[f"exit_time_{suffix}"] = exit_time.where(valid)

    data["label_end_time"] = data["exit_time_4h"].where(
        data[TARGET_4H].notna(), data["exit_time_2h"]
    )
    return data


def build_dataset() -> pd.DataFrame:
    symbol_frames = []

    for symbol in SYMBOLS:
        bars_5m = load_5m(symbol)
        bars_30m = resample_30m(bars_5m)
        featured = add_symbol_features(bars_30m)
        featured["symbol"] = symbol
        symbol_frames.append(featured)

    combined = pd.concat(symbol_frames, axis=0).sort_index()
    combined = add_cross_symbol_features(combined)

    labeled_frames = []
    for symbol in SYMBOLS:
        symbol_data = combined[combined["symbol"] == symbol].copy()
        labeled_frames.append(add_targets(symbol_data))
    data = pd.concat(labeled_frames, axis=0).sort_index()

    for symbol in SYMBOLS:
        data[f"symbol_{symbol}"] = (data["symbol"] == symbol).astype(float)

    required = FEATURE_COLUMNS + [TARGET_2H, "entry_open", "entry_time", "exit_time_2h"]
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=required).copy()

    for column in SYMBOL_FEATURE_COLUMNS:
        data[column] = data[column].astype(float)

    missing_model_features = [c for c in MODEL_FEATURE_COLUMNS if c not in data.columns]
    if missing_model_features:
        raise RuntimeError(f"Missing model features after build: {missing_model_features}")

    data["year"] = pd.DatetimeIndex(data.index).tz_convert(MARKET_TIMEZONE).year
    data["month"] = pd.DatetimeIndex(data.index).tz_convert(MARKET_TIMEZONE).month
    return data.sort_index()
