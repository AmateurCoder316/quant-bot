import numpy as np
import pandas as pd

from strategy_core import MARKET_TIMEZONE, calculate_atr


FEATURE_COLUMNS = [
    "ret_1",
    "ret_3",
    "ret_6",
    "ret_12",
    "range_pct",
    "body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "gap_pct",
    "ma20_gap_pct",
    "ma50_gap_pct",
    "ma200_gap_pct",
    "ma20_ma50_pct",
    "ma50_ma200_pct",
    "ma20_slope_6",
    "ma50_slope_12",
    "ma200_slope_12",
    "atr14_pct",
    "rolling_vol_12",
    "rolling_vol_24",
    "volume_ratio_20",
    "volume_ratio_50",
    "volume_change",
    "dist_high20_atr",
    "dist_high50_atr",
    "dist_high80_atr",
    "dist_low20_atr",
    "minutes_from_open",
    "minutes_to_close",
    "time_sin",
    "time_cos",
]

TARGET_COLUMNS = [
    "future_return_15m",
    "future_return_30m",
    "future_return_60m",
    "future_max_gain_30m",
    "future_max_loss_30m",
]


def _market_local_index(index):
    index = pd.DatetimeIndex(index)
    if index.tz is None:
        index = index.tz_localize("UTC")
    return index.tz_convert(MARKET_TIMEZONE)


def _future_window_stat(series, horizon, mode):
    """Future-only rolling max/min over bars t+1..t+horizon."""
    reversed_series = series.iloc[::-1]
    if mode == "max":
        rolled = reversed_series.rolling(horizon, min_periods=horizon).max()
    elif mode == "min":
        rolled = reversed_series.rolling(horizon, min_periods=horizon).min()
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return rolled.iloc[::-1].shift(-1)


def build_features(data):
    """Create model inputs using only information available at each bar."""
    frame = data.copy().sort_index()
    close = frame["Close"].astype(float)
    high = frame["High"].astype(float)
    low = frame["Low"].astype(float)
    open_ = frame["Open"].astype(float)
    volume = frame["Volume"].astype(float)

    previous_close = close.shift(1)
    candle_top = pd.concat([open_, close], axis=1).max(axis=1)
    candle_bottom = pd.concat([open_, close], axis=1).min(axis=1)

    frame["ret_1"] = close.pct_change(1)
    frame["ret_3"] = close.pct_change(3)
    frame["ret_6"] = close.pct_change(6)
    frame["ret_12"] = close.pct_change(12)

    frame["range_pct"] = (high - low) / close
    frame["body_pct"] = (close - open_) / open_
    frame["upper_wick_pct"] = (high - candle_top) / close
    frame["lower_wick_pct"] = (candle_bottom - low) / close
    frame["gap_pct"] = open_ / previous_close - 1.0

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    frame["ma20_gap_pct"] = close / ma20 - 1.0
    frame["ma50_gap_pct"] = close / ma50 - 1.0
    frame["ma200_gap_pct"] = close / ma200 - 1.0
    frame["ma20_ma50_pct"] = ma20 / ma50 - 1.0
    frame["ma50_ma200_pct"] = ma50 / ma200 - 1.0
    frame["ma20_slope_6"] = ma20 / ma20.shift(6) - 1.0
    frame["ma50_slope_12"] = ma50 / ma50.shift(12) - 1.0
    frame["ma200_slope_12"] = ma200 / ma200.shift(12) - 1.0

    atr14 = calculate_atr(frame, 14)
    frame["atr14_pct"] = atr14 / close

    one_bar_returns = close.pct_change()
    frame["rolling_vol_12"] = one_bar_returns.rolling(12).std()
    frame["rolling_vol_24"] = one_bar_returns.rolling(24).std()

    volume_ma20 = volume.rolling(20).mean()
    volume_ma50 = volume.rolling(50).mean()
    frame["volume_ratio_20"] = volume / volume_ma20
    frame["volume_ratio_50"] = volume / volume_ma50
    frame["volume_change"] = volume.pct_change().replace([np.inf, -np.inf], np.nan)

    for lookback in (20, 50, 80):
        prior_high = high.rolling(lookback).max().shift(1)
        frame[f"dist_high{lookback}_atr"] = (close - prior_high) / atr14

    prior_low20 = low.rolling(20).min().shift(1)
    frame["dist_low20_atr"] = (close - prior_low20) / atr14

    local_index = _market_local_index(frame.index)
    minutes = local_index.hour * 60 + local_index.minute
    market_open_minutes = 9 * 60 + 30
    market_close_minutes = 16 * 60
    session_minutes = market_close_minutes - market_open_minutes

    frame["minutes_from_open"] = minutes - market_open_minutes
    frame["minutes_to_close"] = market_close_minutes - minutes

    angle = 2.0 * np.pi * frame["minutes_from_open"] / session_minutes
    frame["time_sin"] = np.sin(angle)
    frame["time_cos"] = np.cos(angle)

    return frame


def add_future_targets(data):
    """Add future outcomes without allowing targets to cross trading days."""
    frame = data.copy().sort_index()
    local_index = _market_local_index(frame.index)
    frame["session_date"] = local_index.date

    close = frame["Close"].astype(float)

    grouped = frame.groupby("session_date", sort=False)

    for bars, minutes in ((3, 15), (6, 30), (12, 60)):
        future_close = grouped["Close"].shift(-bars)
        frame[f"future_return_{minutes}m"] = future_close / close - 1.0

    future_high_30 = grouped["High"].transform(
        lambda s: _future_window_stat(s, 6, "max")
    )
    future_low_30 = grouped["Low"].transform(
        lambda s: _future_window_stat(s, 6, "min")
    )
    frame["future_max_gain_30m"] = future_high_30 / close - 1.0
    frame["future_max_loss_30m"] = future_low_30 / close - 1.0

    return frame


def build_feature_frame(data):
    """Build both features and training targets for one symbol."""
    return add_future_targets(build_features(data))
