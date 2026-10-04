from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from exp4_config import (
    AUX_2H_BARS,
    BASE_FEATURE_COLUMNS,
    COMMISSION_RATE,
    CONTEXT_SYMBOLS,
    FEATURE_COLUMNS,
    INPUT_BARS_PER_OUTPUT_BAR,
    LEGACY_5M_DIR,
    MAIN_HORIZON_BARS,
    MARKET_TIMEZONE,
    MODEL_FEATURE_COLUMNS,
    SECTOR_ETF_BY_SYMBOL,
    SLIPPAGE_RATE,
    SOURCE_5M_DIR,
    SYMBOL_FEATURE_COLUMNS,
    TARGET_2H,
    TARGET_4H,
    TARGET_SYMBOLS,
)

OHLCV = ["Open", "High", "Low", "Close", "Volume"]
SESSION_OPEN_MINUTE = 9 * 60 + 30
SESSION_BARS = 13


def source_path(symbol: str) -> Path:
    primary = SOURCE_5M_DIR / f"{symbol}.csv.gz"
    if primary.exists():
        return primary
    legacy = LEGACY_5M_DIR / f"{symbol}.csv.gz"
    if legacy.exists():
        return legacy
    raise FileNotFoundError(f"Missing EXP-4 source data for {symbol}. Run download_exp_4_data.py")


def load_5m(symbol: str) -> pd.DataFrame:
    path = source_path(symbol)
    frame = pd.read_csv(path, parse_dates=["Timestamp"]).set_index("Timestamp")
    missing = [column for column in OHLCV if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    frame = frame[OHLCV].copy()
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame[~frame.index.duplicated(keep="last")].sort_index().dropna()


def resample_30m(frame_5m: pd.DataFrame) -> pd.DataFrame:
    local = frame_5m.copy()
    local.index = local.index.tz_convert(MARKET_TIMEZONE)
    minutes = local.index.hour * 60 + local.index.minute
    mask = (local.index.weekday < 5) & (minutes >= SESSION_OPEN_MINUTE) & (minutes < 16 * 60)
    local = local.loc[mask].copy()
    local["session_date"] = pd.Index(local.index.date)
    local["minute_of_day"] = local.index.hour * 60 + local.index.minute
    local["bar_in_session"] = ((local["minute_of_day"] - SESSION_OPEN_MINUTE) // 30).astype(int)

    rows = []
    for (session_date, bar_number), group in local.groupby(["session_date", "bar_in_session"], sort=True):
        group = group.sort_index()
        if not 0 <= int(bar_number) < SESSION_BARS or len(group) != INPUT_BARS_PER_OUTPUT_BAR:
            continue
        rows.append({
            "Timestamp": group.index[0].tz_convert("UTC"),
            "Open": float(group["Open"].iloc[0]),
            "High": float(group["High"].max()),
            "Low": float(group["Low"].min()),
            "Close": float(group["Close"].iloc[-1]),
            "Volume": float(group["Volume"].sum()),
            "session_date": str(session_date),
            "bar_in_session": int(bar_number),
        })
    if not rows:
        return pd.DataFrame(columns=OHLCV)
    result = pd.DataFrame(rows).set_index("Timestamp").sort_index()
    result.index = pd.to_datetime(result.index, utc=True)
    return result


def _atr(data: pd.DataFrame, window: int) -> pd.Series:
    previous_close = data["Close"].shift(1)
    tr = pd.concat([
        data["High"] - data["Low"],
        (data["High"] - previous_close).abs(),
        (data["Low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    close, open_, high, low, volume = data["Close"], data["Open"], data["High"], data["Low"], data["Volume"]
    safe_close = close.replace(0, np.nan)
    for bars in (1, 2, 4, 8, 13, 26):
        data[f"ret_{bars}"] = close.pct_change(bars)
    previous_close = close.shift(1)
    body_high = pd.concat([open_, close], axis=1).max(axis=1)
    body_low = pd.concat([open_, close], axis=1).min(axis=1)
    data["range_pct"] = (high - low) / safe_close
    data["body_pct"] = (close - open_) / open_.replace(0, np.nan)
    data["upper_wick_pct"] = (high - body_high) / safe_close
    data["lower_wick_pct"] = (body_low - low) / safe_close
    data["gap_pct"] = open_ / previous_close.replace(0, np.nan) - 1

    emas = {}
    for span in (4, 8, 13, 26, 52):
        emas[span] = close.ewm(span=span, adjust=False, min_periods=span).mean()
        data[f"ema{span}_gap"] = close / emas[span] - 1
    data["ema4_13_spread"] = emas[4] / emas[13] - 1
    data["ema13_26_spread"] = emas[13] / emas[26] - 1
    data["ema26_52_spread"] = emas[26] / emas[52] - 1
    for span in (8, 13, 26):
        data[f"ema{span}_slope4"] = emas[span] / emas[span].shift(4) - 1

    atr14, atr26 = _atr(data, 14), _atr(data, 26)
    data["atr14_pct"] = atr14 / safe_close
    data["atr26_pct"] = atr26 / safe_close
    for window in (4, 8, 13, 26):
        data[f"rv{window}"] = data["ret_1"].rolling(window).std(ddof=0)
    candle_range = (high - low).replace(0, np.nan)
    data["range_ratio13"] = candle_range / candle_range.rolling(13).mean()
    data["range_ratio26"] = candle_range / candle_range.rolling(26).mean()
    for window in (4, 13, 26):
        data[f"volume_ratio{window}"] = volume / volume.rolling(window).mean()
    data["volume_change"] = volume.pct_change()
    volume_std = volume.rolling(26).std(ddof=0).replace(0, np.nan)
    data["volume_z26"] = (volume - volume.rolling(26).mean()) / volume_std

    for window in (13, 26):
        rolling_high = high.rolling(window).max()
        rolling_low = low.rolling(window).min()
        data[f"dist_high{window}_atr"] = (close - rolling_high) / atr14.replace(0, np.nan)
        data[f"dist_low{window}_atr"] = (close - rolling_low) / atr14.replace(0, np.nan)
        width = (rolling_high - rolling_low).replace(0, np.nan)
        data[f"close_location{window}"] = (close - rolling_low) / width

    session_open = data.groupby("session_date")["Open"].transform("first")
    data["session_return"] = close / session_open - 1
    closes = data.groupby("session_date")["Close"].last()
    gaps = data.groupby("session_date")["Open"].first() / closes.shift(1) - 1
    data["day_gap"] = data["session_date"].map(gaps)
    progress = data["bar_in_session"].astype(float) / (SESSION_BARS - 1)
    data["session_progress"] = progress
    data["session_sin"] = np.sin(2 * np.pi * progress)
    data["session_cos"] = np.cos(2 * np.pi * progress)
    data["bars_to_close"] = (SESSION_BARS - 1 - data["bar_in_session"]).astype(float)
    return data


def _net_return(entry_open: pd.Series, exit_open: pd.Series) -> pd.Series:
    entry_cash = entry_open * (1 + SLIPPAGE_RATE) * (1 + COMMISSION_RATE)
    exit_cash = exit_open * (1 - SLIPPAGE_RATE) * (1 - COMMISSION_RATE)
    return exit_cash / entry_cash - 1


def add_targets(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy().sort_index()
    entry_open = result["Open"].shift(-1)
    entry_time = pd.Series(result.index, index=result.index).shift(-1)
    entry_session = result["session_date"].shift(-1)
    result["entry_open"] = entry_open
    result["entry_time"] = entry_time
    for horizon, name, suffix in ((AUX_2H_BARS, TARGET_2H, "2h"), (MAIN_HORIZON_BARS, TARGET_4H, "4h")):
        shift = -(1 + horizon)
        exit_open = result["Open"].shift(shift)
        exit_time = pd.Series(result.index, index=result.index).shift(shift)
        exit_session = result["session_date"].shift(shift)
        valid = entry_session.eq(exit_session) & entry_open.notna() & exit_open.notna()
        result[name] = _net_return(entry_open, exit_open).where(valid)
        result[f"target_gross_{suffix}"] = (exit_open / entry_open - 1).where(valid)
        result[f"exit_open_{suffix}"] = exit_open.where(valid)
        result[f"exit_time_{suffix}"] = exit_time.where(valid)
    result["label_end_time"] = result["exit_time_4h"]
    return result


def build_dataset() -> pd.DataFrame:
    featured: dict[str, pd.DataFrame] = {}
    for symbol in TARGET_SYMBOLS + CONTEXT_SYMBOLS:
        featured[symbol] = add_features(resample_30m(load_5m(symbol)))

    target_frames = []
    for symbol in TARGET_SYMBOLS:
        frame = featured[symbol].copy()
        frame["symbol"] = symbol
        target_frames.append(frame)
    combined = pd.concat(target_frames).sort_index()
    flat = combined.reset_index().rename(columns={"index": "Timestamp"})
    grouped = flat.groupby("Timestamp", sort=False)
    flat["universe_breadth1"] = grouped["ret_1"].transform(lambda s: (s > 0).mean())
    flat["universe_breadth4"] = grouped["ret_4"].transform(lambda s: (s > 0).mean())
    flat["universe_dispersion1"] = grouped["ret_1"].transform(lambda s: s.std(ddof=0))
    flat["universe_mean_ret1"] = grouped["ret_1"].transform("mean")
    flat["universe_mean_ret4"] = grouped["ret_4"].transform("mean")
    flat["universe_mean_ret8"] = grouped["ret_8"].transform("mean")
    flat["rank_ret1"] = grouped["ret_1"].rank(pct=True)
    flat["rank_ret4"] = grouped["ret_4"].rank(pct=True)
    flat["rank_ret8"] = grouped["ret_8"].rank(pct=True)
    flat["rank_volume13"] = grouped["volume_ratio13"].rank(pct=True)
    combined = flat.set_index("Timestamp").sort_index()

    context = {}
    for symbol in ("SPY", "QQQ", "IWM"):
        frame = featured[symbol]
        prefix = symbol.lower()
        context[prefix] = pd.DataFrame(index=frame.index, data={
            f"{prefix}_ret1": frame["ret_1"],
            f"{prefix}_ret4": frame["ret_4"],
            f"{prefix}_ret8": frame["ret_8"],
            f"{prefix}_rv13": frame["rv13"],
            f"{prefix}_ema13_26": frame["ema13_26_spread"],
        })

    output = []
    for symbol in TARGET_SYMBOLS:
        frame = combined[combined["symbol"] == symbol].copy()
        for ctx in context.values():
            frame = frame.join(ctx, how="left")
        sector = featured[SECTOR_ETF_BY_SYMBOL[symbol]]
        sector_context = pd.DataFrame(index=sector.index, data={
            "sector_ret1": sector["ret_1"], "sector_ret4": sector["ret_4"], "sector_ret8": sector["ret_8"],
            "sector_rv13": sector["rv13"], "sector_ema13_26": sector["ema13_26_spread"],
        })
        frame = frame.join(sector_context, how="left")
        frame["rel_spy_1"] = frame["ret_1"] - frame["spy_ret1"]
        frame["rel_spy_4"] = frame["ret_4"] - frame["spy_ret4"]
        frame["rel_spy_8"] = frame["ret_8"] - frame["spy_ret8"]
        frame["rel_sector_1"] = frame["ret_1"] - frame["sector_ret1"]
        frame["rel_sector_4"] = frame["ret_4"] - frame["sector_ret4"]
        frame["rel_sector_8"] = frame["ret_8"] - frame["sector_ret8"]
        frame["regime_spy_above_ema26"] = (frame["spy_ema13_26"] > 0).astype(float)
        frame["regime_qqq_above_ema26"] = (frame["qqq_ema13_26"] > 0).astype(float)
        spy_vol_median = frame["spy_rv13"].rolling(52, min_periods=26).median()
        frame["regime_spy_vol_high"] = (frame["spy_rv13"] > spy_vol_median).astype(float)
        output.append(add_targets(frame))

    data = pd.concat(output).sort_index()
    for symbol in TARGET_SYMBOLS:
        data[f"symbol_{symbol}"] = (data["symbol"] == symbol).astype(float)
    data = data.replace([np.inf, -np.inf], np.nan)
    required = FEATURE_COLUMNS + [TARGET_4H, "entry_open", "entry_time", "exit_open_4h", "exit_time_4h"]
    data = data.dropna(subset=required).copy()
    for column in SYMBOL_FEATURE_COLUMNS:
        data[column] = data[column].astype(float)
    missing = [c for c in MODEL_FEATURE_COLUMNS if c not in data.columns]
    if missing:
        raise RuntimeError(f"Missing EXP-4 model features: {missing}")
    local = pd.DatetimeIndex(data.index).tz_convert(MARKET_TIMEZONE)
    data["year"] = local.year
    data["month"] = local.month
    return data.sort_index()
