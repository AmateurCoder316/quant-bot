from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd


# ==========================================
# 5-MINUTE SYSTEM CONFIGURATION
# ==========================================

SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
INTERVAL = "5m"
STARTING_CASH = 10000.0

COMMISSION_RATE = 0.001
SLIPPAGE_RATE = 0.0005

MAX_OPEN_POSITIONS = 3
MAX_POSITION_PERCENT = 0.20
RISK_PER_TRADE_PERCENT = 0.005
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

# Strategy filters. These deliberately favor fewer, stronger setups.
BREAKOUT_LOOKBACK = 50
EXIT_LOOKBACK = 20
BREAKOUT_ATR_BUFFER = 0.25
VOLUME_MULTIPLIER = 1.10
MA_SLOPE_LOOKBACK = 20
COOLDOWN_BARS = 6
ENTRY_START = time(10, 0)
ENTRY_END = time(14, 30)

MARKET_TIMEZONE = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class Signal:
    side: str
    reason: str


# ==========================================
# INDICATORS
# ==========================================


def calculate_rsi(series, period=2):
    change = series.diff()
    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)

    average_gain = gains.ewm(alpha=1 / period, adjust=False).mean()
    average_loss = losses.ewm(alpha=1 / period, adjust=False).mean()

    rs = average_gain / average_loss.replace(0, float("nan"))
    return (100 - (100 / (1 + rs))).fillna(100)


def calculate_atr(data, period=ATR_PERIOD):
    previous_close = data["Close"].shift(1)

    true_range = pd.concat(
        [
            data["High"] - data["Low"],
            (data["High"] - previous_close).abs(),
            (data["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def add_indicators(data):
    data = data.copy()

    data["MA50"] = data["Close"].rolling(50).mean()
    data["MA200"] = data["Close"].rolling(200).mean()
    data["HighBreakout"] = data["High"].rolling(BREAKOUT_LOOKBACK).max().shift(1)
    data["LowExit"] = data["Low"].rolling(EXIT_LOOKBACK).min().shift(1)
    data["VolumeMA20"] = data["Volume"].rolling(20).mean()
    data["RSI2"] = calculate_rsi(data["Close"], 2)
    data["ATR14"] = calculate_atr(data, ATR_PERIOD)

    return data


# ==========================================
# SIGNALS
# ==========================================


def bars_since_timestamp(data, timestamp):
    if timestamp is None:
        return None

    try:
        ts = pd.Timestamp(timestamp)
        return int((data.index > ts).sum())
    except Exception:
        return None


def entry_time_allowed(timestamp):
    ts = pd.Timestamp(timestamp)

    if ts.tzinfo is None:
        ts = ts.tz_localize(MARKET_TIMEZONE)
    else:
        ts = ts.tz_convert(MARKET_TIMEZONE)

    current = ts.time()
    return ENTRY_START <= current <= ENTRY_END


def strategy_signal(data, has_position, last_exit_time=None):
    """Return the current 5-minute breakout signal with explicit attribution."""

    latest = data.iloc[-1]

    required = [
        latest.get("MA50"),
        latest.get("MA200"),
        latest.get("HighBreakout"),
        latest.get("LowExit"),
        latest.get("VolumeMA20"),
        latest.get("ATR14"),
    ]

    if any(pd.isna(value) for value in required):
        return Signal("HOLD", "warmup")

    if has_position:
        # Give winners more room than the old RSI exit did. The ATR stop still
        # protects the downside while these exits react to trend deterioration.
        if latest["Close"] < latest["LowExit"]:
            return Signal("SELL", "breakout_20_exit")

        if latest["Close"] < latest["MA50"]:
            return Signal("SELL", "ma50_exit")

        return Signal("HOLD", "hold_position")

    bars_since_exit = bars_since_timestamp(data, last_exit_time)
    if bars_since_exit is not None and bars_since_exit < COOLDOWN_BARS:
        return Signal("HOLD", "cooldown")

    if not entry_time_allowed(data.index[-1]):
        return Signal("HOLD", "outside_entry_window")

    if len(data) <= MA_SLOPE_LOOKBACK:
        return Signal("HOLD", "warmup")

    ma50_then = data["MA50"].iloc[-1 - MA_SLOPE_LOOKBACK]
    if pd.isna(ma50_then):
        return Signal("HOLD", "warmup")

    trend_up = latest["Close"] > latest["MA200"] and latest["MA50"] > latest["MA200"]
    ma50_rising = latest["MA50"] > ma50_then
    volume_confirmed = latest["Volume"] >= latest["VolumeMA20"] * VOLUME_MULTIPLIER
    breakout_level = latest["HighBreakout"] + latest["ATR14"] * BREAKOUT_ATR_BUFFER
    breakout_confirmed = latest["Close"] > breakout_level

    if trend_up and ma50_rising and volume_confirmed and breakout_confirmed:
        return Signal("BUY", "breakout_50_confirmed")

    return Signal("HOLD", "no_entry")


# ==========================================
# RISK MANAGEMENT
# ==========================================


def stop_distance_from_atr(atr):
    if atr is None or pd.isna(atr) or atr <= 0:
        return None
    return float(atr) * ATR_STOP_MULTIPLIER


def stop_price_from_atr(entry_price, atr):
    distance = stop_distance_from_atr(atr)
    if distance is None:
        return None
    return max(0.01, float(entry_price) - distance)


def risk_sized_trade_value(portfolio_value, available_cash, entry_price, atr):
    """
    Size a position so the ATR stop risks at most RISK_PER_TRADE_PERCENT
    of the whole portfolio, while also respecting the max allocation cap.
    """

    stop_distance = stop_distance_from_atr(atr)
    if stop_distance is None or entry_price <= 0:
        return 0.0

    risk_budget = float(portfolio_value) * RISK_PER_TRADE_PERCENT
    shares_by_risk = risk_budget / stop_distance
    value_by_risk = shares_by_risk * float(entry_price)

    allocation_cap = float(portfolio_value) * MAX_POSITION_PERCENT

    return max(
        0.0,
        min(float(available_cash), value_by_risk, allocation_cap),
    )


# ==========================================
# MARKET SESSION
# ==========================================


def market_session_status(now=None):
    """
    Lightweight US regular-session status.

    This intentionally does not attempt to encode the full NYSE holiday
    calendar. The live bot also checks whether fresh bars actually arrive,
    so a holiday naturally behaves as a closed/stale session.
    """

    if now is None:
        now = datetime.now(MARKET_TIMEZONE)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=MARKET_TIMEZONE)
    else:
        now = now.astimezone(MARKET_TIMEZONE)

    if now.weekday() >= 5:
        return "WEEKEND"

    current = now.time()

    if current < time(9, 30):
        return "PREMARKET"

    if current < time(16, 0):
        return "OPEN"

    return "CLOSED"
