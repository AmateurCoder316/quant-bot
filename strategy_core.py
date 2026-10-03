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
    data["High20"] = data["High"].rolling(20).max().shift(1)
    data["Low10"] = data["Low"].rolling(10).min().shift(1)
    data["RSI2"] = calculate_rsi(data["Close"], 2)
    data["ATR14"] = calculate_atr(data, ATR_PERIOD)

    return data


# ==========================================
# SIGNALS
# ==========================================


def strategy_signal(data, has_position):
    """Return a Signal with explicit attribution for every decision."""

    latest = data.iloc[-1]

    if pd.isna(latest["MA200"]) or pd.isna(latest["ATR14"]):
        return Signal("HOLD", "warmup")

    trend_up = latest["Close"] > latest["MA200"]

    if not has_position:
        breakout_buy = (
            trend_up
            and not pd.isna(latest["High20"])
            and latest["Close"] > latest["High20"]
        )

        mean_reversion_buy = trend_up and latest["RSI2"] < 10

        if breakout_buy:
            return Signal("BUY", "breakout_20")

        if mean_reversion_buy:
            return Signal("BUY", "rsi2_mean_reversion")

        return Signal("HOLD", "no_entry")

    breakout_sell = (
        not pd.isna(latest["Low10"])
        and latest["Close"] < latest["Low10"]
    )

    mean_reversion_sell = latest["RSI2"] > 70

    if breakout_sell:
        return Signal("SELL", "breakout_10_exit")

    if mean_reversion_sell:
        return Signal("SELL", "rsi2_exit")

    return Signal("HOLD", "hold_position")


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
