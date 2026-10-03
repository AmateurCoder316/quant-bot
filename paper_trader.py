import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf


# ==========================================
# SECTION 1 — SETTINGS
# ==========================================

SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
INTERVAL = "5m"
PERIOD = "60d"
POLL_SECONDS = 60

STARTING_CASH = 10000.0
COMMISSION_RATE = 0.001      # 0.10%
SLIPPAGE_RATE = 0.0005       # 0.05%

# Risk controls
POSITION_SIZE_PERCENT = 0.20
MAX_OPEN_POSITIONS = 3
STOP_LOSS_PERCENT = 0.05

STATE_FILE = Path("paper_state_5m.json")
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "paper_trader.log"
TRADE_CSV = LOG_DIR / "trades.csv"
EQUITY_CSV = LOG_DIR / "equity.csv"


# ==========================================
# SECTION 2 — LOGGING
# ==========================================


def setup_logging():
    LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def log_event(message):
    logging.info(message)


# ==========================================
# SECTION 3 — MARKET DATA
# ==========================================


def download_data(symbol):
    data = yf.download(
        symbol,
        period=PERIOD,
        interval=INTERVAL,
        auto_adjust=True,
        progress=False,
        prepost=False,
    )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.dropna().copy()

    if len(data) < 2:
        raise ValueError(f"Not enough intraday data for {symbol}.")

    return data


def completed_bars(data):
    # The newest Yahoo candle may still be forming.
    return data.iloc[:-1].copy()


# ==========================================
# SECTION 4 — INDICATORS
# ==========================================


def calculate_rsi(series, period=2):
    change = series.diff()
    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)

    average_gain = gains.ewm(alpha=1 / period, adjust=False).mean()
    average_loss = losses.ewm(alpha=1 / period, adjust=False).mean()

    rs = average_gain / average_loss.replace(0, float("nan"))
    return (100 - (100 / (1 + rs))).fillna(100)


def add_indicators(data):
    data = data.copy()
    data["MA50"] = data["Close"].rolling(50).mean()
    data["MA200"] = data["Close"].rolling(200).mean()
    data["High20"] = data["High"].rolling(20).max().shift(1)
    data["Low10"] = data["Low"].rolling(10).min().shift(1)
    data["RSI2"] = calculate_rsi(data["Close"], 2)
    return data


# ==========================================
# SECTION 5 — STRATEGY
# ==========================================


def strategy_signal(data, has_position):
    latest = data.iloc[-1]

    if pd.isna(latest["MA200"]):
        return "HOLD"

    trend_up = latest["Close"] > latest["MA200"]

    if not has_position:
        breakout_buy = (
            not pd.isna(latest["High20"])
            and latest["Close"] > latest["High20"]
        )
        mean_reversion_buy = trend_up and latest["RSI2"] < 10

        if breakout_buy or mean_reversion_buy:
            return "BUY"

    if has_position:
        breakout_sell = (
            not pd.isna(latest["Low10"])
            and latest["Close"] < latest["Low10"]
        )
        mean_reversion_sell = latest["RSI2"] > 70

        if breakout_sell or mean_reversion_sell:
            return "SELL"

    return "HOLD"


# ==========================================
# SECTION 6 — PERSISTENT STATE
# ==========================================


def fresh_position():
    return {
        "shares": 0.0,
        "entry_price": None,
        "entry_time": None,
        "entry_total_cost": None,
        "stop_price": None,
        "pending_order": None,
        "last_signal_bar": None,
    }


def fresh_state():
    return {
        "created_at": datetime.now().isoformat(),
        "interval": INTERVAL,
        "starting_cash": STARTING_CASH,
        "cash": STARTING_CASH,
        "realized_pnl": 0.0,
        "positions": {symbol: fresh_position() for symbol in SYMBOLS},
        "trades": [],
    }


def load_state():
    if not STATE_FILE.exists():
        return fresh_state()

    with STATE_FILE.open("r", encoding="utf-8") as file:
        state = json.load(file)

    required_keys = {"cash", "realized_pnl", "positions", "trades"}
    if state.get("interval") != INTERVAL or not required_keys.issubset(state):
        log_event("Old or incompatible paper state detected; starting fresh")
        return fresh_state()

    for symbol in SYMBOLS:
        state["positions"].setdefault(symbol, fresh_position())

    return state


def save_state(state):
    with STATE_FILE.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2)


# ==========================================
# SECTION 7 — PORTFOLIO HELPERS
# ==========================================


def open_position_count(state):
    return sum(
        1
        for position in state["positions"].values()
        if position["shares"] > 0
    )


def portfolio_value(state, latest_prices):
    value = state["cash"]

    for symbol, position in state["positions"].items():
        if position["shares"] > 0 and symbol in latest_prices:
            value += position["shares"] * latest_prices[symbol]

    return value


def unrealized_pnl(state, latest_prices):
    total = 0.0

    for symbol, position in state["positions"].items():
        if position["shares"] <= 0 or symbol not in latest_prices:
            continue

        current_value = position["shares"] * latest_prices[symbol]
        total += current_value - position["entry_total_cost"]

    return total


# ==========================================
# SECTION 8 — TRADE RECORDING
# ==========================================


def record_trade(state, symbol, execution_time, execution_price, pnl, return_percent, reason):
    position = state["positions"][symbol]

    trade = {
        "symbol": symbol,
        "entry_time": position["entry_time"],
        "exit_time": execution_time,
        "entry_price": position["entry_price"],
        "exit_price": execution_price,
        "pnl": pnl,
        "return_percent": return_percent,
        "reason": reason,
    }

    state["trades"].append(trade)

    file_exists = TRADE_CSV.exists()
    with TRADE_CSV.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=trade.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(trade)


# ==========================================
# SECTION 9 — ORDER EXECUTION
# ==========================================


def execute_buy(state, symbol, execution_time, market_open, current_portfolio_value):
    position = state["positions"][symbol]

    if position["shares"] > 0:
        return

    if open_position_count(state) >= MAX_OPEN_POSITIONS:
        log_event(f"BUY BLOCKED {symbol}: max open positions reached")
        return

    max_allocation = current_portfolio_value * POSITION_SIZE_PERCENT
    available_for_trade = min(state["cash"], max_allocation)

    if available_for_trade <= 0:
        log_event(f"BUY BLOCKED {symbol}: no available cash")
        return

    execution_price = market_open * (1 + SLIPPAGE_RATE)
    trade_value = available_for_trade / (1 + COMMISSION_RATE)
    commission = trade_value * COMMISSION_RATE
    shares = trade_value / execution_price
    total_cost = trade_value + commission

    state["cash"] -= total_cost

    position["shares"] = shares
    position["entry_price"] = execution_price
    position["entry_time"] = execution_time
    position["entry_total_cost"] = total_cost
    position["stop_price"] = execution_price * (1 - STOP_LOSS_PERCENT)
    position["pending_order"] = None

    log_event(
        f"BUY {symbol} shares={shares:.6f} price={execution_price:.4f} "
        f"cost={total_cost:.2f} commission={commission:.2f}"
    )


def execute_sell(state, symbol, execution_time, market_price, reason):
    position = state["positions"][symbol]

    if position["shares"] <= 0:
        return

    execution_price = market_price * (1 - SLIPPAGE_RATE)
    gross_value = position["shares"] * execution_price
    commission = gross_value * COMMISSION_RATE
    net_value = gross_value - commission

    pnl = net_value - position["entry_total_cost"]
    return_percent = pnl / position["entry_total_cost"] * 100

    state["cash"] += net_value
    state["realized_pnl"] += pnl

    record_trade(
        state,
        symbol,
        execution_time,
        execution_price,
        pnl,
        return_percent,
        reason,
    )

    log_event(
        f"SELL {symbol} price={execution_price:.4f} pnl={pnl:+.2f} "
        f"reason={reason} commission={commission:.2f}"
    )

    state["positions"][symbol] = fresh_position()


# ==========================================
# SECTION 10 — SIGNAL / ORDER PROCESSING
# ==========================================


def timestamp_text(timestamp):
    return pd.Timestamp(timestamp).isoformat()


def try_execute_pending_order(state, symbol, raw_data, latest_prices):
    position = state["positions"][symbol]
    pending = position["pending_order"]

    if pending is None:
        return

    signal_time = pd.Timestamp(pending["signal_time"])
    later_rows = raw_data[raw_data.index > signal_time]

    if len(later_rows) == 0:
        return

    execution_timestamp = later_rows.index[0]
    execution_time = timestamp_text(execution_timestamp)
    market_open = float(later_rows.iloc[0]["Open"])

    if pending["side"] == "BUY":
        execute_buy(
            state,
            symbol,
            execution_time,
            market_open,
            portfolio_value(state, latest_prices),
        )
    elif pending["side"] == "SELL":
        execute_sell(state, symbol, execution_time, market_open, "signal")

    state["positions"][symbol]["pending_order"] = None


def process_stop_loss(state, symbol, latest_price, timestamp):
    position = state["positions"][symbol]

    if position["shares"] <= 0 or position["stop_price"] is None:
        return

    if latest_price <= position["stop_price"]:
        execute_sell(state, symbol, timestamp, latest_price, "stop_loss")


def process_signal(state, symbol, completed):
    position = state["positions"][symbol]
    signal_time = timestamp_text(completed.index[-1])

    if position["last_signal_bar"] == signal_time:
        return

    signal = strategy_signal(completed, position["shares"] > 0)
    position["last_signal_bar"] = signal_time

    log_event(
        f"SIGNAL {symbol} {signal} bar={signal_time} "
        f"close={completed.iloc[-1]['Close']:.4f}"
    )

    if signal in {"BUY", "SELL"}:
        position["pending_order"] = {
            "side": signal,
            "signal_time": signal_time,
        }


# ==========================================
# SECTION 11 — EQUITY HISTORY
# ==========================================


def save_equity_snapshot(state, latest_prices):
    row = {
        "timestamp": datetime.now().isoformat(),
        "cash": state["cash"],
        "portfolio_value": portfolio_value(state, latest_prices),
        "realized_pnl": state["realized_pnl"],
        "unrealized_pnl": unrealized_pnl(state, latest_prices),
        "open_positions": open_position_count(state),
    }

    file_exists = EQUITY_CSV.exists()
    with EQUITY_CSV.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ==========================================
# SECTION 12 — CLEAN TERMINAL OUTPUT
# ==========================================


def print_money_summary(state, latest_prices):
    value = portfolio_value(state, latest_prices)
    total_pnl = value - STARTING_CASH
    return_percent = total_pnl / STARTING_CASH * 100
    unrealized = unrealized_pnl(state, latest_prices)

    print("\033[2J\033[H", end="")
    print("PAPER PORTFOLIO — 5 MIN")
    print("=" * 40)
    print(f"Account value:   ${value:,.2f}")
    print(f"Cash:            ${state['cash']:,.2f}")
    print(f"Total P/L:       ${total_pnl:+,.2f}")
    print(f"Return:          {return_percent:+.2f}%")
    print(f"Realized P/L:    ${state['realized_pnl']:+,.2f}")
    print(f"Unrealized P/L:  ${unrealized:+,.2f}")
    print(f"Open positions:  {open_position_count(state)}/{MAX_OPEN_POSITIONS}")
    print(f"Completed trades:{len(state['trades']):>4}")

    print()
    print("POSITIONS")
    print("-" * 40)

    any_position = False
    for symbol in SYMBOLS:
        position = state["positions"][symbol]
        if position["shares"] <= 0 or symbol not in latest_prices:
            continue

        any_position = True
        market_value = position["shares"] * latest_prices[symbol]
        pnl = market_value - position["entry_total_cost"]
        print(f"{symbol:<6} ${market_value:>9,.2f}  P/L ${pnl:+8,.2f}")

    if not any_position:
        print("None")

    print()
    print("Price refresh: 60s | Signals: every completed 5m candle")
    print("Ctrl+C to stop")


# ==========================================
# SECTION 13 — ONE LIVE CHECK
# ==========================================


def run_once(state):
    market_data = {}
    latest_prices = {}

    for symbol in SYMBOLS:
        raw = download_data(symbol)
        completed = add_indicators(completed_bars(raw))

        if len(completed) < 201:
            raise ValueError(f"Not enough 5-minute history for {symbol}.")

        market_data[symbol] = (raw, completed)
        latest_prices[symbol] = float(raw.iloc[-1]["Close"])

    for symbol in SYMBOLS:
        raw, _ = market_data[symbol]
        try_execute_pending_order(state, symbol, raw, latest_prices)

    for symbol in SYMBOLS:
        raw, _ = market_data[symbol]
        process_stop_loss(
            state,
            symbol,
            latest_prices[symbol],
            timestamp_text(raw.index[-1]),
        )

    for symbol in SYMBOLS:
        _, completed = market_data[symbol]
        process_signal(state, symbol, completed)

    save_state(state)
    save_equity_snapshot(state, latest_prices)
    print_money_summary(state, latest_prices)


# ==========================================
# SECTION 14 — CONTINUOUS BOT LOOP
# ==========================================


def main():
    setup_logging()
    state = load_state()
    log_event("5-minute paper trader started")

    try:
        while True:
            try:
                run_once(state)
            except Exception as error:
                logging.exception("Live check failed")
                print(f"\nCheck failed: {error}")

            time.sleep(POLL_SECONDS)

    except KeyboardInterrupt:
        save_state(state)
        log_event("5-minute paper trader stopped")
        print()
        print("Paper trader stopped.")
        print(f"State saved to {STATE_FILE}")


if __name__ == "__main__":
    main()
