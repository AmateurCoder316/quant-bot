import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from desktop_dashboard.state import write_state_atomic
from market_data import DEFAULT_PROVIDER
from strategy_core import (
    COMMISSION_RATE,
    INTERVAL,
    MAX_OPEN_POSITIONS,
    SLIPPAGE_RATE,
    STARTING_CASH,
    SYMBOLS,
    add_indicators,
    market_session_status,
    risk_sized_trade_value,
    stop_price_from_atr,
    strategy_signal,
)


PERIOD = "60d"
POLL_SECONDS = 60
MIN_BARS = 205

STATE_FILE = Path("paper_state_5m.json")
LIVE_STATE_FILE = Path("live_state.json")
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "paper_trader.log"
TRADE_CSV = LOG_DIR / "trades.csv"
EQUITY_CSV = LOG_DIR / "equity.csv"
MODEL_LABEL = "legacy-rules"
MAX_EQUITY_POINTS = 1500


def setup_logging():
    LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def log_event(message):
    logging.info(message)


def fresh_position(last_exit_time=None):
    return {
        "shares": 0.0,
        "entry_price": None,
        "entry_time": None,
        "entry_total_cost": None,
        "entry_reason": None,
        "stop_price": None,
        "pending_order": None,
        "last_signal_bar": None,
        "last_exit_time": last_exit_time,
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
        "last_prices": {},
        "equity_curve": [],
        "market_status": market_session_status(),
        "last_heartbeat": datetime.now().isoformat(),
    }


def load_state():
    if not STATE_FILE.exists():
        return fresh_state()

    try:
        with STATE_FILE.open("r", encoding="utf-8") as file:
            state = json.load(file)
    except (json.JSONDecodeError, OSError):
        log_event("Unreadable state file; starting fresh")
        return fresh_state()

    required = {"cash", "realized_pnl", "positions", "trades"}
    if state.get("interval") != INTERVAL or not required.issubset(state):
        log_event("Incompatible state file; starting fresh")
        return fresh_state()

    state.setdefault("last_prices", {})
    state.setdefault("equity_curve", [])
    state.setdefault("market_status", market_session_status())
    state.setdefault("last_heartbeat", datetime.now().isoformat())

    for symbol in SYMBOLS:
        position = state["positions"].setdefault(symbol, fresh_position())
        position.setdefault("entry_reason", None)
        position.setdefault("pending_order", None)
        position.setdefault("last_signal_bar", None)
        position.setdefault("last_exit_time", None)

    return state


def save_state(state):
    state["last_heartbeat"] = datetime.now().isoformat()
    with STATE_FILE.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2)


def open_position_count(state):
    return sum(
        float(position.get("shares", 0.0) or 0.0) > 0
        for position in state["positions"].values()
    )


def portfolio_value(state, prices):
    value = float(state["cash"])

    for symbol, position in state["positions"].items():
        shares = float(position.get("shares", 0.0) or 0.0)
        if shares > 0 and symbol in prices:
            value += shares * float(prices[symbol])

    return value


def unrealized_pnl(state, prices):
    total = 0.0

    for symbol, position in state["positions"].items():
        shares = float(position.get("shares", 0.0) or 0.0)
        entry_cost = position.get("entry_total_cost")

        if shares <= 0 or entry_cost is None or symbol not in prices:
            continue

        total += shares * float(prices[symbol]) - float(entry_cost)

    return total


def record_trade(state, symbol, exit_time, exit_price, pnl, return_percent, exit_reason):
    position = state["positions"][symbol]

    trade = {
        "symbol": symbol,
        "entry_time": position["entry_time"],
        "exit_time": exit_time,
        "entry_price": position["entry_price"],
        "exit_price": exit_price,
        "pnl": pnl,
        "return_percent": return_percent,
        "entry_reason": position.get("entry_reason"),
        "exit_reason": exit_reason,
        "reason": exit_reason,
    }

    state["trades"].append(trade)

    file_exists = TRADE_CSV.exists()
    with TRADE_CSV.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=trade.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(trade)


def execute_buy(state, symbol, execution_time, market_open, atr, reason, prices):
    position = state["positions"][symbol]

    if position["shares"] > 0 or open_position_count(state) >= MAX_OPEN_POSITIONS:
        return

    execution_price = float(market_open) * (1 + SLIPPAGE_RATE)
    value = portfolio_value(state, prices)

    allocation = risk_sized_trade_value(
        value,
        state["cash"],
        execution_price,
        atr,
    )

    if allocation <= 0:
        log_event(f"BUY BLOCKED {symbol}: risk sizing returned zero")
        return

    trade_value = allocation / (1 + COMMISSION_RATE)
    commission = trade_value * COMMISSION_RATE
    shares = trade_value / execution_price
    total_cost = trade_value + commission

    state["cash"] -= total_cost

    position["shares"] = shares
    position["entry_price"] = execution_price
    position["entry_time"] = execution_time
    position["entry_total_cost"] = total_cost
    position["entry_reason"] = reason
    position["stop_price"] = stop_price_from_atr(execution_price, atr)
    position["pending_order"] = None

    log_event(
        f"BUY {symbol} reason={reason} shares={shares:.6f} "
        f"price={execution_price:.4f} stop={position['stop_price']:.4f} "
        f"cost={total_cost:.2f}"
    )


def execute_sell(state, symbol, execution_time, market_price, reason):
    position = state["positions"][symbol]

    if position["shares"] <= 0:
        return

    execution_price = float(market_price) * (1 - SLIPPAGE_RATE)
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
        f"SELL {symbol} reason={reason} price={execution_price:.4f} "
        f"pnl={pnl:+.2f}"
    )

    state["positions"][symbol] = fresh_position(last_exit_time=execution_time)


def timestamp_text(timestamp):
    return pd.Timestamp(timestamp).isoformat()


def try_execute_pending_order(state, symbol, raw_data, prices):
    position = state["positions"][symbol]
    pending = position.get("pending_order")

    if pending is None:
        return

    signal_time = pd.Timestamp(pending["signal_time"])
    later_rows = raw_data[raw_data.index > signal_time]

    if len(later_rows) == 0:
        return

    row = later_rows.iloc[0]
    execution_time = timestamp_text(later_rows.index[0])

    if pending["side"] == "BUY":
        execute_buy(
            state,
            symbol,
            execution_time,
            float(row["Open"]),
            pending.get("atr"),
            pending.get("reason", "unknown"),
            prices,
        )
    elif pending["side"] == "SELL":
        execute_sell(
            state,
            symbol,
            execution_time,
            float(row["Open"]),
            pending.get("reason", "signal"),
        )

    state["positions"][symbol]["pending_order"] = None


def process_stop_loss(state, symbol, raw_data):
    position = state["positions"][symbol]
    stop_price = position.get("stop_price")

    if position["shares"] <= 0 or stop_price is None:
        return

    latest = raw_data.iloc[-1]
    if float(latest["Low"]) <= float(stop_price):
        fill_reference = min(float(latest["Open"]), float(stop_price))
        execute_sell(
            state,
            symbol,
            timestamp_text(raw_data.index[-1]),
            fill_reference,
            "atr_stop",
        )


def process_signal(state, symbol, completed):
    position = state["positions"][symbol]
    signal_time = timestamp_text(completed.index[-1])

    if position.get("last_signal_bar") == signal_time:
        return

    signal = strategy_signal(
        completed,
        position["shares"] > 0,
        position.get("last_exit_time"),
    )
    position["last_signal_bar"] = signal_time

    latest = completed.iloc[-1]
    log_event(
        f"SIGNAL {symbol} side={signal.side} reason={signal.reason} "
        f"bar={signal_time} close={float(latest['Close']):.4f}"
    )

    if signal.side in {"BUY", "SELL"}:
        position["pending_order"] = {
            "side": signal.side,
            "reason": signal.reason,
            "signal_time": signal_time,
            "atr": float(latest["ATR14"]) if not pd.isna(latest["ATR14"]) else None,
        }


def save_equity_snapshot(state, prices):
    timestamp = datetime.now().isoformat()
    value = portfolio_value(state, prices)
    row = {
        "timestamp": timestamp,
        "cash": state["cash"],
        "portfolio_value": value,
        "realized_pnl": state["realized_pnl"],
        "unrealized_pnl": unrealized_pnl(state, prices),
        "open_positions": open_position_count(state),
        "market_status": state.get("market_status", "UNKNOWN"),
    }

    curve = state.setdefault("equity_curve", [])
    curve.append({"timestamp": timestamp, "value": value})
    del curve[:-MAX_EQUITY_POINTS]

    file_exists = EQUITY_CSV.exists()
    with EQUITY_CSV.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def publish_dashboard_state(state, prices):
    value = portfolio_value(state, prices)
    total_pnl = value - STARTING_CASH
    positions = []

    for symbol in SYMBOLS:
        position = state["positions"][symbol]
        shares = float(position.get("shares", 0.0) or 0.0)
        if shares <= 0:
            continue

        entry_price = float(position.get("entry_price", 0.0) or 0.0)
        current_price = float(prices.get(symbol, entry_price) or entry_price)
        market_value = shares * current_price
        entry_total_cost = float(position.get("entry_total_cost", 0.0) or 0.0)
        pnl = market_value - entry_total_cost
        pnl_percent = pnl / entry_total_cost * 100.0 if entry_total_cost else 0.0

        positions.append(
            {
                "symbol": symbol,
                "quantity": shares,
                "entry_price": entry_price,
                "current_price": current_price,
                "market_value": market_value,
                "profit_loss": pnl,
                "profit_loss_percent": pnl_percent,
            }
        )

    write_state_atomic(
        LIVE_STATE_FILE,
        {
            "portfolio_value": value,
            "starting_value": STARTING_CASH,
            "profit_loss": total_pnl,
            "profit_loss_percent": total_pnl / STARTING_CASH * 100.0,
            "model": MODEL_LABEL,
            "equity_curve": state.get("equity_curve", []),
            "positions": positions,
        },
    )


def print_money_summary(state, prices):
    value = portfolio_value(state, prices)
    total_pnl = value - STARTING_CASH
    unrealized = unrealized_pnl(state, prices)

    print("\033[2J\033[H", end="")
    print("5-MIN PAPER PORTFOLIO")
    print("=" * 42)
    print(f"Market:          {state.get('market_status', 'UNKNOWN')}")
    print(f"Account value:   ${value:,.2f}")
    print(f"Cash:            ${state['cash']:,.2f}")
    print(f"Total P/L:       ${total_pnl:+,.2f}")
    print(f"Return:          {(total_pnl / STARTING_CASH * 100):+.2f}%")
    print(f"Realized P/L:    ${state['realized_pnl']:+,.2f}")
    print(f"Unrealized P/L:  ${unrealized:+,.2f}")
    print(f"Open positions:  {open_position_count(state)}/{MAX_OPEN_POSITIONS}")
    print(f"Completed trades:{len(state['trades']):>4}")

    print("\nPOSITIONS")
    print("-" * 42)
    any_position = False

    for symbol in SYMBOLS:
        position = state["positions"][symbol]
        if position["shares"] <= 0 or symbol not in prices:
            continue

        any_position = True
        market_value = position["shares"] * prices[symbol]
        pnl = market_value - position["entry_total_cost"]
        print(
            f"{symbol:<6} ${market_value:>9,.2f}  "
            f"P/L ${pnl:+8,.2f}  {position.get('entry_reason') or ''}"
        )

    if not any_position:
        print("None")

    print("\nDesktop dashboard is managed by bot.py")
    print("Ctrl+C to stop")


def run_open_market_check(state):
    market_data = {}
    prices = {}

    for symbol in SYMBOLS:
        raw = DEFAULT_PROVIDER.history(symbol, period=PERIOD)
        if len(raw) < MIN_BARS:
            raise ValueError(f"Not enough 5-minute history for {symbol}.")

        completed = add_indicators(raw.iloc[:-1].copy())
        market_data[symbol] = (raw, completed)
        prices[symbol] = float(raw.iloc[-1]["Close"])

    state["last_prices"] = prices

    for symbol in SYMBOLS:
        raw, _ = market_data[symbol]
        try_execute_pending_order(state, symbol, raw, prices)

    for symbol in SYMBOLS:
        raw, _ = market_data[symbol]
        process_stop_loss(state, symbol, raw)

    for symbol in SYMBOLS:
        _, completed = market_data[symbol]
        process_signal(state, symbol, completed)

    return prices


def run_once(state):
    status = market_session_status()
    state["market_status"] = status

    if status == "OPEN":
        prices = run_open_market_check(state)
    else:
        prices = state.get("last_prices", {})

    save_equity_snapshot(state, prices)
    save_state(state)
    publish_dashboard_state(state, prices)
    print_money_summary(state, prices)


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
                state["market_status"] = "ERROR"
                save_state(state)
                publish_dashboard_state(state, state.get("last_prices", {}))
                print(f"\nCheck failed: {error}")

            time.sleep(POLL_SECONDS)

    except KeyboardInterrupt:
        save_state(state)
        publish_dashboard_state(state, state.get("last_prices", {}))
        log_event("5-minute paper trader stopped")
        print("\nPaper trader stopped.")
        print(f"State saved to {STATE_FILE}")


if __name__ == "__main__":
    main()
