import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf


# ==========================================
# SECTION 1 — SETTINGS
# ==========================================

SYMBOL = "AAPL"
INTERVAL = "15m"
PERIOD = "60d"
POLL_SECONDS = 60
STARTING_CASH = 10000.0

COMMISSION_RATE = 0.001      # 0.10%
SLIPPAGE_RATE = 0.0005       # 0.05%

# Separate state from the older daily paper trader.
STATE_FILE = Path("paper_state_15m.json")

STRATEGIES = {
    "trend_ma50_200": {
        "name": "Trend MA50/200",
    },
    "breakout_20_10": {
        "name": "Breakout 20/10",
    },
    "mean_reversion_rsi2": {
        "name": "Mean Reversion RSI2",
    },
}


# ==========================================
# SECTION 2 — MARKET DATA
# ==========================================


def download_data():
    """Download recent real AAPL 15-minute market candles."""

    data = yf.download(
        SYMBOL,
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
        raise ValueError("Not enough intraday market data was downloaded.")

    return data


def completed_bars(data):
    """
    Exclude the newest Yahoo row because it may still be the currently
    forming 15-minute candle. Only completed candles may create signals.
    """

    return data.iloc[:-1].copy()


# ==========================================
# SECTION 3 — INDICATORS
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
# SECTION 4 — STRATEGY SIGNALS
# ==========================================


def signal_for_strategy(strategy_id, data, has_position):
    latest = data.iloc[-1]
    previous = data.iloc[-2]

    if strategy_id == "trend_ma50_200":
        if pd.isna(latest["MA200"]) or pd.isna(previous["MA200"]):
            return "HOLD"

        crossed_up = (
            previous["MA50"] <= previous["MA200"]
            and latest["MA50"] > latest["MA200"]
        )

        crossed_down = (
            previous["MA50"] >= previous["MA200"]
            and latest["MA50"] < latest["MA200"]
        )

        if crossed_up and not has_position:
            return "BUY"

        if crossed_down and has_position:
            return "SELL"

        return "HOLD"

    if strategy_id == "breakout_20_10":
        if pd.isna(latest["High20"]) or pd.isna(latest["Low10"]):
            return "HOLD"

        if not has_position and latest["Close"] > latest["High20"]:
            return "BUY"

        if has_position and latest["Close"] < latest["Low10"]:
            return "SELL"

        return "HOLD"

    if strategy_id == "mean_reversion_rsi2":
        if pd.isna(latest["MA200"]):
            return "HOLD"

        trend_is_up = latest["Close"] > latest["MA200"]

        if not has_position and trend_is_up and latest["RSI2"] < 10:
            return "BUY"

        if has_position and latest["RSI2"] > 70:
            return "SELL"

        return "HOLD"

    raise ValueError(f"Unknown strategy: {strategy_id}")


# ==========================================
# SECTION 5 — PERSISTENT PAPER STATE
# ==========================================


def fresh_account(strategy_id):
    return {
        "strategy": strategy_id,
        "cash": STARTING_CASH,
        "shares": 0.0,
        "entry_price": None,
        "entry_time": None,
        "entry_total_cost": None,
        "pending_order": None,
        "last_signal_bar": None,
        "realized_pnl": 0.0,
        "trades": [],
    }


def fresh_state():
    return {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "created_at": datetime.now().isoformat(),
        "accounts": {
            strategy_id: fresh_account(strategy_id)
            for strategy_id in STRATEGIES
        },
    }


def load_state():
    if not STATE_FILE.exists():
        return fresh_state()

    with STATE_FILE.open("r", encoding="utf-8") as file:
        state = json.load(file)

    if state.get("symbol") != SYMBOL or state.get("interval") != INTERVAL:
        raise ValueError(
            f"{STATE_FILE} belongs to another symbol or interval. "
            "Delete it to start a fresh 15-minute paper account."
        )

    return state


def save_state(state):
    with STATE_FILE.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2)


# ==========================================
# SECTION 6 — LOCAL FAKE ORDER EXECUTION
# ==========================================


def execute_buy(account, execution_time, market_open):
    execution_price = market_open * (1 + SLIPPAGE_RATE)

    trade_value = account["cash"] / (1 + COMMISSION_RATE)
    commission = trade_value * COMMISSION_RATE
    shares = trade_value / execution_price

    total_cost = trade_value + commission

    account["cash"] -= total_cost
    account["shares"] = shares
    account["entry_price"] = execution_price
    account["entry_time"] = execution_time
    account["entry_total_cost"] = total_cost
    account["pending_order"] = None

    return execution_price


def execute_sell(account, execution_time, market_open):
    execution_price = market_open * (1 - SLIPPAGE_RATE)

    gross_value = account["shares"] * execution_price
    commission = gross_value * COMMISSION_RATE
    net_value = gross_value - commission

    trade_profit = net_value - account["entry_total_cost"]
    trade_return = trade_profit / account["entry_total_cost"] * 100

    account["cash"] += net_value
    account["realized_pnl"] += trade_profit

    account["trades"].append({
        "entry_time": account["entry_time"],
        "exit_time": execution_time,
        "entry_price": account["entry_price"],
        "exit_price": execution_price,
        "pnl": trade_profit,
        "return_percent": trade_return,
    })

    account["shares"] = 0.0
    account["entry_price"] = None
    account["entry_time"] = None
    account["entry_total_cost"] = None
    account["pending_order"] = None

    return execution_price, trade_profit


# ==========================================
# SECTION 7 — NEXT-BAR-OPEN ORDERS
# ==========================================


def timestamp_text(timestamp):
    return pd.Timestamp(timestamp).isoformat()


def try_execute_pending_order(account, raw_data):
    pending = account["pending_order"]

    if pending is None:
        return None

    signal_time = pd.Timestamp(pending["signal_time"])

    later_rows = raw_data[raw_data.index > signal_time]

    if len(later_rows) == 0:
        return None

    execution_timestamp = later_rows.index[0]
    execution_time = timestamp_text(execution_timestamp)
    market_open = float(later_rows.iloc[0]["Open"])

    if pending["side"] == "BUY" and account["shares"] == 0:
        price = execute_buy(account, execution_time, market_open)
        return f"EXECUTED BUY @ ${price:.2f} on {execution_time}"

    if pending["side"] == "SELL" and account["shares"] > 0:
        price, pnl = execute_sell(account, execution_time, market_open)
        return (
            f"EXECUTED SELL @ ${price:.2f} on {execution_time} "
            f"| trade P/L ${pnl:+.2f}"
        )

    account["pending_order"] = None
    return "Discarded stale pending order"


# ==========================================
# SECTION 8 — PROCESS NEW COMPLETED BAR
# ==========================================


def process_signal(strategy_id, account, data):
    signal_timestamp = data.index[-1]
    signal_time = timestamp_text(signal_timestamp)

    if account["last_signal_bar"] == signal_time:
        return None

    signal = signal_for_strategy(
        strategy_id,
        data,
        account["shares"] > 0,
    )

    account["last_signal_bar"] = signal_time

    if signal in {"BUY", "SELL"}:
        account["pending_order"] = {
            "side": signal,
            "signal_time": signal_time,
        }

        return f"NEW {signal} signal on completed bar {signal_time}"

    return f"HOLD on completed bar {signal_time}"


# ==========================================
# SECTION 9 — ACCOUNT VALUATION
# ==========================================


def account_value(account, latest_price):
    return account["cash"] + account["shares"] * latest_price


def unrealized_pnl(account, latest_price):
    if account["shares"] == 0:
        return 0.0

    return account["shares"] * latest_price - account["entry_total_cost"]


# ==========================================
# SECTION 10 — ONE LIVE CHECK
# ==========================================


def run_once(state):
    raw_data = download_data()
    completed = add_indicators(completed_bars(raw_data))

    if len(completed) < 201:
        raise ValueError("Not enough 15-minute history for MA200.")

    latest_price = float(raw_data.iloc[-1]["Close"])
    latest_completed = timestamp_text(completed.index[-1])

    print()
    print("========================================")
    print("15-MINUTE PAPER CHECK")
    print("========================================")
    print(f"Symbol: {SYMBOL}")
    print(f"Latest completed bar: {latest_completed}")
    print(f"Latest available price: ${latest_price:.2f}")

    anything_new = False

    for strategy_id, strategy in STRATEGIES.items():
        account = state["accounts"][strategy_id]

        execution_message = try_execute_pending_order(account, raw_data)
        signal_message = process_signal(strategy_id, account, completed)

        if execution_message or signal_message:
            anything_new = True

        value = account_value(account, latest_price)
        total_return = (value / STARTING_CASH - 1) * 100
        open_pnl = unrealized_pnl(account, latest_price)

        print()
        print(f"----- {strategy['name']} -----")

        if execution_message:
            print(execution_message)

        if signal_message:
            print(signal_message)
        else:
            print("No new completed 15-minute candle")

        print(f"Cash: ${account['cash']:.2f}")
        print(f"Shares: {account['shares']:.4f}")
        print(f"Account value: ${value:.2f}")
        print(f"Return: {total_return:+.2f}%")
        print(f"Realized P/L: ${account['realized_pnl']:+.2f}")
        print(f"Unrealized P/L: ${open_pnl:+.2f}")
        print(f"Completed trades: {len(account['trades'])}")

        if account["pending_order"]:
            pending = account["pending_order"]
            print(
                f"Pending: {pending['side']} after "
                f"{pending['signal_time']}"
            )
        else:
            print("Pending: none")

    save_state(state)
    return anything_new


# ==========================================
# SECTION 11 — CONTINUOUS BOT LOOP
# ==========================================


def main():
    state = load_state()

    print("========================================")
    print("V0.8 CONTINUOUS LOCAL PAPER TRADER")
    print("========================================")
    print(f"Symbol: {SYMBOL}")
    print(f"Candle interval: {INTERVAL}")
    print(f"Polling every {POLL_SECONDS} seconds")
    print("Fake money only. No broker is connected.")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            try:
                run_once(state)
            except Exception as error:
                print()
                print(f"Check failed: {error}")

            print()
            print(f"Sleeping {POLL_SECONDS} seconds...")
            time.sleep(POLL_SECONDS)

    except KeyboardInterrupt:
        save_state(state)
        print()
        print("Paper trader stopped.")
        print(f"State saved to {STATE_FILE}")


if __name__ == "__main__":
    main()
