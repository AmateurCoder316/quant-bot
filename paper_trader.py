import json
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf


# ==========================================
# SECTION 1 — SETTINGS
# ==========================================

SYMBOL = "AAPL"
PERIOD = "1y"
STARTING_CASH = 10000.0

COMMISSION_RATE = 0.001      # 0.10%
SLIPPAGE_RATE = 0.0005       # 0.05%

STATE_FILE = Path("paper_state.json")
MARKET_TIMEZONE = ZoneInfo("America/New_York")

# Each strategy gets its own completely separate fake account.
STRATEGIES = {
    "trend_ma50_200": {
        "name": "Trend MA50/200",
        "family": "trend",
    },
    "breakout_20_10": {
        "name": "Breakout 20/10",
        "family": "breakout",
    },
    "mean_reversion_rsi2": {
        "name": "Mean Reversion RSI2",
        "family": "mean_reversion",
    },
}


# ==========================================
# SECTION 2 — MARKET DATA
# ==========================================


def download_data():
    """Download recent daily AAPL market data."""

    data = yf.download(
        SYMBOL,
        period=PERIOD,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data.dropna().copy()


def latest_completed_bar(data):
    """
    Return data only through the latest completed daily candle.

    During a US trading session, Yahoo can already contain today's row even
    though today's closing price is not known yet. A daily strategy must not
    use that unfinished close to create a signal.
    """

    now_ny = datetime.now(MARKET_TIMEZONE)
    today_ny = pd.Timestamp(now_ny.date())

    completed = data.copy()

    if len(completed) == 0:
        raise ValueError("No market data was downloaded.")

    last_date = pd.Timestamp(completed.index[-1]).tz_localize(None).normalize()

    # Treat today's row as incomplete until shortly after the regular close.
    regular_close = time(16, 15)

    if last_date == today_ny and now_ny.time() < regular_close:
        completed = completed.iloc[:-1]

    if len(completed) == 0:
        raise ValueError("No completed daily candle is available yet.")

    return completed


# ==========================================
# SECTION 3 — INDICATORS
# ==========================================


def calculate_rsi(series, period=2):
    """Calculate a simple RSI using exponentially smoothed gains/losses."""

    change = series.diff()

    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)

    average_gain = gains.ewm(alpha=1 / period, adjust=False).mean()
    average_loss = losses.ewm(alpha=1 / period, adjust=False).mean()

    rs = average_gain / average_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))

    # If there have been gains but no losses, RSI is effectively 100.
    rsi = rsi.fillna(100)

    return rsi


def add_indicators(data):
    """Calculate every indicator required by our paper strategies."""

    data = data.copy()

    # Trend strategy
    data["MA50"] = data["Close"].rolling(50).mean()
    data["MA200"] = data["Close"].rolling(200).mean()

    # Breakout strategy. shift(1) prevents today's price from being included
    # in the previous-high / previous-low calculation.
    data["High20"] = data["High"].rolling(20).max().shift(1)
    data["Low10"] = data["Low"].rolling(10).min().shift(1)

    # Mean-reversion strategy
    data["RSI2"] = calculate_rsi(data["Close"], 2)

    return data


# ==========================================
# SECTION 4 — STRATEGY SIGNALS
# ==========================================


def signal_for_strategy(strategy_id, data, has_position):
    """Return BUY, SELL, or HOLD using the latest completed daily bar."""

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
        # Only buy short-term weakness while the longer-term trend is positive.
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
# SECTION 5 — PERSISTENT PAPER ACCOUNT STATE
# ==========================================


def fresh_account(strategy_id):
    """Create a brand-new fake account for one strategy."""

    return {
        "strategy": strategy_id,
        "cash": STARTING_CASH,
        "shares": 0.0,
        "entry_price": None,
        "entry_date": None,
        "entry_total_cost": None,
        "pending_order": None,
        "last_signal_bar": None,
        "realized_pnl": 0.0,
        "trades": [],
    }


def load_state():
    """Load paper accounts from disk or create them on the first run."""

    if not STATE_FILE.exists():
        return {
            "symbol": SYMBOL,
            "created_at": datetime.now().isoformat(),
            "accounts": {
                strategy_id: fresh_account(strategy_id)
                for strategy_id in STRATEGIES
            },
        }

    with STATE_FILE.open("r", encoding="utf-8") as file:
        state = json.load(file)

    if state.get("symbol") != SYMBOL:
        raise ValueError(
            f"State file belongs to {state.get('symbol')}, not {SYMBOL}. "
            "Delete paper_state.json before changing SYMBOL."
        )

    return state


def save_state(state):
    """Persist fake balances, positions, pending orders, and trade history."""

    with STATE_FILE.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2)


# ==========================================
# SECTION 6 — LOCAL PAPER BROKER
# ==========================================


def execute_buy(account, execution_date, market_open):
    """Simulate buying with all available fake cash at the day's open."""

    execution_price = market_open * (1 + SLIPPAGE_RATE)

    trade_value = account["cash"] / (1 + COMMISSION_RATE)
    commission = trade_value * COMMISSION_RATE

    shares = trade_value / execution_price
    total_cost = trade_value + commission

    account["cash"] -= total_cost
    account["shares"] = shares
    account["entry_price"] = execution_price
    account["entry_date"] = execution_date
    account["entry_total_cost"] = total_cost
    account["pending_order"] = None

    return execution_price


def execute_sell(account, execution_date, market_open):
    """Simulate selling the entire fake position at the day's open."""

    execution_price = market_open * (1 - SLIPPAGE_RATE)

    gross_value = account["shares"] * execution_price
    commission = gross_value * COMMISSION_RATE
    net_value = gross_value - commission

    trade_profit = net_value - account["entry_total_cost"]
    trade_return = trade_profit / account["entry_total_cost"] * 100

    account["cash"] += net_value
    account["realized_pnl"] += trade_profit

    account["trades"].append({
        "entry_date": account["entry_date"],
        "exit_date": execution_date,
        "entry_price": account["entry_price"],
        "exit_price": execution_price,
        "pnl": trade_profit,
        "return_percent": trade_return,
    })

    account["shares"] = 0.0
    account["entry_price"] = None
    account["entry_date"] = None
    account["entry_total_cost"] = None
    account["pending_order"] = None

    return execution_price, trade_profit


# ==========================================
# SECTION 7 — PENDING NEXT-OPEN ORDERS
# ==========================================


def try_execute_pending_order(account, raw_data):
    """
    Execute yesterday's queued signal at the first later daily opening price.

    The strategy generates a signal only after a daily candle has completed.
    The order is therefore queued and filled on the next available trading
    day's open, matching the execution model used by our backtester.
    """

    pending = account["pending_order"]

    if pending is None:
        return None

    signal_date = pd.Timestamp(pending["signal_date"])

    later_rows = raw_data[
        pd.to_datetime(raw_data.index).tz_localize(None) > signal_date
    ]

    if len(later_rows) == 0:
        return None

    execution_date = pd.Timestamp(later_rows.index[0]).date().isoformat()
    market_open = float(later_rows.iloc[0]["Open"])

    if pending["side"] == "BUY" and account["shares"] == 0:
        price = execute_buy(account, execution_date, market_open)
        return f"EXECUTED BUY at ${price:.2f} on {execution_date}"

    if pending["side"] == "SELL" and account["shares"] > 0:
        price, pnl = execute_sell(account, execution_date, market_open)
        return (
            f"EXECUTED SELL at ${price:.2f} on {execution_date} "
            f"| trade P/L ${pnl:+.2f}"
        )

    # Position state no longer matches the queued order, so discard it.
    account["pending_order"] = None
    return "Discarded stale pending order"


# ==========================================
# SECTION 8 — CREATE NEW SIGNALS
# ==========================================


def queue_new_signal(strategy_id, account, completed_data):
    """Generate at most one signal for each newly completed daily candle."""

    signal_date = pd.Timestamp(completed_data.index[-1]).date().isoformat()

    # Running the script repeatedly must not generate duplicate orders from
    # the same completed candle.
    if account["last_signal_bar"] == signal_date:
        return "No new completed daily candle"

    has_position = account["shares"] > 0

    signal = signal_for_strategy(
        strategy_id,
        completed_data,
        has_position,
    )

    account["last_signal_bar"] = signal_date

    if signal in {"BUY", "SELL"}:
        account["pending_order"] = {
            "side": signal,
            "signal_date": signal_date,
        }

        return f"QUEUED {signal} from {signal_date} close"

    return f"HOLD on {signal_date}"


# ==========================================
# SECTION 9 — ACCOUNT VALUATION
# ==========================================


def account_value(account, latest_price):
    return account["cash"] + account["shares"] * latest_price


def unrealized_pnl(account, latest_price):
    if account["shares"] == 0:
        return 0.0

    current_value = account["shares"] * latest_price
    return current_value - account["entry_total_cost"]


# ==========================================
# SECTION 10 — MAIN PAPER-TRADING RUN
# ==========================================


def main():
    print("Downloading latest market data...")

    raw_data = download_data()
    completed_data = latest_completed_bar(raw_data)
    completed_data = add_indicators(completed_data)

    if len(completed_data) < 201:
        raise ValueError("Not enough history yet to calculate MA200.")

    state = load_state()

    latest_market_price = float(raw_data.iloc[-1]["Close"])
    latest_completed_date = pd.Timestamp(completed_data.index[-1]).date().isoformat()

    print()
    print("========================================")
    print("V0.8 LOCAL PAPER TRADER")
    print("========================================")
    print(f"Symbol: {SYMBOL}")
    print(f"Latest completed signal bar: {latest_completed_date}")
    print(f"Latest available price: ${latest_market_price:.2f}")

    for strategy_id, strategy in STRATEGIES.items():
        account = state["accounts"][strategy_id]

        execution_message = try_execute_pending_order(account, raw_data)
        signal_message = queue_new_signal(strategy_id, account, completed_data)

        value = account_value(account, latest_market_price)
        total_return = (value / STARTING_CASH - 1) * 100
        open_pnl = unrealized_pnl(account, latest_market_price)

        print()
        print(f"----- {strategy['name']} -----")

        if execution_message:
            print(execution_message)

        print(signal_message)
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
                f"Pending order: {pending['side']} "
                f"after {pending['signal_date']} close"
            )
        else:
            print("Pending order: none")

    save_state(state)

    print()
    print(f"State saved locally to {STATE_FILE}")
    print("No broker is connected. No real orders were sent.")


if __name__ == "__main__":
    main()
