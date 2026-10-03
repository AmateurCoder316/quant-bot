import time
import yfinance as yf
import pandas as pd


# ==========================================
# SECTION 1 — SETTINGS
# ==========================================

symbol = "AAPL"
period = "2y"
starting_cash = 10000.0
commission_rate = 0.001
slippage_rate = 0.0005

# Number of historical trading days to replay.
replay_days = 180

# Pause between replayed days so you can watch it happen.
# Change to 0 for instant replay.
delay_seconds = 0.15


# ==========================================
# SECTION 2 — STRATEGIES
# ==========================================

strategies = {
    "Trend MA50/200": {
        "cash": starting_cash,
        "shares": 0.0,
        "entry_cost": None,
        "entry_price": None,
        "trades": 0,
        "realized_pl": 0.0,
    },
    "Breakout 20/10": {
        "cash": starting_cash,
        "shares": 0.0,
        "entry_cost": None,
        "entry_price": None,
        "trades": 0,
        "realized_pl": 0.0,
    },
    "Mean Reversion RSI2": {
        "cash": starting_cash,
        "shares": 0.0,
        "entry_cost": None,
        "entry_price": None,
        "trades": 0,
        "realized_pl": 0.0,
    },
}


# ==========================================
# SECTION 3 — DOWNLOAD DATA
# ==========================================

print("Downloading historical market data...")

data = yf.download(
    symbol,
    period=period,
    interval="1d",
    auto_adjust=True,
    progress=False,
)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data = data.dropna().copy()


# ==========================================
# SECTION 4 — INDICATORS
# ==========================================

# Trend strategy indicators

data["MA50"] = data["Close"].rolling(50).mean()
data["MA200"] = data["Close"].rolling(200).mean()

# Breakout strategy indicators use PREVIOUS highs/lows only.
data["High20"] = data["High"].rolling(20).max().shift(1)
data["Low10"] = data["Low"].rolling(10).min().shift(1)

# RSI(2) for mean reversion.
delta = data["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(2).mean()
avg_loss = loss.rolling(2).mean()

rs = avg_gain / avg_loss.replace(0, float("nan"))
data["RSI2"] = 100 - (100 / (1 + rs))

# Long-term filter for mean reversion.
data["MA200_MR"] = data["Close"].rolling(200).mean()


# ==========================================
# SECTION 5 — SIGNAL FUNCTIONS
# ==========================================

def trend_signal(previous_row, current_row):
    if pd.isna(current_row["MA200"]):
        return "HOLD"

    if (
        previous_row["MA50"] <= previous_row["MA200"]
        and current_row["MA50"] > current_row["MA200"]
    ):
        return "BUY"

    if (
        previous_row["MA50"] >= previous_row["MA200"]
        and current_row["MA50"] < current_row["MA200"]
    ):
        return "SELL"

    return "HOLD"


def breakout_signal(current_row):
    if pd.isna(current_row["High20"]) or pd.isna(current_row["Low10"]):
        return "HOLD"

    if current_row["Close"] > current_row["High20"]:
        return "BUY"

    if current_row["Close"] < current_row["Low10"]:
        return "SELL"

    return "HOLD"


def mean_reversion_signal(current_row, holding):
    if pd.isna(current_row["RSI2"]) or pd.isna(current_row["MA200_MR"]):
        return "HOLD"

    # Buy only while the long-term trend is positive.
    if (
        not holding
        and current_row["Close"] > current_row["MA200_MR"]
        and current_row["RSI2"] < 10
    ):
        return "BUY"

    # Exit after the short-term oversold condition mean-reverts.
    if holding and current_row["RSI2"] > 70:
        return "SELL"

    return "HOLD"


# ==========================================
# SECTION 6 — ORDER EXECUTION
# ==========================================

def execute_buy(account, open_price):
    execution_price = open_price * (1 + slippage_rate)

    trade_value = account["cash"] / (1 + commission_rate)
    commission = trade_value * commission_rate

    shares = trade_value / execution_price
    total_cost = trade_value + commission

    account["cash"] -= total_cost
    account["shares"] = shares
    account["entry_cost"] = total_cost
    account["entry_price"] = execution_price

    return execution_price


def execute_sell(account, open_price):
    execution_price = open_price * (1 - slippage_rate)

    gross_value = account["shares"] * execution_price
    commission = gross_value * commission_rate
    net_value = gross_value - commission

    trade_profit = net_value - account["entry_cost"]

    account["cash"] += net_value
    account["realized_pl"] += trade_profit
    account["trades"] += 1

    account["shares"] = 0.0
    account["entry_cost"] = None
    account["entry_price"] = None

    return execution_price, trade_profit


# ==========================================
# SECTION 7 — REPLAY WINDOW
# ==========================================

# We need enough warm-up data for MA200.
minimum_start = 205

start_index = max(
    minimum_start,
    len(data) - replay_days,
)

if start_index >= len(data) - 1:
    raise ValueError("Not enough data for replay.")

print()
print("========================================")
print("V0.8 REPLAY PAPER TRADER")
print("========================================")
print(f"Symbol: {symbol}")
print(f"Replay start: {data.index[start_index].date()}")
print(f"Replay end: {data.index[-1].date()}")
print()
print("Each historical day is revealed one at a time.")
print("Signals use that day's completed data.")
print("Orders execute at the NEXT day's open.")
print()


# ==========================================
# SECTION 8 — REPLAY LOOP
# ==========================================

pending_orders = {
    name: None
    for name in strategies
}

for i in range(start_index, len(data) - 1):
    previous_row = data.iloc[i - 1]
    today = data.iloc[i]
    tomorrow = data.iloc[i + 1]

    today_date = data.index[i]
    tomorrow_date = data.index[i + 1]

    print("=" * 72)
    print(
        f"{today_date.date()} CLOSE ${today['Close']:.2f}"
        f"  -> next open {tomorrow_date.date()}"
    )

    # --------------------------------------
    # 1. EXECUTE ORDERS QUEUED YESTERDAY
    # --------------------------------------

    for name, account in strategies.items():
        pending = pending_orders[name]

        if pending == "BUY" and account["shares"] == 0:
            price = execute_buy(account, today["Open"])
            print(f"{name}: EXECUTED BUY @ ${price:.2f}")

        elif pending == "SELL" and account["shares"] > 0:
            price, trade_profit = execute_sell(account, today["Open"])
            print(
                f"{name}: EXECUTED SELL @ ${price:.2f} "
                f"P/L ${trade_profit:+.2f}"
            )

        pending_orders[name] = None

    # --------------------------------------
    # 2. GENERATE TODAY'S SIGNALS
    # --------------------------------------

    generated_signals = {
        "Trend MA50/200": trend_signal(previous_row, today),
        "Breakout 20/10": breakout_signal(today),
        "Mean Reversion RSI2": mean_reversion_signal(
            today,
            strategies["Mean Reversion RSI2"]["shares"] > 0,
        ),
    }

    for name, signal in generated_signals.items():
        account = strategies[name]

        if signal == "BUY" and account["shares"] == 0:
            pending_orders[name] = "BUY"
            print(f"{name}: BUY signal -> queued for next open")

        elif signal == "SELL" and account["shares"] > 0:
            pending_orders[name] = "SELL"
            print(f"{name}: SELL signal -> queued for next open")

        else:
            print(f"{name}: {signal}")

    # --------------------------------------
    # 3. SHOW ACCOUNT VALUES AT TODAY'S CLOSE
    # --------------------------------------

    for name, account in strategies.items():
        account_value = (
            account["cash"]
            + account["shares"] * today["Close"]
        )

        account_return = (
            account_value / starting_cash - 1
        ) * 100

        print(
            f"  {name}: value ${account_value:.2f} "
            f"({account_return:+.2f}%) "
            f"shares {account['shares']:.4f}"
        )

    if delay_seconds > 0:
        time.sleep(delay_seconds)


# ==========================================
# SECTION 9 — FINAL SUMMARY
# ==========================================

latest_close = data["Close"].iloc[-1]
latest_date = data.index[-1]

print()
print("========================================")
print("REPLAY COMPLETE")
print("========================================")
print(f"Final replay date: {latest_date.date()}")
print(f"Final close: ${latest_close:.2f}")

for name, account in strategies.items():
    final_value = (
        account["cash"]
        + account["shares"] * latest_close
    )

    final_return = (
        final_value / starting_cash - 1
    ) * 100

    unrealized = 0.0

    if account["shares"] > 0:
        marked_value = account["shares"] * latest_close
        unrealized = marked_value - account["entry_cost"]

    print()
    print(f"----- {name} -----")
    print(f"Final account value: ${final_value:.2f}")
    print(f"Return: {final_return:+.2f}%")
    print(f"Realized P/L: ${account['realized_pl']:+.2f}")
    print(f"Unrealized P/L: ${unrealized:+.2f}")
    print(f"Completed trades: {account['trades']}")
    print(f"Shares held: {account['shares']:.4f}")

print()
print("This was a historical replay using fake money only.")
