import yfinance as yf
import pandas as pd
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt


# ==========================================
# SECTION 1 — SETTINGS
# ==========================================

symbols = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
]

period = "5y"
starting_cash = 10000.0
commission_rate = 0.001      # 0.10%
slippage_rate = 0.0005       # 0.05%
train_ratio = 0.70

# We are no longer testing only MA crossovers.
# Each item below is a separate candidate strategy.
strategy_candidates = [
    {
        "name": "Trend MA50/200",
        "family": "trend",
        "fast": 50,
        "slow": 200,
    },
    {
        "name": "Trend MA20/100",
        "family": "trend",
        "fast": 20,
        "slow": 100,
    },
    {
        "name": "Momentum 60/0",
        "family": "momentum",
        "lookback": 60,
        "threshold": 0.0,
    },
    {
        "name": "Momentum 120/0",
        "family": "momentum",
        "lookback": 120,
        "threshold": 0.0,
    },
    {
        "name": "Breakout 55/20",
        "family": "breakout",
        "entry_window": 55,
        "exit_window": 20,
    },
    {
        "name": "Breakout 20/10",
        "family": "breakout",
        "entry_window": 20,
        "exit_window": 10,
    },
    {
        "name": "Mean Reversion RSI2",
        "family": "mean_reversion",
        "rsi_period": 2,
        "buy_rsi": 10,
        "sell_rsi": 70,
        "trend_ma": 200,
    },
]


# ==========================================
# SECTION 2 — DOWNLOAD MARKET DATA
# ==========================================


def download_data(symbol):
    data = yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data.dropna().copy()


market_data = {}

print("Downloading market data...")

for symbol in symbols:
    market_data[symbol] = download_data(symbol)
    print(f"Downloaded {symbol}: {len(market_data[symbol])} days")


# ==========================================
# SECTION 3 — TRAIN / TEST BOUNDARIES
# ==========================================

split_positions = {}
training_data = {}

for symbol, data in market_data.items():
    split_position = int(len(data) * train_ratio)
    split_positions[symbol] = split_position
    training_data[symbol] = data.iloc[:split_position].copy()


# ==========================================
# SECTION 4 — INDICATOR HELPERS
# ==========================================


def calculate_rsi(close, period):
    """Calculate RSI from a closing-price Series."""

    price_change = close.diff()

    gains = price_change.clip(lower=0)
    losses = -price_change.clip(upper=0)

    average_gain = gains.rolling(period).mean()
    average_loss = losses.rolling(period).mean()

    relative_strength = average_gain / average_loss

    return 100 - (100 / (1 + relative_strength))


# ==========================================
# SECTION 5 — STRATEGY SIGNAL GENERATOR
# ==========================================


def generate_signals(data, strategy):
    """
    Add a Signal column for one strategy candidate.

    Signals are created using today's completed daily data.
    The backtester executes them at the NEXT trading day's open.
    """

    data = data.copy()
    data["Signal"] = "HOLD"

    family = strategy["family"]

    # --------------------------------------
    # TREND FOLLOWING
    # --------------------------------------

    if family == "trend":
        fast = strategy["fast"]
        slow = strategy["slow"]

        data["FastMA"] = data["Close"].rolling(fast).mean()
        data["SlowMA"] = data["Close"].rolling(slow).mean()

        previous_fast = data["FastMA"].shift(1)
        previous_slow = data["SlowMA"].shift(1)

        buy = (
            (previous_fast <= previous_slow)
            & (data["FastMA"] > data["SlowMA"])
        )

        sell = (
            (previous_fast >= previous_slow)
            & (data["FastMA"] < data["SlowMA"])
        )

        data.loc[buy, "Signal"] = "BUY"
        data.loc[sell, "Signal"] = "SELL"
        data.loc[data["SlowMA"].isna(), "Signal"] = "WAIT"

    # --------------------------------------
    # MOMENTUM
    # --------------------------------------

    elif family == "momentum":
        lookback = strategy["lookback"]
        threshold = strategy["threshold"]

        data["Momentum"] = data["Close"].pct_change(lookback)

        previous_momentum = data["Momentum"].shift(1)

        buy = (
            (previous_momentum <= threshold)
            & (data["Momentum"] > threshold)
        )

        sell = (
            (previous_momentum >= threshold)
            & (data["Momentum"] < threshold)
        )

        data.loc[buy, "Signal"] = "BUY"
        data.loc[sell, "Signal"] = "SELL"
        data.loc[data["Momentum"].isna(), "Signal"] = "WAIT"

    # --------------------------------------
    # BREAKOUT
    # --------------------------------------

    elif family == "breakout":
        entry_window = strategy["entry_window"]
        exit_window = strategy["exit_window"]

        # shift(1) is critical: today's close is compared only
        # with levels that were already known before today closed.
        data["EntryHigh"] = (
            data["High"]
            .rolling(entry_window)
            .max()
            .shift(1)
        )

        data["ExitLow"] = (
            data["Low"]
            .rolling(exit_window)
            .min()
            .shift(1)
        )

        data.loc[
            data["Close"] > data["EntryHigh"],
            "Signal",
        ] = "BUY"

        data.loc[
            data["Close"] < data["ExitLow"],
            "Signal",
        ] = "SELL"

        data.loc[
            data["EntryHigh"].isna() | data["ExitLow"].isna(),
            "Signal",
        ] = "WAIT"

    # --------------------------------------
    # MEAN REVERSION
    # --------------------------------------

    elif family == "mean_reversion":
        rsi_period = strategy["rsi_period"]
        buy_rsi = strategy["buy_rsi"]
        sell_rsi = strategy["sell_rsi"]
        trend_ma = strategy["trend_ma"]

        data["RSI"] = calculate_rsi(data["Close"], rsi_period)
        data["TrendMA"] = data["Close"].rolling(trend_ma).mean()

        # Only buy short-term weakness while the long-term
        # price trend is still above its trend average.
        data.loc[
            (data["RSI"] < buy_rsi)
            & (data["Close"] > data["TrendMA"]),
            "Signal",
        ] = "BUY"

        data.loc[
            data["RSI"] > sell_rsi,
            "Signal",
        ] = "SELL"

        data.loc[
            data["TrendMA"].isna() | data["RSI"].isna(),
            "Signal",
        ] = "WAIT"

    else:
        raise ValueError(f"Unknown strategy family: {family}")

    return data


# ==========================================
# SECTION 6 — GENERIC BACKTESTER
# ==========================================


def backtest(
    data,
    strategy,
    starting_cash=10000.0,
    trade_start=None,
    force_close_at_end=True,
):
    """
    Backtest any strategy produced by generate_signals().

    trade_start lets us include old rows only for indicator warm-up.
    No order may execute before trade_start.
    """

    data = generate_signals(data, strategy)

    if trade_start is None:
        trade_start = data.index[0]

    trade_start = pd.Timestamp(trade_start)

    trading_dates = data.index[data.index >= trade_start]

    if len(trading_dates) == 0:
        raise ValueError("trade_start is after all available data")

    cash = starting_cash
    shares = 0.0

    entry_date = None
    entry_price = None
    entry_total_cost = None

    trade_history = []
    exposure_days = 0
    forced_exit = False

    first_trading_date = trading_dates[0]
    portfolio_history = [(first_trading_date, starting_cash)]

    # --------------------------------------
    # TRADING LOOP
    # --------------------------------------

    for i in range(len(data) - 1):
        today = data.iloc[i]
        tomorrow = data.iloc[i + 1]
        execution_date = data.index[i + 1]

        if execution_date < trade_start:
            continue

        signal = today["Signal"]
        tomorrow_open = tomorrow["Open"]

        # BUY at next day's open
        if signal == "BUY" and shares == 0:
            execution_price = tomorrow_open * (1 + slippage_rate)

            trade_value = cash / (1 + commission_rate)
            buy_commission = trade_value * commission_rate

            shares = trade_value / execution_price
            entry_total_cost = trade_value + buy_commission

            cash -= entry_total_cost

            entry_date = execution_date
            entry_price = execution_price

        # SELL at next day's open
        elif signal == "SELL" and shares > 0:
            execution_price = tomorrow_open * (1 - slippage_rate)

            gross_sale_value = shares * execution_price
            sell_commission = gross_sale_value * commission_rate
            net_sale_value = gross_sale_value - sell_commission

            trade_profit = net_sale_value - entry_total_cost
            trade_return = (trade_profit / entry_total_cost) * 100

            cash += net_sale_value

            trade_history.append({
                "Entry Date": entry_date,
                "Exit Date": execution_date,
                "Entry Price": entry_price,
                "Exit Price": execution_price,
                "P/L": trade_profit,
                "Return %": trade_return,
                "Exit Reason": "SIGNAL",
            })

            shares = 0.0
            entry_date = None
            entry_price = None
            entry_total_cost = None

        daily_value = cash + shares * tomorrow["Close"]
        portfolio_history.append((execution_date, daily_value))

        if shares > 0:
            exposure_days += 1

    # --------------------------------------
    # FORCE-CLOSE OPEN POSITION
    # --------------------------------------

    unrealized_profit = 0.0

    if shares > 0:
        latest_close = data["Close"].iloc[-1]

        liquidation_price = latest_close * (1 - slippage_rate)
        gross_sale_value = shares * liquidation_price
        sell_commission = gross_sale_value * commission_rate
        net_sale_value = gross_sale_value - sell_commission

        unrealized_profit = net_sale_value - entry_total_cost

        if force_close_at_end:
            cash += net_sale_value

            trade_history.append({
                "Entry Date": entry_date,
                "Exit Date": data.index[-1],
                "Entry Price": entry_price,
                "Exit Price": liquidation_price,
                "P/L": unrealized_profit,
                "Return %": (unrealized_profit / entry_total_cost) * 100,
                "Exit Reason": "END",
            })

            shares = 0.0
            entry_date = None
            entry_price = None
            entry_total_cost = None
            unrealized_profit = 0.0
            forced_exit = True

    # --------------------------------------
    # FINAL VALUE
    # --------------------------------------

    if shares > 0:
        latest_close = data["Close"].iloc[-1]
        estimated_exit = latest_close * (1 - slippage_rate)
        gross_value = shares * estimated_exit
        final_value = cash + gross_value * (1 - commission_rate)
    else:
        final_value = cash

    portfolio_history[-1] = (
        portfolio_history[-1][0],
        final_value,
    )

    total_profit = final_value - starting_cash
    total_return = (total_profit / starting_cash) * 100

    # --------------------------------------
    # TRADE STATISTICS
    # --------------------------------------

    trades = pd.DataFrame(trade_history)
    trade_count = len(trades)

    if trade_count > 0:
        winners = trades[trades["P/L"] > 0]
        losers = trades[trades["P/L"] < 0]

        wins = len(winners)
        losses = len(losers)
        win_rate = (wins / trade_count) * 100

        gross_profit = winners["P/L"].sum()
        gross_loss = abs(losers["P/L"].sum())

        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0

        expectancy = trades["P/L"].mean()
    else:
        wins = 0
        losses = 0
        win_rate = 0.0
        profit_factor = 0.0
        expectancy = 0.0

    # --------------------------------------
    # EQUITY CURVE + RISK METRICS
    # --------------------------------------

    portfolio_df = pd.DataFrame(
        portfolio_history,
        columns=["Date", "Strategy"],
    ).set_index("Date")

    daily_returns = portfolio_df["Strategy"].pct_change().dropna()
    daily_std = daily_returns.std()

    if daily_std > 0:
        sharpe = (
            daily_returns.mean()
            / daily_std
        ) * (252 ** 0.5)
    else:
        sharpe = 0.0

    peaks = portfolio_df["Strategy"].cummax()
    drawdowns = portfolio_df["Strategy"] / peaks - 1
    max_drawdown = drawdowns.min() * 100

    possible_exposure_days = max(len(trading_dates) - 1, 1)
    exposure = (exposure_days / possible_exposure_days) * 100

    total_days = (trading_dates[-1] - trading_dates[0]).days
    years = total_days / 365.25

    if years > 0 and final_value > 0:
        cagr = (
            (final_value / starting_cash) ** (1 / years)
            - 1
        ) * 100
    else:
        cagr = 0.0

    # --------------------------------------
    # BUY-AND-HOLD BENCHMARK
    # --------------------------------------

    start_position = data.index.get_loc(first_trading_date)

    benchmark_cash = starting_cash
    benchmark_shares = 0.0
    benchmark_history = []

    for i in range(start_position, len(data)):
        date = data.index[i]

        if benchmark_shares == 0:
            benchmark_buy_price = (
                data["Open"].iloc[i]
                * (1 + slippage_rate)
            )

            benchmark_trade_value = (
                benchmark_cash
                / (1 + commission_rate)
            )

            benchmark_buy_commission = (
                benchmark_trade_value
                * commission_rate
            )

            benchmark_shares = (
                benchmark_trade_value
                / benchmark_buy_price
            )

            benchmark_cash -= (
                benchmark_trade_value
                + benchmark_buy_commission
            )

        benchmark_value = (
            benchmark_cash
            + benchmark_shares * data["Close"].iloc[i]
        )

        benchmark_history.append((date, benchmark_value))

    benchmark_final_price = (
        data["Close"].iloc[-1]
        * (1 - slippage_rate)
    )

    benchmark_gross = benchmark_shares * benchmark_final_price
    benchmark_sell_commission = benchmark_gross * commission_rate

    benchmark_final_value = (
        benchmark_cash
        + benchmark_gross
        - benchmark_sell_commission
    )

    benchmark_history[-1] = (
        benchmark_history[-1][0],
        benchmark_final_value,
    )

    benchmark_return = (
        benchmark_final_value / starting_cash
        - 1
    ) * 100

    benchmark_df = pd.DataFrame(
        benchmark_history,
        columns=["Date", "Benchmark"],
    ).set_index("Date")

    equity_curve = portfolio_df.join(benchmark_df, how="left")

    # --------------------------------------
    # ACCOUNTING CHECK
    # --------------------------------------

    closed_trade_profit = (
        trades["P/L"].sum()
        if trade_count > 0
        else 0.0
    )

    accounting_error = (
        total_profit
        - closed_trade_profit
        - unrealized_profit
    )

    return {
        "strategy": strategy["name"],
        "family": strategy["family"],
        "final_value": final_value,
        "profit": total_profit,
        "return": total_return,
        "benchmark_return": benchmark_return,
        "trade_count": trade_count,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "exposure": exposure,
        "cagr": cagr,
        "forced_exit": forced_exit,
        "open_position": shares > 0,
        "unrealized_profit": unrealized_profit,
        "accounting_error": accounting_error,
        "trades": trades,
        "equity_curve": equity_curve,
    }


# ==========================================
# SECTION 7 — TRAINING RESEARCH
# ==========================================

training_rows = []

for strategy in strategy_candidates:
    results = []

    for symbol in symbols:
        result = backtest(
            training_data[symbol],
            strategy,
            starting_cash,
        )

        results.append(result)

    training_rows.append({
        "Strategy": strategy["name"],
        "Family": strategy["family"],
        "Average Train Return": sum(
            result["return"] for result in results
        ) / len(results),
        "Average Train Sharpe": sum(
            result["sharpe"] for result in results
        ) / len(results),
        "Average Train Drawdown": sum(
            result["max_drawdown"] for result in results
        ) / len(results),
        "Average Train Trades": sum(
            result["trade_count"] for result in results
        ) / len(results),
    })


training_table = pd.DataFrame(training_rows)


# ==========================================
# SECTION 8 — SELECT ONE CANDIDATE
# ==========================================

# Selection uses training data ONLY.
# The unseen test set has no influence on this choice.
selected_row = (
    training_table
    .sort_values(
        "Average Train Sharpe",
        ascending=False,
    )
    .iloc[0]
)

selected_strategy = next(
    strategy
    for strategy in strategy_candidates
    if strategy["name"] == selected_row["Strategy"]
)


# ==========================================
# SECTION 9 — UNSEEN TEST
# ==========================================

unseen_rows = []
unseen_results = {}

for symbol in symbols:
    full_data = market_data[symbol]
    split_position = split_positions[symbol]
    trade_start = full_data.index[split_position]

    # Important:
    # We pass the FULL dataset so long indicators such as MA200 can
    # use historical warm-up rows. But trade_start prevents any order
    # before the unseen test boundary.
    result = backtest(
        full_data,
        selected_strategy,
        starting_cash,
        trade_start=trade_start,
        force_close_at_end=True,
    )

    unseen_results[symbol] = result

    unseen_rows.append({
        "Symbol": symbol,
        "Test Return": result["return"],
        "Buy & Hold": result["benchmark_return"],
        "Sharpe": result["sharpe"],
        "Drawdown": result["max_drawdown"],
        "Profit Factor": result["profit_factor"],
        "Trades": result["trade_count"],
        "Win Rate": result["win_rate"],
        "Exposure": result["exposure"],
        "Forced Exit": "YES" if result["forced_exit"] else "NO",
        "Accounting Error": result["accounting_error"],
    })


unseen_table = pd.DataFrame(unseen_rows)


# ==========================================
# SECTION 10 — GENERALIZATION SUMMARY
# ==========================================

average_test_return = unseen_table["Test Return"].mean()
average_benchmark_return = unseen_table["Buy & Hold"].mean()
average_test_sharpe = unseen_table["Sharpe"].mean()
average_test_drawdown = unseen_table["Drawdown"].mean()
average_test_trades = unseen_table["Trades"].mean()

if average_test_return <= 0:
    research_status = (
        "REJECT FOR NOW: average unseen return was not positive."
    )
elif average_test_sharpe <= 0:
    research_status = (
        "REJECT FOR NOW: unseen risk-adjusted performance was not positive."
    )
elif average_test_return < average_benchmark_return:
    research_status = (
        "KEEP RESEARCHING: positive unseen results, but the strategy "
        "still underperformed buy & hold."
    )
else:
    research_status = (
        "PAPER-TRADE CANDIDATE: positive unseen results and average "
        "outperformance. This is still not proof of a durable edge."
    )


# ==========================================
# SECTION 11 — OUTPUT
# ==========================================

print()
print("========================================")
print("V0.7 MULTI-STRATEGY RESEARCH")
print("========================================")

print()
print("----- DATA SPLIT -----")
print(f"Training: {train_ratio * 100:.0f}%")
print(f"Testing: {(1 - train_ratio) * 100:.0f}%")

print()
print("----- STRATEGY CANDIDATES -----")
print(training_table.to_string(
    index=False,
    formatters={
        "Average Train Return": lambda x: f"{x:.2f}%",
        "Average Train Sharpe": lambda x: f"{x:.2f}",
        "Average Train Drawdown": lambda x: f"{x:.2f}%",
        "Average Train Trades": lambda x: f"{x:.1f}",
    },
))

print()
print("----- SELECTED CANDIDATE -----")
print(selected_strategy["name"])
print(f"Family: {selected_strategy['family']}")
print("Selected using average TRAINING Sharpe only.")

print()
print("----- UNSEEN TEST RESULTS -----")
print(unseen_table.to_string(
    index=False,
    formatters={
        "Test Return": lambda x: f"{x:.2f}%",
        "Buy & Hold": lambda x: f"{x:.2f}%",
        "Sharpe": lambda x: f"{x:.2f}",
        "Drawdown": lambda x: f"{x:.2f}%",
        "Profit Factor": lambda x: (
            "inf" if x == float("inf") else f"{x:.2f}"
        ),
        "Win Rate": lambda x: f"{x:.2f}%",
        "Exposure": lambda x: f"{x:.2f}%",
        "Accounting Error": lambda x: f"${x:.8f}",
    },
))

print()
print("----- TEST SUMMARY -----")
print(f"Average unseen return: {average_test_return:.2f}%")
print(f"Average buy & hold: {average_benchmark_return:.2f}%")
print(f"Average unseen Sharpe: {average_test_sharpe:.2f}")
print(f"Average unseen drawdown: {average_test_drawdown:.2f}%")
print(f"Average completed trades: {average_test_trades:.1f}")
print(
    "Average vs buy & hold: "
    f"{average_test_return - average_benchmark_return:+.2f} percentage points"
)

print()
print("----- RESEARCH STATUS -----")
print(research_status)

print()
print("----- ACCOUNTING CHECK -----")
max_accounting_error = unseen_table["Accounting Error"].abs().max()
print(f"Maximum accounting error: ${max_accounting_error:.8f}")
print(
    "Accounting check: PASS"
    if max_accounting_error < 0.01
    else "Accounting check: WARNING"
)


# ==========================================
# SECTION 12 — AAPL TEST TRADE HISTORY
# ==========================================

print()
print("----- AAPL UNSEEN TRADE HISTORY -----")

aapl_trades = unseen_results["AAPL"]["trades"]

if len(aapl_trades) == 0:
    print("No completed AAPL trades.")
else:
    print(aapl_trades.to_string(
        index=False,
        formatters={
            "Entry Price": lambda x: f"${x:.2f}",
            "Exit Price": lambda x: f"${x:.2f}",
            "P/L": lambda x: f"${x:.2f}",
            "Return %": lambda x: f"{x:.2f}%",
        },
    ))


# ==========================================
# SECTION 13 — AAPL EQUITY CHART
# ==========================================

chart_equity = unseen_results["AAPL"]["equity_curve"]

plt.figure(figsize=(14, 7))

plt.plot(
    chart_equity.index,
    chart_equity["Strategy"],
    label=selected_strategy["name"],
)

plt.plot(
    chart_equity.index,
    chart_equity["Benchmark"],
    label="Buy & Hold",
)

plt.axhline(
    starting_cash,
    linestyle="--",
    label="Starting balance",
)

plt.title(
    f"AAPL unseen test — {selected_strategy['name']}"
)

plt.xlabel("Date")
plt.ylabel("Portfolio value ($)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
