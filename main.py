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

# First 70% of the history is used for research.
# The final 30% is kept unseen until the final test.
train_ratio = 0.70

candidate_ma_pairs = [
    (5, 20),
    (10, 50),
    (20, 100),
    (50, 200),
]


# ==========================================
# SECTION 2 — DOWNLOAD DATA
# ==========================================

def download_data(symbol):
    data = yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    # yfinance normally gives columns such as:
    # ("Close", "AAPL")
    # We only need the first level: Close, Open, etc.
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
# SECTION 4 — BACKTEST FUNCTION
# ==========================================

def backtest(
    data,
    fast_ma,
    slow_ma,
    starting_cash=10000.0,
    trade_start=None,
    force_close_at_end=True,
):
    """
    Backtest one moving-average crossover strategy.

    data:
        Historical OHLCV data. It may include warm-up history before
        trade_start so indicators are already valid when testing begins.

    trade_start:
        Earliest date on which an order may execute. Data before this date
        is indicator warm-up only and cannot create a trade.

    force_close_at_end:
        If True, any remaining position is sold at the final close with
        slippage and commission. This makes every test fully realized.
    """

    data = data.copy()

    if len(data) <= slow_ma:
        raise ValueError(
            f"Not enough data for MA{slow_ma}: only {len(data)} rows."
        )

    # --------------------------------------
    # INDICATORS
    # --------------------------------------

    data["FastMA"] = data["Close"].rolling(window=fast_ma).mean()
    data["SlowMA"] = data["Close"].rolling(window=slow_ma).mean()

    # --------------------------------------
    # SIGNALS
    # --------------------------------------

    data["Signal"] = "HOLD"

    previous_fast = data["FastMA"].shift(1)
    previous_slow = data["SlowMA"].shift(1)

    buy_condition = (
        (previous_fast <= previous_slow)
        & (data["FastMA"] > data["SlowMA"])
    )

    sell_condition = (
        (previous_fast >= previous_slow)
        & (data["FastMA"] < data["SlowMA"])
    )

    data.loc[buy_condition, "Signal"] = "BUY"
    data.loc[sell_condition, "Signal"] = "SELL"
    data.loc[data["SlowMA"].isna(), "Signal"] = "WAIT"

    # If no explicit start is supplied, the backtest may trade throughout
    # the supplied dataset.
    if trade_start is None:
        trade_start = data.index[0]

    trade_start = pd.Timestamp(trade_start)

    # --------------------------------------
    # ACCOUNT STATE
    # --------------------------------------

    cash = starting_cash
    shares = 0.0

    entry_date = None
    entry_price = None
    entry_total_cost = None

    trade_history = []
    portfolio_history = []

    exposure_days = 0
    forced_exit = False

    # Keep only the equity curve from the actual trading period.
    trading_dates = data.index[data.index >= trade_start]

    if len(trading_dates) == 0:
        raise ValueError("trade_start is after all available data.")

    first_trading_date = trading_dates[0]
    portfolio_history.append((first_trading_date, starting_cash))

    # --------------------------------------
    # TRADING LOOP
    # --------------------------------------

    for i in range(len(data) - 1):
        today = data.iloc[i]
        tomorrow = data.iloc[i + 1]

        execution_date = data.index[i + 1]

        # Rows before the unseen-test boundary may calculate indicators,
        # but they are NOT allowed to execute orders.
        if execution_date < trade_start:
            continue

        signal = today["Signal"]
        tomorrow_open = tomorrow["Open"]

        # ==================================
        # BUY
        # ==================================

        if signal == "BUY" and shares == 0:
            execution_price = tomorrow_open * (1 + slippage_rate)

            # Reserve enough cash to pay commission as well as the shares.
            trade_value = cash / (1 + commission_rate)
            buy_commission = trade_value * commission_rate

            shares = trade_value / execution_price
            entry_total_cost = trade_value + buy_commission

            cash -= entry_total_cost

            entry_date = execution_date
            entry_price = execution_price

        # ==================================
        # SELL
        # ==================================

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

        # ==================================
        # END-OF-DAY EQUITY
        # ==================================

        daily_value = cash + shares * tomorrow["Close"]
        portfolio_history.append((execution_date, daily_value))

        if shares > 0:
            exposure_days += 1

    # ======================================
    # FORCE-CLOSE OPEN POSITION AT THE END
    # ======================================

    latest_date = data.index[-1]
    latest_close = data["Close"].iloc[-1]

    if shares > 0 and force_close_at_end:
        forced_exit = True

        execution_price = latest_close * (1 - slippage_rate)
        gross_sale_value = shares * execution_price
        sell_commission = gross_sale_value * commission_rate
        net_sale_value = gross_sale_value - sell_commission

        trade_profit = net_sale_value - entry_total_cost
        trade_return = (trade_profit / entry_total_cost) * 100

        cash += net_sale_value

        trade_history.append({
            "Entry Date": entry_date,
            "Exit Date": latest_date,
            "Entry Price": entry_price,
            "Exit Price": execution_price,
            "P/L": trade_profit,
            "Return %": trade_return,
            "Exit Reason": "END",
        })

        shares = 0.0
        entry_date = None
        entry_price = None
        entry_total_cost = None

        # Replace the final marked-to-market equity value with the true
        # liquidation value after slippage and commission.
        if portfolio_history and portfolio_history[-1][0] == latest_date:
            portfolio_history[-1] = (latest_date, cash)
        else:
            portfolio_history.append((latest_date, cash))

    open_position = shares > 0

    if open_position:
        # This branch is only used if force_close_at_end=False.
        final_marked_value = cash + shares * latest_close
        unrealized_profit = final_marked_value - entry_total_cost
        final_portfolio_value = final_marked_value
    else:
        unrealized_profit = 0.0
        final_portfolio_value = cash

    total_profit = final_portfolio_value - starting_cash
    total_return = (total_profit / starting_cash) * 100

    # ======================================
    # TRADE STATISTICS
    # ======================================

    trades = pd.DataFrame(trade_history)
    number_of_trades = len(trades)

    if number_of_trades > 0:
        winners = trades[trades["P/L"] > 0]
        losers = trades[trades["P/L"] < 0]

        win_count = len(winners)
        loss_count = len(losers)
        win_rate = (win_count / number_of_trades) * 100

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
        win_count = 0
        loss_count = 0
        win_rate = 0.0
        gross_profit = 0.0
        gross_loss = 0.0
        profit_factor = 0.0
        expectancy = 0.0

    # ======================================
    # EQUITY CURVE + RISK METRICS
    # ======================================

    portfolio_df = pd.DataFrame(
        portfolio_history,
        columns=["Date", "Strategy"],
    )

    # A date can appear twice if trade_start equals a later execution date.
    # Keep the final value for each date.
    portfolio_df = (
        portfolio_df
        .drop_duplicates(subset="Date", keep="last")
        .set_index("Date")
        .sort_index()
    )

    portfolio_df["Daily Return"] = portfolio_df["Strategy"].pct_change()
    daily_returns = portfolio_df["Daily Return"].dropna()

    daily_std = daily_returns.std()

    if pd.notna(daily_std) and daily_std > 0:
        sharpe_ratio = (
            daily_returns.mean() / daily_std
        ) * (252 ** 0.5)
    else:
        sharpe_ratio = 0.0

    peaks = portfolio_df["Strategy"].cummax()
    drawdowns = portfolio_df["Strategy"] / peaks - 1
    max_drawdown = drawdowns.min() * 100

    possible_exposure_days = max(len(portfolio_df) - 1, 1)
    exposure_percent = (exposure_days / possible_exposure_days) * 100

    total_days = (portfolio_df.index[-1] - portfolio_df.index[0]).days
    years = total_days / 365.25

    if years > 0 and final_portfolio_value > 0:
        cagr = (
            (final_portfolio_value / starting_cash) ** (1 / years)
            - 1
        ) * 100
    else:
        cagr = 0.0

    # ======================================
    # BUY-AND-HOLD BENCHMARK
    # ======================================

    # The benchmark begins at the unseen-test boundary too.
    benchmark_rows = data[data.index >= trade_start].copy()

    benchmark_start_date = benchmark_rows.index[0]
    benchmark_start_open = benchmark_rows["Open"].iloc[0]

    benchmark_buy_price = benchmark_start_open * (1 + slippage_rate)
    benchmark_trade_value = starting_cash / (1 + commission_rate)
    benchmark_buy_commission = benchmark_trade_value * commission_rate
    benchmark_shares = benchmark_trade_value / benchmark_buy_price
    benchmark_cash = (
        starting_cash
        - benchmark_trade_value
        - benchmark_buy_commission
    )

    benchmark_values = []

    for date, row in benchmark_rows.iterrows():
        benchmark_value = benchmark_cash + benchmark_shares * row["Close"]
        benchmark_values.append((date, benchmark_value))

    benchmark_final_price = latest_close * (1 - slippage_rate)
    benchmark_gross = benchmark_shares * benchmark_final_price
    benchmark_sell_commission = benchmark_gross * commission_rate
    benchmark_final_value = (
        benchmark_cash
        + benchmark_gross
        - benchmark_sell_commission
    )

    benchmark_df = pd.DataFrame(
        benchmark_values,
        columns=["Date", "Benchmark"],
    ).set_index("Date")

    benchmark_df.iloc[-1, benchmark_df.columns.get_loc("Benchmark")] = (
        benchmark_final_value
    )

    benchmark_return = (
        benchmark_final_value / starting_cash - 1
    ) * 100

    equity_curve = portfolio_df[["Strategy"]].join(
        benchmark_df[["Benchmark"]],
        how="outer",
    ).sort_index()

    # ======================================
    # ACCOUNTING CHECK
    # ======================================

    closed_trade_profit = (
        trades["P/L"].sum()
        if number_of_trades > 0
        else 0.0
    )

    accounted_profit = closed_trade_profit + unrealized_profit
    accounting_error = total_profit - accounted_profit

    # ======================================
    # RETURN RESULTS
    # ======================================

    return {
        "fast_ma": fast_ma,
        "slow_ma": slow_ma,
        "final_value": final_portfolio_value,
        "profit": total_profit,
        "return": total_return,
        "benchmark_value": benchmark_final_value,
        "benchmark_return": benchmark_return,
        "trades": trades,
        "trade_count": number_of_trades,
        "wins": win_count,
        "losses": loss_count,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "sharpe": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "exposure": exposure_percent,
        "cagr": cagr,
        "open_position": open_position,
        "forced_exit": forced_exit,
        "unrealized_profit": unrealized_profit,
        "closed_trade_profit": closed_trade_profit,
        "accounting_error": accounting_error,
        "equity_curve": equity_curve,
        "data": data,
    }


# ==========================================
# SECTION 5 — PARAMETER RESEARCH
# ==========================================

parameter_results = []

for fast_ma, slow_ma in candidate_ma_pairs:
    symbol_results = []

    for symbol in symbols:
        result = backtest(
            training_data[symbol],
            fast_ma,
            slow_ma,
            starting_cash,
        )

        symbol_results.append(result)

    average_return = sum(
        result["return"]
        for result in symbol_results
    ) / len(symbol_results)

    average_sharpe = sum(
        result["sharpe"]
        for result in symbol_results
    ) / len(symbol_results)

    average_drawdown = sum(
        result["max_drawdown"]
        for result in symbol_results
    ) / len(symbol_results)

    parameter_results.append({
        "Fast MA": fast_ma,
        "Slow MA": slow_ma,
        "Average Train Return": average_return,
        "Average Train Sharpe": average_sharpe,
        "Average Train Drawdown": average_drawdown,
    })


parameter_table = pd.DataFrame(parameter_results)


# ==========================================
# SECTION 6 — SELECT TRAINING WINNER
# ==========================================

# IMPORTANT:
# We select the strategy using TRAINING DATA ONLY.
# The unseen test data does not influence this choice.
best_row = (
    parameter_table
    .sort_values(
        "Average Train Sharpe",
        ascending=False,
    )
    .iloc[0]
)

selected_fast_ma = int(best_row["Fast MA"])
selected_slow_ma = int(best_row["Slow MA"])


# ==========================================
# SECTION 7 — UNSEEN TEST
# ==========================================

test_results = []
detailed_test_results = {}

for symbol in symbols:
    full_data = market_data[symbol]
    split_position = split_positions[symbol]

    test_start = full_data.index[split_position]

    # Give the test enough PREVIOUS history to calculate MA200 etc.
    # This warm-up history is read-only: backtest() forbids orders before
    # test_start, so we are not leaking test performance into training.
    warmup_rows = selected_slow_ma + 2
    warmup_start = max(0, split_position - warmup_rows)

    test_with_warmup = full_data.iloc[warmup_start:].copy()

    train_result = backtest(
        training_data[symbol],
        selected_fast_ma,
        selected_slow_ma,
        starting_cash,
    )

    test_result = backtest(
        test_with_warmup,
        selected_fast_ma,
        selected_slow_ma,
        starting_cash,
        trade_start=test_start,
        force_close_at_end=True,
    )

    detailed_test_results[symbol] = test_result

    test_results.append({
        "Symbol": symbol,
        "Train Return": train_result["return"],
        "Test Return": test_result["return"],
        "Test Buy & Hold": test_result["benchmark_return"],
        "Test Sharpe": test_result["sharpe"],
        "Test Drawdown": test_result["max_drawdown"],
        "Test Profit Factor": test_result["profit_factor"],
        "Completed Trades": test_result["trade_count"],
        "Forced Exit": "YES" if test_result["forced_exit"] else "NO",
        "Open Position": "YES" if test_result["open_position"] else "NO",
        "Unrealized P/L": test_result["unrealized_profit"],
        "Accounting Error": test_result["accounting_error"],
    })


test_table = pd.DataFrame(test_results)


# ==========================================
# SECTION 8 — GENERALIZATION CHECK
# ==========================================

average_train_return = test_table["Train Return"].mean()
average_test_return = test_table["Test Return"].mean()
average_test_benchmark = test_table["Test Buy & Hold"].mean()
average_test_sharpe = test_table["Test Sharpe"].mean()

if average_train_return > 0 and average_test_return <= 0:
    research_status = (
        "REJECT FOR NOW: positive training performance "
        "did not survive unseen testing."
    )
elif average_test_return <= 0:
    research_status = (
        "REJECT FOR NOW: average unseen test return was not positive."
    )
else:
    research_status = (
        "KEEP RESEARCHING: unseen test return was positive, "
        "but this is not proof of an edge."
    )


# ==========================================
# SECTION 9 — OUTPUT
# ==========================================

print()
print("========================================")
print("V0.6.1 QUANT RESEARCH")
print("========================================")

print()
print("----- DATA SPLIT -----")
print(f"Training: {train_ratio * 100:.0f}%")
print(f"Testing: {(1 - train_ratio) * 100:.0f}%")

print()
print("----- PARAMETER TRAINING RESULTS -----")
print(
    parameter_table.to_string(
        index=False,
        formatters={
            "Average Train Return": lambda x: f"{x:.2f}%",
            "Average Train Sharpe": lambda x: f"{x:.2f}",
            "Average Train Drawdown": lambda x: f"{x:.2f}%",
        },
    )
)

print()
print("----- SELECTED STRATEGY -----")
print(f"MA{selected_fast_ma} / MA{selected_slow_ma}")
print("Selected using training Sharpe only.")

print()
print("----- UNSEEN TEST RESULTS -----")
print(
    test_table.to_string(
        index=False,
        formatters={
            "Train Return": lambda x: f"{x:.2f}%",
            "Test Return": lambda x: f"{x:.2f}%",
            "Test Buy & Hold": lambda x: f"{x:.2f}%",
            "Test Sharpe": lambda x: f"{x:.2f}",
            "Test Drawdown": lambda x: f"{x:.2f}%",
            "Test Profit Factor": lambda x: f"{x:.2f}",
            "Unrealized P/L": lambda x: f"${x:.2f}",
            "Accounting Error": lambda x: f"${x:.6f}",
        },
    )
)

print()
print("----- TEST SUMMARY -----")
print(f"Average train return: {average_train_return:.2f}%")
print(f"Average unseen test return: {average_test_return:.2f}%")
print(f"Average unseen buy & hold: {average_test_benchmark:.2f}%")
print(f"Average unseen Sharpe: {average_test_sharpe:.2f}")

print()
print("----- RESEARCH STATUS -----")
print(research_status)


# ==========================================
# SECTION 10 — ACCOUNTING CHECK
# ==========================================

print()
print("----- ACCOUNTING CHECK -----")

max_accounting_error = test_table["Accounting Error"].abs().max()

print(f"Maximum accounting error: ${max_accounting_error:.8f}")

if max_accounting_error < 0.01:
    print("Accounting check: PASS")
else:
    print("Accounting check: WARNING")


# ==========================================
# SECTION 11 — EXAMPLE TRADE HISTORY
# ==========================================

chart_symbol = symbols[0]
chart_result = detailed_test_results[chart_symbol]
chart_trades = chart_result["trades"]

print()
print(f"----- {chart_symbol} UNSEEN TRADE HISTORY -----")

if len(chart_trades) == 0:
    print("No trades.")
else:
    print(
        chart_trades.to_string(
            index=False,
            formatters={
                "Entry Price": lambda x: f"${x:.2f}",
                "Exit Price": lambda x: f"${x:.2f}",
                "P/L": lambda x: f"${x:.2f}",
                "Return %": lambda x: f"{x:.2f}%",
            },
        )
    )


# ==========================================
# SECTION 12 — EXAMPLE EQUITY CHART
# ==========================================

equity = chart_result["equity_curve"]

plt.figure(figsize=(14, 7))

plt.plot(
    equity.index,
    equity["Strategy"],
    label="Strategy",
)

plt.plot(
    equity.index,
    equity["Benchmark"],
    label="Buy & Hold",
)

plt.axhline(
    starting_cash,
    linestyle="--",
    label="Starting balance",
)

plt.title(
    f"{chart_symbol} — Unseen Test Equity Curve "
    f"(MA{selected_fast_ma}/MA{selected_slow_ma})"
)

plt.xlabel("Date")
plt.ylabel("Portfolio value ($)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
