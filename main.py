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

# First 70% of history = research/training.
# Last 30% = unseen test data.
train_ratio = 0.70

# Strategies we're allowed to test.
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

    # Flatten yfinance's MultiIndex columns:
    # ("Close", "AAPL") -> "Close"
    data.columns = data.columns.get_level_values(0)

    return data


market_data = {}

print("Downloading market data...")

for symbol in symbols:
    market_data[symbol] = download_data(symbol)
    print(f"Downloaded {symbol}: {len(market_data[symbol])} days")


# ==========================================
# SECTION 3 — TRAIN / TEST SPLIT
# ==========================================

training_data = {}
testing_data = {}

for symbol, data in market_data.items():

    split_index = int(
        len(data) * train_ratio
    )

    training_data[symbol] = (
        data.iloc[:split_index].copy()
    )

    testing_data[symbol] = (
        data.iloc[split_index:].copy()
    )


# ==========================================
# SECTION 4 — BACKTEST FUNCTION
# ==========================================

def backtest(
    data,
    fast_ma,
    slow_ma,
    starting_cash=10000.0,
):

    data = data.copy()

    # --------------------------------------
    # INDICATORS
    # --------------------------------------

    data["FastMA"] = (
        data["Close"]
        .rolling(window=fast_ma)
        .mean()
    )

    data["SlowMA"] = (
        data["Close"]
        .rolling(window=slow_ma)
        .mean()
    )


    # --------------------------------------
    # SIGNALS
    # --------------------------------------

    data["Signal"] = "HOLD"

    previous_fast = (
        data["FastMA"].shift(1)
    )

    previous_slow = (
        data["SlowMA"].shift(1)
    )


    # BUY when the fast MA crosses ABOVE
    # the slow MA.
    buy_condition = (
        (previous_fast <= previous_slow)
        &
        (data["FastMA"] > data["SlowMA"])
    )


    # SELL when the fast MA crosses BELOW
    # the slow MA.
    sell_condition = (
        (previous_fast >= previous_slow)
        &
        (data["FastMA"] < data["SlowMA"])
    )


    data.loc[
        buy_condition,
        "Signal"
    ] = "BUY"


    data.loc[
        sell_condition,
        "Signal"
    ] = "SELL"


    data.loc[
        data["SlowMA"].isna(),
        "Signal"
    ] = "WAIT"


    # --------------------------------------
    # ACCOUNT STATE
    # --------------------------------------

    cash = starting_cash
    shares = 0.0

    entry_price = None
    entry_date = None
    entry_total_cost = None

    trade_history = []

    portfolio_history = [
        (data.index[0], starting_cash)
    ]

    exposure_days = 0


    # --------------------------------------
    # TRADING LOOP
    # --------------------------------------

    for i in range(len(data) - 1):

        today = data.iloc[i]
        tomorrow = data.iloc[i + 1]

        signal = today["Signal"]

        execution_date = (
            data.index[i + 1]
        )

        tomorrow_open = (
            tomorrow["Open"]
        )


        # ==================================
        # BUY
        # ==================================

        if (
            signal == "BUY"
            and shares == 0
        ):

            # Slippage means we pay slightly
            # MORE than the quoted open.
            execution_price = (
                tomorrow_open
                * (1 + slippage_rate)
            )


            # We cannot spend all cash directly
            # because commission must also be paid.
            trade_value = (
                cash
                / (1 + commission_rate)
            )

            buy_commission = (
                trade_value
                * commission_rate
            )

            shares = (
                trade_value
                / execution_price
            )


            # Total money leaving account
            entry_total_cost = (
                trade_value
                + buy_commission
            )

            cash -= entry_total_cost

            entry_price = execution_price
            entry_date = execution_date


        # ==================================
        # SELL
        # ==================================

        elif (
            signal == "SELL"
            and shares > 0
        ):

            # Slippage means we receive slightly
            # LESS than the quoted open.
            execution_price = (
                tomorrow_open
                * (1 - slippage_rate)
            )

            gross_sale_value = (
                shares
                * execution_price
            )

            sell_commission = (
                gross_sale_value
                * commission_rate
            )

            net_sale_value = (
                gross_sale_value
                - sell_commission
            )


            # This is the TRUE P/L for this trade:
            #
            # money received from selling
            # minus
            # all money spent entering.
            trade_profit = (
                net_sale_value
                - entry_total_cost
            )

            trade_return = (
                trade_profit
                / entry_total_cost
            ) * 100


            cash += net_sale_value


            trade_history.append({
                "Entry Date": entry_date,
                "Exit Date": execution_date,
                "Entry Price": entry_price,
                "Exit Price": execution_price,
                "P/L": trade_profit,
                "Return %": trade_return,
            })


            shares = 0.0

            entry_price = None
            entry_date = None
            entry_total_cost = None


        # ==================================
        # END-OF-DAY PORTFOLIO VALUE
        # ==================================

        daily_value = (
            cash
            + shares * tomorrow["Close"]
        )

        portfolio_history.append(
            (
                execution_date,
                daily_value,
            )
        )

        if shares > 0:
            exposure_days += 1


    # ======================================
    # FINAL ACCOUNT VALUE
    # ======================================

    latest_close = (
        data["Close"].iloc[-1]
    )


    unrealized_profit = 0.0


    # If we still own shares at the end,
    # estimate what we'd receive if we
    # liquidated them now.
    if shares > 0:

        final_sell_price = (
            latest_close
            * (1 - slippage_rate)
        )

        final_gross_value = (
            shares
            * final_sell_price
        )

        final_commission = (
            final_gross_value
            * commission_rate
        )

        final_net_value = (
            final_gross_value
            - final_commission
        )

        final_portfolio_value = (
            cash
            + final_net_value
        )

        unrealized_profit = (
            final_net_value
            - entry_total_cost
        )

    else:

        final_portfolio_value = cash


    # Make the final equity point reflect
    # liquidation costs too.
    portfolio_history[-1] = (
        portfolio_history[-1][0],
        final_portfolio_value,
    )


    total_profit = (
        final_portfolio_value
        - starting_cash
    )

    total_return = (
        total_profit
        / starting_cash
    ) * 100


    # ======================================
    # TRADE STATISTICS
    # ======================================

    trades = pd.DataFrame(
        trade_history
    )

    number_of_trades = len(trades)


    if number_of_trades > 0:

        winners = trades[
            trades["P/L"] > 0
        ]

        losers = trades[
            trades["P/L"] < 0
        ]


        win_count = len(winners)
        loss_count = len(losers)


        win_rate = (
            win_count
            / number_of_trades
        ) * 100


        gross_profit = (
            winners["P/L"].sum()
        )

        gross_loss = abs(
            losers["P/L"].sum()
        )


        if gross_loss > 0:

            profit_factor = (
                gross_profit
                / gross_loss
            )

        else:

            profit_factor = float("inf")


        expectancy = (
            trades["P/L"].mean()
        )


    else:

        win_count = 0
        loss_count = 0

        win_rate = 0
        gross_profit = 0
        gross_loss = 0

        profit_factor = 0
        expectancy = 0


    # ======================================
    # EQUITY CURVE
    # ======================================

    portfolio_df = pd.DataFrame(
        portfolio_history,
        columns=[
            "Date",
            "Strategy",
        ],
    )

    portfolio_df.set_index(
        "Date",
        inplace=True,
    )


    # ======================================
    # DAILY RETURNS
    # ======================================

    portfolio_df[
        "Daily Return"
    ] = (
        portfolio_df["Strategy"]
        .pct_change()
    )

    daily_returns = (
        portfolio_df["Daily Return"]
        .dropna()
    )


    # ======================================
    # SHARPE RATIO
    # ======================================

    daily_std = (
        daily_returns.std()
    )

    if daily_std > 0:

        sharpe_ratio = (
            daily_returns.mean()
            / daily_std
        ) * (252 ** 0.5)

    else:

        sharpe_ratio = 0


    # ======================================
    # MAX DRAWDOWN
    # ======================================

    peaks = (
        portfolio_df["Strategy"]
        .cummax()
    )

    drawdowns = (
        portfolio_df["Strategy"]
        / peaks
        - 1
    )

    max_drawdown = (
        drawdowns.min()
        * 100
    )


    # ======================================
    # EXPOSURE
    # ======================================

    exposure_percent = (
        exposure_days
        / max(len(data) - 1, 1)
    ) * 100


    # ======================================
    # CAGR
    # ======================================

    total_days = (
        data.index[-1]
        - data.index[0]
    ).days

    years = (
        total_days
        / 365.25
    )


    if years > 0:

        cagr = (
            (
                final_portfolio_value
                / starting_cash
            ) ** (1 / years)
            - 1
        ) * 100

    else:

        cagr = 0


    # ======================================
    # BUY AND HOLD BENCHMARK
    # ======================================

    benchmark_history = []

    benchmark_cash = starting_cash
    benchmark_shares = 0.0


    first_valid_index = (
        data["SlowMA"]
        .first_valid_index()
    )


    if first_valid_index is not None:

        first_valid_position = (
            data.index.get_loc(
                first_valid_index
            )
        )

        benchmark_buy_position = (
            first_valid_position + 1
        )

    else:

        benchmark_buy_position = (
            len(data)
        )


    for i in range(len(data)):

        date = data.index[i]


        # Before benchmark entry,
        # it simply stays in cash.
        if i < benchmark_buy_position:

            benchmark_value = (
                starting_cash
            )


        else:

            # Buy once at the first allowed
            # next-day open.
            if benchmark_shares == 0:

                benchmark_buy_price = (
                    data["Open"].iloc[i]
                    * (1 + slippage_rate)
                )

                benchmark_trade_value = (
                    benchmark_cash
                    / (1 + commission_rate)
                )

                buy_commission = (
                    benchmark_trade_value
                    * commission_rate
                )

                benchmark_shares = (
                    benchmark_trade_value
                    / benchmark_buy_price
                )

                benchmark_cash -= (
                    benchmark_trade_value
                    + buy_commission
                )


            benchmark_value = (
                benchmark_cash
                + benchmark_shares
                * data["Close"].iloc[i]
            )


        benchmark_history.append(
            (
                date,
                benchmark_value,
            )
        )


    benchmark_df = pd.DataFrame(
        benchmark_history,
        columns=[
            "Date",
            "Benchmark",
        ],
    )

    benchmark_df.set_index(
        "Date",
        inplace=True,
    )


    # Include final selling costs
    if benchmark_shares > 0:

        benchmark_final_price = (
            latest_close
            * (1 - slippage_rate)
        )

        benchmark_gross = (
            benchmark_shares
            * benchmark_final_price
        )

        benchmark_commission = (
            benchmark_gross
            * commission_rate
        )

        benchmark_final_value = (
            benchmark_cash
            + benchmark_gross
            - benchmark_commission
        )

    else:

        benchmark_final_value = (
            starting_cash
        )


    benchmark_df.iloc[
        -1,
        benchmark_df.columns.get_loc(
            "Benchmark"
        )
    ] = benchmark_final_value


    benchmark_return = (
        (
            benchmark_final_value
            / starting_cash
        )
        - 1
    ) * 100


    # Combine both equity curves
    equity_curve = (
        portfolio_df[["Strategy"]]
        .join(
            benchmark_df[["Benchmark"]],
            how="left",
        )
    )


    # ======================================
    # ACCOUNTING CHECK
    # ======================================

    closed_trade_profit = (
        trades["P/L"].sum()
        if number_of_trades > 0
        else 0
    )

    accounted_profit = (
        closed_trade_profit
        + unrealized_profit
    )

    accounting_error = (
        total_profit
        - accounted_profit
    )


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

        symbol_results.append(
            result
        )


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


parameter_table = pd.DataFrame(
    parameter_results
)


# ==========================================
# SECTION 6 — SELECT TRAINING WINNER
# ==========================================

# We choose using TRAINING DATA ONLY.
#
# The test data has not influenced this choice.

best_row = (
    parameter_table
    .sort_values(
        "Average Train Sharpe",
        ascending=False,
    )
    .iloc[0]
)


selected_fast_ma = int(
    best_row["Fast MA"]
)

selected_slow_ma = int(
    best_row["Slow MA"]
)


# ==========================================
# SECTION 7 — UNSEEN TEST
# ==========================================

test_results = []


for symbol in symbols:

    train_result = backtest(
        training_data[symbol],
        selected_fast_ma,
        selected_slow_ma,
        starting_cash,
    )


    test_result = backtest(
        testing_data[symbol],
        selected_fast_ma,
        selected_slow_ma,
        starting_cash,
    )


    test_results.append({
        "Symbol": symbol,

        "Train Return": train_result["return"],
        "Test Return": test_result["return"],

        "Test Buy & Hold": test_result["benchmark_return"],

        "Test Sharpe": test_result["sharpe"],

        "Test Drawdown": test_result["max_drawdown"],

        "Test Profit Factor": test_result["profit_factor"],

        "Test Trades": test_result["trade_count"],

        "Accounting Error": test_result["accounting_error"],
    })


test_table = pd.DataFrame(
    test_results
)


# ==========================================
# SECTION 8 — GENERALIZATION CHECK
# ==========================================

average_train_return = (
    test_table["Train Return"].mean()
)

average_test_return = (
    test_table["Test Return"].mean()
)

average_test_benchmark = (
    test_table["Test Buy & Hold"].mean()
)

average_test_sharpe = (
    test_table["Test Sharpe"].mean()
)


if (
    average_train_return > 0
    and average_test_return <= 0
):

    research_status = (
        "REJECT FOR NOW: "
        "positive training performance "
        "did not survive unseen testing."
    )

elif average_test_return <= 0:

    research_status = (
        "REJECT FOR NOW: "
        "average unseen test return "
        "was not positive."
    )

else:

    research_status = (
        "KEEP RESEARCHING: "
        "unseen test return was positive, "
        "but this is not proof of an edge."
    )


# ==========================================
# SECTION 9 — OUTPUT
# ==========================================

print()
print("========================================")
print("V0.6 QUANT RESEARCH")
print("========================================")


print()
print("----- DATA SPLIT -----")

print(
    f"Training: {train_ratio * 100:.0f}%"
)

print(
    f"Testing: "
    f"{(1 - train_ratio) * 100:.0f}%"
)


print()
print("----- PARAMETER TRAINING RESULTS -----")

print(
    parameter_table.to_string(
        index=False,
        formatters={
            "Average Train Return":
                lambda x: f"{x:.2f}%",

            "Average Train Sharpe":
                lambda x: f"{x:.2f}",

            "Average Train Drawdown":
                lambda x: f"{x:.2f}%",
        },
    )
)


print()
print("----- SELECTED STRATEGY -----")

print(
    f"MA{selected_fast_ma} / "
    f"MA{selected_slow_ma}"
)

print(
    "Selected using training Sharpe only."
)


print()
print("----- UNSEEN TEST RESULTS -----")

print(
    test_table.to_string(
        index=False,
        formatters={
            "Train Return":
                lambda x: f"{x:.2f}%",

            "Test Return":
                lambda x: f"{x:.2f}%",

            "Test Buy & Hold":
                lambda x: f"{x:.2f}%",

            "Test Sharpe":
                lambda x: f"{x:.2f}",

            "Test Drawdown":
                lambda x: f"{x:.2f}%",

            "Test Profit Factor":
                lambda x: f"{x:.2f}",

            "Accounting Error":
                lambda x: f"${x:.6f}",
        },
    )
)


print()
print("----- TEST SUMMARY -----")

print(
    f"Average train return: "
    f"{average_train_return:.2f}%"
)

print(
    f"Average unseen test return: "
    f"{average_test_return:.2f}%"
)

print(
    f"Average unseen buy & hold: "
    f"{average_test_benchmark:.2f}%"
)

print(
    f"Average unseen Sharpe: "
    f"{average_test_sharpe:.2f}"
)


print()
print("----- RESEARCH STATUS -----")
print(research_status)


# ==========================================
# SECTION 10 — ACCOUNTING CHECK
# ==========================================

print()
print("----- ACCOUNTING CHECK -----")

max_accounting_error = (
    test_table["Accounting Error"]
    .abs()
    .max()
)

print(
    f"Maximum accounting error: "
    f"${max_accounting_error:.8f}"
)

if max_accounting_error < 0.01:

    print(
        "Accounting check: PASS"
    )

else:

    print(
        "Accounting check: WARNING"
    )


# ==========================================
# SECTION 11 — EXAMPLE EQUITY CHART
# ==========================================

# Show one stock in detail so the chart
# doesn't become unreadable.

chart_symbol = symbols[0]

chart_result = backtest(
    testing_data[chart_symbol],
    selected_fast_ma,
    selected_slow_ma,
    starting_cash,
)

equity = (
    chart_result["equity_curve"]
)


plt.figure(
    figsize=(14, 7)
)

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
    f"{chart_symbol} — "
    f"Unseen Test Equity Curve"
)

plt.xlabel("Date")
plt.ylabel("Portfolio value ($)")

plt.legend()
plt.grid()

plt.tight_layout()
plt.show()