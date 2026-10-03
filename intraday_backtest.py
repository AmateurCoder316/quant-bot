import math
from collections import Counter, defaultdict

import pandas as pd

from market_data import DEFAULT_PROVIDER
from strategy_core import (
    COMMISSION_RATE,
    MAX_OPEN_POSITIONS,
    SLIPPAGE_RATE,
    STARTING_CASH,
    SYMBOLS,
    add_indicators,
    risk_sized_trade_value,
    stop_price_from_atr,
    strategy_signal,
)


PERIOD = "60d"
MIN_BARS = 205
BARS_PER_YEAR = 78 * 252


def fresh_position():
    return {
        "shares": 0.0,
        "entry_price": None,
        "entry_time": None,
        "entry_cost": None,
        "stop_price": None,
        "entry_reason": None,
        "pending": None,
    }


def portfolio_value(cash, positions, close_prices):
    value = cash
    for symbol, position in positions.items():
        if position["shares"] > 0:
            value += position["shares"] * close_prices[symbol]
    return value


def open_count(positions):
    return sum(position["shares"] > 0 for position in positions.values())


def download_universe():
    data = {}

    for symbol in SYMBOLS:
        raw = DEFAULT_PROVIDER.history(symbol, period=PERIOD)
        if len(raw) < MIN_BARS:
            raise ValueError(f"Not enough 5-minute data for {symbol}: {len(raw)} bars")
        data[symbol] = add_indicators(raw)

    common_index = data[SYMBOLS[0]].index
    for symbol in SYMBOLS[1:]:
        common_index = common_index.intersection(data[symbol].index)

    common_index = common_index.sort_values()
    if len(common_index) < MIN_BARS:
        raise ValueError("Not enough common timestamps across the symbol universe.")

    return {symbol: frame.loc[common_index].copy() for symbol, frame in data.items()}, common_index


def run_backtest():
    frames, index = download_universe()

    cash = STARTING_CASH
    positions = {symbol: fresh_position() for symbol in SYMBOLS}
    trades = []
    equity = []

    for i in range(MIN_BARS, len(index)):
        timestamp = index[i]
        previous_timestamp = index[i - 1]

        opens = {symbol: float(frames[symbol].loc[timestamp, "Open"]) for symbol in SYMBOLS}
        closes = {symbol: float(frames[symbol].loc[timestamp, "Close"]) for symbol in SYMBOLS}
        lows = {symbol: float(frames[symbol].loc[timestamp, "Low"]) for symbol in SYMBOLS}

        # 1) Execute signals generated on the previous completed bar.
        for symbol in SYMBOLS:
            position = positions[symbol]
            pending = position["pending"]
            if pending is None:
                continue

            if pending["side"] == "BUY" and position["shares"] == 0:
                if open_count(positions) < MAX_OPEN_POSITIONS:
                    marked_value = portfolio_value(cash, positions, opens)
                    atr = float(frames[symbol].loc[previous_timestamp, "ATR14"])
                    execution_price = opens[symbol] * (1 + SLIPPAGE_RATE)
                    allocation = risk_sized_trade_value(
                        marked_value,
                        cash,
                        execution_price,
                        atr,
                    )

                    if allocation > 0:
                        trade_value = allocation / (1 + COMMISSION_RATE)
                        commission = trade_value * COMMISSION_RATE
                        shares = trade_value / execution_price
                        total_cost = trade_value + commission

                        cash -= total_cost
                        positions[symbol] = {
                            "shares": shares,
                            "entry_price": execution_price,
                            "entry_time": timestamp,
                            "entry_cost": total_cost,
                            "stop_price": stop_price_from_atr(execution_price, atr),
                            "entry_reason": pending["reason"],
                            "pending": None,
                        }
                    else:
                        position["pending"] = None
                else:
                    position["pending"] = None

            elif pending["side"] == "SELL" and position["shares"] > 0:
                execution_price = opens[symbol] * (1 - SLIPPAGE_RATE)
                gross = position["shares"] * execution_price
                commission = gross * COMMISSION_RATE
                net = gross - commission
                pnl = net - position["entry_cost"]
                return_pct = pnl / position["entry_cost"] * 100

                cash += net
                trades.append({
                    "symbol": symbol,
                    "entry_time": position["entry_time"],
                    "exit_time": timestamp,
                    "entry_reason": position["entry_reason"],
                    "exit_reason": pending["reason"],
                    "entry_price": position["entry_price"],
                    "exit_price": execution_price,
                    "pnl": pnl,
                    "return_percent": return_pct,
                    "holding_minutes": (timestamp - position["entry_time"]).total_seconds() / 60,
                })
                positions[symbol] = fresh_position()

        # 2) ATR stop-losses during the current bar.
        for symbol in SYMBOLS:
            position = positions[symbol]
            if position["shares"] <= 0 or position["stop_price"] is None:
                continue

            if lows[symbol] <= position["stop_price"]:
                stop_market_price = min(opens[symbol], position["stop_price"])
                execution_price = stop_market_price * (1 - SLIPPAGE_RATE)
                gross = position["shares"] * execution_price
                commission = gross * COMMISSION_RATE
                net = gross - commission
                pnl = net - position["entry_cost"]
                return_pct = pnl / position["entry_cost"] * 100

                cash += net
                trades.append({
                    "symbol": symbol,
                    "entry_time": position["entry_time"],
                    "exit_time": timestamp,
                    "entry_reason": position["entry_reason"],
                    "exit_reason": "atr_stop",
                    "entry_price": position["entry_price"],
                    "exit_price": execution_price,
                    "pnl": pnl,
                    "return_percent": return_pct,
                    "holding_minutes": (timestamp - position["entry_time"]).total_seconds() / 60,
                })
                positions[symbol] = fresh_position()

        # 3) Generate signals from this completed bar for next-bar execution.
        for symbol in SYMBOLS:
            history = frames[symbol].iloc[: i + 1]
            position = positions[symbol]
            signal = strategy_signal(history, position["shares"] > 0)

            if signal.side == "BUY" and position["shares"] == 0:
                position["pending"] = {"side": "BUY", "reason": signal.reason}
            elif signal.side == "SELL" and position["shares"] > 0:
                position["pending"] = {"side": "SELL", "reason": signal.reason}

        value = portfolio_value(cash, positions, closes)
        equity.append({"timestamp": timestamp, "value": value})

    # Mark open positions at the final close for reporting only.
    final_prices = {symbol: float(frames[symbol].iloc[-1]["Close"]) for symbol in SYMBOLS}
    final_value = portfolio_value(cash, positions, final_prices)

    return pd.DataFrame(equity), pd.DataFrame(trades), final_value, positions


def calculate_metrics(equity, trades, final_value):
    total_return = (final_value / STARTING_CASH - 1) * 100

    if equity.empty:
        return {}

    values = equity["value"].astype(float)
    returns = values.pct_change().dropna()
    running_max = values.cummax()
    drawdowns = values / running_max - 1
    max_drawdown = drawdowns.min() * 100

    sharpe = 0.0
    if len(returns) > 1 and returns.std() > 0:
        sharpe = returns.mean() / returns.std() * math.sqrt(BARS_PER_YEAR)

    if trades.empty:
        return {
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe": sharpe,
            "trades": 0,
        }

    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] < 0]

    gross_profit = wins["pnl"].sum()
    gross_loss = abs(losses["pnl"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "trades": len(trades),
        "win_rate": len(wins) / len(trades) * 100,
        "profit_factor": profit_factor,
        "expectancy": trades["pnl"].mean(),
        "average_trade_return": trades["return_percent"].mean(),
        "average_holding_minutes": trades["holding_minutes"].mean(),
    }


def print_report(equity, trades, final_value, positions):
    metrics = calculate_metrics(equity, trades, final_value)

    print("=" * 64)
    print("5-MINUTE PORTFOLIO BACKTEST")
    print("=" * 64)
    print(f"Symbols:             {', '.join(SYMBOLS)}")
    print(f"Starting cash:       ${STARTING_CASH:,.2f}")
    print(f"Final value:         ${final_value:,.2f}")
    print(f"Total return:        {metrics.get('total_return', 0):+.2f}%")
    print(f"Max drawdown:        {metrics.get('max_drawdown', 0):+.2f}%")
    print(f"Sharpe:              {metrics.get('sharpe', 0):.2f}")
    print(f"Completed trades:    {metrics.get('trades', 0)}")

    if not trades.empty:
        print(f"Win rate:            {metrics['win_rate']:.2f}%")
        print(f"Profit factor:       {metrics['profit_factor']:.2f}")
        print(f"Expectancy/trade:    ${metrics['expectancy']:+.2f}")
        print(f"Avg trade return:    {metrics['average_trade_return']:+.2f}%")
        print(f"Avg holding time:    {metrics['average_holding_minutes']:.1f} min")

        print("\nBY SYMBOL")
        for symbol, group in trades.groupby("symbol"):
            print(
                f"{symbol:<6} trades={len(group):>3} "
                f"P/L=${group['pnl'].sum():+8.2f} "
                f"win={((group['pnl'] > 0).mean() * 100):5.1f}%"
            )

        print("\nBY ENTRY SIGNAL")
        for reason, group in trades.groupby("entry_reason"):
            print(
                f"{reason:<24} trades={len(group):>3} "
                f"P/L=${group['pnl'].sum():+8.2f} "
                f"win={((group['pnl'] > 0).mean() * 100):5.1f}%"
            )

        print("\nEXIT REASONS")
        counts = Counter(trades["exit_reason"])
        for reason, count in counts.items():
            print(f"{reason:<24} {count}")

    open_symbols = [symbol for symbol, position in positions.items() if position["shares"] > 0]
    print(f"\nOpen at end:         {', '.join(open_symbols) if open_symbols else 'none'}")
    print("\nNote: yfinance 5-minute history is limited, so this is a short intraday sample.")


if __name__ == "__main__":
    equity, trades, final_value, positions = run_backtest()
    print_report(equity, trades, final_value, positions)
