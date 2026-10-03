import math
from collections import Counter

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
TRADING_DAYS_PER_YEAR = 252


def fresh_position(last_exit_time=None):
    return {
        "shares": 0.0,
        "entry_price": None,
        "entry_time": None,
        "entry_cost": None,
        "entry_raw_price": None,
        "entry_commission": 0.0,
        "entry_slippage_cost": 0.0,
        "stop_price": None,
        "entry_reason": None,
        "pending": None,
        "last_exit_time": last_exit_time,
    }


def portfolio_value(cash, positions, prices):
    value = cash
    for symbol, position in positions.items():
        if position["shares"] > 0:
            value += position["shares"] * prices[symbol]
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


def run_backtest(frames, index, commission_rate, slippage_rate):
    cash = STARTING_CASH
    positions = {symbol: fresh_position() for symbol in SYMBOLS}
    trades = []
    equity = []

    total_commission = 0.0
    total_slippage_cost = 0.0

    for i in range(MIN_BARS, len(index)):
        timestamp = index[i]
        previous_timestamp = index[i - 1]

        opens = {symbol: float(frames[symbol].loc[timestamp, "Open"]) for symbol in SYMBOLS}
        closes = {symbol: float(frames[symbol].loc[timestamp, "Close"]) for symbol in SYMBOLS}
        lows = {symbol: float(frames[symbol].loc[timestamp, "Low"]) for symbol in SYMBOLS}

        for symbol in SYMBOLS:
            position = positions[symbol]
            pending = position["pending"]
            if pending is None:
                continue

            if pending["side"] == "BUY" and position["shares"] == 0:
                if open_count(positions) < MAX_OPEN_POSITIONS:
                    marked_value = portfolio_value(cash, positions, opens)
                    atr = float(frames[symbol].loc[previous_timestamp, "ATR14"])
                    raw_open = opens[symbol]
                    execution_price = raw_open * (1 + slippage_rate)
                    allocation = risk_sized_trade_value(
                        marked_value,
                        cash,
                        execution_price,
                        atr,
                    )

                    if allocation > 0:
                        trade_value = allocation / (1 + commission_rate)
                        commission = trade_value * commission_rate
                        shares = trade_value / execution_price
                        total_cost = trade_value + commission
                        slippage_cost = shares * max(0.0, execution_price - raw_open)

                        cash -= total_cost
                        total_commission += commission
                        total_slippage_cost += slippage_cost

                        positions[symbol] = {
                            "shares": shares,
                            "entry_price": execution_price,
                            "entry_time": timestamp,
                            "entry_cost": total_cost,
                            "entry_raw_price": raw_open,
                            "entry_commission": commission,
                            "entry_slippage_cost": slippage_cost,
                            "stop_price": stop_price_from_atr(execution_price, atr),
                            "entry_reason": pending["reason"],
                            "pending": None,
                            "last_exit_time": position.get("last_exit_time"),
                        }
                    else:
                        position["pending"] = None
                else:
                    position["pending"] = None

            elif pending["side"] == "SELL" and position["shares"] > 0:
                raw_open = opens[symbol]
                execution_price = raw_open * (1 - slippage_rate)
                gross = position["shares"] * execution_price
                commission = gross * commission_rate
                net = gross - commission
                pnl = net - position["entry_cost"]
                return_pct = pnl / position["entry_cost"] * 100
                exit_slippage_cost = position["shares"] * max(0.0, raw_open - execution_price)

                cash += net
                total_commission += commission
                total_slippage_cost += exit_slippage_cost

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
                    "commission": position["entry_commission"] + commission,
                    "slippage_cost": position["entry_slippage_cost"] + exit_slippage_cost,
                })
                positions[symbol] = fresh_position(last_exit_time=timestamp)

        for symbol in SYMBOLS:
            position = positions[symbol]
            if position["shares"] <= 0 or position["stop_price"] is None:
                continue

            if lows[symbol] <= position["stop_price"]:
                raw_exit = min(opens[symbol], position["stop_price"])
                execution_price = raw_exit * (1 - slippage_rate)
                gross = position["shares"] * execution_price
                commission = gross * commission_rate
                net = gross - commission
                pnl = net - position["entry_cost"]
                return_pct = pnl / position["entry_cost"] * 100
                exit_slippage_cost = position["shares"] * max(0.0, raw_exit - execution_price)

                cash += net
                total_commission += commission
                total_slippage_cost += exit_slippage_cost

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
                    "commission": position["entry_commission"] + commission,
                    "slippage_cost": position["entry_slippage_cost"] + exit_slippage_cost,
                })
                positions[symbol] = fresh_position(last_exit_time=timestamp)

        for symbol in SYMBOLS:
            history = frames[symbol].iloc[: i + 1]
            position = positions[symbol]
            signal = strategy_signal(
                history,
                position["shares"] > 0,
                position.get("last_exit_time"),
            )

            if signal.side == "BUY" and position["shares"] == 0:
                position["pending"] = {"side": "BUY", "reason": signal.reason}
            elif signal.side == "SELL" and position["shares"] > 0:
                position["pending"] = {"side": "SELL", "reason": signal.reason}

        value = portfolio_value(cash, positions, closes)
        equity.append({"timestamp": timestamp, "value": value})

    final_prices = {symbol: float(frames[symbol].iloc[-1]["Close"]) for symbol in SYMBOLS}
    final_value = portfolio_value(cash, positions, final_prices)

    diagnostics = {
        "total_commission": total_commission,
        "total_slippage_cost": total_slippage_cost,
    }

    return pd.DataFrame(equity), pd.DataFrame(trades), final_value, positions, diagnostics


def calculate_metrics(equity, trades, final_value):
    total_return = (final_value / STARTING_CASH - 1) * 100

    if equity.empty:
        return {}

    equity = equity.copy()
    equity["timestamp"] = pd.to_datetime(equity["timestamp"])
    values = equity["value"].astype(float)

    running_max = values.cummax()
    drawdowns = values / running_max - 1
    max_drawdown = drawdowns.min() * 100

    daily = equity.set_index("timestamp")["value"].resample("1D").last().dropna()
    daily_returns = daily.pct_change().dropna()
    sharpe = 0.0
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = daily_returns.mean() / daily_returns.std() * math.sqrt(TRADING_DAYS_PER_YEAR)

    trading_days = max(1, len(daily))

    if trades.empty:
        return {
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe": sharpe,
            "trades": 0,
            "trades_per_day": 0.0,
            "trading_days": trading_days,
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
        "trades_per_day": len(trades) / trading_days,
        "trading_days": trading_days,
        "win_rate": len(wins) / len(trades) * 100,
        "profit_factor": profit_factor,
        "expectancy": trades["pnl"].mean(),
        "average_trade_return": trades["return_percent"].mean(),
        "average_holding_minutes": trades["holding_minutes"].mean(),
    }


def print_one_report(title, equity, trades, final_value, positions, diagnostics):
    metrics = calculate_metrics(equity, trades, final_value)

    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)
    print(f"Final value:          ${final_value:,.2f}")
    print(f"Total return:         {metrics.get('total_return', 0):+.2f}%")
    print(f"Max drawdown:         {metrics.get('max_drawdown', 0):+.2f}%")
    print(f"Daily Sharpe:         {metrics.get('sharpe', 0):.2f}")
    print(f"Trading days:         {metrics.get('trading_days', 0)}")
    print(f"Completed trades:     {metrics.get('trades', 0)}")
    print(f"Trades / day:         {metrics.get('trades_per_day', 0):.1f}")
    print(f"Commission paid:      ${diagnostics['total_commission']:,.2f}")
    print(f"Slippage cost est.:   ${diagnostics['total_slippage_cost']:,.2f}")
    print(f"Total friction est.:  ${diagnostics['total_commission'] + diagnostics['total_slippage_cost']:,.2f}")

    if not trades.empty:
        print(f"Win rate:             {metrics['win_rate']:.2f}%")
        print(f"Profit factor:        {metrics['profit_factor']:.2f}")
        print(f"Expectancy / trade:   ${metrics['expectancy']:+.2f}")
        print(f"Avg trade return:     {metrics['average_trade_return']:+.3f}%")
        print(f"Avg holding time:     {metrics['average_holding_minutes']:.1f} min")

        print("\nBY SYMBOL")
        for symbol, group in trades.groupby("symbol"):
            print(
                f"{symbol:<6} trades={len(group):>4} "
                f"P/L=${group['pnl'].sum():+9.2f} "
                f"win={((group['pnl'] > 0).mean() * 100):5.1f}%"
            )

        print("\nBY ENTRY SIGNAL")
        for reason, group in trades.groupby("entry_reason"):
            print(
                f"{reason:<24} trades={len(group):>4} "
                f"P/L=${group['pnl'].sum():+9.2f} "
                f"win={((group['pnl'] > 0).mean() * 100):5.1f}%"
            )

        print("\nEXIT REASONS")
        for reason, count in Counter(trades["exit_reason"]).items():
            print(f"{reason:<24} {count}")

    open_symbols = [symbol for symbol, position in positions.items() if position["shares"] > 0]
    print(f"\nOpen at end:          {', '.join(open_symbols) if open_symbols else 'none'}")


def main():
    frames, index = download_universe()

    print("=" * 68)
    print("5-MINUTE BREAKOUT STRATEGY DIAGNOSTIC")
    print("=" * 68)
    print(f"Symbols:              {', '.join(SYMBOLS)}")
    print(f"Starting cash:        ${STARTING_CASH:,.2f}")
    print(f"Configured costs:     commission={COMMISSION_RATE:.3%} / side, slippage={SLIPPAGE_RATE:.3%} / side")

    live = run_backtest(frames, index, COMMISSION_RATE, SLIPPAGE_RATE)
    frictionless = run_backtest(frames, index, 0.0, 0.0)

    print_one_report("CONFIGURED COST MODEL", *live)
    print_one_report("FRICTIONLESS DIAGNOSTIC", *frictionless)

    live_return = (live[2] / STARTING_CASH - 1) * 100
    zero_return = (frictionless[2] / STARTING_CASH - 1) * 100

    print("\n" + "=" * 68)
    print("COST DIAGNOSIS")
    print("=" * 68)
    print(f"Return with costs:    {live_return:+.2f}%")
    print(f"Return without costs: {zero_return:+.2f}%")
    print(f"Cost-model drag:      {live_return - zero_return:+.2f} percentage points")
    print("\nThis comparison is diagnostic only. A frictionless result is not a realistic trading result.")
    print("yfinance 5-minute history is limited, so this remains a short intraday sample.")


if __name__ == "__main__":
    main()
