import math
from dataclasses import dataclass
from itertools import product

import pandas as pd

from market_data import DEFAULT_PROVIDER
from strategy_core import (
    COMMISSION_RATE,
    SLIPPAGE_RATE,
    STARTING_CASH,
    SYMBOLS,
)


PERIOD = "60d"
TRAIN_FRACTION = 0.70
MIN_WARMUP_BARS = 220
MAX_OPEN_POSITIONS = 3
MAX_POSITION_PERCENT = 0.20
RISK_PER_TRADE_PERCENT = 0.005
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class Params:
    breakout_lookback: int
    exit_lookback: int
    breakout_atr_buffer: float
    volume_multiplier: float
    atr_stop_multiplier: float
    ma_slope_lookback: int

    @property
    def name(self):
        return (
            f"B{self.breakout_lookback}/E{self.exit_lookback} "
            f"buf={self.breakout_atr_buffer:.2f} "
            f"vol={self.volume_multiplier:.2f} "
            f"stop={self.atr_stop_multiplier:.1f}ATR "
            f"slope={self.ma_slope_lookback}"
        )


CANDIDATES = [
    Params(*values)
    for values in product(
        [30, 50, 80],          # breakout lookback
        [20, 40],              # exit lookback
        [0.00, 0.25, 0.50],    # extra breakout distance in ATRs
        [1.00, 1.10, 1.25],    # volume / 20-bar average
        [2.0, 3.0],            # ATR stop multiple
        [10, 20],              # MA50 slope lookback
    )
]


def flatten(data):
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data.dropna().copy()


def calculate_atr(data, period=14):
    previous_close = data["Close"].shift(1)
    true_range = pd.concat(
        [
            data["High"] - data["Low"],
            (data["High"] - previous_close).abs(),
            (data["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def prepare_symbol(raw):
    data = flatten(raw)
    data["MA50"] = data["Close"].rolling(50).mean()
    data["MA200"] = data["Close"].rolling(200).mean()
    data["VolumeMA20"] = data["Volume"].rolling(20).mean()
    data["ATR14"] = calculate_atr(data)
    return data


def download_universe():
    frames = {}
    for symbol in SYMBOLS:
        raw = DEFAULT_PROVIDER.history(symbol, period=PERIOD)
        if len(raw) < MIN_WARMUP_BARS:
            raise ValueError(f"Not enough 5-minute data for {symbol}: {len(raw)} bars")
        frames[symbol] = prepare_symbol(raw)

    common_index = frames[SYMBOLS[0]].index
    for symbol in SYMBOLS[1:]:
        common_index = common_index.intersection(frames[symbol].index)
    common_index = common_index.sort_values()

    if len(common_index) < MIN_WARMUP_BARS:
        raise ValueError("Not enough common timestamps across the symbol universe.")

    frames = {symbol: frame.loc[common_index].copy() for symbol, frame in frames.items()}
    return frames, common_index


def trading_days(index):
    return list(dict.fromkeys(pd.Timestamp(ts).date() for ts in index))


def split_days(index):
    days = trading_days(index)
    split = max(1, min(len(days) - 1, int(len(days) * TRAIN_FRACTION)))
    return days[:split], days[split:]


def add_candidate_columns(frame, params):
    data = frame.copy()
    data["HighBreakout"] = (
        data["High"].rolling(params.breakout_lookback).max().shift(1)
    )
    data["LowExit"] = (
        data["Low"].rolling(params.exit_lookback).min().shift(1)
    )
    return data


def entry_allowed(timestamp):
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("America/New_York")
    current = ts.time()
    return pd.Timestamp("10:00").time() <= current <= pd.Timestamp("14:30").time()


def stop_distance(atr, params):
    if atr is None or pd.isna(atr) or atr <= 0:
        return None
    return float(atr) * params.atr_stop_multiplier


def size_position(portfolio_value, cash, entry_price, atr, params):
    distance = stop_distance(atr, params)
    if distance is None or entry_price <= 0:
        return 0.0

    risk_budget = portfolio_value * RISK_PER_TRADE_PERCENT
    shares_by_risk = risk_budget / distance
    value_by_risk = shares_by_risk * entry_price
    allocation_cap = portfolio_value * MAX_POSITION_PERCENT
    return max(0.0, min(cash, value_by_risk, allocation_cap))


def fresh_position(last_exit_bar=None):
    return {
        "shares": 0.0,
        "entry_price": None,
        "entry_time": None,
        "entry_cost": None,
        "entry_commission": 0.0,
        "entry_slippage": 0.0,
        "entry_reason": None,
        "stop_price": None,
        "pending": None,
        "last_exit_bar": last_exit_bar,
    }


def open_count(positions):
    return sum(position["shares"] > 0 for position in positions.values())


def portfolio_value(cash, positions, prices):
    value = cash
    for symbol, position in positions.items():
        if position["shares"] > 0 and symbol in prices:
            value += position["shares"] * prices[symbol]
    return value


def signal_for(data, i, position, params):
    latest = data.iloc[i]

    required = [
        latest["MA50"],
        latest["MA200"],
        latest["HighBreakout"],
        latest["LowExit"],
        latest["VolumeMA20"],
        latest["ATR14"],
    ]
    if any(pd.isna(value) for value in required):
        return None

    if position["shares"] > 0:
        if latest["Close"] < latest["LowExit"]:
            return {"side": "SELL", "reason": "breakout_exit"}
        if latest["Close"] < latest["MA50"]:
            return {"side": "SELL", "reason": "ma50_exit"}
        return None

    if not entry_allowed(data.index[i]):
        return None

    if i <= params.ma_slope_lookback:
        return None

    ma50_then = data["MA50"].iloc[i - params.ma_slope_lookback]
    if pd.isna(ma50_then):
        return None

    trend_up = latest["Close"] > latest["MA200"] and latest["MA50"] > latest["MA200"]
    ma50_rising = latest["MA50"] > ma50_then
    volume_ok = latest["Volume"] >= latest["VolumeMA20"] * params.volume_multiplier
    breakout_level = latest["HighBreakout"] + latest["ATR14"] * params.breakout_atr_buffer
    breakout_ok = latest["Close"] > breakout_level

    if trend_up and ma50_rising and volume_ok and breakout_ok:
        return {"side": "BUY", "reason": "breakout"}

    return None


def backtest(frames, common_index, allowed_days, params, commission_rate, slippage_rate):
    prepared = {symbol: add_candidate_columns(frame, params) for symbol, frame in frames.items()}

    cash = STARTING_CASH
    positions = {symbol: fresh_position() for symbol in SYMBOLS}
    trades = []
    equity = []
    total_commission = 0.0
    total_slippage = 0.0

    allowed_days = set(allowed_days)
    first_allowed = min(allowed_days)
    last_allowed = max(allowed_days)

    for i in range(MIN_WARMUP_BARS, len(common_index)):
        timestamp = common_index[i]
        day = pd.Timestamp(timestamp).date()
        if day < first_allowed or day > last_allowed:
            continue

        opens = {symbol: float(prepared[symbol].iloc[i]["Open"]) for symbol in SYMBOLS}
        closes = {symbol: float(prepared[symbol].iloc[i]["Close"]) for symbol in SYMBOLS}
        lows = {symbol: float(prepared[symbol].iloc[i]["Low"]) for symbol in SYMBOLS}

        # Execute orders queued from the previous completed bar.
        for symbol in SYMBOLS:
            position = positions[symbol]
            pending = position["pending"]
            if pending is None:
                continue

            if pending["side"] == "BUY" and position["shares"] == 0:
                if open_count(positions) < MAX_OPEN_POSITIONS:
                    raw_open = opens[symbol]
                    fill = raw_open * (1 + slippage_rate)
                    marked_value = portfolio_value(cash, positions, opens)
                    allocation = size_position(marked_value, cash, fill, pending["atr"], params)

                    if allocation > 0:
                        trade_value = allocation / (1 + commission_rate)
                        commission = trade_value * commission_rate
                        shares = trade_value / fill
                        total_cost = trade_value + commission
                        slip = shares * max(0.0, fill - raw_open)

                        cash -= total_cost
                        total_commission += commission
                        total_slippage += slip

                        distance = stop_distance(pending["atr"], params)
                        positions[symbol] = {
                            "shares": shares,
                            "entry_price": fill,
                            "entry_time": timestamp,
                            "entry_cost": total_cost,
                            "entry_commission": commission,
                            "entry_slippage": slip,
                            "entry_reason": pending["reason"],
                            "stop_price": max(0.01, fill - distance),
                            "pending": None,
                            "last_exit_bar": position["last_exit_bar"],
                        }
                    else:
                        position["pending"] = None
                else:
                    position["pending"] = None

            elif pending["side"] == "SELL" and position["shares"] > 0:
                raw_open = opens[symbol]
                fill = raw_open * (1 - slippage_rate)
                gross = position["shares"] * fill
                commission = gross * commission_rate
                net = gross - commission
                pnl = net - position["entry_cost"]
                slip = position["shares"] * max(0.0, raw_open - fill)

                cash += net
                total_commission += commission
                total_slippage += slip
                trades.append({
                    "symbol": symbol,
                    "entry_time": position["entry_time"],
                    "exit_time": timestamp,
                    "pnl": pnl,
                    "return_percent": pnl / position["entry_cost"] * 100,
                    "holding_minutes": (timestamp - position["entry_time"]).total_seconds() / 60,
                    "exit_reason": pending["reason"],
                    "commission": position["entry_commission"] + commission,
                    "slippage": position["entry_slippage"] + slip,
                })
                positions[symbol] = fresh_position(last_exit_bar=i)

        # Intrabar ATR stop.
        for symbol in SYMBOLS:
            position = positions[symbol]
            if position["shares"] <= 0 or position["stop_price"] is None:
                continue
            if lows[symbol] <= position["stop_price"]:
                raw_exit = min(opens[symbol], position["stop_price"])
                fill = raw_exit * (1 - slippage_rate)
                gross = position["shares"] * fill
                commission = gross * commission_rate
                net = gross - commission
                pnl = net - position["entry_cost"]
                slip = position["shares"] * max(0.0, raw_exit - fill)

                cash += net
                total_commission += commission
                total_slippage += slip
                trades.append({
                    "symbol": symbol,
                    "entry_time": position["entry_time"],
                    "exit_time": timestamp,
                    "pnl": pnl,
                    "return_percent": pnl / position["entry_cost"] * 100,
                    "holding_minutes": (timestamp - position["entry_time"]).total_seconds() / 60,
                    "exit_reason": "atr_stop",
                    "commission": position["entry_commission"] + commission,
                    "slippage": position["entry_slippage"] + slip,
                })
                positions[symbol] = fresh_position(last_exit_bar=i)

        # Only generate new signals on days in this partition.
        if day in allowed_days:
            for symbol in SYMBOLS:
                position = positions[symbol]

                # Six 5-minute bars = 30-minute cooldown after an exit.
                if position["shares"] == 0 and position["last_exit_bar"] is not None:
                    if i - position["last_exit_bar"] < 6:
                        continue

                signal = signal_for(prepared[symbol], i, position, params)
                if signal is None:
                    continue

                if signal["side"] == "BUY" and position["shares"] == 0:
                    position["pending"] = {
                        "side": "BUY",
                        "reason": signal["reason"],
                        "atr": float(prepared[symbol].iloc[i]["ATR14"]),
                    }
                elif signal["side"] == "SELL" and position["shares"] > 0:
                    position["pending"] = signal

        equity.append({"timestamp": timestamp, "value": portfolio_value(cash, positions, closes)})

    # Mark any open positions at the final close for reporting only.
    final_i = max(i for i, ts in enumerate(common_index) if pd.Timestamp(ts).date() <= last_allowed)
    final_prices = {symbol: float(prepared[symbol].iloc[final_i]["Close"]) for symbol in SYMBOLS}
    final_value = portfolio_value(cash, positions, final_prices)

    return pd.DataFrame(equity), pd.DataFrame(trades), final_value, total_commission, total_slippage


def metrics(equity, trades, final_value):
    result = {
        "return": (final_value / STARTING_CASH - 1) * 100,
        "trades": len(trades),
        "sharpe": 0.0,
        "profit_factor": 0.0,
        "avg_trade": 0.0,
        "win_rate": 0.0,
        "max_drawdown": 0.0,
    }

    if not equity.empty:
        eq = equity.copy()
        eq["timestamp"] = pd.to_datetime(eq["timestamp"])
        values = eq["value"].astype(float)
        result["max_drawdown"] = ((values / values.cummax()) - 1).min() * 100
        daily = eq.set_index("timestamp")["value"].resample("1D").last().dropna()
        returns = daily.pct_change().dropna()
        if len(returns) > 1 and returns.std() > 0:
            result["sharpe"] = returns.mean() / returns.std() * math.sqrt(TRADING_DAYS_PER_YEAR)

    if not trades.empty:
        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] < 0]
        gross_profit = wins["pnl"].sum()
        gross_loss = abs(losses["pnl"].sum())
        result["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        result["avg_trade"] = trades["return_percent"].mean()
        result["win_rate"] = len(wins) / len(trades) * 100

    return result


def research_score(m):
    # Avoid selecting a lucky strategy with almost no observations.
    if m["trades"] < 12:
        return -999.0
    return m["sharpe"] + 0.15 * m["return"] + 0.20 * (m["profit_factor"] - 1)


def run_candidate(frames, index, days, params, commission, slippage):
    result = backtest(frames, index, days, params, commission, slippage)
    m = metrics(result[0], result[1], result[2])
    return result, m


def main():
    frames, index = download_universe()
    train_days, test_days = split_days(index)

    print("=" * 92)
    print("5-MINUTE BREAKOUT WALK-FORWARD RESEARCH")
    print("=" * 92)
    print(f"Trading days: {len(train_days) + len(test_days)} | train={len(train_days)} | unseen test={len(test_days)}")
    print(f"Candidates:   {len(CANDIDATES)}")
    print(
        f"Cost model:   commission={COMMISSION_RATE:.3%}/side, "
        f"slippage={SLIPPAGE_RATE:.3%}/side"
    )
    print("Selection uses TRAIN ONLY. Test results are printed only after ranking is finished.\n")

    ranked = []
    for number, params in enumerate(CANDIDATES, 1):
        _, train_metrics = run_candidate(
            frames,
            index,
            train_days,
            params,
            COMMISSION_RATE,
            SLIPPAGE_RATE,
        )
        ranked.append((research_score(train_metrics), params, train_metrics))
        if number % 25 == 0 or number == len(CANDIDATES):
            print(f"Evaluated {number:>3}/{len(CANDIDATES)} candidates")

    ranked.sort(key=lambda row: row[0], reverse=True)
    finalists = ranked[:5]

    print("\n" + "=" * 92)
    print("TOP 5 ON TRAINING PERIOD")
    print("=" * 92)
    for rank, (score, params, m) in enumerate(finalists, 1):
        print(
            f"{rank}. {params.name:<55} "
            f"ret={m['return']:+6.2f}% sharpe={m['sharpe']:+5.2f} "
            f"PF={m['profit_factor']:.2f} trades={m['trades']:>3} avg={m['avg_trade']:+.3f}%"
        )

    print("\n" + "=" * 92)
    print("UNSEEN TEST PERIOD")
    print("=" * 92)

    for rank, (_, params, train_m) in enumerate(finalists, 1):
        test_result, test_m = run_candidate(
            frames,
            index,
            test_days,
            params,
            COMMISSION_RATE,
            SLIPPAGE_RATE,
        )
        zero_result, zero_m = run_candidate(
            frames,
            index,
            test_days,
            params,
            0.0,
            0.0,
        )

        trades = test_result[1]
        total_friction = test_result[3] + test_result[4]
        notional_round_trips = 0.0
        if not trades.empty:
            # Break-even round-trip friction approximation from gross average trade return.
            break_even_round_trip = max(0.0, zero_m["avg_trade"])
            break_even_per_side = break_even_round_trip / 2
        else:
            break_even_round_trip = 0.0
            break_even_per_side = 0.0

        print(f"\n#{rank} {params.name}")
        print(
            f"  train: ret={train_m['return']:+.2f}% sharpe={train_m['sharpe']:+.2f} "
            f"PF={train_m['profit_factor']:.2f} trades={train_m['trades']}"
        )
        print(
            f"  test costs: ret={test_m['return']:+.2f}% sharpe={test_m['sharpe']:+.2f} "
            f"PF={test_m['profit_factor']:.2f} trades={test_m['trades']} avg={test_m['avg_trade']:+.3f}%"
        )
        print(
            f"  test gross: ret={zero_m['return']:+.2f}% sharpe={zero_m['sharpe']:+.2f} "
            f"PF={zero_m['profit_factor']:.2f} trades={zero_m['trades']} avg={zero_m['avg_trade']:+.3f}%"
        )
        print(f"  test friction paid: ${total_friction:,.2f}")
        print(
            f"  approx break-even friction: {break_even_round_trip:.3f}% round trip "
            f"({break_even_per_side:.3f}%/side)"
        )

    print("\n" + "=" * 92)
    print("IMPORTANT")
    print("=" * 92)
    print("Do not choose a strategy because it has the highest TEST return.")
    print("The test period is for validation only; repeatedly optimizing against it would turn it into training data.")
    print("With only ~60 days of yfinance 5-minute history, even a good-looking result is still weak evidence.")


if __name__ == "__main__":
    main()
