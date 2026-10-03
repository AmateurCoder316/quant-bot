import json
import time
import webbrowser
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
from threading import Timer
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from dash import Dash, Input, Output, State, ctx, dcc, html, dash_table


# ==========================================
# SETTINGS
# ==========================================

SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
INTERVAL = "5m"
STARTING_CASH = 10000.0
COMMISSION_RATE = 0.001
SLIPPAGE_RATE = 0.0005
POSITION_SIZE_PERCENT = 0.20
MAX_OPEN_POSITIONS = 3
STOP_LOSS_PERCENT = 0.05

STATE_FILE = Path("paper_state_5m.json")
EQUITY_FILE = Path("logs/equity.csv")
TRADES_FILE = Path("logs/trades.csv")

NY = ZoneInfo("America/New_York")

# One visual system. Green/red are reserved for financial meaning only.
BG = "#0b0d10"
SURFACE = "#111418"
SURFACE_2 = "#161a1f"
BORDER = "#232830"
TEXT = "#f4f5f7"
MUTED = "#8b929c"
ACCENT = "#8aa4ff"
GREEN = "#42c985"
RED = "#ef6673"
YELLOW = "#d7b55b"
FONT = "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
RADIUS = "10px"
SPACE = "16px"

# Local single-user replay engine. It never touches live paper state.
REPLAY = {
    "date": None,
    "data": {},
    "timeline": [],
    "step": -1,
    "playing": False,
    "speed": 1,
    "state": None,
    "equity": [],
    "error": None,
}


# ==========================================
# FILE + MARKET HELPERS
# ==========================================


def read_state():
    if not STATE_FILE.exists():
        return fresh_state()
    try:
        with STATE_FILE.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return fresh_state()


def read_csv(path):
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError):
        return pd.DataFrame()


def flatten(data):
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data.dropna().copy()


def download_live(symbol):
    return flatten(
        yf.download(
            symbol,
            period="5d",
            interval=INTERVAL,
            auto_adjust=True,
            progress=False,
            prepost=False,
        )
    )


def latest_live_prices():
    prices = {}
    for symbol in SYMBOLS:
        try:
            data = flatten(
                yf.download(
                    symbol,
                    period="1d",
                    interval="1m",
                    auto_adjust=True,
                    progress=False,
                    prepost=False,
                )
            )
            if not data.empty:
                prices[symbol] = float(data.iloc[-1]["Close"])
        except Exception:
            pass
    return prices


def market_status():
    now = datetime.now(NY)
    if now.weekday() >= 5:
        return "CLOSED", MUTED, "Weekend"
    if dt_time(9, 30) <= now.time() < dt_time(16, 0):
        return "OPEN", GREEN, "US regular session"
    return "CLOSED", MUTED, "Outside regular session"


# ==========================================
# STRATEGY + PORTFOLIO HELPERS
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


def strategy_signal(data, has_position):
    latest = data.iloc[-1]
    if pd.isna(latest["MA200"]):
        return "HOLD"

    trend_up = latest["Close"] > latest["MA200"]

    if not has_position:
        breakout_buy = not pd.isna(latest["High20"]) and latest["Close"] > latest["High20"]
        mean_reversion_buy = trend_up and latest["RSI2"] < 10
        if breakout_buy or mean_reversion_buy:
            return "BUY"

    if has_position:
        breakout_sell = not pd.isna(latest["Low10"]) and latest["Close"] < latest["Low10"]
        mean_reversion_sell = latest["RSI2"] > 70
        if breakout_sell or mean_reversion_sell:
            return "SELL"

    return "HOLD"


def fresh_position():
    return {
        "shares": 0.0,
        "entry_price": None,
        "entry_time": None,
        "entry_total_cost": None,
        "stop_price": None,
        "pending_order": None,
    }


def fresh_state():
    return {
        "cash": STARTING_CASH,
        "realized_pnl": 0.0,
        "positions": {symbol: fresh_position() for symbol in SYMBOLS},
        "trades": [],
    }


def position_count(state):
    return sum(1 for p in state.get("positions", {}).values() if float(p.get("shares", 0) or 0) > 0)


def portfolio_value(state, prices):
    value = float(state.get("cash", STARTING_CASH))
    for symbol, position in state.get("positions", {}).items():
        shares = float(position.get("shares", 0) or 0)
        if shares > 0 and symbol in prices:
            value += shares * prices[symbol]
    return value


def unrealized_pnl(state, prices):
    total = 0.0
    for symbol, position in state.get("positions", {}).items():
        shares = float(position.get("shares", 0) or 0)
        cost = position.get("entry_total_cost")
        if shares > 0 and cost is not None and symbol in prices:
            total += shares * prices[symbol] - float(cost)
    return total


def pnl_color(value):
    return GREEN if value > 0 else RED if value < 0 else TEXT


# ==========================================
# REPLAY ENGINE
# ==========================================


def load_replay_day(date_text):
    try:
        selected = pd.Timestamp(date_text).date()
        warmup_start = selected - timedelta(days=14)
        end = selected + timedelta(days=1)

        prepared = {}
        for symbol in SYMBOLS:
            data = flatten(
                yf.download(
                    symbol,
                    start=warmup_start.isoformat(),
                    end=end.isoformat(),
                    interval=INTERVAL,
                    auto_adjust=True,
                    progress=False,
                    prepost=False,
                )
            )
            if data.empty:
                continue
            data = add_indicators(data)
            prepared[symbol] = data

        if "AAPL" not in prepared:
            raise ValueError("No 5-minute data available for that date.")

        aapl = prepared["AAPL"]
        timeline = [
            ts for ts in aapl.index
            if pd.Timestamp(ts).date() == selected
        ]
        if not timeline:
            raise ValueError("No regular-session candles for that date.")

        REPLAY.update({
            "date": date_text,
            "data": prepared,
            "timeline": timeline,
            "step": -1,
            "playing": False,
            "state": fresh_state(),
            "equity": [],
            "error": None,
        })
    except Exception as error:
        REPLAY.update({"error": str(error), "playing": False})


def execute_replay_buy(state, symbol, timestamp, market_open, current_value):
    position = state["positions"][symbol]
    if position["shares"] > 0 or position_count(state) >= MAX_OPEN_POSITIONS:
        position["pending_order"] = None
        return

    allocation = min(state["cash"], current_value * POSITION_SIZE_PERCENT)
    if allocation <= 0:
        position["pending_order"] = None
        return

    price = market_open * (1 + SLIPPAGE_RATE)
    trade_value = allocation / (1 + COMMISSION_RATE)
    commission = trade_value * COMMISSION_RATE
    shares = trade_value / price
    total_cost = trade_value + commission

    state["cash"] -= total_cost
    position.update({
        "shares": shares,
        "entry_price": price,
        "entry_time": pd.Timestamp(timestamp).isoformat(),
        "entry_total_cost": total_cost,
        "stop_price": price * (1 - STOP_LOSS_PERCENT),
        "pending_order": None,
    })


def execute_replay_sell(state, symbol, timestamp, market_price, reason):
    position = state["positions"][symbol]
    if position["shares"] <= 0:
        position["pending_order"] = None
        return

    price = market_price * (1 - SLIPPAGE_RATE)
    gross = position["shares"] * price
    commission = gross * COMMISSION_RATE
    net = gross - commission
    pnl = net - position["entry_total_cost"]
    ret = pnl / position["entry_total_cost"] * 100

    state["cash"] += net
    state["realized_pnl"] += pnl
    state["trades"].append({
        "symbol": symbol,
        "entry_time": position["entry_time"],
        "exit_time": pd.Timestamp(timestamp).isoformat(),
        "entry_price": position["entry_price"],
        "exit_price": price,
        "pnl": pnl,
        "return_percent": ret,
        "reason": reason,
    })
    state["positions"][symbol] = fresh_position()


def replay_prices_at(timestamp):
    prices = {}
    for symbol, data in REPLAY["data"].items():
        rows = data.loc[:timestamp]
        if not rows.empty:
            prices[symbol] = float(rows.iloc[-1]["Close"])
    return prices


def advance_replay(steps=1):
    if REPLAY["state"] is None or not REPLAY["timeline"]:
        return

    for _ in range(max(1, int(steps))):
        if REPLAY["step"] >= len(REPLAY["timeline"]) - 1:
            REPLAY["playing"] = False
            break

        REPLAY["step"] += 1
        timestamp = REPLAY["timeline"][REPLAY["step"]]
        state = REPLAY["state"]
        prices = replay_prices_at(timestamp)

        # 1) Pending signals fill at this candle's open.
        for symbol, data in REPLAY["data"].items():
            if timestamp not in data.index:
                continue
            row = data.loc[timestamp]
            position = state["positions"][symbol]
            pending = position.get("pending_order")
            if pending == "BUY":
                execute_replay_buy(state, symbol, timestamp, float(row["Open"]), portfolio_value(state, prices))
            elif pending == "SELL":
                execute_replay_sell(state, symbol, timestamp, float(row["Open"]), "signal")

        # 2) Stop loss inside this candle.
        for symbol, data in REPLAY["data"].items():
            if timestamp not in data.index:
                continue
            row = data.loc[timestamp]
            position = state["positions"][symbol]
            stop = position.get("stop_price")
            if position["shares"] > 0 and stop is not None and float(row["Low"]) <= float(stop):
                execute_replay_sell(state, symbol, timestamp, float(stop), "stop_loss")

        # 3) Generate signals from the completed candle.
        for symbol, data in REPLAY["data"].items():
            if timestamp not in data.index:
                continue
            history = data.loc[:timestamp]
            if len(history) < 201:
                continue
            position = state["positions"][symbol]
            signal = strategy_signal(history, position["shares"] > 0)
            if signal in {"BUY", "SELL"}:
                position["pending_order"] = signal

        prices = replay_prices_at(timestamp)
        REPLAY["equity"].append({
            "timestamp": pd.Timestamp(timestamp).isoformat(),
            "portfolio_value": portfolio_value(state, prices),
        })


# ==========================================
# UI HELPERS
# ==========================================


def panel(children, **extra):
    style = {
        "background": SURFACE,
        "border": f"1px solid {BORDER}",
        "borderRadius": RADIUS,
        "padding": SPACE,
    }
    style.update(extra.pop("style", {}))
    return html.Div(children, style=style, **extra)


def metric(label, value, color=TEXT):
    return html.Div([
        html.Div(label, style={"fontSize": "12px", "color": MUTED, "marginBottom": "6px"}),
        html.Div(value, style={"fontSize": "22px", "fontWeight": "650", "color": color}),
    ])


def badge(text, color):
    return html.Span(
        text,
        style={
            "fontSize": "11px",
            "fontWeight": "700",
            "color": color,
            "border": f"1px solid {BORDER}",
            "background": SURFACE_2,
            "borderRadius": "999px",
            "padding": "5px 8px",
        },
    )


def empty_figure():
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"family": FONT, "color": TEXT, "size": 12},
        margin={"l": 45, "r": 16, "t": 20, "b": 36},
        xaxis={"gridcolor": BORDER, "zeroline": False},
        yaxis={"gridcolor": BORDER, "zeroline": False},
        height=330,
        hovermode="x unified",
        showlegend=False,
    )
    return fig


def portfolio_figure(rows):
    fig = empty_figure()
    if rows:
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["portfolio_value"],
            mode="lines",
            line={"color": ACCENT, "width": 2},
            name="Portfolio",
        ))
    fig.add_hline(y=STARTING_CASH, line_dash="dot", line_color=MUTED)
    return fig


def live_stock_figure(symbol, trades_df):
    fig = empty_figure()
    try:
        data = download_live(symbol)
    except Exception:
        return fig
    if data.empty:
        return fig

    data = data.tail(160)
    fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode="lines", line={"color": TEXT, "width": 2}))

    if not trades_df.empty and "symbol" in trades_df.columns:
        t = trades_df[trades_df["symbol"] == symbol]
        if not t.empty:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(t["entry_time"], errors="coerce"), y=t["entry_price"], mode="markers",
                marker={"symbol": "triangle-up", "size": 9, "color": GREEN}, name="Buy"
            ))
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(t["exit_time"], errors="coerce"), y=t["exit_price"], mode="markers",
                marker={"symbol": "triangle-down", "size": 9, "color": RED}, name="Sell"
            ))
    return fig


def replay_stock_figure(symbol):
    fig = empty_figure()
    if REPLAY["state"] is None or symbol not in REPLAY["data"] or REPLAY["step"] < 0:
        return fig

    end_ts = REPLAY["timeline"][REPLAY["step"]]
    selected_date = pd.Timestamp(REPLAY["date"]).date()
    data = REPLAY["data"][symbol]
    visible = data[(data.index <= end_ts) & (pd.Index(data.index).date == selected_date)]
    if visible.empty:
        return fig

    fig.add_trace(go.Scatter(x=visible.index, y=visible["Close"], mode="lines", line={"color": TEXT, "width": 2}))

    trades = [t for t in REPLAY["state"]["trades"] if t["symbol"] == symbol]
    if trades:
        t = pd.DataFrame(trades)
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(t["entry_time"]), y=t["entry_price"], mode="markers",
            marker={"symbol": "triangle-up", "size": 9, "color": GREEN}
        ))
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(t["exit_time"]), y=t["exit_price"], mode="markers",
            marker={"symbol": "triangle-down", "size": 9, "color": RED}
        ))
    return fig


def positions_rows(state, prices):
    rows = []
    for symbol in SYMBOLS:
        p = state.get("positions", {}).get(symbol, {})
        shares = float(p.get("shares", 0) or 0)
        if shares <= 0:
            continue
        current = prices.get(symbol)
        cost = p.get("entry_total_cost")
        pnl = None if current is None or cost is None else shares * current - float(cost)
        rows.append({
            "Symbol": symbol,
            "Value": None if current is None else round(shares * current, 2),
            "P/L": None if pnl is None else round(pnl, 2),
            "Entry": None if p.get("entry_price") is None else round(float(p["entry_price"]), 2),
        })
    return rows


# ==========================================
# DASH APP
# ==========================================


app = Dash(__name__)
app.title = "Quant Bot"

CONTROL_STYLE = {
    "height": "36px",
    "borderRadius": "8px",
    "border": f"1px solid {BORDER}",
    "background": SURFACE_2,
    "color": TEXT,
    "fontFamily": FONT,
}
BUTTON_STYLE = {
    **CONTROL_STYLE,
    "padding": "0 12px",
    "cursor": "pointer",
    "fontWeight": "600",
}
TABLE = {
    "style_table": {"overflowX": "auto"},
    "style_cell": {
        "backgroundColor": SURFACE,
        "color": TEXT,
        "border": "none",
        "borderBottom": f"1px solid {BORDER}",
        "padding": "9px 6px",
        "fontFamily": FONT,
        "fontSize": "12px",
        "textAlign": "left",
    },
    "style_header": {
        "backgroundColor": SURFACE,
        "color": MUTED,
        "border": "none",
        "borderBottom": f"1px solid {BORDER}",
        "fontWeight": "600",
    },
}

app.layout = html.Div(
    style={"minHeight": "100vh", "background": BG, "color": TEXT, "fontFamily": FONT},
    children=[
        dcc.Interval(id="live-refresh", interval=15_000, n_intervals=0),
        dcc.Interval(id="replay-tick", interval=1000, n_intervals=0),
        dcc.Store(id="replay-version", data=0),

        html.Div(
            style={"maxWidth": "1180px", "margin": "0 auto", "padding": "28px 20px 48px"},
            children=[
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "gap": SPACE, "marginBottom": "20px"},
                    children=[
                        html.Div([
                            html.Div("QUANT BOT", style={"fontSize": "11px", "fontWeight": "700", "letterSpacing": "0.12em", "color": MUTED}),
                            html.H1("Paper portfolio", style={"fontSize": "26px", "fontWeight": "650", "margin": "4px 0 0"}),
                        ]),
                        html.Div(id="header-status", style={"display": "flex", "gap": "8px", "alignItems": "center"}),
                    ],
                ),

                panel([
                    html.Div(
                        style={"display": "flex", "gap": "10px", "alignItems": "center", "flexWrap": "wrap"},
                        children=[
                            dcc.RadioItems(
                                id="mode",
                                options=[{"label": "Live", "value": "live"}, {"label": "Replay", "value": "replay"}],
                                value="live",
                                inline=True,
                                inputStyle={"marginRight": "5px"},
                                labelStyle={"marginRight": "14px", "fontSize": "13px", "color": TEXT},
                            ),
                            dcc.DatePickerSingle(
                                id="replay-date",
                                date=datetime.now().date().isoformat(),
                                max_date_allowed=datetime.now().date(),
                                display_format="YYYY-MM-DD",
                            ),
                            html.Button("Load", id="load-replay", n_clicks=0, style=BUTTON_STYLE),
                            html.Button("▶", id="play-replay", n_clicks=0, style=BUTTON_STYLE),
                            html.Button("Ⅱ", id="pause-replay", n_clicks=0, style=BUTTON_STYLE),
                            html.Button("→", id="next-replay", n_clicks=0, style=BUTTON_STYLE),
                            dcc.Dropdown(
                                id="replay-speed",
                                options=[{"label": "1x", "value": 1}, {"label": "5x", "value": 5}, {"label": "20x", "value": 20}],
                                value=1,
                                clearable=False,
                                style={"width": "90px", "color": "#111"},
                            ),
                            html.Div(id="replay-status", style={"marginLeft": "auto", "fontSize": "12px", "color": MUTED}),
                        ],
                    )
                ], style={"marginBottom": SPACE}),

                panel(
                    html.Div(id="metrics", style={"display": "grid", "gridTemplateColumns": "repeat(6, minmax(0, 1fr))", "gap": "18px"}),
                    style={"marginBottom": SPACE},
                ),

                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": SPACE, "marginBottom": SPACE},
                    children=[
                        panel([
                            html.Div("Portfolio", style={"fontSize": "12px", "fontWeight": "600", "color": MUTED, "marginBottom": "6px"}),
                            dcc.Graph(id="portfolio-chart", config={"displayModeBar": False}, style={"height": "330px"}),
                        ]),
                        panel([
                            html.Div(
                                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "6px"},
                                children=[
                                    html.Div("Price", style={"fontSize": "12px", "fontWeight": "600", "color": MUTED}),
                                    dcc.Dropdown(
                                        id="symbol",
                                        options=[{"label": s, "value": s} for s in SYMBOLS],
                                        value="AAPL",
                                        clearable=False,
                                        style={"width": "110px", "color": "#111"},
                                    ),
                                ],
                            ),
                            dcc.Graph(id="stock-chart", config={"displayModeBar": False}, style={"height": "330px"}),
                        ]),
                    ],
                ),

                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": SPACE},
                    children=[
                        panel([
                            html.Div("Open positions", style={"fontSize": "12px", "fontWeight": "600", "color": MUTED, "marginBottom": "8px"}),
                            dash_table.DataTable(id="positions", page_size=6, **TABLE),
                        ]),
                        panel([
                            html.Div("Recent trades", style={"fontSize": "12px", "fontWeight": "600", "color": MUTED, "marginBottom": "8px"}),
                            dash_table.DataTable(id="trades", page_size=6, **TABLE),
                        ]),
                    ],
                ),
            ],
        ),
    ],
)


# ==========================================
# CALLBACKS
# ==========================================


@app.callback(
    Output("replay-version", "data"),
    Output("replay-status", "children"),
    Input("load-replay", "n_clicks"),
    Input("play-replay", "n_clicks"),
    Input("pause-replay", "n_clicks"),
    Input("next-replay", "n_clicks"),
    Input("replay-speed", "value"),
    Input("replay-tick", "n_intervals"),
    State("replay-date", "date"),
    State("mode", "value"),
    prevent_initial_call=True,
)
def control_replay(load_clicks, play_clicks, pause_clicks, next_clicks, speed, tick, date_text, mode):
    trigger = ctx.triggered_id
    REPLAY["speed"] = speed or 1

    if trigger == "load-replay":
        load_replay_day(date_text)
    elif trigger == "play-replay" and REPLAY["state"] is not None:
        REPLAY["playing"] = True
    elif trigger == "pause-replay":
        REPLAY["playing"] = False
    elif trigger == "next-replay" and mode == "replay":
        advance_replay(1)
    elif trigger == "replay-tick" and mode == "replay" and REPLAY["playing"]:
        advance_replay(REPLAY["speed"])

    if REPLAY["error"]:
        status = REPLAY["error"]
    elif REPLAY["state"] is None:
        status = "Choose a date and load replay"
    elif REPLAY["step"] < 0:
        status = f"{REPLAY['date']} · ready"
    else:
        ts = REPLAY["timeline"][REPLAY["step"]]
        status = f"{pd.Timestamp(ts).strftime('%H:%M')} · {REPLAY['step'] + 1}/{len(REPLAY['timeline'])}"
        if REPLAY["playing"]:
            status += f" · playing {REPLAY['speed']}x"

    return time.time(), status


@app.callback(
    Output("header-status", "children"),
    Output("metrics", "children"),
    Output("portfolio-chart", "figure"),
    Output("stock-chart", "figure"),
    Output("positions", "data"),
    Output("positions", "columns"),
    Output("trades", "data"),
    Output("trades", "columns"),
    Input("live-refresh", "n_intervals"),
    Input("replay-version", "data"),
    Input("mode", "value"),
    Input("symbol", "value"),
)
def render_dashboard(_refresh, _replay_version, mode, symbol):
    if mode == "replay" and REPLAY["state"] is not None:
        state = REPLAY["state"]
        prices = {}
        if REPLAY["step"] >= 0:
            prices = replay_prices_at(REPLAY["timeline"][REPLAY["step"]])
        value = portfolio_value(state, prices)
        unrealized = unrealized_pnl(state, prices)
        total_pnl = value - STARTING_CASH
        trades_df = pd.DataFrame(state["trades"])
        equity_rows = REPLAY["equity"]
        stock_fig = replay_stock_figure(symbol)
        status_children = [badge("REPLAY", ACCENT)]
    else:
        state = read_state()
        prices = latest_live_prices()
        value = portfolio_value(state, prices)
        unrealized = unrealized_pnl(state, prices)
        total_pnl = value - STARTING_CASH
        trades_df = read_csv(TRADES_FILE)
        equity_df = read_csv(EQUITY_FILE)
        equity_rows = equity_df.to_dict("records") if not equity_df.empty else []
        stock_fig = live_stock_figure(symbol, trades_df)
        market_text, market_color, market_note = market_status()
        age = None
        if STATE_FILE.exists():
            age = max(0, int(time.time() - STATE_FILE.stat().st_mtime))
        bot_text = "LIVE" if age is not None and age <= 90 else "STALE" if age is not None and age <= 240 else "OFFLINE"
        bot_color = GREEN if bot_text == "LIVE" else YELLOW if bot_text == "STALE" else MUTED
        status_children = [badge(f"MARKET {market_text}", market_color), badge(f"BOT {bot_text}", bot_color)]

    return_pct = total_pnl / STARTING_CASH * 100
    realized = float(state.get("realized_pnl", 0.0))

    metrics = [
        metric("Account", f"${value:,.2f}"),
        metric("Cash", f"${float(state.get('cash', 0)):,.2f}"),
        metric("Total P/L", f"${total_pnl:+,.2f}", pnl_color(total_pnl)),
        metric("Return", f"{return_pct:+.2f}%", pnl_color(return_pct)),
        metric("Realized", f"${realized:+,.2f}", pnl_color(realized)),
        metric("Unrealized", f"${unrealized:+,.2f}", pnl_color(unrealized)),
    ]

    pos_rows = positions_rows(state, prices)
    pos_cols = [{"name": c, "id": c} for c in ["Symbol", "Value", "P/L", "Entry"]]

    if trades_df.empty:
        trade_rows = []
        trade_cols = [{"name": c, "id": c} for c in ["symbol", "exit_time", "pnl", "reason"]]
    else:
        cols = [c for c in ["symbol", "exit_time", "pnl", "reason"] if c in trades_df.columns]
        trade_rows = trades_df.tail(6).iloc[::-1][cols].to_dict("records")
        trade_cols = [{"name": c.replace("_", " ").title(), "id": c} for c in cols]

    return (
        status_children,
        metrics,
        portfolio_figure(equity_rows),
        stock_fig,
        pos_rows,
        pos_cols,
        trade_rows,
        trade_cols,
    )


# ==========================================
# START
# ==========================================


def open_browser():
    webbrowser.open("http://127.0.0.1:8050")


if __name__ == "__main__":
    Timer(1.0, open_browser).start()
    app.run(debug=False, host="127.0.0.1", port=8050)
