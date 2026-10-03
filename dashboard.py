import json
import time
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path
from threading import Timer

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html, dash_table

from market_data import DEFAULT_PROVIDER
from strategy_core import (
    COMMISSION_RATE,
    INTERVAL,
    MAX_OPEN_POSITIONS,
    SLIPPAGE_RATE,
    STARTING_CASH,
    SYMBOLS,
    add_indicators,
    market_session_status,
    risk_sized_trade_value,
    stop_price_from_atr,
    strategy_signal,
)


# ==========================================
# SETTINGS
# ==========================================

STATE_FILE = Path("paper_state_5m.json")
EQUITY_FILE = Path("logs/equity.csv")
TRADES_FILE = Path("logs/trades.csv")

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
# FILE / PORTFOLIO HELPERS
# ==========================================


def fresh_position():
    return {
        "shares": 0.0,
        "entry_price": None,
        "entry_time": None,
        "entry_total_cost": None,
        "entry_reason": None,
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


def position_count(state):
    return sum(
        float(position.get("shares", 0.0) or 0.0) > 0
        for position in state.get("positions", {}).values()
    )


def portfolio_value(state, prices):
    value = float(state.get("cash", STARTING_CASH))
    for symbol, position in state.get("positions", {}).items():
        shares = float(position.get("shares", 0.0) or 0.0)
        if shares > 0 and symbol in prices:
            value += shares * float(prices[symbol])
    return value


def unrealized_pnl(state, prices):
    total = 0.0
    for symbol, position in state.get("positions", {}).items():
        shares = float(position.get("shares", 0.0) or 0.0)
        cost = position.get("entry_total_cost")
        if shares > 0 and cost is not None and symbol in prices:
            total += shares * float(prices[symbol]) - float(cost)
    return total


def pnl_color(value):
    return GREEN if value > 0 else RED if value < 0 else TEXT


def latest_live_prices():
    state = read_state()
    saved = state.get("last_prices", {})
    if saved:
        return {symbol: float(price) for symbol, price in saved.items()}

    prices = {}
    for symbol in SYMBOLS:
        try:
            data = DEFAULT_PROVIDER.history(symbol, period="1d", interval="1m")
            if not data.empty:
                prices[symbol] = float(data.iloc[-1]["Close"])
        except Exception:
            pass
    return prices


# ==========================================
# REPLAY ENGINE — SAME CORE AS LIVE BOT
# ==========================================


def load_replay_day(date_text):
    try:
        selected = pd.Timestamp(date_text).date()
        warmup_start = selected - timedelta(days=14)
        end = selected + timedelta(days=1)

        prepared = {}
        for symbol in SYMBOLS:
            data = DEFAULT_PROVIDER.history(
                symbol,
                period=None,
                interval=INTERVAL,
            ) if False else None

            # yfinance needs explicit start/end for historical replay.
            import yfinance as yf
            raw = yf.download(
                symbol,
                start=warmup_start.isoformat(),
                end=end.isoformat(),
                interval=INTERVAL,
                auto_adjust=True,
                progress=False,
                prepost=False,
            )
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.dropna().copy()
            if not raw.empty:
                prepared[symbol] = add_indicators(raw)

        if not prepared:
            raise ValueError("No 5-minute data available for that date.")

        reference = prepared[next(iter(prepared))]
        timeline = [ts for ts in reference.index if pd.Timestamp(ts).date() == selected]
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


def replay_prices_at(timestamp):
    prices = {}
    for symbol, data in REPLAY["data"].items():
        rows = data.loc[:timestamp]
        if not rows.empty:
            prices[symbol] = float(rows.iloc[-1]["Close"])
    return prices


def execute_replay_buy(state, symbol, timestamp, market_open, atr, reason, prices):
    position = state["positions"][symbol]
    if position["shares"] > 0 or position_count(state) >= MAX_OPEN_POSITIONS:
        position["pending_order"] = None
        return

    execution_price = float(market_open) * (1 + SLIPPAGE_RATE)
    allocation = risk_sized_trade_value(
        portfolio_value(state, prices),
        state["cash"],
        execution_price,
        atr,
    )
    if allocation <= 0:
        position["pending_order"] = None
        return

    trade_value = allocation / (1 + COMMISSION_RATE)
    commission = trade_value * COMMISSION_RATE
    shares = trade_value / execution_price
    total_cost = trade_value + commission

    state["cash"] -= total_cost
    position.update({
        "shares": shares,
        "entry_price": execution_price,
        "entry_time": pd.Timestamp(timestamp).isoformat(),
        "entry_total_cost": total_cost,
        "entry_reason": reason,
        "stop_price": stop_price_from_atr(execution_price, atr),
        "pending_order": None,
    })


def execute_replay_sell(state, symbol, timestamp, market_price, reason):
    position = state["positions"][symbol]
    if position["shares"] <= 0:
        return

    execution_price = float(market_price) * (1 - SLIPPAGE_RATE)
    gross = position["shares"] * execution_price
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
        "exit_price": execution_price,
        "pnl": pnl,
        "return_percent": ret,
        "entry_reason": position.get("entry_reason"),
        "exit_reason": reason,
        "reason": reason,
    })
    state["positions"][symbol] = fresh_position()


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

        # 1) Execute orders from previous completed candle.
        for symbol, data in REPLAY["data"].items():
            if timestamp not in data.index:
                continue
            row = data.loc[timestamp]
            position = state["positions"][symbol]
            pending = position.get("pending_order")
            if not pending:
                continue

            if pending["side"] == "BUY":
                execute_replay_buy(
                    state,
                    symbol,
                    timestamp,
                    float(row["Open"]),
                    pending.get("atr"),
                    pending.get("reason", "unknown"),
                    prices,
                )
            elif pending["side"] == "SELL":
                execute_replay_sell(
                    state,
                    symbol,
                    timestamp,
                    float(row["Open"]),
                    pending.get("reason", "signal"),
                )

        # 2) Intrabar ATR stop.
        for symbol, data in REPLAY["data"].items():
            if timestamp not in data.index:
                continue
            row = data.loc[timestamp]
            position = state["positions"][symbol]
            stop = position.get("stop_price")
            if position["shares"] > 0 and stop is not None and float(row["Low"]) <= float(stop):
                fill_reference = min(float(row["Open"]), float(stop))
                execute_replay_sell(state, symbol, timestamp, fill_reference, "atr_stop")

        # 3) Generate current completed-candle signal for next bar.
        for symbol, data in REPLAY["data"].items():
            if timestamp not in data.index:
                continue
            history = data.loc[:timestamp]
            if len(history) < 205:
                continue

            position = state["positions"][symbol]
            signal = strategy_signal(history, position["shares"] > 0)
            if signal.side in {"BUY", "SELL"}:
                latest = history.iloc[-1]
                position["pending_order"] = {
                    "side": signal.side,
                    "reason": signal.reason,
                    "atr": float(latest["ATR14"]) if not pd.isna(latest["ATR14"]) else None,
                }

        prices = replay_prices_at(timestamp)
        REPLAY["equity"].append({
            "timestamp": pd.Timestamp(timestamp).isoformat(),
            "portfolio_value": portfolio_value(state, prices),
        })


# ==========================================
# CHART / UI HELPERS
# ==========================================


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


def panel(children, style=None):
    base = {
        "background": SURFACE,
        "border": f"1px solid {BORDER}",
        "borderRadius": "10px",
        "padding": "16px",
    }
    if style:
        base.update(style)
    return html.Div(children, style=base)


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
        ))
    fig.add_hline(y=STARTING_CASH, line_dash="dot", line_color=MUTED)
    return fig


def stock_figure(symbol, mode, trades_df):
    fig = empty_figure()

    if mode == "replay" and REPLAY["state"] is not None and REPLAY["step"] >= 0:
        if symbol not in REPLAY["data"]:
            return fig
        end_ts = REPLAY["timeline"][REPLAY["step"]]
        selected_date = pd.Timestamp(REPLAY["date"]).date()
        data = REPLAY["data"][symbol]
        visible = data[(data.index <= end_ts) & (pd.Index(data.index).date == selected_date)]
        trades_df = pd.DataFrame([
            trade for trade in REPLAY["state"]["trades"]
            if trade["symbol"] == symbol
        ])
    else:
        try:
            visible = DEFAULT_PROVIDER.history(symbol, period="5d").tail(160)
        except Exception:
            return fig

    if visible.empty:
        return fig

    fig.add_trace(go.Scatter(
        x=visible.index,
        y=visible["Close"],
        mode="lines",
        line={"color": TEXT, "width": 2},
    ))

    if not trades_df.empty and "symbol" in trades_df.columns:
        selected = trades_df[trades_df["symbol"] == symbol]
        if not selected.empty:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(selected["entry_time"], errors="coerce"),
                y=selected["entry_price"],
                mode="markers",
                marker={"symbol": "triangle-up", "size": 9, "color": GREEN},
            ))
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(selected["exit_time"], errors="coerce"),
                y=selected["exit_price"],
                mode="markers",
                marker={"symbol": "triangle-down", "size": 9, "color": RED},
            ))

    return fig


def positions_rows(state, prices):
    rows = []
    for symbol in SYMBOLS:
        position = state.get("positions", {}).get(symbol, {})
        shares = float(position.get("shares", 0.0) or 0.0)
        if shares <= 0:
            continue
        current = prices.get(symbol)
        cost = position.get("entry_total_cost")
        pnl = None if current is None or cost is None else shares * current - float(cost)
        rows.append({
            "Symbol": symbol,
            "Value": None if current is None else round(shares * current, 2),
            "P/L": None if pnl is None else round(pnl, 2),
            "Entry": None if position.get("entry_price") is None else round(float(position["entry_price"]), 2),
            "Reason": position.get("entry_reason") or "",
        })
    return rows


# ==========================================
# APP
# ==========================================


app = Dash(__name__)
app.title = "Quant Bot"

BUTTON_STYLE = {
    "height": "36px",
    "borderRadius": "8px",
    "border": f"1px solid {BORDER}",
    "background": SURFACE_2,
    "color": TEXT,
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
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "20px"},
                    children=[
                        html.Div([
                            html.Div("QUANT BOT", style={"fontSize": "11px", "fontWeight": "700", "letterSpacing": "0.12em", "color": MUTED}),
                            html.H1("5-minute portfolio", style={"fontSize": "26px", "fontWeight": "650", "margin": "4px 0 0"}),
                        ]),
                        html.Div(id="header-status", style={"display": "flex", "gap": "8px"}),
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
                                labelStyle={"marginRight": "14px", "fontSize": "13px"},
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
                                style={"width": "90px"},
                            ),
                            html.Div(id="replay-status", style={"marginLeft": "auto", "fontSize": "12px", "color": MUTED}),
                        ],
                    )
                ], style={"marginBottom": "16px"}),

                panel(
                    html.Div(id="metrics", style={"display": "grid", "gridTemplateColumns": "repeat(6, minmax(0, 1fr))", "gap": "18px"}),
                    style={"marginBottom": "16px"},
                ),

                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px", "marginBottom": "16px"},
                    children=[
                        panel([
                            html.Div("Portfolio", style={"fontSize": "12px", "color": MUTED, "marginBottom": "6px"}),
                            dcc.Graph(id="portfolio-chart", config={"displayModeBar": False}),
                        ]),
                        panel([
                            html.Div(
                                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "6px"},
                                children=[
                                    html.Div("Price", style={"fontSize": "12px", "color": MUTED}),
                                    dcc.Dropdown(
                                        id="symbol",
                                        options=[{"label": symbol, "value": symbol} for symbol in SYMBOLS],
                                        value="AAPL",
                                        clearable=False,
                                        style={"width": "110px"},
                                    ),
                                ],
                            ),
                            dcc.Graph(id="stock-chart", config={"displayModeBar": False}),
                        ]),
                    ],
                ),

                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
                    children=[
                        panel([
                            html.Div("Open positions", style={"fontSize": "12px", "color": MUTED, "marginBottom": "8px"}),
                            dash_table.DataTable(id="positions", page_size=6, **TABLE),
                        ]),
                        panel([
                            html.Div("Recent trades", style={"fontSize": "12px", "color": MUTED, "marginBottom": "8px"}),
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
def control_replay(_load, _play, _pause, _next, speed, _tick, date_text, mode):
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
        timestamp = REPLAY["timeline"][REPLAY["step"]]
        status = f"{pd.Timestamp(timestamp).strftime('%H:%M')} · {REPLAY['step'] + 1}/{len(REPLAY['timeline'])}"
        if REPLAY["playing"]:
            status += f" · {REPLAY['speed']}x"

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
        prices = replay_prices_at(REPLAY["timeline"][REPLAY["step"]]) if REPLAY["step"] >= 0 else {}
        trades_df = pd.DataFrame(state["trades"])
        equity_rows = REPLAY["equity"]
        status_children = [badge("REPLAY", ACCENT)]
    else:
        state = read_state()
        prices = latest_live_prices()
        trades_df = read_csv(TRADES_FILE)
        equity_df = read_csv(EQUITY_FILE)
        equity_rows = equity_df.to_dict("records") if not equity_df.empty else []

        session = state.get("market_status") or market_session_status()
        market_color = GREEN if session == "OPEN" else MUTED

        age = max(0, int(time.time() - STATE_FILE.stat().st_mtime)) if STATE_FILE.exists() else None
        bot_text = "LIVE" if age is not None and age <= 90 else "STALE" if age is not None and age <= 240 else "OFFLINE"
        bot_color = GREEN if bot_text == "LIVE" else YELLOW if bot_text == "STALE" else MUTED
        status_children = [badge(f"MARKET {session}", market_color), badge(f"BOT {bot_text}", bot_color)]

    value = portfolio_value(state, prices)
    total_pnl = value - STARTING_CASH
    unrealized = unrealized_pnl(state, prices)
    realized = float(state.get("realized_pnl", 0.0))
    return_pct = total_pnl / STARTING_CASH * 100

    metrics = [
        metric("Account", f"${value:,.2f}"),
        metric("Cash", f"${float(state.get('cash', 0)):,.2f}"),
        metric("Total P/L", f"${total_pnl:+,.2f}", pnl_color(total_pnl)),
        metric("Return", f"{return_pct:+.2f}%", pnl_color(return_pct)),
        metric("Realized", f"${realized:+,.2f}", pnl_color(realized)),
        metric("Unrealized", f"${unrealized:+,.2f}", pnl_color(unrealized)),
    ]

    pos_rows = positions_rows(state, prices)
    pos_cols = [{"name": c, "id": c} for c in ["Symbol", "Value", "P/L", "Entry", "Reason"]]

    if trades_df.empty:
        trade_rows = []
        trade_cols = [{"name": c, "id": c} for c in ["symbol", "exit_time", "pnl", "entry_reason", "exit_reason"]]
    else:
        cols = [
            column for column in ["symbol", "exit_time", "pnl", "entry_reason", "exit_reason", "reason"]
            if column in trades_df.columns
        ]
        trade_rows = trades_df.tail(6).iloc[::-1][cols].to_dict("records")
        trade_cols = [{"name": column.replace("_", " ").title(), "id": column} for column in cols]

    return (
        status_children,
        metrics,
        portfolio_figure(equity_rows),
        stock_figure(symbol, mode, trades_df),
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
