import json
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from threading import Timer

import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from dash import Dash, Input, Output, dcc, html, dash_table


# ==========================================
# SECTION 1 — SETTINGS
# ==========================================

SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
INTERVAL = "5m"
STARTING_CASH = 10000.0
REFRESH_MS = 15_000

STATE_FILE = Path("paper_state_5m.json")
LOG_FILE = Path("logs/paper_trader.log")
EQUITY_FILE = Path("logs/equity.csv")
TRADES_FILE = Path("logs/trades.csv")

BG = "#07111f"
PANEL = "#0d1b2a"
PANEL_2 = "#102235"
BORDER = "#1d3550"
TEXT = "#edf6ff"
MUTED = "#8fa9bd"
GREEN = "#35d07f"
RED = "#ff6174"
BLUE = "#4da3ff"
YELLOW = "#f4c95d"


# ==========================================
# SECTION 2 — SAFE FILE READERS
# ==========================================


def read_state():
    if not STATE_FILE.exists():
        return {
            "cash": STARTING_CASH,
            "realized_pnl": 0.0,
            "positions": {},
            "trades": [],
        }

    try:
        with STATE_FILE.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return {
            "cash": STARTING_CASH,
            "realized_pnl": 0.0,
            "positions": {},
            "trades": [],
        }


def read_csv(path):
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError):
        return pd.DataFrame()


# ==========================================
# SECTION 3 — MARKET DATA
# ==========================================


def download_symbol_data(symbol):
    data = yf.download(
        symbol,
        period="5d",
        interval=INTERVAL,
        auto_adjust=True,
        progress=False,
        prepost=False,
    )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.dropna().copy()

    if len(data) > 0:
        data["MA20"] = data["Close"].rolling(20).mean()
        data["MA50"] = data["Close"].rolling(50).mean()

    return data


def latest_prices():
    prices = {}

    for symbol in SYMBOLS:
        try:
            data = yf.download(
                symbol,
                period="1d",
                interval="1m",
                auto_adjust=True,
                progress=False,
                prepost=False,
            )

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            data = data.dropna()
            if len(data) > 0:
                prices[symbol] = float(data.iloc[-1]["Close"])
        except Exception:
            continue

    return prices


# ==========================================
# SECTION 4 — PORTFOLIO HELPERS
# ==========================================


def position_count(state):
    return sum(
        1
        for position in state.get("positions", {}).values()
        if float(position.get("shares", 0.0) or 0.0) > 0
    )


def portfolio_value(state, prices):
    value = float(state.get("cash", STARTING_CASH))

    for symbol, position in state.get("positions", {}).items():
        shares = float(position.get("shares", 0.0) or 0.0)
        if shares > 0 and symbol in prices:
            value += shares * prices[symbol]

    return value


def unrealized_pnl(state, prices):
    total = 0.0

    for symbol, position in state.get("positions", {}).items():
        shares = float(position.get("shares", 0.0) or 0.0)
        entry_cost = position.get("entry_total_cost")

        if shares <= 0 or entry_cost is None or symbol not in prices:
            continue

        total += shares * prices[symbol] - float(entry_cost)

    return total


def pnl_color(value):
    if value > 0:
        return GREEN
    if value < 0:
        return RED
    return TEXT


# ==========================================
# SECTION 5 — UI HELPERS
# ==========================================


def metric_card(label, value, accent=TEXT, note=None):
    children = [
        html.Div(label, style={"fontSize": "12px", "color": MUTED, "fontWeight": "600"}),
        html.Div(value, style={"fontSize": "25px", "fontWeight": "750", "color": accent, "marginTop": "5px"}),
    ]

    if note:
        children.append(
            html.Div(note, style={"fontSize": "11px", "color": MUTED, "marginTop": "5px"})
        )

    return html.Div(
        children,
        style={
            "background": PANEL,
            "border": f"1px solid {BORDER}",
            "borderRadius": "14px",
            "padding": "16px 18px",
            "minHeight": "78px",
        },
    )


def status_badge():
    if not STATE_FILE.exists():
        return "WAITING", YELLOW, "No state file yet"

    age_seconds = time.time() - STATE_FILE.stat().st_mtime

    if age_seconds <= 90:
        return "LIVE", GREEN, f"Updated {int(age_seconds)}s ago"
    if age_seconds <= 240:
        return "STALE", YELLOW, f"Updated {int(age_seconds)}s ago"

    return "OFFLINE", RED, f"Updated {int(age_seconds)}s ago"


def recent_activity(limit=8):
    if not LOG_FILE.exists():
        return []

    try:
        lines = LOG_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []

    events = []

    for line in reversed(lines):
        if " BUY " not in line and " SELL " not in line and "BUY BLOCKED" not in line:
            continue

        parts = line.split(" | ", 2)
        timestamp = parts[0] if parts else ""
        message = parts[-1] if parts else line

        event_type = "INFO"
        accent = BLUE
        if message.startswith("BUY "):
            event_type = "BUY"
            accent = GREEN
        elif message.startswith("SELL "):
            event_type = "SELL"
            accent = RED
        elif "BUY BLOCKED" in message:
            event_type = "BLOCKED"
            accent = YELLOW

        events.append((timestamp, event_type, message, accent))
        if len(events) >= limit:
            break

    return events


# ==========================================
# SECTION 6 — CHARTS
# ==========================================


def empty_figure(title):
    fig = go.Figure()
    fig.update_layout(
        title=title,
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font={"color": TEXT},
        xaxis={"gridcolor": BORDER},
        yaxis={"gridcolor": BORDER},
        margin={"l": 45, "r": 20, "t": 55, "b": 40},
        height=390,
    )
    return fig


def portfolio_figure(equity_df):
    fig = empty_figure("Portfolio value")

    if equity_df.empty or "portfolio_value" not in equity_df.columns:
        return fig

    data = equity_df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = data.dropna(subset=["timestamp", "portfolio_value"])

    fig.add_trace(
        go.Scatter(
            x=data["timestamp"],
            y=data["portfolio_value"],
            mode="lines",
            name="Portfolio",
            line={"color": BLUE, "width": 2.5},
            fill="tozeroy",
            fillcolor="rgba(77,163,255,0.07)",
        )
    )

    fig.add_hline(
        y=STARTING_CASH,
        line_dash="dot",
        line_color=MUTED,
        annotation_text="Start $10,000",
        annotation_font_color=MUTED,
    )

    fig.update_layout(showlegend=False)
    return fig


def stock_figure(symbol, state, trades_df):
    fig = empty_figure(f"{symbol} · 5 minute price")

    try:
        data = download_symbol_data(symbol)
    except Exception:
        return fig

    if data.empty:
        return fig

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            name="Price",
            line={"color": TEXT, "width": 2.2},
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MA20"],
            mode="lines",
            name="MA20",
            line={"color": BLUE, "width": 1.2},
            opacity=0.8,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MA50"],
            mode="lines",
            name="MA50",
            line={"color": YELLOW, "width": 1.2},
            opacity=0.8,
        )
    )

    if not trades_df.empty and "symbol" in trades_df.columns:
        symbol_trades = trades_df[trades_df["symbol"] == symbol].copy()

        if not symbol_trades.empty:
            entries = pd.to_datetime(symbol_trades["entry_time"], errors="coerce")
            exits = pd.to_datetime(symbol_trades["exit_time"], errors="coerce")

            fig.add_trace(
                go.Scatter(
                    x=entries,
                    y=symbol_trades["entry_price"],
                    mode="markers",
                    name="Buy",
                    marker={"symbol": "triangle-up", "size": 12, "color": GREEN},
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=exits,
                    y=symbol_trades["exit_price"],
                    mode="markers",
                    name="Sell",
                    marker={"symbol": "triangle-down", "size": 12, "color": RED},
                )
            )

    position = state.get("positions", {}).get(symbol, {})
    shares = float(position.get("shares", 0.0) or 0.0)
    entry_price = position.get("entry_price")

    if shares > 0 and entry_price is not None:
        fig.add_hline(
            y=float(entry_price),
            line_dash="dash",
            line_color=GREEN,
            annotation_text="Open entry",
            annotation_font_color=GREEN,
        )

    fig.update_layout(
        legend={"orientation": "h", "y": 1.08, "x": 0},
        hovermode="x unified",
    )

    return fig


# ==========================================
# SECTION 7 — DASH APP
# ==========================================


app = Dash(__name__)
app.title = "Quant Bot Dashboard"

TABLE_STYLE = {
    "style_table": {"overflowX": "auto", "borderRadius": "12px"},
    "style_cell": {
        "backgroundColor": PANEL,
        "color": TEXT,
        "border": f"1px solid {BORDER}",
        "padding": "10px",
        "fontFamily": "Inter, system-ui, sans-serif",
        "fontSize": "13px",
        "textAlign": "left",
    },
    "style_header": {
        "backgroundColor": PANEL_2,
        "color": MUTED,
        "fontWeight": "700",
        "border": f"1px solid {BORDER}",
    },
}

app.layout = html.Div(
    style={
        "minHeight": "100vh",
        "background": BG,
        "color": TEXT,
        "fontFamily": "Inter, system-ui, sans-serif",
        "padding": "26px",
    },
    children=[
        dcc.Interval(id="refresh", interval=REFRESH_MS, n_intervals=0),

        html.Div(
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "gap": "20px",
                "marginBottom": "22px",
                "flexWrap": "wrap",
            },
            children=[
                html.Div([
                    html.Div("QUANT BOT", style={"fontSize": "12px", "letterSpacing": "0.16em", "color": BLUE, "fontWeight": "800"}),
                    html.H1("Paper Trading Dashboard", style={"margin": "3px 0 0", "fontSize": "30px"}),
                    html.Div("5-minute local paper portfolio", style={"color": MUTED, "fontSize": "13px", "marginTop": "4px"}),
                ]),
                html.Div(id="bot-status"),
            ],
        ),

        html.Div(
            id="metrics",
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(170px, 1fr))",
                "gap": "12px",
                "marginBottom": "18px",
            },
        ),

        html.Div(
            id="symbol-cards",
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(170px, 1fr))",
                "gap": "10px",
                "marginBottom": "18px",
            },
        ),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "minmax(0, 1.05fr) minmax(0, 1fr)",
                "gap": "16px",
                "marginBottom": "18px",
            },
            children=[
                html.Div(
                    dcc.Graph(id="portfolio-chart", config={"displayModeBar": False}),
                    style={"background": PANEL, "border": f"1px solid {BORDER}", "borderRadius": "14px", "overflow": "hidden"},
                ),
                html.Div(
                    [
                        html.Div(
                            dcc.Dropdown(
                                id="symbol-dropdown",
                                options=[{"label": symbol, "value": symbol} for symbol in SYMBOLS],
                                value="AAPL",
                                clearable=False,
                                style={"color": "#101828", "width": "140px"},
                            ),
                            style={"padding": "12px 12px 0"},
                        ),
                        dcc.Graph(id="stock-chart", config={"displayModeBar": False}),
                    ],
                    style={"background": PANEL, "border": f"1px solid {BORDER}", "borderRadius": "14px", "overflow": "hidden"},
                ),
            ],
        ),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "minmax(0, 1.2fr) minmax(280px, 0.8fr)",
                "gap": "16px",
                "marginBottom": "18px",
            },
            children=[
                html.Div([
                    html.H3("Open positions", style={"margin": "0 0 10px"}),
                    dash_table.DataTable(id="positions-table", **TABLE_STYLE),
                ]),
                html.Div([
                    html.H3("Recent activity", style={"margin": "0 0 10px"}),
                    html.Div(id="activity-list"),
                ]),
            ],
        ),

        html.Div([
            html.H3("Recent completed trades", style={"margin": "0 0 10px"}),
            dash_table.DataTable(id="trades-table", page_size=8, **TABLE_STYLE),
        ]),
    ],
)


# ==========================================
# SECTION 8 — CALLBACKS
# ==========================================


@app.callback(
    Output("bot-status", "children"),
    Output("metrics", "children"),
    Output("symbol-cards", "children"),
    Output("portfolio-chart", "figure"),
    Output("positions-table", "data"),
    Output("positions-table", "columns"),
    Output("trades-table", "data"),
    Output("trades-table", "columns"),
    Output("activity-list", "children"),
    Input("refresh", "n_intervals"),
)
def refresh_dashboard(_):
    state = read_state()
    equity_df = read_csv(EQUITY_FILE)
    trades_df = read_csv(TRADES_FILE)
    prices = latest_prices()

    value = portfolio_value(state, prices)
    cash = float(state.get("cash", STARTING_CASH))
    realized = float(state.get("realized_pnl", 0.0))
    unrealized = unrealized_pnl(state, prices)
    total_pnl = value - STARTING_CASH
    return_pct = total_pnl / STARTING_CASH * 100
    open_count = position_count(state)
    completed_count = len(state.get("trades", []))

    status_text, status_color, status_note = status_badge()
    status = html.Div(
        [
            html.Span("●", style={"color": status_color, "marginRight": "7px"}),
            html.Span(status_text, style={"fontWeight": "800", "color": status_color}),
            html.Span(f" · {status_note}", style={"color": MUTED, "fontSize": "12px"}),
        ],
        style={"background": PANEL, "border": f"1px solid {BORDER}", "padding": "10px 14px", "borderRadius": "999px"},
    )

    metrics = [
        metric_card("Account value", f"${value:,.2f}"),
        metric_card("Cash", f"${cash:,.2f}"),
        metric_card("Total P/L", f"${total_pnl:+,.2f}", pnl_color(total_pnl)),
        metric_card("Return", f"{return_pct:+.2f}%", pnl_color(return_pct)),
        metric_card("Realized P/L", f"${realized:+,.2f}", pnl_color(realized)),
        metric_card("Unrealized P/L", f"${unrealized:+,.2f}", pnl_color(unrealized)),
        metric_card("Open positions", f"{open_count}/3"),
        metric_card("Completed trades", str(completed_count)),
    ]

    symbol_cards = []
    positions_rows = []

    for symbol in SYMBOLS:
        position = state.get("positions", {}).get(symbol, {})
        shares = float(position.get("shares", 0.0) or 0.0)
        price = prices.get(symbol)

        if shares > 0 and price is not None:
            market_value = shares * price
            entry_cost = float(position.get("entry_total_cost", 0.0) or 0.0)
            position_pnl = market_value - entry_cost
            card_note = f"Position ${market_value:,.2f} · P/L ${position_pnl:+,.2f}"
            card_accent = pnl_color(position_pnl)
            state_label = "OPEN"

            positions_rows.append({
                "Symbol": symbol,
                "Shares": round(shares, 4),
                "Entry": round(float(position.get("entry_price") or 0.0), 2),
                "Price": round(price, 2),
                "Value": round(market_value, 2),
                "P/L": round(position_pnl, 2),
                "Stop": round(float(position.get("stop_price") or 0.0), 2),
            })
        else:
            card_note = "No open position"
            card_accent = MUTED
            state_label = "FLAT"

        symbol_cards.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(symbol, style={"fontWeight": "800", "fontSize": "15px"}),
                            html.Span(state_label, style={"fontSize": "10px", "fontWeight": "800", "color": card_accent}),
                        ],
                        style={"display": "flex", "justifyContent": "space-between"},
                    ),
                    html.Div(
                        f"${price:,.2f}" if price is not None else "—",
                        style={"fontSize": "21px", "fontWeight": "750", "marginTop": "7px"},
                    ),
                    html.Div(card_note, style={"fontSize": "11px", "color": card_accent, "marginTop": "5px"}),
                ],
                style={"background": PANEL, "border": f"1px solid {BORDER}", "borderRadius": "12px", "padding": "13px 15px"},
            )
        )

    position_columns = [{"name": col, "id": col} for col in ["Symbol", "Shares", "Entry", "Price", "Value", "P/L", "Stop"]]

    if not trades_df.empty:
        recent_trades = trades_df.tail(8).iloc[::-1].copy()
        for column in ["entry_price", "exit_price", "pnl", "return_percent"]:
            if column in recent_trades.columns:
                recent_trades[column] = pd.to_numeric(recent_trades[column], errors="coerce").round(2)
        trade_rows = recent_trades.to_dict("records")
        trade_columns = [{"name": col.replace("_", " ").title(), "id": col} for col in recent_trades.columns]
    else:
        trade_rows = []
        trade_columns = []

    activity_components = []
    events = recent_activity()

    if not events:
        activity_components.append(html.Div("No trade activity yet", style={"color": MUTED, "padding": "14px 0"}))
    else:
        for timestamp, event_type, message, accent in events:
            short_time = timestamp[-8:] if len(timestamp) >= 8 else timestamp
            activity_components.append(
                html.Div(
                    [
                        html.Div(event_type, style={"fontSize": "10px", "fontWeight": "900", "color": accent, "minWidth": "55px"}),
                        html.Div(message, style={"fontSize": "12px", "color": TEXT, "flex": "1"}),
                        html.Div(short_time, style={"fontSize": "10px", "color": MUTED}),
                    ],
                    style={"display": "flex", "gap": "9px", "alignItems": "center", "padding": "9px 0", "borderBottom": f"1px solid {BORDER}"},
                )
            )

    return (
        status,
        metrics,
        symbol_cards,
        portfolio_figure(equity_df),
        positions_rows,
        position_columns,
        trade_rows,
        trade_columns,
        activity_components,
    )


@app.callback(
    Output("stock-chart", "figure"),
    Input("symbol-dropdown", "value"),
    Input("refresh", "n_intervals"),
)
def refresh_stock_chart(symbol, _):
    return stock_figure(symbol, read_state(), read_csv(TRADES_FILE))


# ==========================================
# SECTION 9 — START DASHBOARD
# ==========================================


def open_browser():
    webbrowser.open("http://127.0.0.1:8050")


if __name__ == "__main__":
    Timer(1.0, open_browser).start()
    app.run(host="127.0.0.1", port=8050, debug=False)
