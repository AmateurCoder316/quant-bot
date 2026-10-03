import json
import time
import webbrowser
from datetime import date, datetime
from pathlib import Path
from threading import Timer
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from dash import Dash, Input, Output, State, dcc, html, dash_table, no_update

SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
INTERVAL = "5m"
PERIOD = "60d"
STARTING_CASH = 10000.0
COMMISSION_RATE = 0.001
SLIPPAGE_RATE = 0.0005
POSITION_SIZE_PERCENT = 0.20
MAX_OPEN_POSITIONS = 3
STOP_LOSS_PERCENT = 0.05
REFRESH_MS = 30_000

STATE_FILE = Path("paper_state_5m.json")
LOG_FILE = Path("logs/paper_trader.log")
EQUITY_FILE = Path("logs/equity.csv")
TRADES_FILE = Path("logs/trades.csv")

NY = ZoneInfo("America/New_York")

BG = "#090b0f"
PANEL = "#101318"
BORDER = "#252a33"
TEXT = "#f4f6f8"
MUTED = "#8b93a1"
GREEN = "#37d67a"
RED = "#ff5c6c"
BLUE = "#6ea8fe"
YELLOW = "#e7b955"


def blank_position():
    return {
        "shares": 0.0,
        "entry_price": None,
        "entry_time": None,
        "entry_total_cost": None,
        "stop_price": None,
        "pending_order": None,
        "last_signal_bar": None,
    }


def fresh_state():
    return {
        "cash": STARTING_CASH,
        "realized_pnl": 0.0,
        "positions": {symbol: blank_position() for symbol in SYMBOLS},
        "trades": [],
    }


def read_state():
    if not STATE_FILE.exists():
        return fresh_state()
    try:
        with STATE_FILE.open("r", encoding="utf-8") as file:
            state = json.load(file)
    except (json.JSONDecodeError, OSError):
        return fresh_state()

    state.setdefault("cash", STARTING_CASH)
    state.setdefault("realized_pnl", 0.0)
    state.setdefault("trades", [])
    state.setdefault("positions", {})
    for symbol in SYMBOLS:
        state["positions"].setdefault(symbol, blank_position())
    return state


def read_csv(path):
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError):
        return pd.DataFrame()


def flatten_columns(data):
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


def download_data(symbol, period=PERIOD):
    data = yf.download(
        symbol,
        period=period,
        interval=INTERVAL,
        auto_adjust=True,
        progress=False,
        prepost=False,
    )
    return flatten_columns(data).dropna().copy()


def add_indicators(data):
    data = data.copy()
    change = data["Close"].diff()
    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / 2, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / 2, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    data["RSI2"] = (100 - (100 / (1 + rs))).fillna(100)
    data["MA20"] = data["Close"].rolling(20).mean()
    data["MA50"] = data["Close"].rolling(50).mean()
    data["MA200"] = data["Close"].rolling(200).mean()
    data["High20"] = data["High"].rolling(20).max().shift(1)
    data["Low10"] = data["Low"].rolling(10).min().shift(1)
    return data


def as_ny_index(index):
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        return idx.tz_localize(NY)
    return idx.tz_convert(NY)


def day_slice(data, selected_date):
    if data.empty:
        return data
    ny_index = as_ny_index(data.index)
    mask = ny_index.date == selected_date
    sliced = data.loc[mask].copy()
    sliced.index = ny_index[mask]
    return sliced


def latest_prices_and_times():
    prices = {}
    timestamps = {}
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
            data = flatten_columns(data).dropna()
            if data.empty:
                continue
            prices[symbol] = float(data.iloc[-1]["Close"])
            timestamps[symbol] = as_ny_index(data.index)[-1]
        except Exception:
            continue
    return prices, timestamps


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


def strategy_signal(row, has_position):
    if pd.isna(row["MA200"]):
        return "HOLD"
    trend_up = row["Close"] > row["MA200"]
    if not has_position:
        breakout_buy = not pd.isna(row["High20"]) and row["Close"] > row["High20"]
        mean_reversion_buy = trend_up and row["RSI2"] < 10
        if breakout_buy or mean_reversion_buy:
            return "BUY"
    if has_position:
        breakout_sell = not pd.isna(row["Low10"]) and row["Close"] < row["Low10"]
        mean_reversion_sell = row["RSI2"] > 70
        if breakout_sell or mean_reversion_sell:
            return "SELL"
    return "HOLD"


def sim_execute_buy(state, symbol, timestamp, market_open, prices):
    position = state["positions"][symbol]
    if position["shares"] > 0 or position_count(state) >= MAX_OPEN_POSITIONS:
        return
    total_value = portfolio_value(state, prices)
    allocation = min(state["cash"], total_value * POSITION_SIZE_PERCENT)
    if allocation <= 0:
        return
    execution_price = market_open * (1 + SLIPPAGE_RATE)
    trade_value = allocation / (1 + COMMISSION_RATE)
    commission = trade_value * COMMISSION_RATE
    shares = trade_value / execution_price
    total_cost = trade_value + commission
    state["cash"] -= total_cost
    position["shares"] = shares
    position["entry_price"] = execution_price
    position["entry_time"] = timestamp.isoformat()
    position["entry_total_cost"] = total_cost
    position["stop_price"] = execution_price * (1 - STOP_LOSS_PERCENT)
    position["pending_order"] = None


def sim_execute_sell(state, symbol, timestamp, market_price, reason):
    position = state["positions"][symbol]
    if position["shares"] <= 0:
        return
    execution_price = market_price * (1 - SLIPPAGE_RATE)
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
        "exit_time": timestamp.isoformat(),
        "entry_price": position["entry_price"],
        "exit_price": execution_price,
        "pnl": pnl,
        "return_percent": ret,
        "reason": reason,
    })
    state["positions"][symbol] = blank_position()


def simulate_day(selected_date_text):
    selected_date = date.fromisoformat(selected_date_text)
    state = fresh_state()
    day_data = {}
    for symbol in SYMBOLS:
        raw = add_indicators(download_data(symbol))
        day_data[symbol] = day_slice(raw, selected_date)

    available = [frame for frame in day_data.values() if not frame.empty]
    if not available:
        return {
            "ok": False,
            "date": selected_date_text,
            "message": "No regular-session 5-minute candles are available for this date.",
        }

    all_times = sorted(set().union(*(set(frame.index) for frame in available)))
    latest_prices = {}
    equity = []

    for timestamp in all_times:
        for symbol in SYMBOLS:
            frame = day_data[symbol]
            if timestamp not in frame.index:
                continue
            row = frame.loc[timestamp]
            position = state["positions"][symbol]
            pending = position["pending_order"]
            if pending is not None and pd.Timestamp(pending["signal_time"]) < timestamp:
                if pending["side"] == "BUY":
                    sim_execute_buy(state, symbol, timestamp, float(row["Open"]), latest_prices)
                elif pending["side"] == "SELL":
                    sim_execute_sell(state, symbol, timestamp, float(row["Open"]), "signal")
                state["positions"][symbol]["pending_order"] = None

        for symbol in SYMBOLS:
            frame = day_data[symbol]
            if timestamp not in frame.index:
                continue
            row = frame.loc[timestamp]
            close = float(row["Close"])
            latest_prices[symbol] = close
            position = state["positions"][symbol]
            if position["shares"] > 0 and position["stop_price"] is not None and close <= position["stop_price"]:
                sim_execute_sell(state, symbol, timestamp, close, "stop_loss")

        for symbol in SYMBOLS:
            frame = day_data[symbol]
            if timestamp not in frame.index:
                continue
            row = frame.loc[timestamp]
            position = state["positions"][symbol]
            signal = strategy_signal(row, position["shares"] > 0)
            if signal in {"BUY", "SELL"}:
                position["pending_order"] = {"side": signal, "signal_time": timestamp.isoformat()}

        equity.append({
            "timestamp": timestamp.isoformat(),
            "portfolio_value": portfolio_value(state, latest_prices),
            "cash": state["cash"],
            "realized_pnl": state["realized_pnl"],
            "unrealized_pnl": unrealized_pnl(state, latest_prices),
            "open_positions": position_count(state),
        })

    final_prices = {}
    stock_data = {}
    for symbol, frame in day_data.items():
        if frame.empty:
            continue
        final_prices[symbol] = float(frame.iloc[-1]["Close"])
        stock_data[symbol] = [
            {
                "timestamp": ts.isoformat(),
                "close": float(row["Close"]),
                "ma20": None if pd.isna(row["MA20"]) else float(row["MA20"]),
                "ma50": None if pd.isna(row["MA50"]) else float(row["MA50"]),
            }
            for ts, row in frame.iterrows()
        ]

    return {
        "ok": True,
        "date": selected_date_text,
        "state": state,
        "prices": final_prices,
        "equity": equity,
        "trades": state["trades"],
        "stock_data": stock_data,
    }


def market_status(timestamps):
    now_ny = datetime.now(NY)
    if not timestamps:
        return "NO DATA", RED, "No market data available"
    latest = max(timestamps.values())
    if latest.date() != now_ny.date():
        return "MARKET CLOSED", MUTED, f"Last candle {latest.strftime('%a %H:%M ET')}"
    if now_ny.weekday() >= 5:
        return "MARKET CLOSED", MUTED, f"Last candle {latest.strftime('%H:%M ET')}"
    minutes = now_ny.hour * 60 + now_ny.minute
    if 570 <= minutes < 960:
        return "MARKET OPEN", GREEN, f"Latest data {latest.strftime('%H:%M ET')}"
    return "MARKET CLOSED", MUTED, f"Last candle {latest.strftime('%H:%M ET')}"


def bot_status():
    if not STATE_FILE.exists():
        return "BOT WAITING", YELLOW
    age = time.time() - STATE_FILE.stat().st_mtime
    if age <= 100:
        return "BOT LIVE", GREEN
    if age <= 240:
        return "BOT STALE", YELLOW
    return "BOT OFFLINE", RED


def metric(label, value, accent=TEXT):
    return html.Div([
        html.Div(label, style={"fontSize": "11px", "color": MUTED}),
        html.Div(value, style={"fontSize": "22px", "fontWeight": "700", "color": accent, "marginTop": "5px"}),
    ], style={"background": PANEL, "border": f"1px solid {BORDER}", "borderRadius": "10px", "padding": "14px 16px"})


def badge(text, color):
    return html.Span(text, style={
        "fontSize": "11px", "fontWeight": "700", "color": color,
        "border": f"1px solid {color}", "borderRadius": "999px", "padding": "5px 9px"
    })


def base_figure(height=330):
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font={"color": MUTED, "family": "Inter, system-ui, sans-serif", "size": 11},
        margin={"l": 42, "r": 16, "t": 20, "b": 36},
        xaxis={"gridcolor": BORDER, "zeroline": False},
        yaxis={"gridcolor": BORDER, "zeroline": False},
        height=height,
        hovermode="x unified",
    )
    return fig


def portfolio_chart(equity):
    fig = base_figure()
    if not equity:
        return fig
    data = pd.DataFrame(equity)
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    fig.add_trace(go.Scatter(
        x=data["timestamp"], y=data["portfolio_value"], mode="lines",
        line={"color": BLUE, "width": 2}, name="Portfolio"
    ))
    fig.add_hline(y=STARTING_CASH, line_dash="dot", line_color="#4c5360")
    fig.update_layout(showlegend=False)
    return fig


def stock_chart_from_frame(symbol, data, state, trades_df):
    fig = base_figure()
    if data.empty:
        return fig
    fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Price", line={"color": TEXT, "width": 2}))
    if "MA20" in data:
        fig.add_trace(go.Scatter(x=data.index, y=data["MA20"], mode="lines", name="MA20", line={"color": BLUE, "width": 1}, opacity=0.7))
    if "MA50" in data:
        fig.add_trace(go.Scatter(x=data.index, y=data["MA50"], mode="lines", name="MA50", line={"color": YELLOW, "width": 1}, opacity=0.7))
    if not trades_df.empty and "symbol" in trades_df.columns:
        symbol_trades = trades_df[trades_df["symbol"] == symbol]
        if not symbol_trades.empty:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(symbol_trades["entry_time"], errors="coerce"),
                y=symbol_trades["entry_price"], mode="markers", name="Buy",
                marker={"symbol": "triangle-up", "size": 10, "color": GREEN}
            ))
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(symbol_trades["exit_time"], errors="coerce"),
                y=symbol_trades["exit_price"], mode="markers", name="Sell",
                marker={"symbol": "triangle-down", "size": 10, "color": RED}
            ))
    position = state.get("positions", {}).get(symbol, {})
    if float(position.get("shares", 0.0) or 0.0) > 0 and position.get("entry_price") is not None:
        fig.add_hline(y=float(position["entry_price"]), line_dash="dash", line_color=GREEN)
    fig.update_layout(legend={"orientation": "h", "x": 0, "y": 1.08})
    return fig


def live_stock_chart(symbol, state, trades_df):
    try:
        data = add_indicators(download_data(symbol, period="5d"))
    except Exception:
        return base_figure()
    return stock_chart_from_frame(symbol, data, state, trades_df)


def simulation_stock_frame(sim_data, symbol):
    rows = sim_data.get("stock_data", {}).get(symbol, [])
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.set_index("timestamp")
    return frame.rename(columns={"close": "Close", "ma20": "MA20", "ma50": "MA50"})


def positions_rows(state, prices):
    rows = []
    for symbol in SYMBOLS:
        position = state.get("positions", {}).get(symbol, {})
        shares = float(position.get("shares", 0.0) or 0.0)
        if shares <= 0:
            continue
        current = prices.get(symbol)
        entry_cost = position.get("entry_total_cost")
        value = shares * current if current is not None else None
        pnl = value - float(entry_cost) if value is not None and entry_cost is not None else None
        rows.append({
            "Symbol": symbol,
            "Value": None if value is None else round(value, 2),
            "P/L": None if pnl is None else round(pnl, 2),
            "Entry": position.get("entry_price"),
            "Stop": position.get("stop_price"),
        })
    return rows


def live_activity(limit=6):
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
        events.append(line.split(" | ")[-1])
        if len(events) >= limit:
            break
    return events


def activity_components(events):
    if not events:
        return [html.Div("No trades yet", style={"color": MUTED, "fontSize": "13px"})]
    result = []
    for event in events:
        color = GREEN if event.startswith("BUY ") else RED if event.startswith("SELL ") else YELLOW
        result.append(html.Div(
            event,
            style={
                "fontSize": "12px", "color": TEXT, "padding": "8px 0",
                "borderBottom": f"1px solid {BORDER}", "borderLeft": f"2px solid {color}",
                "paddingLeft": "10px"
            }
        ))
    return result


app = Dash(__name__)
app.title = "Quant Bot"

TABLE_STYLE = {
    "style_table": {"overflowX": "auto"},
    "style_cell": {
        "backgroundColor": PANEL, "color": TEXT, "border": "none",
        "borderBottom": f"1px solid {BORDER}", "padding": "9px 6px",
        "fontFamily": "Inter, system-ui, sans-serif", "fontSize": "12px", "textAlign": "left"
    },
    "style_header": {
        "backgroundColor": PANEL, "color": MUTED, "fontWeight": "600",
        "border": "none", "borderBottom": f"1px solid {BORDER}"
    },
}

BUTTON_STYLE = {
    "height": "36px", "background": TEXT, "color": BG, "border": "none",
    "borderRadius": "7px", "padding": "0 14px", "fontWeight": "700", "cursor": "pointer"
}

app.layout = html.Div(
    style={"minHeight": "100vh", "background": BG, "color": TEXT, "fontFamily": "Inter, system-ui, sans-serif", "padding": "22px"},
    children=[
        dcc.Interval(id="refresh", interval=REFRESH_MS, n_intervals=0),
        dcc.Store(id="simulation-store"),

        html.Div(
            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "gap": "16px", "flexWrap": "wrap", "marginBottom": "18px"},
            children=[
                html.Div([
                    html.H1("Quant Bot", style={"margin": 0, "fontSize": "24px", "fontWeight": "700"}),
                    html.Div("5-minute paper trading", style={"fontSize": "12px", "color": MUTED, "marginTop": "3px"}),
                ]),
                html.Div(id="status-area", style={"display": "flex", "gap": "8px", "alignItems": "center", "flexWrap": "wrap"}),
            ],
        ),

        html.Div(
            style={"display": "flex", "gap": "10px", "alignItems": "end", "flexWrap": "wrap", "marginBottom": "18px", "padding": "12px", "background": PANEL, "border": f"1px solid {BORDER}", "borderRadius": "10px"},
            children=[
                html.Div([
                    html.Div("Mode", style={"fontSize": "10px", "color": MUTED, "marginBottom": "6px"}),
                    dcc.RadioItems(
                        id="view-mode",
                        options=[{"label": " Live", "value": "live"}, {"label": " Simulate", "value": "simulate"}],
                        value="live", inline=True,
                        style={"fontSize": "13px", "display": "flex", "gap": "12px"},
                    ),
                ]),
                html.Div([
                    html.Div("Trading day", style={"fontSize": "10px", "color": MUTED, "marginBottom": "6px"}),
                    dcc.DatePickerSingle(
                        id="simulation-date",
                        date=date.today().isoformat(),
                        max_date_allowed=date.today().isoformat(),
                        display_format="YYYY-MM-DD",
                    ),
                ], id="date-control"),
                html.Button("Run simulation", id="run-simulation", n_clicks=0, style=BUTTON_STYLE),
                html.Div(id="simulation-message", style={"fontSize": "12px", "color": MUTED, "paddingBottom": "8px"}),
            ],
        ),

        html.Div(id="metrics", style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(145px, 1fr))", "gap": "10px", "marginBottom": "14px"}),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "minmax(0, 1fr) minmax(0, 1fr)", "gap": "14px", "marginBottom": "14px"},
            children=[
                html.Div([
                    html.Div("Portfolio", style={"fontSize": "12px", "color": MUTED, "padding": "12px 14px 0"}),
                    dcc.Graph(id="portfolio-chart", config={"displayModeBar": False}),
                ], style={"background": PANEL, "border": f"1px solid {BORDER}", "borderRadius": "10px", "overflow": "hidden"}),
                html.Div([
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "padding": "10px 12px 0"},
                        children=[
                            html.Div("Price", style={"fontSize": "12px", "color": MUTED}),
                            dcc.Dropdown(
                                id="symbol-dropdown",
                                options=[{"label": s, "value": s} for s in SYMBOLS],
                                value="AAPL", clearable=False,
                                style={"width": "120px", "color": "#111"},
                            ),
                        ],
                    ),
                    dcc.Graph(id="stock-chart", config={"displayModeBar": False}),
                ], style={"background": PANEL, "border": f"1px solid {BORDER}", "borderRadius": "10px", "overflow": "hidden"}),
            ],
        ),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "minmax(0, 1.2fr) minmax(260px, 0.8fr)", "gap": "14px"},
            children=[
                html.Div([
                    html.Div("Positions", style={"fontSize": "12px", "color": MUTED, "marginBottom": "8px"}),
                    dash_table.DataTable(id="positions-table", **TABLE_STYLE),
                ], style={"background": PANEL, "border": f"1px solid {BORDER}", "borderRadius": "10px", "padding": "12px"}),
                html.Div([
                    html.Div("Activity", style={"fontSize": "12px", "color": MUTED, "marginBottom": "4px"}),
                    html.Div(id="activity"),
                ], style={"background": PANEL, "border": f"1px solid {BORDER}", "borderRadius": "10px", "padding": "12px"}),
            ],
        ),
    ],
)


@app.callback(
    Output("simulation-store", "data"),
    Output("simulation-message", "children"),
    Input("run-simulation", "n_clicks"),
    State("simulation-date", "date"),
    prevent_initial_call=True,
)
def run_simulation_callback(_, selected_date):
    if not selected_date:
        return no_update, "Select a date first."
    try:
        result = simulate_day(selected_date)
    except Exception as error:
        return None, f"Simulation failed: {error}"
    if not result.get("ok"):
        return result, result.get("message", "No data for that day.")
    return result, f"Simulated {selected_date} · {len(result.get('trades', []))} completed trade(s)"


@app.callback(
    Output("status-area", "children"),
    Output("metrics", "children"),
    Output("portfolio-chart", "figure"),
    Output("stock-chart", "figure"),
    Output("positions-table", "data"),
    Output("positions-table", "columns"),
    Output("activity", "children"),
    Output("date-control", "style"),
    Output("run-simulation", "style"),
    Input("refresh", "n_intervals"),
    Input("view-mode", "value"),
    Input("simulation-store", "data"),
    Input("symbol-dropdown", "value"),
)
def render_dashboard(_, mode, simulation, symbol):
    if mode == "simulate":
        date_style = {"display": "block"}
        button_style = BUTTON_STYLE
        if not simulation or not simulation.get("ok"):
            state = fresh_state()
            prices = {}
            equity = []
            trades_df = pd.DataFrame()
            status = [badge("SIMULATION", BLUE)]
            events = []
        else:
            state = simulation["state"]
            prices = simulation.get("prices", {})
            equity = simulation.get("equity", [])
            trades_df = pd.DataFrame(simulation.get("trades", []))
            status = [badge(f"SIM · {simulation['date']}", BLUE)]
            events = [
                f"{trade['symbol']} SELL · P/L ${trade['pnl']:+.2f} · {trade['reason']}"
                for trade in reversed(simulation.get("trades", []))
            ][:6]
        stock_fig = stock_chart_from_frame(symbol, simulation_stock_frame(simulation or {}, symbol), state, trades_df)
        portfolio_fig = portfolio_chart(equity)
    else:
        date_style = {"display": "none"}
        button_style = {**BUTTON_STYLE, "display": "none"}
        state = read_state()
        prices, timestamps = latest_prices_and_times()
        equity_df = read_csv(EQUITY_FILE)
        equity = equity_df.to_dict("records") if not equity_df.empty else []
        trades_df = read_csv(TRADES_FILE)
        market_text, market_color, market_note = market_status(timestamps)
        bot_text, bot_color = bot_status()
        status = [
            badge(market_text, market_color),
            badge(bot_text, bot_color),
            html.Span(market_note, style={"fontSize": "11px", "color": MUTED}),
        ]
        events = live_activity()
        portfolio_fig = portfolio_chart(equity)
        stock_fig = live_stock_chart(symbol, state, trades_df)

    value = portfolio_value(state, prices)
    total_pnl = value - STARTING_CASH
    realized = float(state.get("realized_pnl", 0.0))
    unrealized = unrealized_pnl(state, prices)
    return_pct = total_pnl / STARTING_CASH * 100

    cards = [
        metric("Account", f"${value:,.2f}"),
        metric("Cash", f"${float(state.get('cash', 0)):,.2f}"),
        metric("P/L", f"${total_pnl:+,.2f}", pnl_color(total_pnl)),
        metric("Return", f"{return_pct:+.2f}%", pnl_color(return_pct)),
        metric("Realized", f"${realized:+,.2f}", pnl_color(realized)),
        metric("Unrealized", f"${unrealized:+,.2f}", pnl_color(unrealized)),
        metric("Positions", f"{position_count(state)}/{MAX_OPEN_POSITIONS}"),
        metric("Trades", str(len(state.get("trades", [])))),
    ]

    rows = positions_rows(state, prices)
    columns = [{"name": key, "id": key} for key in rows[0].keys()] if rows else []

    return (
        status, cards, portfolio_fig, stock_fig, rows, columns,
        activity_components(events), date_style, button_style,
    )


def open_browser():
    webbrowser.open("http://127.0.0.1:8050/")


if __name__ == "__main__":
    Timer(1.0, open_browser).start()
    app.run(debug=False, host="127.0.0.1", port=8050)
