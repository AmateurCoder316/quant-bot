import os
from pathlib import Path

import pandas as pd
import yfinance as yf

from strategy_core import INTERVAL


DATA_DIR = Path("data") / INTERVAL
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def normalize_ohlcv(data):
    """Return a clean OHLCV frame with a sorted DatetimeIndex."""
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    frame = data[REQUIRED_COLUMNS].copy()
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame.dropna()


class YahooMarketData:
    """Live/development provider using yfinance."""

    name = "yfinance"

    def history(self, symbol, period="60d", interval=INTERVAL):
        data = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            prepost=False,
        )
        return normalize_ohlcv(data)

    def day(self, symbol, date_text, interval=INTERVAL):
        start = pd.Timestamp(date_text)
        end = start + pd.Timedelta(days=1)

        data = yf.download(
            symbol,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=True,
            progress=False,
            prepost=False,
        )
        return normalize_ohlcv(data)


class LocalMarketData:
    """Research provider backed by normalized local CSV.gz files."""

    name = "local"

    def __init__(self, root=DATA_DIR):
        self.root = Path(root)

    def path_for(self, symbol):
        return self.root / f"{symbol.upper()}.csv.gz"

    def history(self, symbol, period=None, interval=INTERVAL):
        if interval != INTERVAL:
            raise ValueError(f"Local dataset is stored as {INTERVAL}, not {interval}.")

        path = self.path_for(symbol)
        if not path.exists():
            raise FileNotFoundError(
                f"No local {INTERVAL} dataset for {symbol}: {path}. "
                "Run historical_data.py first."
            )

        data = pd.read_csv(path, index_col="Timestamp", parse_dates=["Timestamp"])
        return normalize_ohlcv(data)

    def day(self, symbol, date_text, interval=INTERVAL):
        data = self.history(symbol, interval=interval)
        target = pd.Timestamp(date_text).date()
        mask = [pd.Timestamp(ts).date() == target for ts in data.index]
        return data.loc[mask].copy()


YAHOO_PROVIDER = YahooMarketData()
LOCAL_PROVIDER = LocalMarketData()


def provider_from_environment():
    """
    Select the provider without coupling strategy code to a data source.

    QUANT_DATA_SOURCE=yahoo  -> yfinance (default, suitable for live paper bot)
    QUANT_DATA_SOURCE=local  -> files under data/5m/ (recommended for research)
    """
    source = os.getenv("QUANT_DATA_SOURCE", "yahoo").strip().lower()

    if source in {"yahoo", "yfinance"}:
        return YAHOO_PROVIDER
    if source == "local":
        return LOCAL_PROVIDER

    raise ValueError(
        f"Unknown QUANT_DATA_SOURCE={source!r}. Expected 'yahoo' or 'local'."
    )


DEFAULT_PROVIDER = provider_from_environment()
