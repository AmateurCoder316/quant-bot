import os
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from market_data import DATA_DIR, normalize_ohlcv
from strategy_core import INTERVAL, SYMBOLS


BASE_URL = "https://data.alpaca.markets/v2/stocks/{symbol}/bars"
TIMEFRAME = "5Min"
FEED = "iex"
START_DATE = "2023-01-01"
END_DATE = date.today().isoformat()
PAGE_LIMIT = 10000
REQUEST_PAUSE_SECONDS = 0.35


def credentials():
    key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")

    if not key or not secret:
        raise RuntimeError(
            "Missing Alpaca credentials. Set APCA_API_KEY_ID and "
            "APCA_API_SECRET_KEY in your shell or .env loader."
        )

    return key, secret


def fetch_symbol(symbol, start=START_DATE, end=END_DATE):
    key, secret = credentials()
    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }

    rows = []
    page_token = None

    while True:
        params = {
            "timeframe": TIMEFRAME,
            "start": start,
            "end": end,
            "limit": PAGE_LIMIT,
            "adjustment": "all",
            "feed": FEED,
            "sort": "asc",
        }
        if page_token:
            params["page_token"] = page_token

        response = requests.get(
            BASE_URL.format(symbol=symbol),
            headers=headers,
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()

        bars = payload.get("bars") or []
        rows.extend(bars)
        page_token = payload.get("next_page_token")

        if not page_token:
            break

        time.sleep(REQUEST_PAUSE_SECONDS)

    if not rows:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    frame = pd.DataFrame(rows)
    frame = frame.rename(
        columns={
            "t": "Timestamp",
            "o": "Open",
            "h": "High",
            "l": "Low",
            "c": "Close",
            "v": "Volume",
        }
    )
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], utc=True)
    frame = frame.set_index("Timestamp")

    return normalize_ohlcv(frame)


def save_symbol(symbol, data):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{symbol}.csv.gz"
    data.to_csv(path, index_label="Timestamp", compression="gzip")
    return path


def main():
    print("=" * 72)
    print("HISTORICAL 5-MINUTE DATA DOWNLOAD")
    print("=" * 72)
    print(f"Provider: Alpaca ({FEED.upper()} feed)")
    print(f"Interval: {INTERVAL}")
    print(f"Range:    {START_DATE} -> {END_DATE}")
    print(f"Symbols:  {', '.join(SYMBOLS)}")
    print()

    total_rows = 0

    for symbol in SYMBOLS:
        print(f"Downloading {symbol} ...", flush=True)
        data = fetch_symbol(symbol)
        if data.empty:
            print(f"  no data returned for {symbol}")
            continue

        path = save_symbol(symbol, data)
        total_rows += len(data)
        print(
            f"  {len(data):,} bars | "
            f"{data.index.min()} -> {data.index.max()} | {path}"
        )

    print()
    print(f"Total bars saved: {total_rows:,}")
    print("Research usage:")
    print("  QUANT_DATA_SOURCE=local python strategy_research.py")
    print("  QUANT_DATA_SOURCE=local python intraday_backtest.py")


if __name__ == "__main__":
    main()
