import os
import time
from datetime import date

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from market_data import DATA_DIR, normalize_ohlcv
from strategy_core import INTERVAL, SYMBOLS


BASE_URL = "https://data.alpaca.markets/v2/stocks/{symbol}/bars"
TIMEFRAME = "5Min"
FEED = "iex"
START_DATE = "2023-01-01"
END_DATE = date.today().isoformat()
PAGE_LIMIT = 5000
REQUEST_PAUSE_SECONDS = 0.40
REQUEST_TIMEOUT_SECONDS = 90
MAX_PAGE_ATTEMPTS = 6


def credentials():
    key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")

    if not key or not secret:
        raise RuntimeError(
            "Missing Alpaca credentials. Set APCA_API_KEY_ID and "
            "APCA_API_SECRET_KEY in your shell or .env loader."
        )

    return key, secret


def build_session():
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
    )

    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def request_page(session, symbol, headers, params):
    last_error = None

    for attempt in range(1, MAX_PAGE_ATTEMPTS + 1):
        try:
            response = session.get(
                BASE_URL.format(symbol=symbol),
                headers=headers,
                params=params,
                timeout=(15, REQUEST_TIMEOUT_SECONDS),
            )
            response.raise_for_status()
            return response.json()
        except (requests.Timeout, requests.ConnectionError) as error:
            last_error = error
            if attempt == MAX_PAGE_ATTEMPTS:
                break

            wait_seconds = min(30, 2 ** attempt)
            print(
                f"  network issue on {symbol}; retry {attempt}/{MAX_PAGE_ATTEMPTS} "
                f"in {wait_seconds}s ...",
                flush=True,
            )
            time.sleep(wait_seconds)

    raise RuntimeError(
        f"Failed to download a page for {symbol} after {MAX_PAGE_ATTEMPTS} attempts"
    ) from last_error


def load_existing(symbol):
    path = DATA_DIR / f"{symbol}.csv.gz"
    if not path.exists():
        return None

    data = pd.read_csv(path, parse_dates=["Timestamp"])
    data = data.set_index("Timestamp")
    return normalize_ohlcv(data)


def fetch_symbol(symbol, start=START_DATE, end=END_DATE):
    key, secret = credentials()
    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }

    session = build_session()
    rows = []
    page_token = None
    page_number = 0

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

        payload = request_page(session, symbol, headers, params)
        bars = payload.get("bars") or []
        rows.extend(bars)
        page_number += 1

        if bars:
            print(
                f"  page {page_number:>2}: +{len(bars):,} bars "
                f"({len(rows):,} total)",
                flush=True,
            )

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


def symbol_is_complete(data, end):
    if data is None or data.empty:
        return False

    last_day = pd.Timestamp(data.index.max()).date()
    target_day = pd.Timestamp(end).date()

    # The requested end may be a weekend/holiday. Treat data reaching within
    # four calendar days of the target as complete enough for this snapshot.
    return (target_day - last_day).days <= 4


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
        existing = load_existing(symbol)
        if symbol_is_complete(existing, END_DATE):
            print(
                f"Skipping {symbol}: already saved "
                f"({len(existing):,} bars through {existing.index.max()})"
            )
            total_rows += len(existing)
            continue

        print(f"Downloading {symbol} ...", flush=True)

        try:
            data = fetch_symbol(symbol)
        except Exception as error:
            print(f"  FAILED {symbol}: {error}")
            print("  Continuing with the next symbol. Re-run later to retry failures.")
            continue

        if data.empty:
            print(f"  no data returned for {symbol}")
            continue

        path = save_symbol(symbol, data)
        total_rows += len(data)
        print(
            f"  saved {len(data):,} bars | "
            f"{data.index.min()} -> {data.index.max()} | {path}"
        )

    print()
    print(f"Total bars currently available: {total_rows:,}")
    print("Research usage:")
    print("  QUANT_DATA_SOURCE=local python strategy_research.py")
    print("  QUANT_DATA_SOURCE=local python intraday_backtest.py")


if __name__ == "__main__":
    main()
