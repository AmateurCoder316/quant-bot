from __future__ import annotations

import os
import shutil
import time
from datetime import date

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from exp4_config import ALL_SYMBOLS, LEGACY_5M_DIR, SOURCE_5M_DIR

BASE_URL = "https://data.alpaca.markets/v2/stocks/{symbol}/bars"
TIMEFRAME = "5Min"
FEED = "iex"
START_DATE = "2023-01-01"
END_DATE = date.today().isoformat()
PAGE_LIMIT = 5000
REQUEST_PAUSE_SECONDS = 0.25
MAX_ATTEMPTS = 6


def credentials() -> tuple[str, str]:
    key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in your shell/.env loader.")
    return key, secret


def build_session() -> requests.Session:
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


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["Open", "High", "Low", "Close", "Volume"]
    frame = frame[required].copy()
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame[~frame.index.duplicated(keep="last")].sort_index().dropna()


def read_saved(path):
    if not path.exists():
        return None
    frame = pd.read_csv(path, parse_dates=["Timestamp"]).set_index("Timestamp")
    return normalize(frame)


def is_complete(frame: pd.DataFrame | None) -> bool:
    if frame is None or frame.empty:
        return False
    last_day = pd.Timestamp(frame.index.max()).date()
    return (pd.Timestamp(END_DATE).date() - last_day).days <= 4


def request_page(session, symbol, headers, params):
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = session.get(
                BASE_URL.format(symbol=symbol),
                headers=headers,
                params=params,
                timeout=(15, 90),
            )
            response.raise_for_status()
            return response.json()
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = exc
            if attempt == MAX_ATTEMPTS:
                break
            sleep = min(30, 2 ** attempt)
            print(f"    network retry {attempt}/{MAX_ATTEMPTS} in {sleep}s", flush=True)
            time.sleep(sleep)
    raise RuntimeError(f"failed to download {symbol}") from last_error


def fetch_symbol(session, symbol, headers):
    rows = []
    token = None
    page = 0
    while True:
        params = {
            "timeframe": TIMEFRAME,
            "start": START_DATE,
            "end": END_DATE,
            "limit": PAGE_LIMIT,
            "adjustment": "all",
            "feed": FEED,
            "sort": "asc",
        }
        if token:
            params["page_token"] = token
        payload = request_page(session, symbol, headers, params)
        bars = payload.get("bars") or []
        rows.extend(bars)
        page += 1
        if bars:
            print(f"    page {page:>2}: +{len(bars):,} ({len(rows):,})", flush=True)
        token = payload.get("next_page_token")
        if not token:
            break
        time.sleep(REQUEST_PAUSE_SECONDS)

    if not rows:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    frame = pd.DataFrame(rows).rename(columns={
        "t": "Timestamp", "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"
    })
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], utc=True)
    frame = frame.set_index("Timestamp")
    return normalize(frame)


def main() -> None:
    key, secret = credentials()
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    session = build_session()
    SOURCE_5M_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("EXP-4 - DOWNLOAD LARGE-UNIVERSE 5-MINUTE DATA")
    print("=" * 100)
    print(f"Symbols: {len(ALL_SYMBOLS)} | {START_DATE} -> {END_DATE} | Alpaca {FEED.upper()}")
    print("Existing complete legacy files are reused; missing symbols are downloaded into EXP-4 storage.\n")

    downloaded = reused = skipped = 0
    total_rows = 0
    for number, symbol in enumerate(ALL_SYMBOLS, start=1):
        destination = SOURCE_5M_DIR / f"{symbol}.csv.gz"
        existing = read_saved(destination)
        if is_complete(existing):
            print(f"[{number:02d}/{len(ALL_SYMBOLS)}] {symbol:<5} already complete ({len(existing):,} bars)")
            skipped += 1
            total_rows += len(existing)
            continue

        legacy = LEGACY_5M_DIR / f"{symbol}.csv.gz"
        legacy_data = read_saved(legacy)
        if is_complete(legacy_data):
            shutil.copy2(legacy, destination)
            print(f"[{number:02d}/{len(ALL_SYMBOLS)}] {symbol:<5} reused legacy data ({len(legacy_data):,} bars)")
            reused += 1
            total_rows += len(legacy_data)
            continue

        print(f"[{number:02d}/{len(ALL_SYMBOLS)}] {symbol:<5} downloading...", flush=True)
        try:
            data = fetch_symbol(session, symbol, headers)
        except Exception as exc:
            print(f"    FAILED: {exc}")
            continue
        if data.empty:
            print("    no bars returned")
            continue
        data.to_csv(destination, index_label="Timestamp", compression="gzip")
        downloaded += 1
        total_rows += len(data)
        print(f"    saved {len(data):,} bars -> {destination}")

    print("\n" + "=" * 100)
    print(f"Downloaded: {downloaded} | reused: {reused} | already present: {skipped}")
    print(f"Total available rows counted: {total_rows:,}")
    print("Next: python build_exp_4_dataset.py")


if __name__ == "__main__":
    main()
