from __future__ import annotations

import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests


ECB_DAILY_FX_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml"
CACHE_PATH = Path(".usd_eur_rate.json")
CACHE_TTL_SECONDS = 6 * 60 * 60


def _read_cache() -> tuple[float, float] | None:
    try:
        payload = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        rate = float(payload["usd_to_eur"])
        timestamp = float(payload["timestamp"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
    if rate <= 0:
        return None
    return rate, timestamp


def _write_cache(rate: float) -> None:
    try:
        CACHE_PATH.write_text(
            json.dumps({"usd_to_eur": rate, "timestamp": time.time()}),
            encoding="utf-8",
        )
    except OSError:
        pass


def _fetch_ecb_rate() -> float:
    response = requests.get(ECB_DAILY_FX_URL, timeout=6)
    response.raise_for_status()
    root = ET.fromstring(response.content)
    usd_per_eur = None
    for element in root.iter():
        if element.attrib.get("currency") == "USD":
            usd_per_eur = float(element.attrib["rate"])
            break
    if usd_per_eur is None or usd_per_eur <= 0:
        raise ValueError("ECB response did not contain a valid USD rate")
    return 1.0 / usd_per_eur


def get_usd_to_eur_rate() -> float:
    override = os.getenv("USD_TO_EUR_RATE")
    if override:
        try:
            rate = float(override)
            if rate > 0:
                return rate
        except ValueError:
            pass

    cached = _read_cache()
    if cached is not None:
        rate, timestamp = cached
        if time.time() - timestamp <= CACHE_TTL_SECONDS:
            return rate

    try:
        rate = _fetch_ecb_rate()
    except Exception:
        if cached is not None:
            return cached[0]
        # Last-resort fallback keeps the UI alive. Set USD_TO_EUR_RATE to override
        # this explicitly if the machine cannot reach the ECB endpoint.
        return 0.85

    _write_cache(rate)
    return rate


__all__ = ["get_usd_to_eur_rate"]
