from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_STATE_PATH = Path("live_state.json")


@dataclass
class Position:
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    market_value: float = 0.0
    profit_loss: float = 0.0
    profit_loss_percent: float = 0.0

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "Position":
        quantity = float(value.get("quantity", value.get("qty", 0.0)) or 0.0)
        entry_price = float(value.get("entry_price", value.get("avg_entry_price", 0.0)) or 0.0)
        current_price = float(value.get("current_price", value.get("last_price", 0.0)) or 0.0)
        market_value = float(value.get("market_value", quantity * current_price) or 0.0)

        explicit_pl = value.get("profit_loss", value.get("unrealized_pl"))
        if explicit_pl is None:
            profit_loss = (current_price - entry_price) * quantity
        else:
            profit_loss = float(explicit_pl or 0.0)

        explicit_pl_percent = value.get(
            "profit_loss_percent",
            value.get("unrealized_pl_percent"),
        )
        if explicit_pl_percent is None:
            basis = abs(entry_price * quantity)
            profit_loss_percent = (profit_loss / basis * 100.0) if basis else 0.0
        else:
            profit_loss_percent = float(explicit_pl_percent or 0.0)

        return cls(
            symbol=str(value.get("symbol", "")).upper(),
            quantity=quantity,
            entry_price=entry_price,
            current_price=current_price,
            market_value=market_value,
            profit_loss=profit_loss,
            profit_loss_percent=profit_loss_percent,
        )


@dataclass
class EquityPoint:
    timestamp: float
    value: float

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "EquityPoint":
        raw_timestamp = value.get("timestamp")
        timestamp = _parse_timestamp(raw_timestamp)
        return cls(timestamp=timestamp, value=float(value.get("value", 0.0) or 0.0))


@dataclass
class DashboardState:
    portfolio_value: float = 0.0
    profit_loss: float = 0.0
    profit_loss_percent: float = 0.0
    model: str = "—"
    positions: list[Position] = field(default_factory=list)
    equity_curve: list[EquityPoint] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "DashboardState":
        portfolio_value = float(value.get("portfolio_value", 0.0) or 0.0)
        starting_value = float(value.get("starting_value", 0.0) or 0.0)

        raw_pl = value.get("profit_loss")
        if raw_pl is None and starting_value:
            profit_loss = portfolio_value - starting_value
        else:
            profit_loss = float(raw_pl or 0.0)

        raw_pl_percent = value.get("profit_loss_percent")
        if raw_pl_percent is None and starting_value:
            profit_loss_percent = profit_loss / starting_value * 100.0
        else:
            profit_loss_percent = float(raw_pl_percent or 0.0)

        positions = [
            Position.from_mapping(item)
            for item in value.get("positions", [])
            if isinstance(item, dict)
        ]
        equity_curve = [
            EquityPoint.from_mapping(item)
            for item in value.get("equity_curve", [])
            if isinstance(item, dict)
        ]
        equity_curve.sort(key=lambda item: item.timestamp)

        return cls(
            portfolio_value=portfolio_value,
            profit_loss=profit_loss,
            profit_loss_percent=profit_loss_percent,
            model=str(value.get("model", "—") or "—"),
            positions=positions,
            equity_curve=equity_curve,
        )


def _parse_timestamp(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0.0
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return 0.0
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    return 0.0


def load_state(path: Path) -> DashboardState:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Dashboard state must be a JSON object.")
    return DashboardState.from_mapping(payload)


def write_state_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace the dashboard state file.

    The live trader should use this helper rather than writing directly to the
    watched JSON path. The dashboard can never observe a half-written document.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


__all__ = [
    "DEFAULT_STATE_PATH",
    "DashboardState",
    "EquityPoint",
    "Position",
    "load_state",
    "write_state_atomic",
]
