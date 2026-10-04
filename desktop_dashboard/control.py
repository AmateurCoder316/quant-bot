from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .state import write_state_atomic


DEFAULT_CONTROL_PATH = Path("bot_control.json")


@dataclass(frozen=True)
class ControlState:
    running: bool = True


def load_control(path: Path) -> ControlState:
    path = Path(path)
    if not path.exists():
        return ControlState(running=True)

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ControlState(running=True)

    if not isinstance(payload, dict):
        return ControlState(running=True)
    return ControlState(running=bool(payload.get("running", True)))


def write_control(path: Path, running: bool) -> None:
    write_state_atomic(
        Path(path),
        {
            "running": bool(running),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


__all__ = ["ControlState", "DEFAULT_CONTROL_PATH", "load_control", "write_control"]
