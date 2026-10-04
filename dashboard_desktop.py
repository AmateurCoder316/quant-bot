from __future__ import annotations

import sys
from pathlib import Path


DASHBOARD_DIR = Path(__file__).resolve().parent / "desktop_dashboard"
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from app import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
