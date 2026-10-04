from __future__ import annotations

import argparse
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from .state import DEFAULT_STATE_PATH, write_state_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write harmless fake data for dashboard preview")
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--model", default="EXP-2-M2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(42)

    starting_value = 10_000.0
    portfolio_value = starting_value
    curve = []
    step = 0

    try:
        while True:
            drift = 0.45 + math.sin(step / 8.0) * 1.8
            noise = random.gauss(0.0, 5.5)
            portfolio_value = max(100.0, portfolio_value + drift + noise)
            profit_loss = portfolio_value - starting_value
            profit_loss_percent = profit_loss / starting_value * 100.0

            now = datetime.now(timezone.utc)
            curve.append({"timestamp": now.isoformat(), "value": portfolio_value})
            curve = curve[-900:]

            aapl_last = 287.40 + math.sin(step / 5.0) * 1.9
            nvda_last = 214.75 + math.cos(step / 7.0) * 2.4
            aapl_entry = 284.10
            nvda_entry = 216.25
            aapl_qty = 7.0
            nvda_qty = 5.0

            positions = [
                {
                    "symbol": "AAPL",
                    "quantity": aapl_qty,
                    "entry_price": aapl_entry,
                    "current_price": aapl_last,
                    "market_value": aapl_qty * aapl_last,
                    "profit_loss": (aapl_last - aapl_entry) * aapl_qty,
                    "profit_loss_percent": (aapl_last / aapl_entry - 1.0) * 100.0,
                },
                {
                    "symbol": "NVDA",
                    "quantity": nvda_qty,
                    "entry_price": nvda_entry,
                    "current_price": nvda_last,
                    "market_value": nvda_qty * nvda_last,
                    "profit_loss": (nvda_last - nvda_entry) * nvda_qty,
                    "profit_loss_percent": (nvda_last / nvda_entry - 1.0) * 100.0,
                },
            ]

            write_state_atomic(
                args.state,
                {
                    "portfolio_value": portfolio_value,
                    "starting_value": starting_value,
                    "profit_loss": profit_loss,
                    "profit_loss_percent": profit_loss_percent,
                    "model": args.model,
                    "equity_curve": curve,
                    "positions": positions,
                },
            )

            step += 1
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
