from __future__ import annotations

import argparse
import importlib.util
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from desktop_dashboard.control import load_control, write_control


ROOT = Path(__file__).resolve().parent
LIVE_STATE = ROOT / "live_state.json"
CONTROL_FILE = ROOT / "bot_control.json"
DASHBOARD_MODULE = "desktop_dashboard"
TRADER_SCRIPT = ROOT / "paper_trader.py"
DEMO_MODULE = "desktop_dashboard.demo_state"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Quant Bot and its desktop dashboard")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the dashboard with fake moving data instead of the paper trader.",
    )
    parser.add_argument(
        "--dashboard-only",
        action="store_true",
        help="Launch only the desktop dashboard.",
    )
    return parser.parse_args()


def check_dashboard_dependencies() -> None:
    missing = [
        package
        for package in ("PySide6", "pyqtgraph")
        if importlib.util.find_spec(package) is None
    ]
    if missing:
        names = ", ".join(missing)
        raise SystemExit(
            f"Missing dashboard dependencies: {names}\n"
            "Install them with:\n"
            "  pip install -r requirements-dashboard.txt"
        )


def start_process(command: list[str], *, name: str) -> subprocess.Popen:
    try:
        return subprocess.Popen(
            command,
            cwd=ROOT,
            env=os.environ.copy(),
        )
    except Exception as exc:
        raise SystemExit(f"Could not start {name}: {exc}") from exc


def stop_process(process: subprocess.Popen | None, *, interrupt: bool = False) -> None:
    if process is None or process.poll() is not None:
        return

    try:
        if interrupt and os.name != "nt":
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=4)
                return
            except subprocess.TimeoutExpired:
                pass

        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
    except ProcessLookupError:
        pass


def worker_command(args: argparse.Namespace) -> tuple[list[str], str, bool]:
    if args.demo:
        return (
            [
                sys.executable,
                "-m",
                DEMO_MODULE,
                "--state",
                str(LIVE_STATE),
            ],
            "dashboard demo feed",
            False,
        )

    if not TRADER_SCRIPT.exists():
        raise SystemExit(f"Missing trader: {TRADER_SCRIPT}")
    return [sys.executable, str(TRADER_SCRIPT)], "paper trader", True


def main() -> int:
    args = parse_args()
    check_dashboard_dependencies()

    dashboard = None
    worker = None
    worker_needs_interrupt = False

    # Every normal/demo launch starts in the running state. The button may then
    # stop and restart the worker while keeping the dashboard alive.
    write_control(CONTROL_FILE, not args.dashboard_only)

    try:
        dashboard = start_process(
            [
                sys.executable,
                "-m",
                DASHBOARD_MODULE,
                "--state",
                str(LIVE_STATE),
                "--control",
                str(CONTROL_FILE),
            ],
            name="desktop dashboard",
        )

        if args.dashboard_only:
            print("Desktop dashboard started.")
            print("Close the window or press Ctrl+C here to stop.")
        else:
            command, name, worker_needs_interrupt = worker_command(args)
            worker = start_process(command, name=name)
            if args.demo:
                print("Quant Bot dashboard demo started.")
            else:
                print("Quant Bot started.")
                print("Paper trader + desktop dashboard are running.")
            print("Use Start/Stop in the dashboard or Ctrl+C here.")

        while True:
            if dashboard.poll() is not None:
                return_code = dashboard.returncode
                if return_code not in (0, None):
                    print(f"Dashboard exited with code {return_code}.")
                break

            if not args.dashboard_only:
                desired_running = load_control(CONTROL_FILE).running

                if worker is not None and worker.poll() is not None:
                    return_code = worker.returncode
                    print(f"Worker exited with code {return_code}.")
                    worker = None
                    if desired_running:
                        write_control(CONTROL_FILE, False)
                        desired_running = False

                if desired_running and worker is None:
                    command, name, worker_needs_interrupt = worker_command(args)
                    worker = start_process(command, name=name)
                    print(f"{name.capitalize()} started.")
                elif not desired_running and worker is not None:
                    stop_process(worker, interrupt=worker_needs_interrupt)
                    worker = None
                    print("Worker stopped. Dashboard remains open.")

            time.sleep(0.25)

    except KeyboardInterrupt:
        print("\nStopping Quant Bot...")
    finally:
        write_control(CONTROL_FILE, False)
        stop_process(worker, interrupt=worker_needs_interrupt)
        stop_process(dashboard)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
