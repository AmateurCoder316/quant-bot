from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Import PySide6 before pyqtgraph so pyqtgraph binds to PySide6 even if another
# Qt wrapper happens to be installed in the environment.
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
import pyqtgraph as pg

from .control import DEFAULT_CONTROL_PATH, load_control, write_control
from .currency import get_usd_to_eur_rate
from .state import DEFAULT_STATE_PATH, DashboardState, Position, load_state
from .styles import (
    ACCENT,
    APP_STYLESHEET,
    BORDER,
    GREEN,
    MUTED,
    RED,
    SURFACE,
    TEXT,
)


POLL_INTERVAL_MS = 500


class MetricCard(QFrame):
    def __init__(self, label: str, *, model_card: bool = False, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(94)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 15, 18, 15)
        layout.setSpacing(7)

        label_widget = QLabel(label)
        label_widget.setObjectName("metricLabel")

        self.value = QLabel("—")
        self.value.setObjectName("modelValue" if model_card else "metricValue")
        self.value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        layout.addWidget(label_widget)
        layout.addWidget(self.value)
        layout.addStretch(1)

    def set_value(self, text: str, color: str | None = None) -> None:
        self.value.setText(text)
        if color:
            self.value.setStyleSheet(f"color: {color}; background: transparent;")
        else:
            self.value.setStyleSheet("")


class MoneyAxis(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):  # noqa: N802 - Qt/pyqtgraph API
        output = []
        for value in values:
            absolute = abs(value)
            if absolute >= 1_000_000:
                output.append(f"€{value / 1_000_000:.1f}m")
            elif absolute >= 1_000:
                output.append(f"€{value / 1_000:.1f}k")
            else:
                output.append(f"€{value:,.0f}")
        return output


class EquityGraph(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("graphPanel")
        self.setMinimumHeight(315)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 8)
        layout.setSpacing(0)

        date_axis = pg.DateAxisItem(orientation="bottom")
        money_axis = MoneyAxis(orientation="left")
        self.plot = pg.PlotWidget(axisItems={"bottom": date_axis, "left": money_axis})
        self.plot.setBackground(SURFACE)
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=True, y=False)
        self.plot.hideButtons()
        self.plot.setClipToView(True)
        self.plot.setDownsampling(auto=True, mode="peak")
        self.plot.showGrid(x=True, y=True, alpha=0.12)

        plot_item = self.plot.getPlotItem()
        plot_item.setContentsMargins(4, 4, 4, 4)
        plot_item.setDefaultPadding(0.025)

        for axis_name in ("left", "bottom"):
            axis = plot_item.getAxis(axis_name)
            axis.setPen(pg.mkPen(BORDER, width=1))
            axis.setTextPen(pg.mkPen(MUTED))
            axis.setStyle(tickLength=0)

        self.curve = self.plot.plot(
            [],
            [],
            pen=pg.mkPen(ACCENT, width=2),
            antialias=True,
            connect="finite",
        )
        layout.addWidget(self.plot)

    def set_state(self, state: DashboardState, usd_to_eur: float) -> None:
        points = [point for point in state.equity_curve if point.timestamp > 0]
        if not points:
            self.curve.setData([], [])
            return

        x = [point.timestamp for point in points]
        y = [point.value * usd_to_eur for point in points]
        color = GREEN if state.profit_loss > 0 else RED if state.profit_loss < 0 else ACCENT
        self.curve.setPen(pg.mkPen(color, width=2))
        self.curve.setData(x, y)

        plot_item = self.plot.getPlotItem()
        plot_item.enableAutoRange(axis="x", enable=True)
        plot_item.enableAutoRange(axis="y", enable=True)


class PositionsTable(QFrame):
    HEADERS = ("Symbol", "Qty", "Entry", "Last", "P/L", "P/L %")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("tablePanel")
        self.setMinimumHeight(225)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)

        self.table = QTableWidget(0, len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.setAlternatingRowColors(False)
        self.table.setVerticalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self.table.setHorizontalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self.table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        header = self.table.horizontalHeader()
        header.setHighlightSections(False)
        header.setStretchLastSection(False)
        for column in range(len(self.HEADERS)):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)

        layout.addWidget(self.table)

    @staticmethod
    def _cell(text: str, *, color: str | None = None, bold: bool = False) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
        if color:
            item.setForeground(QColor(color))
        if bold:
            font = item.font()
            font.setBold(True)
            item.setFont(font)
        return item

    def set_positions(self, positions: list[Position], usd_to_eur: float) -> None:
        self.table.setUpdatesEnabled(False)
        try:
            self.table.setRowCount(len(positions))
            for row, position in enumerate(positions):
                pl_color = GREEN if position.profit_loss > 0 else RED if position.profit_loss < 0 else TEXT

                symbol_item = self._cell(position.symbol, bold=True)
                symbol_item.setTextAlignment(
                    Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
                )
                self.table.setItem(row, 0, symbol_item)
                self.table.setItem(row, 1, self._cell(_format_quantity(position.quantity)))
                self.table.setItem(row, 2, self._cell(_format_price(position.entry_price * usd_to_eur)))
                self.table.setItem(row, 3, self._cell(_format_price(position.current_price * usd_to_eur)))
                self.table.setItem(
                    row,
                    4,
                    self._cell(
                        _format_signed_money(position.profit_loss * usd_to_eur),
                        color=pl_color,
                    ),
                )
                self.table.setItem(
                    row,
                    5,
                    self._cell(_format_signed_percent(position.profit_loss_percent), color=pl_color),
                )
                self.table.setRowHeight(row, 42)
        finally:
            self.table.setUpdatesEnabled(True)


class DashboardWindow(QMainWindow):
    def __init__(self, state_path: Path, control_path: Path):
        super().__init__()
        self.state_path = Path(state_path)
        self.control_path = Path(control_path)
        self._last_mtime_ns: int | None = None
        self._desired_running = True
        self.usd_to_eur = get_usd_to_eur_rate()

        self.setWindowTitle("Quant Bot")
        self.setMinimumSize(1020, 680)
        self.resize(1280, 800)

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        metrics = QGridLayout()
        metrics.setHorizontalSpacing(14)
        metrics.setVerticalSpacing(0)
        metrics.setColumnStretch(0, 1)
        metrics.setColumnStretch(1, 1)
        metrics.setColumnStretch(2, 1)
        metrics.setColumnStretch(3, 0)

        self.portfolio_card = MetricCard("Portfolio value")
        self.profit_card = MetricCard("P/L")
        self.model_card = MetricCard("Model", model_card=True)
        self.control_button = QPushButton()
        self.control_button.setObjectName("controlButton")
        self.control_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.control_button.clicked.connect(self.toggle_running)

        metrics.addWidget(self.portfolio_card, 0, 0)
        metrics.addWidget(self.profit_card, 0, 1)
        metrics.addWidget(self.model_card, 0, 2)
        metrics.addWidget(
            self.control_button,
            0,
            3,
            alignment=Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
        )

        self.graph = EquityGraph()
        self.positions = PositionsTable()

        layout.addLayout(metrics)
        layout.addWidget(self.graph, stretch=5)
        layout.addWidget(self.positions, stretch=3)

        self.timer = QTimer(self)
        self.timer.setInterval(POLL_INTERVAL_MS)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()

        self.apply_state(DashboardState())
        self.refresh_control()
        self.refresh(force=True)

    def toggle_running(self) -> None:
        self._desired_running = not self._desired_running
        write_control(self.control_path, self._desired_running)
        self.apply_control_state()

    def refresh_control(self) -> None:
        control = load_control(self.control_path)
        if control.running != self._desired_running:
            self._desired_running = control.running
            self.apply_control_state()
        elif not self.control_button.text():
            self.apply_control_state()

    def apply_control_state(self) -> None:
        self.control_button.setText("Stop" if self._desired_running else "Start")
        self.control_button.setProperty(
            "runState",
            "running" if self._desired_running else "stopped",
        )
        self.control_button.style().unpolish(self.control_button)
        self.control_button.style().polish(self.control_button)

    def refresh(self, force: bool = False) -> None:
        self.refresh_control()
        try:
            stat = self.state_path.stat()
        except FileNotFoundError:
            return
        except OSError:
            return

        if not force and stat.st_mtime_ns == self._last_mtime_ns:
            return

        try:
            state = load_state(self.state_path)
        except (OSError, ValueError, TypeError):
            return

        self._last_mtime_ns = stat.st_mtime_ns
        self.apply_state(state)

    def apply_state(self, state: DashboardState) -> None:
        self.portfolio_card.set_value(
            _format_money(state.portfolio_value * self.usd_to_eur)
        )

        pl_color = GREEN if state.profit_loss > 0 else RED if state.profit_loss < 0 else TEXT
        self.profit_card.set_value(
            f"{_format_signed_money(state.profit_loss * self.usd_to_eur)}  "
            f"{_format_signed_percent(state.profit_loss_percent)}",
            pl_color,
        )
        self.model_card.set_value(state.model)
        self.graph.set_state(state, self.usd_to_eur)
        self.positions.set_positions(state.positions, self.usd_to_eur)


def _format_money(value: float) -> str:
    return f"€{value:,.2f}"


def _format_signed_money(value: float) -> str:
    sign = "+" if value > 0 else "-" if value < 0 else ""
    return f"{sign}€{abs(value):,.2f}"


def _format_signed_percent(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}%"


def _format_price(value: float) -> str:
    return f"€{value:,.2f}"


def _format_quantity(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.4f}".rstrip("0").rstrip(".")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal live paper-trading desktop dashboard")
    parser.add_argument(
        "--state",
        type=Path,
        default=DEFAULT_STATE_PATH,
        help=f"Live state JSON path (default: {DEFAULT_STATE_PATH})",
    )
    parser.add_argument(
        "--control",
        type=Path,
        default=DEFAULT_CONTROL_PATH,
        help=f"Bot control JSON path (default: {DEFAULT_CONTROL_PATH})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    pg.setConfigOptions(antialias=True)
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Quant Bot")
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLESHEET)

    window = DashboardWindow(args.state, args.control)
    window.show()
    window.raise_()
    window.activateWindow()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
