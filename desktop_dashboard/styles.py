BACKGROUND = "#0a0b0d"
SURFACE = "#111317"
SURFACE_ALT = "#0e1013"
BORDER = "#20242a"
TEXT = "#f3f4f6"
MUTED = "#858b96"
GREEN = "#42c985"
RED = "#f06a72"
ACCENT = "#d8dde6"
GRID = "#252a31"


APP_STYLESHEET = f"""
QMainWindow {{
    background: {BACKGROUND};
}}

QWidget {{
    background: {BACKGROUND};
    color: {TEXT};
    font-family: Inter, "Noto Sans", "Segoe UI", sans-serif;
    font-size: 13px;
}}

QFrame#card,
QFrame#graphPanel,
QFrame#tablePanel {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}

QLabel#metricLabel {{
    background: transparent;
    color: {MUTED};
    font-size: 12px;
    font-weight: 500;
}}

QLabel#metricValue {{
    background: transparent;
    color: {TEXT};
    font-size: 25px;
    font-weight: 650;
}}

QLabel#modelValue {{
    background: transparent;
    color: {TEXT};
    font-size: 20px;
    font-weight: 600;
}}

QTableWidget {{
    background: {SURFACE};
    alternate-background-color: {SURFACE_ALT};
    border: none;
    outline: none;
    gridline-color: transparent;
    selection-background-color: {BORDER};
    selection-color: {TEXT};
}}

QTableWidget::item {{
    border: none;
    padding: 0 10px;
}}

QHeaderView::section {{
    background: {SURFACE};
    color: {MUTED};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 9px 10px;
    font-size: 11px;
    font-weight: 600;
}}

QScrollBar:vertical {{
    width: 8px;
    background: transparent;
    margin: 2px;
}}

QScrollBar::handle:vertical {{
    background: {BORDER};
    min-height: 30px;
    border-radius: 4px;
}}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {{
    background: transparent;
    height: 0;
}}
"""


__all__ = [
    "ACCENT",
    "APP_STYLESHEET",
    "BACKGROUND",
    "BORDER",
    "GREEN",
    "GRID",
    "MUTED",
    "RED",
    "SURFACE",
    "TEXT",
]
