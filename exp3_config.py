from pathlib import Path


MODEL_NAME = "exp-3-m1"
EXPERIMENT_FAMILY = "exp-3"
SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

SOURCE_5M_DIR = Path("data") / "5m"
OUTPUT_DIR = Path("data") / "experiments" / MODEL_NAME
DATASET_PATH = OUTPUT_DIR / "dataset.parquet"
MODEL_PATH = OUTPUT_DIR / "model.pt"
SCALER_PATH = OUTPUT_DIR / "scaler.joblib"
METADATA_PATH = OUTPUT_DIR / "metadata.json"
SEARCH_RESULTS_PATH = OUTPUT_DIR / "search_results.json"
EXECUTION_PATH = OUTPUT_DIR / "execution.json"
STUDY_PATH = OUTPUT_DIR / "study.sqlite3"

MARKET_TIMEZONE = "America/New_York"
BAR_MINUTES = 30
INPUT_BARS_PER_OUTPUT_BAR = 6
MAIN_HORIZON_BARS = 4       # 2 hours after entry
AUX_1H_BARS = 2
AUX_4H_BARS = 8
PURGE_GAP_MINUTES = 240     # longest auxiliary label horizon

COMMISSION_RATE = 0.001
SLIPPAGE_RATE = 0.0005
STARTING_CASH = 10_000.0
MAX_OPEN_POSITIONS = 3
MAX_POSITION_PERCENT = 0.20

TRAIN_YEARS = [2023, 2024]
CV_VALIDATION_YEAR = 2024
CV_BLOCK_MONTHS = 2
CV_MIN_TRAIN_ROWS = 4_000
TUNE_YEAR = 2025
TEST_YEAR = 2026

SEARCH_SEED = 3301
DEFAULT_TRIALS = 24
MAX_EPOCHS = 90
EARLY_STOPPING_PATIENCE = 12
MIN_EPOCHS = 12

# Model search is allowed to choose only within this deliberately modest family.
HIDDEN_LAYOUTS = [
    [96, 48, 24],
    [128, 64],
    [128, 64, 32],
    [192, 96, 48],
]
ACTIVATIONS = ["gelu", "relu"]
BATCH_SIZES = [256, 512, 1024]

# A trial is economically eligible only when its top-ranked opportunities are
# actually profitable after the configured costs and score ordering agrees with
# realized net returns.
PRIMARY_TAIL_FRACTION = 0.10
SECONDARY_TAIL_FRACTION = 0.05
ECONOMIC_RETURN_SCALE = 0.0030

# 2025 may tune one thing only: the minimum predicted 2-hour net return required
# to open a position. These thresholds are frozen before 2026 is evaluated.
EXECUTION_THRESHOLDS = [0.0000, 0.0010, 0.0020, 0.0030, 0.0040]
MIN_TRADES_FOR_DEPLOYMENT = 40

TARGET_1H = "target_net_1h"
TARGET_2H = "target_net_2h"
TARGET_4H = "target_net_4h"
TARGET_COLUMNS = [TARGET_1H, TARGET_2H, TARGET_4H]

FEATURE_COLUMNS = [
    # Price action / momentum
    "ret_1", "ret_2", "ret_4", "ret_8", "ret_13", "ret_26",
    "range_pct", "body_pct", "upper_wick_pct", "lower_wick_pct", "gap_pct",
    # Trend
    "ema4_gap", "ema8_gap", "ema13_gap", "ema26_gap", "ema52_gap",
    "ema4_13_spread", "ema13_26_spread", "ema26_52_spread",
    "ema8_slope4", "ema13_slope4", "ema26_slope4",
    # Volatility
    "atr14_pct", "atr26_pct", "rv4", "rv8", "rv13", "rv26",
    "range_ratio13", "range_ratio26",
    # Volume
    "volume_ratio4", "volume_ratio13", "volume_ratio26", "volume_change",
    "volume_z26",
    # Location / intraday structure
    "dist_high13_atr", "dist_low13_atr", "dist_high26_atr", "dist_low26_atr",
    "close_location13", "close_location26", "session_return", "day_gap",
    "session_progress", "session_sin", "session_cos", "bars_to_close",
    # Five-stock market / relative context
    "basket_ret1", "basket_ret2", "basket_ret4", "basket_ret8",
    "breadth_up1", "breadth_up4", "basket_volatility",
    "relative_ret1", "relative_ret4", "relative_ret8",
    "rank_ret1", "rank_ret4", "rank_volume13",
]

SYMBOL_FEATURE_COLUMNS = [f"symbol_{symbol}" for symbol in SYMBOLS]
MODEL_FEATURE_COLUMNS = FEATURE_COLUMNS + SYMBOL_FEATURE_COLUMNS
