from pathlib import Path

MODEL_NAME = "exp-4-m1"
EXPERIMENT_FAMILY = "exp-4"

TARGET_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMD", "AVGO", "CRM", "ORCL", "ADBE",
    "GOOGL", "META", "NFLX",
    "AMZN", "TSLA", "HD", "MCD",
    "JPM", "BAC", "V", "MA",
    "XOM", "CVX",
    "LLY", "JNJ", "UNH",
    "CAT", "GE", "BA",
    "COST", "WMT", "PG",
]

SECTOR_ETF_BY_SYMBOL = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AMD": "XLK", "AVGO": "XLK", "CRM": "XLK", "ORCL": "XLK", "ADBE": "XLK",
    "GOOGL": "XLC", "META": "XLC", "NFLX": "XLC",
    "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY", "MCD": "XLY",
    "JPM": "XLF", "BAC": "XLF", "V": "XLF", "MA": "XLF",
    "XOM": "XLE", "CVX": "XLE",
    "LLY": "XLV", "JNJ": "XLV", "UNH": "XLV",
    "CAT": "XLI", "GE": "XLI", "BA": "XLI",
    "COST": "XLP", "WMT": "XLP", "PG": "XLP",
}

SECTOR_ETFS = ["XLK", "XLC", "XLY", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLRE", "XLB"]
MARKET_ETFS = ["SPY", "QQQ", "IWM"]
CONTEXT_SYMBOLS = MARKET_ETFS + SECTOR_ETFS
ALL_SYMBOLS = TARGET_SYMBOLS + CONTEXT_SYMBOLS

LEGACY_5M_DIR = Path("data") / "5m"
SOURCE_5M_DIR = Path("data") / "experiments" / "exp-4" / "5m"
OUTPUT_DIR = Path("data") / "experiments" / MODEL_NAME
DATASET_PATH = OUTPUT_DIR / "dataset.parquet"
MODEL_PATH = OUTPUT_DIR / "model.pt"
SCALER_PATH = OUTPUT_DIR / "scaler.joblib"
METADATA_PATH = OUTPUT_DIR / "metadata.json"
SEARCH_RESULTS_PATH = OUTPUT_DIR / "search_results.json"
EXECUTION_PATH = OUTPUT_DIR / "execution.json"
STUDY_PATH = OUTPUT_DIR / "study.sqlite3"
FROZEN_MARKER_PATH = OUTPUT_DIR / "FROZEN_AFTER_2025.txt"

MARKET_TIMEZONE = "America/New_York"
BAR_MINUTES = 30
INPUT_BARS_PER_OUTPUT_BAR = 6
MAIN_HORIZON_BARS = 8  # 4 hours
AUX_2H_BARS = 4
PURGE_GAP_MINUTES = 240

COMMISSION_RATE = 0.001
SLIPPAGE_RATE = 0.0005
STARTING_CASH = 10_000.0
MAX_OPEN_POSITIONS = 3
MAX_POSITION_PERCENT = 0.20

TRAIN_YEARS = [2023, 2024]
CV_VALIDATION_YEAR = 2024
CV_BLOCK_MONTHS = 2
CV_MIN_TRAIN_ROWS = 30_000
TUNE_YEAR = 2025
TEST_YEAR = 2026

SEARCH_SEED = 4401
DEFAULT_TRIALS = 12
MIN_EPOCHS = 25
MAX_EPOCHS = 120
EARLY_STOPPING_PATIENCE = 20
FINAL_MIN_EPOCHS = 40
FINAL_MAX_EPOCHS = 100

WIDTHS = [384, 512, 640]
BLOCKS = [2, 3, 4]
BATCH_SIZES = [512, 1024, 2048]
EXPANSION = 2
HEAD_WEIGHTS = [0.35, 1.0]
RANK_MIN_TARGET_GAP_STD = 0.35
RANK_TEMPERATURE = 0.50

PRIMARY_TAIL_FRACTION = 0.05
SECONDARY_TAIL_FRACTION = 0.02
ECONOMIC_RETURN_SCALE = 0.0040

EXECUTION_PERCENTILES = [0.90, 0.925, 0.95, 0.975, 0.99]
ROLLING_EVENT_WINDOW = 1500
MIN_ROLLING_EVENT_HISTORY = 400
MIN_TRADES_FOR_DEPLOYMENT = 75

TARGET_2H = "target_net_2h"
TARGET_4H = "target_net_4h"
TARGET_COLUMNS = [TARGET_2H, TARGET_4H]

BASE_FEATURE_COLUMNS = [
    "ret_1", "ret_2", "ret_4", "ret_8", "ret_13", "ret_26",
    "range_pct", "body_pct", "upper_wick_pct", "lower_wick_pct", "gap_pct",
    "ema4_gap", "ema8_gap", "ema13_gap", "ema26_gap", "ema52_gap",
    "ema4_13_spread", "ema13_26_spread", "ema26_52_spread",
    "ema8_slope4", "ema13_slope4", "ema26_slope4",
    "atr14_pct", "atr26_pct", "rv4", "rv8", "rv13", "rv26",
    "range_ratio13", "range_ratio26",
    "volume_ratio4", "volume_ratio13", "volume_ratio26", "volume_change", "volume_z26",
    "dist_high13_atr", "dist_low13_atr", "dist_high26_atr", "dist_low26_atr",
    "close_location13", "close_location26", "session_return", "day_gap",
    "session_progress", "session_sin", "session_cos", "bars_to_close",
]

MARKET_CONTEXT_FEATURES = [
    "spy_ret1", "spy_ret4", "spy_ret8", "spy_rv13", "spy_ema13_26",
    "qqq_ret1", "qqq_ret4", "qqq_ret8", "qqq_rv13", "qqq_ema13_26",
    "iwm_ret1", "iwm_ret4", "iwm_ret8", "iwm_rv13", "iwm_ema13_26",
    "sector_ret1", "sector_ret4", "sector_ret8", "sector_rv13", "sector_ema13_26",
    "rel_spy_1", "rel_spy_4", "rel_spy_8",
    "rel_sector_1", "rel_sector_4", "rel_sector_8",
    "universe_breadth1", "universe_breadth4", "universe_dispersion1",
    "universe_mean_ret1", "universe_mean_ret4", "universe_mean_ret8",
    "rank_ret1", "rank_ret4", "rank_ret8", "rank_volume13",
    "regime_spy_above_ema26", "regime_qqq_above_ema26", "regime_spy_vol_high",
]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + MARKET_CONTEXT_FEATURES
SYMBOL_FEATURE_COLUMNS = [f"symbol_{symbol}" for symbol in TARGET_SYMBOLS]
MODEL_FEATURE_COLUMNS = FEATURE_COLUMNS + SYMBOL_FEATURE_COLUMNS
