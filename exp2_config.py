from pathlib import Path

from features import FEATURE_COLUMNS


MODEL_NAME = "exp-2-m1"
EXPERIMENT_FAMILY = "exp-2"

BASE_DATASET_PATH = Path("data") / "ml" / "features.parquet"
OUTPUT_DIR = Path("data") / "experiments" / MODEL_NAME
EVENT_DATASET_PATH = OUTPUT_DIR / "events.parquet"
MODEL_PATH = OUTPUT_DIR / "model.joblib"
METADATA_PATH = OUTPUT_DIR / "metadata.json"
EXECUTION_PATH = OUTPUT_DIR / "execution.json"

TRAIN_YEARS = [2023, 2024]
META_OOF_YEAR = 2024
TUNE_YEAR = 2025
TEST_YEAR = 2026

# -----------------------------------------------------------------------------
# Event sampling
# -----------------------------------------------------------------------------
# Calendar-bar training massively duplicates overlapping outcomes. EXP-2 samples
# information events with a symmetric CUSUM filter. The threshold scales with
# trailing 5-minute realized volatility and uses only information available at t.
CUSUM_VOL_MULTIPLIER = 1.50
CUSUM_MIN_THRESHOLD = 0.0005

# -----------------------------------------------------------------------------
# Path-dependent trade / label definition
# -----------------------------------------------------------------------------
# Signal at t -> enter at the next 5-minute bar open. Horizontal barriers scale
# with ATR known at signal time and have floors/caps so transaction costs never
# dominate the reward target in very quiet regimes.
HORIZON_BARS = 24
HORIZON_MINUTES = 120

PROFIT_ATR_MULTIPLIER = 3.0
STOP_ATR_MULTIPLIER = 1.5
PROFIT_FLOOR = 0.0060
PROFIT_CAP = 0.0180
STOP_FLOOR = 0.0030
STOP_CAP = 0.0090

# Meta-label: require a real margin after modeled round-trip costs, not merely a
# one-tick positive P&L.
ECONOMIC_NET_THRESHOLD = 0.0015

# -----------------------------------------------------------------------------
# Validation / stacking
# -----------------------------------------------------------------------------
PURGE_GAP_MINUTES = HORIZON_MINUTES
OOF_MIN_TRAIN_ROWS = 5000

# -----------------------------------------------------------------------------
# Execution selection
# -----------------------------------------------------------------------------
# Only ONE axis is tuned on 2025: the causal rolling percentile of the final
# meta-score. This is intentionally tiny compared with old threshold grids.
EXECUTION_PERCENTILES = [0.90, 0.925, 0.95, 0.975, 0.99]
ROLLING_EVENT_WINDOW = 500
MIN_ROLLING_EVENT_HISTORY = 150
MIN_TRADES_FOR_DEPLOYMENT = 75

# At a timestamp, correlated simultaneous signals compete for a single new slot.
MAX_NEW_ENTRIES_PER_TIMESTAMP = 1

# -----------------------------------------------------------------------------
# Feature set
# -----------------------------------------------------------------------------
EXP2_DERIVED_FEATURE_COLUMNS = [
    "ret_1_vol_scaled",
    "ret_3_vol_scaled",
    "ret_6_vol_scaled",
    "ret_12_vol_scaled",
    "vol_ratio_12_24",
    "trend_strength_atr",
    "ma_stack_strength_atr",
    "volume_ratio_spread",
    "breakout_pressure",
    "session_progress",
    "market_ret_1_mean",
    "market_ret_6_mean",
    "market_ret_12_mean",
    "market_ret_12_dispersion",
    "relative_ret_1",
    "relative_ret_6",
    "relative_ret_12",
    "ret_12_cross_rank",
    "trend_cross_rank",
    "volume_cross_rank",
    "atr_cross_rank",
    "breadth_ret_1_positive",
    "breadth_above_ma50",
    "breadth_above_ma200",
]

EXP2_FEATURE_COLUMNS = list(FEATURE_COLUMNS) + EXP2_DERIVED_FEATURE_COLUMNS

# Stable context features exposed to the meta-model. Base learners see the full
# feature set; keeping the meta context compact reduces stacking overfit.
META_CONTEXT_COLUMNS = [
    "atr14_pct",
    "rolling_vol_24",
    "vol_ratio_12_24",
    "volume_ratio_20",
    "ma50_gap_pct",
    "dist_high50_atr",
    "ret_12_vol_scaled",
    "market_ret_12_mean",
    "market_ret_12_dispersion",
    "relative_ret_12",
    "breadth_above_ma50",
    "session_progress",
]

EVENT_TARGET_COLUMNS = [
    "barrier_success",
    "economic_success",
    "net_positive",
    "event_gross_return",
    "event_net_return",
]
