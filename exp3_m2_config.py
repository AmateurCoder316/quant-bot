from pathlib import Path

from exp3_config import (
    FEATURE_COLUMNS,
    MARKET_TIMEZONE,
    MODEL_FEATURE_COLUMNS,
    SYMBOLS,
    TARGET_1H,
    TARGET_2H,
    TARGET_4H,
)


MODEL_NAME = "exp-3-m2"
EXPERIMENT_FAMILY = "exp-3"

# EXP-3-M2 deliberately reuses the already-built 30-minute dataset. The data
# definition stays fixed; only the neural-network/training core changes.
DATASET_PATH = Path("data") / "experiments" / "exp-3-m1" / "dataset.parquet"
OUTPUT_DIR = Path("data") / "experiments" / MODEL_NAME
MODEL_PATH = OUTPUT_DIR / "model.pt"
SCALER_PATH = OUTPUT_DIR / "scaler.joblib"
METADATA_PATH = OUTPUT_DIR / "metadata.json"
SEARCH_RESULTS_PATH = OUTPUT_DIR / "search_results.json"
EXECUTION_PATH = OUTPUT_DIR / "execution.json"
STUDY_PATH = OUTPUT_DIR / "study.sqlite3"
FROZEN_MARKER_PATH = OUTPUT_DIR / "FROZEN_AFTER_2025.txt"

TRAIN_YEARS = [2023, 2024]
CV_VALIDATION_YEAR = 2024
CV_BLOCK_MONTHS = 2
CV_MIN_TRAIN_ROWS = 4_000
TUNE_YEAR = 2025
TEST_YEAR = 2026
PURGE_GAP_MINUTES = 240

STARTING_CASH = 10_000.0
COMMISSION_RATE = 0.001
SLIPPAGE_RATE = 0.0005
MAX_OPEN_POSITIONS = 3
MAX_POSITION_PERCENT = 0.20

TARGET_COLUMNS = [TARGET_1H, TARGET_2H, TARGET_4H]

# M2 is intentionally much larger than M1. Width/block combinations range from
# roughly one million to several million parameters. Residual blocks use a 2x
# internal expansion, LayerNorm, GELU and dropout.
WIDTHS = [384, 512, 640]
RESIDUAL_BLOCKS = [2, 3, 4]
EXPANSION = 2
DROPOUT_RANGE = (0.08, 0.32)
BATCH_SIZES = [256, 512, 1024]

SEARCH_SEED = 3302
DEFAULT_TRIALS = 18
MIN_EPOCHS = 30
MAX_EPOCHS = 180
EARLY_STOPPING_PATIENCE = 30
FINAL_MIN_EPOCHS = 50
FINAL_MAX_EPOCHS = 150

# The 2h head remains primary. 1h/4h are auxiliary representation-learning
# targets. Ranking loss is applied directly to the 2h head so higher scores are
# explicitly trained to correspond to higher realized returns.
HEAD_WEIGHTS = [0.20, 1.00, 0.20]
RANK_WEIGHT_RANGE = (0.15, 0.45)
RANK_TEMPERATURE = 0.75
RANK_MIN_TARGET_GAP_STD = 0.10

PRIMARY_TAIL_FRACTION = 0.10
SECONDARY_TAIL_FRACTION = 0.05
ECONOMIC_RETURN_SCALE = 0.0030

# M2 execution tuning uses the causal rank of each prediction instead of an
# absolute return cutoff. This avoids calibration collapse while still asking a
# strict question: are the model's highest-ranked opportunities profitable?
EXECUTION_PERCENTILES = [0.90, 0.925, 0.95, 0.975, 0.99]
ROLLING_SCORE_WINDOW = 1_000
MIN_SCORE_HISTORY = 250
MIN_TRADES_FOR_DEPLOYMENT = 40
