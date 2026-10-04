from pathlib import Path

from exp2_config import EXP2_FEATURE_COLUMNS


MODEL_NAME = "exp-2-m2"
EXPERIMENT_FAMILY = "exp-2"

SOURCE_EVENT_DATASET_PATH = (
    Path("data") / "experiments" / "exp-2-m1" / "events.parquet"
)
OUTPUT_DIR = Path("data") / "experiments" / MODEL_NAME
MODEL_PATH = OUTPUT_DIR / "model.joblib"
METADATA_PATH = OUTPUT_DIR / "metadata.json"
EXECUTION_PATH = OUTPUT_DIR / "execution.json"
SEARCH_RESULTS_PATH = OUTPUT_DIR / "search_results.json"
STUDY_DIR = OUTPUT_DIR / "studies"

TARGET_COLUMN = "economic_success"
RETURN_COLUMN = "event_net_return"
TRAIN_YEARS = [2023, 2024]
TUNE_YEAR = 2025
TEST_YEAR = 2026

# Labels can remain active for up to 120 minutes. Every CV training fold is
# purged by this amount before the validation block and also requires exit_time
# to resolve before validation begins.
PURGE_GAP_MINUTES = 120
CV_VALIDATION_YEAR = 2024
CV_BLOCK_MONTHS = 2
CV_MIN_TRAIN_ROWS = 10_000

# Hyperparameter-search budget. This is intentionally substantial: each family
# gets its own TPE study so one model family cannot monopolize a mixed search
# space. The environment variable EXP2_M2_TRIALS can override this locally.
DEFAULT_TRIALS_PER_FAMILY = 30
SEARCH_SEED = 2201

MODEL_FAMILIES = [
    "hist_gradient_boosting",
    "extra_trees",
    "logistic",
    "xgboost",
    "lightgbm",
    "catboost",
]

# One winner per model family is compared. The strongest distinct families are
# then blended from strictly OOF predictions rather than in-sample predictions.
MAX_FINALIST_FAMILIES = 4
MIN_FINALIST_FAMILIES = 2

# Training objective diagnostics.
PRIMARY_TAIL_FRACTION = 0.10
SECONDARY_TAIL_FRACTION = 0.05
ECONOMIC_SCALE_RETURN = 0.0030

# Execution selection remains deliberately tiny and independent from the model
# search. 2025 chooses only one causal percentile after the entire model is frozen.
EXECUTION_PERCENTILES = [0.90, 0.925, 0.95, 0.975, 0.99]
ROLLING_EVENT_WINDOW = 500
MIN_ROLLING_EVENT_HISTORY = 150
MIN_TRADES_FOR_DEPLOYMENT = 75

FEATURE_COLUMNS = list(EXP2_FEATURE_COLUMNS)
