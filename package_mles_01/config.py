from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Constants
TARGET = "price"
TEST_PERCENTAGE = 0.2
SEED = 0
DROP_FEATURES = ["Unnamed: 0","flight"]
STRATIFY_FEATURE = "class"
NOMINAL_FEATURES = ["airline","source_city","destination_city"]
ORDINAL_FEATURES = ["departure_time","stops","arrival_time","class"]
NUMERICAL_FEATURES = ["duration","days_left"]
CUSTOM_ORDER = {
    "departure_time": ["Early_Morning","Morning","Afternoon","Evening","Night","Late_Night"],
    "stops": ["zero","one","two_or_more"],
    "arrival_time": ["Early_Morning","Morning","Afternoon","Evening","Night","Late_Night"],
    "class": ["Economy","Business"]
}

BASE_MODEL_NAME = "Linear Regression"
MODEL_01_NAME = "Decission Tree Regression"
MODEL_02_NAME = "K-Neighbors Regression"
TRAINING_DATA_LABEL = "Training Data"
TESTING_DATA_LABEL = "Testing Data"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass