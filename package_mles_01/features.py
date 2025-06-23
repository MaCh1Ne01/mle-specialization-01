from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from package_mles_01.utils.helpers import *

import joblib

app = typer.Typer()


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "splitted_data.joblib",
    output_path: Path = PROCESSED_DATA_DIR / "train_test_data.joblib"
):
    
    # ---- Feature Engineering ----

    # ---- Loading splitted datasets ----
    logger.info("Loading datasets...")
    splitted_data = joblib.load(input_path)
    X_train = splitted_data["X_train"]
    X_test= splitted_data["X_test"]
    y_train = splitted_data["y_train"]
    y_test = splitted_data["y_test"]
    logger.success("Loading datasets complete.")
    # -----------------------------------------

    # ---- Nominal Encoding ----
    logger.info("Nominal Encoding...")
    X_train, X_test = encoding_nominal_features(features_train=X_train, features_test=X_test, nominal_features=NOMINAL_FEATURES)
    logger.success("Nominal Encoding complete.")
    # -----------------------------------------

    # ---- Ordinal Encoding ----
    logger.info("Ordinal Encoding...")
    categories_list = [CUSTOM_ORDER[col] for col in ORDINAL_FEATURES]
    X_train, X_test = encoding_ordinal_features(features_train=X_train, features_test=X_test, ordinal_features=ORDINAL_FEATURES, categories_list=categories_list)
    logger.success("Ordinal Encoding complete.")
    # -----------------------------------------

    # ---- Scaling ----
    logger.info("Scaling...")
    X_train, X_test = scaling_numerical_features(features_train=X_train, features_test=X_test, features_to_scale=NUMERICAL_FEATURES+ORDINAL_FEATURES)
    logger.success("Scaling complete.")
    # -----------------------------------------

    # ---- Saving processed datasets ----
    logger.info("Saving artifacts...")
    processed_data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }
    joblib.dump(processed_data, output_path)
    logger.success("Saving artifacts complete.")
    # -----------------------------------------
    

if __name__ == "__main__":
    app()
