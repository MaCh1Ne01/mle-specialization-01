from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import pandas as pd
import numpy as np
import joblib
from package_mles_01.utils.helpers import *

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "flights_dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "splitted_data.joblib"
):
    
    # ---- Loading and Cleaning ----
    logger.info("Loading dataset...")
    df_flights_raw = pd.read_csv(input_path)
    logger.success("Loading dataset complete.")

    logger.info("Cleaning dataset...")
    df_flights = df_flights_raw.copy()
    df_flights.drop(DROP_FEATURES, axis=1, inplace=True)
    logger.success("Cleaning dataset complete.")
    # -----------------------------------------

    # ---- Data Splitting ----
    logger.info("Splitting dataset...")
    X_train, X_test, y_train, y_test = split_dataset(target_feature=TARGET, dataframe=df_flights,
                                                    test_percentage=TEST_PERCENTAGE, seed=SEED, stratify_feature=STRATIFY_FEATURE)
    logger.success("Splitting dataset complete.")
    # -----------------------------------------

    # ---- Saving datasets ----
    logger.info("Splitting dataset...")
    splitted_data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }
    joblib.dump(splitted_data, output_path)
    logger.success("Saving datasets complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
