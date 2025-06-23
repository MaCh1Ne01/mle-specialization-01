from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from package_mles_01.utils.helpers import *

import joblib

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "train_test_data.joblib",
    models_path: Path = MODELS_DIR / "models.joblib",
    base_model_predictions_path: Path = PROCESSED_DATA_DIR / "base_model_predictions_path.csv",
    model_01_predictions_path: Path = PROCESSED_DATA_DIR / "model_01_predictions_path.csv",
    model_02_predictions_path: Path = PROCESSED_DATA_DIR / "model_02_predictions_path.csv"
):

    # ---- Loading test datasets----
    logger.info("Loading test datasets...")
    processed_data = joblib.load(features_path)
    X_test = processed_data["X_test"]
    y_test = processed_data["y_test"]
    logger.success("Loading test datasets complete.")
    # -----------------------------------------

    # ---- Loading fitted models ----
    logger.info("Loading fitted models...")
    fitted_models = joblib.load(models_path)
    base_model = fitted_models[BASE_MODEL_NAME]
    model_01 = fitted_models[MODEL_01_NAME]
    model_02 = fitted_models[MODEL_02_NAME]
    logger.success("Loading fitted models complete.")
    # -----------------------------------------

    # ---- Evaluating testing data ----
    logger.info("Evaluating testing data...")
    base_model_pred = evaluating_model(model=base_model, model_name=BASE_MODEL_NAME, X=X_test, y=y_test, label_data=TESTING_DATA_LABEL)
    model_01_pred = evaluating_model(model=model_01, model_name=MODEL_01_NAME, X=X_test, y=y_test, label_data=TESTING_DATA_LABEL)
    model_02_pred = evaluating_model(model=model_02, model_name=MODEL_02_NAME, X=X_test, y=y_test, label_data=TESTING_DATA_LABEL)
    logger.success("Evaluating testing data complete.")
    # -----------------------------------------

    # ---- Saving predictions ----
    logger.info("Saving predictions...")
    base_model_pred.to_csv(base_model_predictions_path, index=False)
    model_01_pred.to_csv(model_01_predictions_path, index=False)
    model_02_pred.to_csv(model_02_predictions_path, index=False)
    logger.success("Saving predictions complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
