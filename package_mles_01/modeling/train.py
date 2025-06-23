from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from package_mles_01.config import MODELS_DIR, PROCESSED_DATA_DIR, SEED, BASE_MODEL_NAME, MODEL_01_NAME, MODEL_02_NAME

import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "train_test_data.joblib",
    models_path: Path = MODELS_DIR / "models.joblib"
):
    
    # ---- Loading processed train datasets ----
    logger.info("Loading processed datasets...")
    processed_data = joblib.load(features_path)
    X_train = processed_data["X_train"]
    y_train = processed_data["y_train"]
    logger.success("Loading processed train datasets complete.")
    # -----------------------------------------

    # ---- Models ----
    logger.info("Initializing models...")
    base_model = LinearRegression()
    model_01 = DecisionTreeRegressor(random_state=SEED)
    model_02 = KNeighborsRegressor()
    logger.success("Initializing models complete.")
    # -----------------------------------------

    # ---- Fitting ----
    logger.info("Fitting models...")
    base_model.fit(X_train, y_train)
    model_01.fit(X_train, y_train)
    model_02.fit(X_train, y_train)
    logger.success("Fitting models complete.")
    # -----------------------------------------

    # ---- Saving models ----
    logger.info("Saving fitted models...")
    fitted_models = {
        BASE_MODEL_NAME: base_model,
        MODEL_01_NAME: model_01,
        MODEL_02_NAME: model_02
    }
    joblib.dump(fitted_models, models_path)
    logger.success("Saving fitted models complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
