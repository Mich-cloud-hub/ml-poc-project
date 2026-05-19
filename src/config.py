import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"

for directory in [MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

DROP_COLS = [
    "gameId", "redFirstBlood", "redKills", "redDeaths", 
    "redGoldDiff", "redExperienceDiff", "redTotalGold", 
    "redTotalExperience", "redGoldPerMin", "redCSPerMin"
]

MODELS = {
    "logistic_regression": {
        "name": "Logistic Regression Baseline",
        "description": "Standardized features passed into a logistic regression model.",
        "path": MODELS_DIR / "log_reg.pkl",
    },
    "xgboost": {
        "name": "XGBoost Classifier",
        "description": "Tree ensemble optimized via RandomizedSearchCV.",
        "path": MODELS_DIR / "xgboost.pkl",
    }
}
