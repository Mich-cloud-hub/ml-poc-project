import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

from src.config import MODELS_DIR, PLOTS_DIR
from src.data import load_dataset_split

def train_and_save_models():
    scaled_features_train, scaled_features_test, targets_train, targets_test = load_dataset_split()
    
    logistic_model = LogisticRegression(random_state=42, max_iter=1000)
    logistic_model.fit(scaled_features_train, targets_train)
    joblib.dump(logistic_model, MODELS_DIR / "log_reg.pkl")

    xgb_base = XGBClassifier(eval_metric="logloss", random_state=42)
    param_distributions = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 1.0]
    }
    xgb_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_distributions,
        n_iter=5,
        scoring="roc_auc",
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    xgb_search.fit(scaled_features_train, targets_train)
    best_xgb_model = xgb_search.best_estimator_
    joblib.dump(best_xgb_model, MODELS_DIR / "xgboost.pkl")

    explainer = shap.TreeExplainer(best_xgb_model)
    shap_values = explainer(scaled_features_test)

    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    plt.savefig(PLOTS_DIR / "shap_beeswarm.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    plt.savefig(PLOTS_DIR / "shap_waterfall.png", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    train_and_save_models()
