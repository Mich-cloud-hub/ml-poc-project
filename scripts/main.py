import sys
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import load_dataset_split

def evaluate_model(model, X_test, y_test, name="Model"):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    print(f"\n--- {name} Results ---")
    print(classification_report(y_test, preds))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}")

def run_experiments():
    X_train, X_test, y_train, y_test = load_dataset_split()
    
    # Baseline
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)
    evaluate_model(log_reg, X_test, y_test, "Logistic Regression (Baseline)")
    
    # Champion actuel : Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    evaluate_model(rf, X_test, y_test, "Random Forest")

if __name__ == "__main__":
    run_experiments()