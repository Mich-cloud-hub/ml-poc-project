import sys
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import load_dataset_split

def run_experiment():
    X_train, X_test, y_train, y_test = load_dataset_split()
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("Baseline: Logistic Regression")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

if __name__ == "__main__":
    run_experiment()