import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import subprocess
import pandas as pd
import joblib

from src.config import MODELS, RESULTS_DIR
from src.data import load_dataset_split
from src.metrics import compute_metrics


def run_evaluation_pipeline() -> None:
    _, scaled_features_test, _, targets_test = load_dataset_split()
    
    evaluation_results = []
    
    for model_id, model_info in MODELS.items():
        model_path = model_info["path"]
        
        if not model_path.exists():
            print(f"Model file missing for {model_id}: {model_path}")
            continue
            
        model = joblib.load(model_path)
        predictions = model.predict(scaled_features_test)
        
        metrics = compute_metrics(targets_test, predictions)
        metrics["model_name"] = model_info["name"]
        metrics["model_id"] = model_id
        
        print(f"--- Metrics for {model_info['name']} ---")
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                print(f"{metric_name}: {metric_value:.4f}")
            else:
                print(f"{metric_name}: {metric_value}")
        print()
        
        evaluation_results.append(metrics)
        
    if evaluation_results:
        results_df = pd.DataFrame(evaluation_results)
        cols = ["model_id", "model_name"] + [c for c in results_df.columns if c not in ["model_id", "model_name"]]
        results_df = results_df[cols]
        results_df.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)
        print(f"Results saved to {RESULTS_DIR / 'model_metrics.csv'}")

    print("Launching Streamlit App...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/app.py"])


if __name__ == "__main__":
    run_evaluation_pipeline()