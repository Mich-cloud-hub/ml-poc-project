from __future__ import annotations
import pandas as pd
from typing import Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    df = pd.read_csv('high_diamond_ranked_10min.csv')
    
    df = df.drop(columns=['gameId'])
    
    redundant_red_stats = [col for col in df.columns if 'red' in col and ('Diff' in col or 'Experience' in col or 'Gold' in col)]
    df = df.drop(columns=redundant_red_stats)

    X = df.drop(columns=['blueWins'])
    y = df['blueWins']

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test
    