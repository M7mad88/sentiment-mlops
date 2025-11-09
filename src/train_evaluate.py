# src/train_evaluate.py
import os
import json
import yaml
import pickle
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

def main():
    # Load params
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    
    # Load data
    train_data = pd.read_csv("data/features/train_bow.csv")
    test_data = pd.read_csv("data/features/test_bow.csv")
    
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values
    
    # Train model
    clf = GradientBoostingClassifier(
        n_estimators=params["model"]["n_estimators"],
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_pred_proba)
    }
    
    # Save model and metrics
    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(clf, f)
    
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    print("✅ Training and evaluation completed!")
    print(f"Metrics: {metrics}")

if __name__ == "__main__":
    main()