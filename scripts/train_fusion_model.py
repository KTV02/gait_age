#uses prediction file from individual weak learners to train fusion model 

#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score

def load_predictions(prediction_file):
    if not os.path.exists(prediction_file):
        print(f"⚠️  Missing prediction file: {prediction_file}")
        return {}
    df = pd.read_csv(prediction_file)
    return dict(zip(df["ID"], df["Predicted_Age"]))

def load_ground_truth(patients_csv, target_column):
    df = pd.read_csv(patients_csv)
    return dict(zip(df["ID"], df[target_column]))

def main():
    parser = argparse.ArgumentParser(description="Train boosted fusion model from modality predictions")
    parser.add_argument("--results_dir", required=True, help="Root folder with per-modality prediction CSVs")
    parser.add_argument("--patients_csv", required=True, help="CSV with ground-truth ages")
    parser.add_argument("--target_column", default="Age", help="Name of the target column in the patients CSV")
    parser.add_argument("--output_model", default="fusion_model.joblib", help="Path to save the trained model")
    parser.add_argument("--use_alphapose", action="store_true", help="Include AlphaPose predictions if available")
    args = parser.parse_args()

    modalities = ["silhouette", "semantic_segmentation", "optical_flow"]
    if args.use_alphapose:
        modalities.append("alphapose")

    # Load predictions per modality
    modality_preds = []
    modality_names = []
    for m in modalities:
        pred_path = os.path.join(args.results_dir, m, "predictions.csv")
        preds = load_predictions(pred_path)
        if preds:
            modality_preds.append(preds)
            modality_names.append(m)
        else:
            print(f"⚠️ Skipping modality {m} (missing predictions)")

    if not modality_preds:
        raise ValueError("No valid predictions found")

    # Load ground truth
    y_true = load_ground_truth(args.patients_csv, args.target_column)

    # Merge into training matrix
    all_ids = list(set.intersection(*[set(p.keys()) for p in modality_preds], set(y_true.keys())))
    X = []
    Y = []
    for pid in sorted(all_ids):
        X.append([preds[pid] for preds in modality_preds])
        Y.append(y_true[pid])
    X = np.array(X)
    Y = np.array(Y)

    print(f"✅ Training fusion model on {len(X)} samples with features: {modality_names}")

    # Train fusion model (e.g. Gradient Boosting)
    model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X, Y)

    # Evaluate
    y_pred = model.predict(X)
    print(f"📊 MAE: {mean_absolute_error(Y, y_pred):.2f}")
    print(f"📊 MSE: {mean_squared_error(Y, y_pred):.2f}")
    cv_scores = cross_val_score(model, X, Y, cv=5, scoring="neg_mean_absolute_error")
    print(f"📉 Cross-validated MAE: {-np.mean(cv_scores):.2f}")

    # Save
    os.makedirs(args.results_dir, exist_ok=True)
    model_path = os.path.join(args.results_dir, args.output_model)
    joblib.dump(model, model_path)
    print(f"✅ Saved fusion model to {model_path}")

if __name__ == "__main__":
    main()