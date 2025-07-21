#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import pandas as pd
from types import SimpleNamespace
from scripts import train_MovieNet_regression

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

import json

def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal MoviNet + AlphaPose training with boosting fusion")
    parser.add_argument("--dataset", required=True, help="root folder containing modalities and AlphaPose CSVs")
    parser.add_argument("--patients_info", required=True, help="CSV with participant metadata")
    parser.add_argument("--partitions", default="partitions/Age/partition_0.json", help="JSON with splits")
    parser.add_argument("--save_dir", default="results", help="output directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_frames", type=int, default=20)
    parser.add_argument("--model_id", default="a0")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--optical_flow_method", type=str, help="Only used if modality is optical_flow")
    return parser.parse_args()


def run_movinet_for_modality(modality, args):
    optical_flow_method = args.optical_flow_method if modality == "optical_flow" else None
    movinet_args = SimpleNamespace(
        device=args.gpu,
        epochs=args.epochs,
        img_size=224,
        model_id=args.model_id,
        save_dir=os.path.join(args.save_dir, modality),
        data_path=args.dataset,
        data_type=modality,
        num_frames=args.num_frames,
        batch_size=args.batch_size,
        units=512,
        id_partition=0,
        learning_rate=0.001,
        patients_info=args.patients_info,
        partitions_file=args.partitions,
        data_class="both",
        optical_flow_method=optical_flow_method,
        id_experiment=f"{modality}_regression",
        target="Age",
        wandb_project="age_regression",
        experiment_name=f"movinet_{modality}",
        seed=27,
    )
    y_preds, y_true = train_MovieNet_regression.main(movinet_args)
    return np.array(y_preds).reshape(-1, 1), np.array(y_true)

def load_alphapose_features(dataset_path, partitions_file):
    with open(partitions_file) as f:
        partitions = json.load(f)

    train_ids = partitions["train"]
    test_ids = partitions["test"]

    def load_features(ids):
        features, labels = [], []
        for pid in ids:
            json_path = os.path.join(dataset_path, "alphapose_csv", f"{pid}.csv")
            if not os.path.exists(json_path):
                continue

            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️  Skipping malformed file: {json_path}")
                continue

            joints_all = []
            for frame in data:
                joints = frame.get("joints")
                if not joints:
                    continue
                coords = [(kp["x"], kp["y"]) for kp in joints.values() if kp]
                if coords:
                    joints_all.append([v for xy in coords for v in xy])  # flatten x, y

            if not joints_all:
                print(f"⚠️  No valid joints in: {json_path}")
                continue

            feat = np.mean(joints_all, axis=0)
            features.append(feat)

            try:
                age = int(pid.split("_")[0])  # assumes age is encoded in ID
            except ValueError:
                print(f"⚠️  Could not parse age from {pid}")
                continue
            labels.append(age)

        return np.array(features), np.array(labels)

    return load_features(train_ids), load_features(test_ids)


def main():
    args = parse_args()
    modality_preds = []
    ground_truth = None
    for modality in ["silhouette", "semantic_segmentation","optical_flow"]:
        modality_path = os.path.join(args.dataset, modality)
        if os.path.isdir(modality_path):
            print(f"Training MoviNet on: {modality}")
            if modality == 'optical_flow' and args.optical_flow_method is None:
                args.optical_flow_method = 'GMFLOW'
            preds, y_true = run_movinet_for_modality(modality, args)
            modality_preds.append(preds)
            if ground_truth is None:
                ground_truth = y_true
        else:
            print(f"Skipping modality {modality}: folder not found at {modality_path}")

    if os.path.isdir(os.path.join(args.dataset, "alphapose_csv")):
        print("Training RandomForest on AlphaPose CSVs...")
        (X_train, y_train), (X_test, y_test) = load_alphapose_features(args.dataset, args.partitions)
        rf = RandomForestRegressor(n_estimators=100)
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test).reshape(-1, 1)
        modality_preds.append(rf_preds)
        if ground_truth is None:
            ground_truth = y_test

    X_fusion = np.hstack(modality_preds)
    y_fusion = ground_truth[:len(X_fusion)]

    print("Training final Gradient Boosting Regressor...")
    X_train, X_val, y_train, y_val = train_test_split(X_fusion, y_fusion, test_size=0.2, random_state=42)
    boost = GradientBoostingRegressor()
    boost.fit(X_train, y_train)
    y_pred = boost.predict(X_val)

    print(f"Fusion MAE: {mean_absolute_error(y_val, y_pred):.3f}")
    os.makedirs(args.save_dir, exist_ok=True)
    joblib.dump(boost, os.path.join(args.save_dir, "fusion_model.joblib"))


if __name__ == "__main__":
    main()
