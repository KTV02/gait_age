#!/usr/bin/env python3
"""Train four modality-specific age models and a boosting ensemble.

The script trains one MoviNet regressor for each visual modality
(silhouette, semantic segmentation and GMFlow optical flow) and a
RandomForest regressor from gait parameters estimated with AlphaPose.
Predictions from the validation split are then used to fit a
``GradientBoostingRegressor`` that combines the four modalities.

Only the dataset root path is required. ``participants_measures.csv``
and ``gait_parameters.csv`` are assumed to live inside that folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import tensorflow as tf
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from code.train import train_MoviNet_regression
from code.train import train_random_forest_regression as rf


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def train_movinet_modality(dataset: Path, modality: str, partitions: Path, save_dir: Path) -> Path:
    """Train a MoviNet regression model for the given modality."""
    args = SimpleNamespace(
        device=0,
        epochs=20,
        img_size=224,
        model_id="a0",
        save_dir=str(save_dir),
        data_path=str(dataset),
        data_type=modality,
        num_frames=20,
        batch_size=32,
        units=512,
        id_partition=0,
        learning_rate=0.001,
        patients_info=str(dataset / "patients_measures.csv"),
        partitions_file=str(partitions),
        data_class="both",
        optical_flow_method="GMFLOW" if modality == "optical_flow" else None,
        id_experiment=f"{modality}_age",
        target="Age",
        wandb_project="age_regression",
        experiment_name=f"{modality}_age",
        seed=27,
    )
    train_MoviNet_regression.main(args)
    return save_dir / args.experiment_name / "Partition 0" / "model"


def movinet_predictions(
    model_path: Path,
    dataset: Path,
    modality: str,
    partitions: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return validation and test predictions and targets for a modality."""
    data = train_MoviNet_regression.get_data(
        str(partitions),
        str(dataset / modality),
        modality,
        "both",
        "GMFLOW" if modality == "optical_flow" else None,
        str(dataset / "patients_measures.csv"),
        "Age",
    )
    min_train = np.min(data["train"]["target"])
    max_train = np.max(data["train"]["target"])
    val_targets = data["validation"]["target"].astype(float)
    test_targets = data["test"]["target"].astype(float)
    norm_val = (val_targets - min_train) / (max_train - min_train)
    norm_test = (test_targets - min_train) / (max_train - min_train)
    output_sig = (
        tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
    )
    val_ds = (
        tf.data.Dataset.from_generator(
            train_MoviNet_regression.FrameGenerator(
                data["validation"]["path"], norm_val, 20, (224, 224)
            ),
            output_signature=output_sig,
        ).batch(32)
    )
    test_ds = (
        tf.data.Dataset.from_generator(
            train_MoviNet_regression.FrameGenerator(
                data["test"]["path"], norm_test, 20, (224, 224)
            ),
            output_signature=output_sig,
        ).batch(32)
    )
    model = tf.keras.models.load_model(model_path)
    val_pred = model.predict(val_ds).ravel() * (max_train - min_train) + min_train
    test_pred = model.predict(test_ds).ravel() * (max_train - min_train) + min_train
    return val_pred, val_targets, test_pred, test_targets


def train_rf_modality(dataset: Path, partitions: Path) -> tuple[Path, dict]:
    """Train a RandomForest regressor on gait parameters."""
    gait_csv = dataset / "gait_parameters.csv"
    measures_csv = dataset / "patients_measures.csv"
    data = rf.get_data(
        str(gait_csv),
        str(measures_csv),
        str(partitions),
        ["Stride_UGS", "Cadence_UGS"],
        "Age",
        task="age",
    )
    data, target_scaler, imp, scaler = rf.preprocess_data(data, task="age", scale_target=True)
    X_train = np.concatenate([data["train"]["data"], data["val"]["data"]])
    y_train = np.concatenate([data["train"]["target"], data["val"]["target"]])
    model = RandomForestRegressor(n_estimators=100, random_state=27)
    model.fit(X_train, y_train)
    model_dir = dataset / "rf_model.joblib"
    joblib.dump({"model": model, "imputer": imp, "scaler": scaler, "target_scaler": target_scaler}, model_dir)
    return model_dir, data


def rf_predictions(model_path: Path, data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return validation and test predictions from a trained RF model."""
    pack = joblib.load(model_path)
    model = pack["model"]
    imp = pack["imputer"]
    scaler = pack["scaler"]
    target_scaler = pack["target_scaler"]

    def predict(split: str) -> np.ndarray:
        X = imp.transform(data[split]["data"])
        X = scaler.transform(X)
        y = model.predict(X).reshape(-1, 1)
        return target_scaler.inverse_transform(y).ravel()

    val_pred = predict("val")
    val_targets = target_scaler.inverse_transform(data["val"]["target"].reshape(-1, 1)).ravel()
    test_pred = predict("test")
    test_targets = target_scaler.inverse_transform(data["test"]["target"].reshape(-1, 1)).ravel()
    return val_pred, val_targets, test_pred, test_targets


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-modal age ensemble")
    parser.add_argument("dataset", type=Path, help="root folder of the dataset")
    parser.add_argument(
        "--partitions",
        type=Path,
        default=Path("partitions/Age/partition_0.json"),
        help="JSON file with train/val/test splits",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=Path("results"),
        help="directory to store intermediate models",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    save_dir = args.save_dir.resolve()
    partitions = args.partitions.resolve()

    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Train individual models
    sil_model = train_movinet_modality(dataset, "silhouette", partitions, save_dir)
    seg_model = train_movinet_modality(dataset, "semantic_segmentation", partitions, save_dir)
    flow_model = train_movinet_modality(dataset, "optical_flow", partitions, save_dir)
    rf_model, rf_data = train_rf_modality(dataset, partitions)

    # 2. Collect predictions on validation/test splits
    sil_val, y_val, sil_test, y_test = movinet_predictions(sil_model, dataset, "silhouette", partitions)
    seg_val, _, seg_test, _ = movinet_predictions(seg_model, dataset, "semantic_segmentation", partitions)
    flow_val, _, flow_test, _ = movinet_predictions(flow_model, dataset, "optical_flow", partitions)
    rf_val, _, rf_test, _ = rf_predictions(rf_model, rf_data)

    X_val = np.column_stack([sil_val, seg_val, flow_val, rf_val])
    X_test = np.column_stack([sil_test, seg_test, flow_test, rf_test])

    # 3. Train boosting ensemble
    booster = GradientBoostingRegressor(random_state=27)
    booster.fit(X_val, y_val)
    final_pred = booster.predict(X_test)

    mse = mean_squared_error(y_test, final_pred)
    mae = mean_absolute_error(y_test, final_pred)
    print(f"Ensemble MSE: {mse:.2f}\nEnsemble MAE: {mae:.2f}")

    joblib.dump(booster, save_dir / "age_booster.joblib")


if __name__ == "__main__":
    main()
