#!/usr/bin/env python3
"""Run inference for age and gender from gait features.

The script relies on the repository's ``calculate_parameters`` function to
extract gait features from AlphaPose, sensor and segmentation data. Age and
gender models are loaded from ``joblib`` files. These files may contain any
scikit-learn estimator (RandomForest, Lasso, etc.) or even a Keras model
together with the fitted ``imputer``, ``scaler`` and optionally a
``target_scaler`` for age regression.
"""

import argparse
import joblib
from pathlib import Path
import sys
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from gait_estimation.gait_parameters_estimation import calculate_parameters


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict age and gender from pre-extracted gait features"
    )
    parser.add_argument(
        "--pose_path",
        help="directory containing AlphaPose JSON files",
    )
    parser.add_argument(
        "--sensors",
        help="directory with OptoGait sensor bounding boxes",
    )
    parser.add_argument(
        "--segmentation_path",
        help="directory containing DensePose segmentations",
    )
    parser.add_argument(
        "--video",
        help="path to a dataset video. If given, pose, sensor and segmentation paths are inferred",
    )
    parser.add_argument(
        "--dataset_root",
        help="root directory of the dataset (defaults to the video parent)",
    )
    parser.add_argument(
        "--age_model",
        help="Path to a joblib file with the age regression model",
    )
    parser.add_argument(
        "--gender_model",
        help="Path to a joblib file with the gender classification model",
    )
    parser.add_argument(
        "--scale", type=float, default=4.2, help="scene scale used for gait estimation"
    )
    parser.add_argument(
        "--fps", type=float, help="video frame rate used to compute parameters"
    )
    args = parser.parse_args()

    if args.video:
        video_path = Path(args.video).resolve()
        dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else video_path.parents[2]
        patient = video_path.parents[1].name
        gait_type = video_path.parent.name

        if args.pose_path is None:
            args.pose_path = str(dataset_root / "pose" / patient / gait_type)
        if args.sensors is None:
            args.sensors = str(dataset_root / "sensors_bboxes" / patient / gait_type)
        if args.segmentation_path is None:
            args.segmentation_path = str(dataset_root / "semantic_segmentation" / patient / gait_type)
        if args.fps is None:
            cap = cv2.VideoCapture(str(video_path))
            args.fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
            cap.release()
    else:
        if args.fps is None:
            args.fps = 29.97

    df = calculate_parameters(
        args.pose_path, args.sensors, args.segmentation_path, args.scale, args.fps
    )
    base_features = df.drop(columns=["ID"]).to_numpy()

    if args.age_model:
        age_data = joblib.load(args.age_model)
        if isinstance(age_data, dict):
            age_model = age_data.get("age_model") or age_data.get("model")
            age_imputer = age_data.get("imputer")
            age_scaler = age_data.get("scaler")
            target_scaler = age_data.get("target_scaler")
        else:
            age_model = age_data
            age_imputer = None
            age_scaler = None
            target_scaler = None
        age_features = base_features.copy()
        if age_imputer is not None:
            age_features = age_imputer.transform(age_features)
        if age_scaler is not None:
            age_features = age_scaler.transform(age_features)

        age_pred = age_model.predict(age_features)
        if target_scaler is not None:
            age_pred = target_scaler.inverse_transform(age_pred.reshape(-1, 1)).ravel()
        age_pred = age_pred[0]
        print(f"Predicted age: {age_pred:.2f} years")

    if args.gender_model:
        gender_data = joblib.load(args.gender_model)
        if isinstance(gender_data, dict):
            gender_model = gender_data.get("gender_model") or gender_data.get("model")
            gender_imputer = gender_data.get("imputer")
            gender_scaler = gender_data.get("scaler")
        else:
            gender_model = gender_data
            gender_imputer = None
            gender_scaler = None
        gender_features = base_features.copy()
        if gender_imputer is not None:
            gender_features = gender_imputer.transform(gender_features)
        if gender_scaler is not None:
            gender_features = gender_scaler.transform(gender_features)

        gender_pred = gender_model.predict(gender_features)[0]
        print("Predicted gender:", "Male" if gender_pred == 1 else "Female")


if __name__ == "__main__":
    main()
