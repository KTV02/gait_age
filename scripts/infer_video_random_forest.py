#!/usr/bin/env python3
"""Run inference for age and gender from gait features.

The script relies on the repository's ``calculate_parameters`` function to
extract gait features from AlphaPose, sensor and segmentation data. The trained
model is loaded from a ``joblib`` file, which can contain any scikit-learn
estimator (RandomForest, Lasso, etc.) or even a Keras model. The saved object
should include the fitted ``imputer`` and ``scaler`` used during training and
optionally a ``target_scaler`` for age regression.
"""

import argparse
import joblib

from code.gait_estimation.gait_parameters_estimation import calculate_parameters


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict age and gender from pre-extracted gait features"
    )
    parser.add_argument(
        "--pose_path",
        required=True,
        help="directory containing AlphaPose JSON files",
    )
    parser.add_argument(
        "--sensors",
        required=True,
        help="directory with OptoGait sensor bounding boxes",
    )
    parser.add_argument(
        "--segmentation_path",
        required=True,
        help="directory containing DensePose segmentations",
    )
    parser.add_argument(
        "--model_file",
        required=True,
        help="Path to a joblib file with the trained model(s) and preprocessors",
    )
    parser.add_argument(
        "--scale", type=float, default=4.2, help="scene scale used for gait estimation"
    )
    parser.add_argument(
        "--fps", type=float, default=29.97, help="video frame rate used to compute parameters"
    )
    args = parser.parse_args()

    df = calculate_parameters(
        args.pose_path, args.sensors, args.segmentation_path, args.scale, args.fps
    )
    features = df.drop(columns=["ID"]).to_numpy()

    model_data = joblib.load(args.model_file)
    imputer = model_data.get("imputer")
    scaler = model_data.get("scaler")
    target_scaler = model_data.get("target_scaler")
    age_model = model_data.get("age_model") or model_data.get("model")
    gender_model = model_data.get("gender_model")

    if imputer:
        features = imputer.transform(features)
    if scaler:
        features = scaler.transform(features)

    if age_model is not None:
        age_pred = age_model.predict(features)
        if target_scaler is not None:
            age_pred = target_scaler.inverse_transform(age_pred.reshape(-1, 1)).ravel()
        age_pred = age_pred[0]
        print(f"Predicted age: {age_pred:.2f} years")

    if gender_model is not None:
        gender_pred = gender_model.predict(features)[0]
        print("Predicted gender:", "Male" if gender_pred == 1 else "Female")


if __name__ == "__main__":
    main()
