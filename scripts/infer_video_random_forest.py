#!/usr/bin/env python3
"""Run Random Forest inference for age and gender from gait features.

This script delegates gait-parameter estimation to the repository's
``calculate_parameters`` function and only handles model loading and
prediction.
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
        "--age_model",
        required=True,
        help="path to the trained RandomForestRegressor",
    )
    parser.add_argument(
        "--gender_model",
        required=True,
        help="path to the trained RandomForestClassifier",
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

    age_model = joblib.load(args.age_model)
    gender_model = joblib.load(args.gender_model)

    age_pred = age_model.predict(features)[0]
    gender_pred = gender_model.predict(features)[0]

    print(f"Predicted age: {age_pred:.2f} years")
    print("Predicted gender:", "Male" if gender_pred == 1 else "Female")


if __name__ == "__main__":
    main()
