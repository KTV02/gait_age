#!/usr/bin/env python3
"""Simplified age predictor training script.

This script trains a MoviNet regression model using the video-based
features provided in a Health&Gait-style dataset folder. It wraps the
``train_MoviNet_regression`` module from this repository so that the
user only needs to supply the dataset path and CSV files.
"""

import argparse
from types import SimpleNamespace

from code.train import train_MoviNet_regression


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MoviNet age regression from video features",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="root folder containing the modalities (silhouette, semantic_segmentation, optical_flow)",
    )
    parser.add_argument(
        "--patients_info",
        required=True,
        help="CSV file with participant measurements (participants_measures.csv)",
    )
    parser.add_argument(
        "--partitions",
        default="partitions/Age/partition_0.json",
        help="JSON file with train/val/test splits",
    )
    parser.add_argument(
        "--save_dir",
        default="results",
        help="directory where training artifacts will be stored",
    )
    parser.add_argument(
        "--data_type",
        choices=["silhouette", "semantic_segmentation", "optical_flow"],
        default="silhouette",
        help="feature modality to use",
    )
    parser.add_argument(
        "--data_class",
        choices=["WoJ", "WJ", "both"],
        default="both",
        help="walking condition (WoJ/WJ/both)",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_frames", type=int, default=20)
    parser.add_argument("--model_id", default="a0")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    movinet_args = SimpleNamespace(
        device=args.gpu,
        epochs=args.epochs,
        img_size=224,
        model_id=args.model_id,
        save_dir=args.save_dir,
        data_path=args.dataset,
        data_type=args.data_type,
        num_frames=args.num_frames,
        batch_size=args.batch_size,
        units=512,
        id_partition=0,
        learning_rate=0.001,
        patients_info=args.patients_info,
        partitions_file=args.partitions,
        data_class=args.data_class,
        optical_flow_method=None,
        id_experiment="age_regression",
        target="Age",
        wandb_project="age_regression",
        experiment_name="movinet_age",
        seed=27,
    )

    train_MoviNet_regression.main(movinet_args)


if __name__ == "__main__":
    main()
