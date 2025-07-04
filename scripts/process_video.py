#!/usr/bin/env python3
"""Convert a raw video into the folder structure used by Health&Gait."""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare video data in the Health&Gait folder structure",
    )
    parser.add_argument("--video", required=True, help="path to input video")
    parser.add_argument(
        "--participant",
        default="PA001",
        help="participant identifier used for the output folders",
    )
    parser.add_argument(
        "--condition",
        default="UGS_WoJ",
        help="gait condition name (e.g. UGS_WoJ)",
    )
    parser.add_argument(
        "--direction",
        default="1",
        help="direction index used inside the condition folder",
    )
    parser.add_argument(
        "--alphapose_dir",
        default="AlphaPose",
        help="location of the AlphaPose repository",
    )
    parser.add_argument(
        "--densepose_dir",
        default="DensePose",
        help="location of the DensePose repository",
    )
    return parser.parse_args()


# === Folder Setup ===
def main() -> None:
    args = parse_args()

    participant_id = args.participant
    condition = args.condition
    direction = args.direction

    output_base = Path(participant_id)
    modalities = ["pose", "semantic_segmentation", "optical_flow", "silhouette"]
    for modality in modalities:
        (output_base / modality / condition / direction).mkdir(
            parents=True, exist_ok=True
        )

    # === Extract frames ===

    frame_dir = "temp_frames"
    os.makedirs(frame_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{frame_dir}/frame_{frame_idx:04d}.jpg", frame)
        frame_idx += 1
    cap.release()

    # === Run AlphaPose ===
    subprocess.run(
        [
            "python",
            "scripts/demo_inference.py",
            "--indir",
            frame_dir,
            "--outdir",
            "alphapose_temp",
            "--detector",
            "yolo3",
            "--pose_model",
            "resnet50",
            "--save_img",
            "False",
        ],
        check=True,
        cwd=args.alphapose_dir,
        env=dict(os.environ, PYTHONPATH=args.alphapose_dir),
    )
    # Move JSONs
    for f in os.listdir("alphapose_temp"):
        if f.endswith(".json"):
            shutil.move(
                os.path.join("alphapose_temp", f),
                output_base / "pose" / condition / direction / f,
            )

    # === Run DensePose ===
    os.makedirs("densepose_temp", exist_ok=True)
    for f in sorted(os.listdir(frame_dir)):
        input_img = os.path.join(frame_dir, f)
        output_img = os.path.join("densepose_temp", f.replace(".jpg", "_IUV.png"))
        subprocess.run(
            [
                "python",
                "tools/infer_simple.py",
                "--cfg",
                str(
                    Path(args.densepose_dir)
                    / "configs"
                    / "DensePose_ResNet101_FPN_s1x-e2e.yaml"
                ),
                "--output-dir",
                "densepose_temp",
                "--image-ext",
                "jpg",
                "--wts",
                "https://dl.fbaipublicfiles.com/densepose/DensePose_ResNet101_FPN_s1x-e2e.pkl",
                input_img,
            ],
            check=True,
            cwd=args.densepose_dir,
            env=dict(os.environ, PYTHONPATH=args.densepose_dir),
        )
        moved = Path(output_img)
        if moved.exists():
            shutil.move(
                str(moved),
                output_base
                / "semantic_segmentation"
                / condition
                / direction
                / moved.name,
            )

    # === Run Optical Flow (Dual TV-L1 using OpenCV) ===
    prev_frame = None
    for f in sorted(os.listdir(frame_dir)):
        current_frame = cv2.imread(os.path.join(frame_dir, f))
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            flow = cv2.calcOpticalFlowDualTVL1(prev_frame, gray, None)
            hsv = np.zeros_like(current_frame)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 1] = 255
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            out_path = (
                output_base
                / "optical_flow"
                / condition
                / direction
                / f.replace(".jpg", "_flow.png")
            )
            cv2.imwrite(str(out_path), flow_rgb)
        prev_frame = gray

    # === Run YOLOv8x-seg for Silhouette ===
    yolo_script = Path("yolov8seg/apply_yolo_seg.py")
    if yolo_script.exists():
        for f in sorted(os.listdir(frame_dir)):
            input_img = os.path.join(frame_dir, f)
            output_path = (
                output_base
                / "silhouette"
                / condition
                / direction
                / f.replace(".jpg", "_silhouette.jpg")
            )
            subprocess.run(
                [
                    "python",
                    str(yolo_script),
                    "--image",
                    input_img,
                    "--output",
                    str(output_path),
                ]
            )
    else:
        print("⚠️ YOLOv8 segmentation script not found; skipping silhouettes")

    # === Clean Up ===
    shutil.rmtree("alphapose_temp", ignore_errors=True)
    shutil.rmtree("densepose_temp", ignore_errors=True)
    shutil.rmtree(frame_dir, ignore_errors=True)

    print("✅ Video processed into Health&Gait format!")


if __name__ == "__main__":
    main()
