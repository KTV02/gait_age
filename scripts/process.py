import os
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# AlphaPose + DensePose imports
from alphapose.models import builder
from alphapose.utils.presets import SimpleTransform
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

# ---------------------- ARGUMENT PARSING ----------------------
parser = argparse.ArgumentParser(description="Process video to Health&Gait format")
parser.add_argument("video", type=str, help="Path to input video")
parser.add_argument("participant_id", type=str, default="PA001", help="Participant ID")
parser.add_argument("output_root", type=str, default="Health_Gait", help="Root output folder")
args = parser.parse_args()

video_path = Path(args.video)
participant_id = args.participant_id
output_root = Path(args.output_root)
video_name = video_path.stem

# ---------------------- FOLDER STRUCTURE ----------------------
segm_dir = output_root / "semantic_segmentation" / participant_id / "FGS" / f"{video_name}_DensePose"
silhouette_dir = output_root / "silhouette" / participant_id / "FGS" / f"{video_name}_YOLOV8"
gei_dir = output_root / "gait_energy_image" / participant_id / "FGS" / f"{video_name}_GEI"

for path in [segm_dir, silhouette_dir, gei_dir]:
    path.mkdir(parents=True, exist_ok=True)

# ---------------------- INIT MODELS ----------------------

def init_densepose():
    cfg = get_cfg()
    cfg.merge_from_file("detectron2_repo/configs/densepose_rcnn_R_50_FPN_s1x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/137260150/model_final_162be9.pkl"
    return DefaultPredictor(cfg)

def init_alphapose():
    pose_model_cfg = 'configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml'
    pose_model = builder.build_sppe(pose_model_cfg, preset='simple')
    pose_model.load_state_dict(torch.load('pretrained/pose_model.pth'))
    pose_model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    transform = SimpleTransform(
        pose_model.cfg.DATASET.IMAGE_SIZE,
        pose_model.cfg.DATASET.IMAGE_SIZE,
        pose_model.cfg.DATASET.IMAGE_MEAN,
        pose_model.cfg.DATASET.IMAGE_STD
    )
    return pose_model, transform

# ---------------------- PROCESS VIDEO ----------------------

def process(video_path, dp, ap_model, ap_trans):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    silhouettes = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame.copy())
        outputs = dp(frame)
        boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy().astype(int)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            crop = frame[y1:y2, x1:x2]
            sil_path = silhouette_dir / f"{frame_id:03d}.jpg"
            cv2.imwrite(str(sil_path), crop)
            silhouettes.append(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))

            # DensePose output
            dp_mask = outputs["instances"].pred_densepose[i].segm.cpu().numpy()
            dp_img = cv2.applyColorMap((dp_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(str(segm_dir / f"{frame_id:03d}.png"), dp_img)

            # AlphaPose (optional - only storing keypoints)
            input_pose = ap_trans.test_transform(crop, [x2 - x1, y2 - y1]).unsqueeze(0).to(device)
            with torch.no_grad():
                _ = ap_model(input_pose)

        frame_id += 1

    cap.release()
    generate_gei(silhouettes, gei_dir)

# ---------------------- GEI GENERATION ----------------------

def generate_gei(silhouettes, output_dir):
    if not silhouettes:
        return
    sil_stack = np.stack([cv2.resize(s, (64, 128)) for s in silhouettes], axis=0)
    avg_sil = np.mean(sil_stack, axis=0).astype(np.uint8)
    gei_path = output_dir / f"{video_name}.png"
    cv2.imwrite(str(gei_path), avg_sil)

# ---------------------- RUN ----------------------

densepose = init_densepose()
alphapose_model, alphapose_trans = init_alphapose()
process(video_path, densepose, alphapose_model, alphapose_trans)
