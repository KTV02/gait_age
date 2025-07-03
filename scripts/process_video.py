import os
import cv2
import subprocess
import shutil

# === Configurations ===
video_path = "your_video.mp4"
participant_id = "PA001"
condition = "UGS_WoJ"
direction = "1"

# === Folder Setup ===
modalities = ["pose", "semantic_segmentation", "optical_flow", "silhouette"]
for modality in modalities:
    os.makedirs(f"{participant_id}/{modality}/{condition}/{direction}", exist_ok=True)

# === Extract frames ===
frame_dir = "temp_frames"
os.makedirs(frame_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(f"{frame_dir}/frame_{frame_idx:04d}.jpg", frame)
    frame_idx += 1
cap.release()

# === Run AlphaPose ===
subprocess.run([
    "python", "AlphaPose/demo.py",
    "--indir", frame_dir,
    "--outdir", "alphapose_temp",
    "--detector", "yolo3",
    "--pose_model", "resnet50",
    "--save_img", "False"
])
# Move JSONs
for f in os.listdir("alphapose_temp"):
    if f.endswith(".json"):
        shutil.move(f"alphapose_temp/{f}", f"{participant_id}/pose/{condition}/{direction}/{f}")

# === Run DensePose ===
os.makedirs("densepose_temp", exist_ok=True)
for f in sorted(os.listdir(frame_dir)):
    input_img = os.path.join(frame_dir, f)
    output_img = os.path.join("densepose_temp", f.replace(".jpg", "_densepose.png"))
    subprocess.run([
        "python", "DensePose/apply_densepose.py",
        "--image", input_img,
        "--output", output_img
    ])
    shutil.move(output_img, f"{participant_id}/semantic_segmentation/{condition}/{direction}/" + os.path.basename(output_img))

# === Run Optical Flow (Dual TV-L1 using OpenCV) ===
prev_frame = None
for f in sorted(os.listdir(frame_dir)):
    current_frame = cv2.imread(os.path.join(frame_dir, f))
    gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    if prev_frame is not None:
        flow = cv2.calcOpticalFlowDualTVL1(prev_frame, gray, None)
        hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        out_path = f"{participant_id}/optical_flow/{condition}/{direction}/" + f.replace(".jpg", "_flow.png")
        cv2.imwrite(out_path, flow_rgb)
    prev_frame = gray

# === Run YOLOv8x-seg for Silhouette ===
for f in sorted(os.listdir(frame_dir)):
    input_img = os.path.join(frame_dir, f)
    output_path = f"{participant_id}/silhouette/{condition}/{direction}/" + f.replace(".jpg", "_silhouette.jpg")
    subprocess.run([
        "python", "yolov8seg/apply_yolo_seg.py",
        "--image", input_img,
        "--output", output_path
    ])

# === Clean Up ===
shutil.rmtree("alphapose_temp")
shutil.rmtree("densepose_temp")
shutil.rmtree(frame_dir)

print("✅ Video processed into Health&Gait format!")
