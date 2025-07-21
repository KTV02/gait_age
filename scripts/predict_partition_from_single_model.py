import os
import json
import argparse
import numpy as np
import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
from tqdm import tqdm
import cv2

class MoviNetFeatureExtractor(tf.keras.layers.Layer):
    def __init__(self, model_url):
        super().__init__()
        self.model = hub.load(model_url)
    def call(self, x):
        return self.model({"image": x})

def create_model(model_id, img_size=224, num_frames=20, units=512):
    inputs = tf.keras.Input(shape=(num_frames, img_size, img_size, 3), name="image")
    movinet_url = f"https://tfhub.dev/tensorflow/movinet/{model_id}/base/kinetics-600/classification/3"
    x = MoviNetFeatureExtractor(movinet_url)(inputs)
    x = tf.keras.layers.Dense(units, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='linear')(x)
    return tf.keras.Model(inputs, outputs)

def load_frames(folder, num_frames=20, img_size=224):
    files = sorted([f for f in os.listdir(folder) if f.endswith((".png", ".jpg"))])
    if len(files) < num_frames:
        raise ValueError("Not enough frames")
    step = len(files) // num_frames
    selected = [files[i * step] for i in range(num_frames)]
    frames = [cv2.resize(cv2.imread(os.path.join(folder, f)), (img_size, img_size)) for f in selected]
    frames = [tf.image.convert_image_dtype(f, tf.float32).numpy()[..., ::-1] for f in frames]
    return np.stack(frames)

def predict_for_patient(model, patient_folder, num_frames, img_size):
    all_dirs = []
    for root, dirs, _ in os.walk(patient_folder):
        for d in dirs:
            full_path = os.path.join(root, d)
            if len(os.listdir(full_path)) >= num_frames:
                all_dirs.append(full_path)

    preds = []
    for clip_path in all_dirs:
        try:
            frames = load_frames(clip_path, num_frames, img_size)
            pred = model.predict(np.expand_dims(frames, 0), verbose=0)[0][0]
            preds.append(pred)
        except Exception as e:
            print(f"Skipping {clip_path}: {e}")
    return np.mean(preds) if preds else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Root directory containing modality subfolders")
    parser.add_argument("--results_dir", required=True, help="Path to results/{modality}/")
    parser.add_argument("--partition_json", required=True, help="Path to partition_0.json")
    parser.add_argument("--modality", required=True, choices=["silhouette", "semantic_segmentation", "optical_flow"])
    parser.add_argument("--model_id", default="a0")
    parser.add_argument("--num_frames", type=int, default=20)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--units", type=int, default=512)
    args = parser.parse_args()

    modality_path = os.path.join(args.data_dir, args.modality)
    weight_path = os.path.join(args.results_dir, args.modality, "checkpoint", "model.weights.h5")
    norm_path = os.path.join(args.results_dir, args.modality, "checkpoint", "norm_params.npz")
    print(weight_path)
    print(norm_path)
    if not (os.path.exists(weight_path) and os.path.exists(norm_path)):
        print(f"Missing model or normalization files for {args.modality}")
        return

    model = create_model(args.model_id, args.img_size, args.num_frames, args.units)
    model.load_weights(weight_path)

    norm = np.load(norm_path)
    min_train, max_train = norm["min_train"], norm["max_train"]

    with open(args.partition_json) as f:
        partition = json.load(f)
    patient_ids = partition.get("train", []) + partition.get("validation", []) + partition.get("test", [])

    predictions = []
    for pid in tqdm(patient_ids):
        patient_dir = os.path.join(modality_path, pid)
        if not os.path.isdir(patient_dir):
            print(f"Missing: {patient_dir}")
            continue
        pred_norm = predict_for_patient(model, patient_dir, args.num_frames, args.img_size)
        if pred_norm is None:
            continue
        age = pred_norm * (max_train - min_train) + min_train
        predictions.append((pid, age))

    # Save predictions
    out_path = os.path.join(args.results_dir, args.modality, "predictions.csv")
    pd.DataFrame(predictions, columns=["ID", "Predicted_Age"]).to_csv(out_path, index=False)
    print(f"✅ Saved predictions to {out_path}")

if __name__ == "__main__":
    main()