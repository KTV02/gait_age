#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import joblib
from sklearn.ensemble import RandomForestRegressor

# ------------------------ Helper Classes & Functions ------------------------

class MoviNetFeatureExtractor(tf.keras.layers.Layer):
    def __init__(self, model_url):
        super().__init__()
        self.movinet = hub.load(model_url)

    def call(self, inputs):
        return self.movinet({"image": inputs})


def create_model(model_id, img_size=224, num_frames=20, learning_rate=0.001, units=512):
    inputs = tf.keras.Input(shape=(num_frames, img_size, img_size, 3), name="image")
    movinet_url = f"https://tfhub.dev/tensorflow/movinet/{model_id}/base/kinetics-600/classification/3"
    x = MoviNetFeatureExtractor(movinet_url)(inputs)
    x = tf.keras.layers.Dense(units, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='linear')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='mse', metrics=['mae'])
    return model


def load_frames(path, num_frames=10, img_size=224):
    files = sorted([f for f in os.listdir(path) if f.endswith(".png") or f.endswith(".jpg")])
    total = len(files)
    if total < num_frames:
        raise ValueError(f"Not enough frames in {path}")

    step = total // num_frames
    selected = [files[i * step] for i in range(num_frames)]
    frames = [cv2.imread(os.path.join(path, f)) for f in selected]
    frames = [cv2.resize(f, (img_size, img_size)) for f in frames]
    frames = [tf.image.convert_image_dtype(f, tf.float32) for f in frames]
    return tf.stack(frames)[..., ::-1].numpy()  # BGR → RGB

def predict_movinet(model, modality_dir, num_frames, img_size):
    """
    Predict age for each patient (i.e., subdirectory under modality_dir)
    """
    preds = []
    for patient_id in sorted(os.listdir(modality_dir)):
        patient_path = os.path.join(modality_dir, patient_id)
        if not os.path.isdir(patient_path):
            continue

        files = sorted([
            f for f in os.listdir(patient_path)
            if f.endswith(".png") or f.endswith(".jpg")
        ])
        if not files:
            print(f"⚠️  No images in {patient_path}")
            continue

        # Handle cases with fewer images than required
        if len(files) < num_frames:
            files += [files[-1]] * (num_frames - len(files))

        step = len(files) // num_frames
        selected = [files[i * step] for i in range(num_frames)]

        frames = []
        for f in selected:
            img = cv2.imread(os.path.join(patient_path, f))
            if img is None:
                print(f"⚠️  Could not read {f}")
                continue
            img = cv2.resize(img, (img_size, img_size))
            img = tf.image.convert_image_dtype(img, tf.float32)
            frames.append(img)

        if len(frames) != num_frames:
            print(f"⚠️  Skipping {patient_id}, invalid frame count after loading.")
            continue

        video = tf.stack(frames)[..., ::-1].numpy()  # BGR → RGB
        video = np.expand_dims(video, axis=0)

        pred = model.predict(video, verbose=0)[0][0]
        preds.append((patient_id, pred))
    
    return preds


def predict_alphapose(alphapose_model, csv_dir):
    preds = []
    for fname in sorted(os.listdir(csv_dir)):
        if not fname.endswith(".csv"):
            continue
        path = os.path.join(csv_dir, fname)
        try:
            df = pd.read_csv(path)
            if df.empty:
                continue
            feat = df.mean().values.reshape(1, -1)
            pred = alphapose_model.predict(feat)[0]
            pid = fname.replace(".csv", "")
            preds.append((pid, pred))
        except Exception as e:
            print(f"Skipping {path}: {e}")
    return preds


def merge_predictions(*modalities):
    merged = {}
    for modality in modalities:
        for pid, pred in modality:
            if pid not in merged:
                merged[pid] = []
            merged[pid].append(pred)
    return merged


# ------------------------ Main Function ------------------------

def main():
    parser = argparse.ArgumentParser(description="Inference with boosted age regression model")
    parser.add_argument("--data_dir", required=True, help="Unseen data root folder")
    parser.add_argument("--model_dir", default="results", help="Directory with saved models")
    parser.add_argument("--num_frames", type=int, default=20)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--model_id", default="a0", help="MoviNet model ID")
    parser.add_argument("--output_csv", default="boosted_predictions.csv")
    args = parser.parse_args()

    # Load boosted model
    fusion_model_path = os.path.join(args.model_dir, "fusion_model.joblib")
    if not os.path.exists(fusion_model_path):
        raise FileNotFoundError(f"Missing boosted model: {fusion_model_path}")
    fusion_model = joblib.load(fusion_model_path)

    predictions_by_modality = []

    for modality in ["silhouette", "semantic_segmentation", "optical_flow"]:
        modality_path = os.path.join(args.data_dir, modality)
        checkpoint_path = os.path.join(args.model_dir, modality, "checkpoint", "model.weights.h5")
        norm_path = os.path.join(args.model_dir, modality, "checkpoint", "norm_params.npz")

        if os.path.isdir(modality_path) and os.path.exists(checkpoint_path) and os.path.exists(norm_path):
            print(f"→ Loading MoviNet model for {modality}")
            model = create_model(args.model_id, img_size=args.img_size, num_frames=args.num_frames)
            model.load_weights(checkpoint_path)

            # ⏬ Load normalization parameters
            norm_data = np.load(norm_path)
            min_train = norm_data["min_train"].item()
            max_train = norm_data["max_train"].item()

            # 🔄 Run prediction for all patient subfolders
            raw_preds = predict_movinet(model, modality_path, num_frames=args.num_frames, img_size=args.img_size)

            # 🧮 Denormalize each prediction
            denorm_preds = [
                (pid, float(pred) * (max_train - min_train) + min_train)
                for pid, pred in raw_preds
            ]

            predictions_by_modality.append(denorm_preds)
        else:
            print(f"⚠️  Skipping modality '{modality}' (data, model, or norm_params missing)")


    alphapose_path = os.path.join(args.data_dir, "alphapose_csv")
    alphapose_model_path = os.path.join(args.model_dir, "alphapose_rf.joblib")

    if os.path.isdir(alphapose_path) and os.path.exists(alphapose_model_path):
        print("→ Loading AlphaPose model")
        alphapose_model = joblib.load(alphapose_model_path)
        preds = predict_alphapose(alphapose_model, alphapose_path)
        predictions_by_modality.append(preds)
    else:
        print("⚠️  Skipping AlphaPose (data or model missing)")

    # Merge predictions
    merged = merge_predictions(*predictions_by_modality)

    # Stack into array
    final_preds = []
    pids = []
    for pid, preds in merged.items():
        if len(preds) != fusion_model.n_features_in_:
            print(f"⚠️  Skipping {pid}: expected {fusion_model.n_features_in_} features, got {len(preds)}")
            continue
        print(f"\n🧩 Patient: {pid}")
        for i, value in enumerate(preds):
            print(f"  Modality {i+1}: {value:.4f}")

        pred = fusion_model.predict([preds])[0]
        final_preds.append((pid, pred))
        pids.append(pid)

    # Save to CSV
    df = pd.DataFrame(final_preds, columns=["ID", "Predicted_Age"])
    df.to_csv(args.output_csv, index=False)
    print(f"\n✅ Predictions saved to {args.output_csv}")


if __name__ == "__main__":
    import cv2
    main()