import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# === Constants ===
MOVINET_MODELS = {
    "silhouette": "https://tfhub.dev/tensorflow/movinet/a0/base/kinetics-600/classification/3",
    "semantic":   "https://tfhub.dev/tensorflow/movinet/a0/base/kinetics-600/classification/3",
    "flow":       "https://tfhub.dev/tensorflow/movinet/a0/base/kinetics-600/classification/3"
}

NUM_FRAMES = 20
IMG_SIZE = 224
UNITS = 512


# === Load Video Feature Model ===
def load_movinet_model(url):
    module = hub.load(url)
    def extractor(video_tensor):  # video_tensor: [T, H, W, 3]
        x = tf.convert_to_tensor(video_tensor[np.newaxis], dtype=tf.float32)
        out = module({"image": x})  # shape: (1, 600)
        return tf.squeeze(out).numpy()
    return extractor


# === Load AlphaPose CSV Features ===
def load_alphapose_features(csv_path):
    df = pd.read_csv(csv_path)
    return df.values.flatten()


# === Ensemble Training ===
def train_all(X, y, present_modalities):
    base_preds = []
    models = {}

    for name, inputs in X.items():
        if name == "alphapose":
            rf = RandomForestRegressor(n_estimators=100)
            rf.fit(inputs, y)
            base_preds.append(rf.predict(inputs).reshape(-1, 1))
            models[name] = rf
        else:
            movinet = tf.keras.Sequential([
                tf.keras.Input(shape=(NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3)),
                tf.keras.layers.Lambda(lambda x: hub.load(MOVINET_MODELS[name])({"image": x})),
                tf.keras.layers.Dense(UNITS, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
            movinet.compile(optimizer='adam', loss='mse')
            movinet.fit(inputs, y, epochs=10, batch_size=4)
            base_preds.append(movinet.predict(inputs))
            models[name] = movinet

    stacked_X = np.hstack(base_preds)
    booster = GradientBoostingRegressor(n_estimators=100)
    booster.fit(stacked_X, y)
    return models, booster


# === Predict with missing modalities support ===
def predict(models, booster, inputs):
    preds = []
    for name, model in models.items():
        if name not in inputs:
            continue
        x = inputs[name]
        if isinstance(model, tf.keras.Model):
            pred = model.predict(x[np.newaxis])[0]
        else:
            pred = model.predict(x.reshape(1, -1))[0]
        preds.append(pred)
    return booster.predict(np.array(preds).reshape(1, -1))[0]


# === Example Usage ===
if __name__ == "__main__":
    # Dummy data loading section
    # Replace with real video loading and AlphaPose parsing
    silhouette_data = np.random.rand(100, NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3)
    semantic_data   = np.random.rand(100, NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3)
    flow_data       = np.random.rand(100, NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3)
    alphapose_data  = np.random.rand(100, 90)  # e.g. 90 pose-based features

    labels = np.random.uniform(20, 80, 100)

    X = {
        "silhouette": silhouette_data,
        "semantic": semantic_data,
        "flow": flow_data,
        "alphapose": alphapose_data
    }

    # Train
    models, booster = train_all(X, labels, present_modalities=list(X.keys()))

    # Save booster and models
    joblib.dump(booster, "booster.pkl")
    joblib.dump(models["alphapose"], "rf_alphapose.pkl")

    # Example inference with missing 'flow'
    test_input = {
        "silhouette": silhouette_data[0],
        "semantic": semantic_data[0],
        "alphapose": alphapose_data[0]
    }
    pred = predict(models, booster, test_input)
    print(f"Predicted age: {pred:.2f}")
