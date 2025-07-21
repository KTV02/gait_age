#!/usr/bin/env python3
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import argparse
import cv2
import os

class MoviNetFeatureExtractor(tf.keras.layers.Layer):
    def __init__(self, model_url):
        super().__init__()
        self.movinet = hub.load(model_url)

    def call(self, inputs):
        return self.movinet({"image": inputs})


def create_model(model_id="a0", img_size=224, num_frames=20, learning_rate=0.001, units=512):
    inputs = tf.keras.Input(shape=(num_frames, img_size, img_size, 3), name="image")
    movinet_url = f"https://tfhub.dev/tensorflow/movinet/{model_id}/base/kinetics-600/classification/3"
    x = MoviNetFeatureExtractor(movinet_url)(inputs)
    x = tf.keras.layers.Dense(units, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='linear')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='mse')
    return model


def preprocess_image(image_path, img_size=224):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    image = cv2.resize(image, (img_size, img_size))
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = image[..., ::-1]  # BGR → RGB
    return image.numpy()


def main():
    parser = argparse.ArgumentParser(description="Run single image through trained MoviNet regression model")
    parser.add_argument("--image", required=True, help="Path to single image")
    parser.add_argument("--weights", required=True, help="Path to MoviNet weights (.weights.h5)")
    parser.add_argument("--model_id", default="a0", help="MoviNet model ID (e.g., a0)")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_frames", type=int, default=20)
    parser.add_argument("--normalize", action="store_true", help="Apply normalization using min/max params")
    parser.add_argument("--norm_path", help="Path to norm_params.npz (if --normalize is set)")
    args = parser.parse_args()

    model = create_model(model_id=args.model_id, img_size=args.img_size, num_frames=args.num_frames)
    model.load_weights(args.weights)

    # Prepare fake clip from one image (repeat the frame num_frames times)
    frame = preprocess_image(args.image, img_size=args.img_size)
    clip = np.repeat(np.expand_dims(frame, axis=0), args.num_frames, axis=0)
    clip = np.expand_dims(clip, axis=0)  # shape: (1, num_frames, H, W, 3)

    pred = model.predict(clip, verbose=0)[0][0]

    if args.normalize and args.norm_path:
        npz = np.load(args.norm_path)
        min_train = npz["min_train"]
        max_train = npz["max_train"]
        pred = pred * (max_train - min_train) + min_train

    print(f"\n🎯 Predicted age: {pred:.2f}")


if __name__ == "__main__":
    main()
