# src/evaluate.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import json
import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)

# ── Load data ──────────────────────────────────────────────
X = np.arange(-100, 100, 4).reshape(-1, 1)
y = np.arange(-90,  110, 4).reshape(-1, 1)

split = int(len(X) * params["data"]["train_split"])
X_test, y_test = X[split:], y[split:]

# ── Load saved model ───────────────────────────────────────
model = tf.keras.models.load_model("models/model.keras")

# ── Evaluate ───────────────────────────────────────────────
y_preds = model.predict(X_test, verbose=0)

mae = float(tf.keras.losses.mae(y_test.squeeze(), y_preds.squeeze()))
mse = float(tf.keras.losses.mse(y_test.squeeze(), y_preds.squeeze()))

metrics = {"mae": round(mae, 4), "mse": round(mse, 4)}
print(f"MAE: {metrics['mae']}  MSE: {metrics['mse']}")

# ── Write metrics.json — DVC reads this for experiment tracking ──
metrics_path = params["evaluate"]["metrics_path"]
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Metrics written to {metrics_path}")