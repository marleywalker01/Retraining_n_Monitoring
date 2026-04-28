# src/model.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import yaml
import json
from dvclive import Live

# ── Load hyperparameters from params.yaml ──────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

EPOCHS        = params["model"]["epochs"]
LEARNING_RATE = params["model"]["learning_rate"]
DENSE_UNITS   = params["model"]["dense_units"]
TRAIN_SPLIT   = params["data"]["train_split"]
SEED          = params["data"]["random_seed"]

# ── Data ───────────────────────────────────────────────────
X = np.arange(-100, 100, 4).reshape(-1, 1)
y = np.arange(-90,  110, 4).reshape(-1, 1)

split = int(len(X) * TRAIN_SPLIT)
X_train, y_train = X[:split], y[:split]
X_test,  y_test  = X[split:], y[split:]

# ── Model ──────────────────────────────────────────────────
tf.random.set_seed(SEED)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(DENSE_UNITS, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(
    loss=tf.keras.losses.mae,
    optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE),
    metrics=['mae']
)

# ── Train with DVCLive logging ─────────────────────────────
# DVCLive automatically logs metrics each epoch and saves them
# in a format DVC can read and compare across runs.
with Live(dir="dvclive", report="html") as live:
    for epoch in range(EPOCHS):
        history = model.fit(X_train, y_train, epochs=1, verbose=0)
        train_mae = history.history['mae'][0]

        val_loss  = model.evaluate(X_test, y_test, verbose=0)
        live.log_metric("train_mae", train_mae)
        live.log_metric("val_mae",   val_loss[1])
        live.next_step()

# ── Save model ─────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
model.save("models/model.keras")
print("Model saved to models/model.keras")

# ── Save predictions plot ──────────────────────────────────
y_preds = model.predict(X_test)

plt.figure(figsize=(6, 5))
plt.scatter(X_train, y_train, c="b", label="Training data")
plt.scatter(X_test,  y_test,  c="g", label="Testing data")
plt.scatter(X_test,  y_preds, c="r", label="Predictions")
plt.legend(shadow=True)
plt.grid(which='major', c='#cccccc', linestyle='--', alpha=0.5)
plt.title('Model Results', family='Arial', fontsize=14)
plt.xlabel('X axis values', family='Arial', fontsize=11)
plt.ylabel('Y axis values', family='Arial', fontsize=11)
plt.savefig('model_results.png', dpi=120)