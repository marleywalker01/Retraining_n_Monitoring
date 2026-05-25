import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import json
from datetime import datetime

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load hyperparameters
with open("params.yaml") as f:
    params = yaml.safe_load(f)

TEST_SIZE    = params["data"]["test_size"]
BATCH_SIZE   = params["model"]["batch_size"]
EPOCHS       = params["model"]["epochs"]
METRICS_PATH = params["evaluate"]["metrics_path"]

artifacts_dir = "artifacts"

# Check required files exist
for path in ["artifacts/X_test_cnn.npy", "artifacts/y_test.npy",
             "artifacts/training_history.json", "models/model.keras"]:
    if not os.path.exists(path):
        print(f"ERROR: {path} not found — run model.py first")
        sys.exit(1)

print("Loading test data and model...")
X_test = np.load("artifacts/X_test_cnn.npy")
y_test = np.load("artifacts/y_test.npy")
print(f"X_test: {X_test.shape}  y_test: {y_test.shape}")

with open("artifacts/training_history.json") as f:
    history_dict = json.load(f)

# Load model
model = tf.keras.models.load_model("models/model.keras")
print("Model loaded successfully")

# Evaluate
score = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss: {score[0]:.4f}")
print(f"Test Accuracy: {score[1]:.4f}")

# Predictions
preds_prob = model.predict(X_test, verbose=0)
preds = np.argmax(preds_prob, axis=1)

# Metrics
acc = accuracy_score(y_test, preds)
print(f"\nAccuracy: {acc:.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, preds)}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, preds)}")

# Plot training curves
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history_dict['accuracy'], label="Train Accuracy")
plt.plot(history_dict['val_accuracy'], label="Val Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history_dict['loss'], label="Train Loss")
plt.plot(history_dict['val_loss'], label="Val Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig(f'{artifacts_dir}/training_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: training_curves.png")

# Save metrics.json
metrics_out = {
    "timestamp":  datetime.now().isoformat(),
    "model_type": "ANN Classifier",
    "test_size":  TEST_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs":     EPOCHS,
    "metrics": {
        "accuracy": float(acc),
        "loss":     float(score[0])
    },
    "training_history": {
        "final_train_loss":     float(history_dict['loss'][-1]),
        "final_val_loss":       float(history_dict['val_loss'][-1]),
        "final_train_accuracy": float(history_dict['accuracy'][-1]),
        "final_val_accuracy":   float(history_dict['val_accuracy'][-1])
    }
}

with open(METRICS_PATH, 'w', encoding='utf-8') as f:
    json.dump(metrics_out, f, indent=4)
print(f"Saved: {METRICS_PATH}")

# Save metrics.txt
with open('metrics.txt', 'w') as f:
    f.write("=" * 50 + "\n")
    f.write("MODEL PERFORMANCE METRICS\n")
    f.write("=" * 50 + "\n")
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"Loss:     {score[0]:.4f}\n")
    f.write("=" * 50 + "\n")
print("Saved: metrics.txt")