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
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix,
                             classification_report)

# ── Load hyperparameters ───────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

TEST_SIZE    = params["data"]["test_size"]
BATCH_SIZE   = params["model"]["batch_size"]
EPOCHS       = params["model"]["epochs"]
METRICS_PATH = params["evaluate"]["metrics_path"]

artifacts_dir = "artifacts"

# ── Load test data saved by model.py ──────────────────────
for path in ["artifacts/X_test_cnn.npy", "artifacts/y_test.npy",
             "artifacts/training_history.json", "models/model.keras"]:
    if not os.path.exists(path):
        print(f"ERROR: {path} not found — run model.py first")
        sys.exit(1)

print("Loading test data and model...")
X_test_cnn = np.load("artifacts/X_test_cnn.npy")
y_test     = np.load("artifacts/y_test.npy")
print(f"X_test_cnn: {X_test_cnn.shape}  y_test: {y_test.shape}")

with open("artifacts/training_history.json") as f:
    history_dict = json.load(f)

# ── Load model ─────────────────────────────────────────────
model = tf.keras.models.load_model("models/model.keras")
print("Model loaded successfully")

# ── Evaluate ───────────────────────────────────────────────
score = model.evaluate(X_test_cnn, y_test, verbose=0)
print(f"\nTest Loss:     {score[0]:.4f}")
print(f"Test Accuracy: {score[1]:.4f}")

# ── Predictions ────────────────────────────────────────────
preds_prob = model.predict(X_test_cnn, verbose=0)
preds = np.argmax(preds_prob, axis=1)

# ── Classification metrics ─────────────────────────────────
acc       = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, average='weighted', zero_division=0)
recall    = recall_score(y_test, preds, average='weighted', zero_division=0)
f1        = f1_score(y_test, preds, average='weighted', zero_division=0)

print(f"\nAccuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, preds, zero_division=0)}")

# ── Confusion matrix plot ──────────────────────────────────
cm = confusion_matrix(y_test, preds)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(title='Confusion Matrix',
       ylabel='True Label',
       xlabel='Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png',                  dpi=300, bbox_inches='tight')
plt.savefig(f'{artifacts_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: confusion_matrix.png")

# ── metrics.txt ───────────────────────────────────────────
with open('metrics.txt', 'w') as f:
    f.write("=" * 50 + "\n")
    f.write("MODEL PERFORMANCE METRICS\n")
    f.write("=" * 50 + "\n")
    f.write(f"Accuracy:  {acc:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"F1 Score:  {f1:.4f}\n")
    f.write("=" * 50 + "\n\n")
    f.write("TRAINING HISTORY\n")
    f.write("=" * 50 + "\n")
    f.write(f"Final Train Loss:     {history_dict['loss'][-1]:.4f}\n")
    f.write(f"Final Val Loss:       {history_dict['val_loss'][-1]:.4f}\n")
    if 'accuracy' in history_dict:
        f.write(f"Final Train Accuracy: {history_dict['accuracy'][-1]:.4f}\n")
        f.write(f"Final Val Accuracy:   {history_dict['val_accuracy'][-1]:.4f}\n")
print("Saved: metrics.txt")

# ── metrics.json — read by DVC ─────────────────────────────
metrics_out = {
    "timestamp":   datetime.now().isoformat(),
    "model_type":  "1D CNN Classifier",
    "test_size":   TEST_SIZE,
    "batch_size":  BATCH_SIZE,
    "epochs":      EPOCHS,
    "final_epoch": len(history_dict['loss']),
    "metrics": {
        "accuracy":  float(acc),
        "precision": float(precision),
        "recall":    float(recall),
        "f1_score":  float(f1)
    },
    "training_history": {
        "final_train_loss": float(history_dict['loss'][-1]),
        "final_val_loss":   float(history_dict['val_loss'][-1])
    }
}
if 'accuracy' in history_dict:
    metrics_out["training_history"]["final_train_accuracy"] = float(history_dict['accuracy'][-1])
    metrics_out["training_history"]["final_val_accuracy"]   = float(history_dict['val_accuracy'][-1])

with open(METRICS_PATH, 'w', encoding='utf-8') as f:
    json.dump(metrics_out, f, indent=4)
print(f"Saved: {METRICS_PATH}")
