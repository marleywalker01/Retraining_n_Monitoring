import os
import sys
import json
import numpy as np
import joblib
import yaml
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ── Load hyperparameters ───────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

SEED = params["data"]["random_seed"]

artifacts_dir = "artifacts"
os.makedirs(f"{artifacts_dir}/models", exist_ok=True)
os.makedirs(f"{artifacts_dir}/metrics", exist_ok=True)

print("=" * 50)
print("STARTING RANDOM FOREST CLASSIFIER — TRAINING")
print("=" * 50)

# ── Load data saved by model.py ────────────────────────────
for path in ["artifacts/data/X_train_scaled.npy", "artifacts/data/y_train.npy",
             "artifacts/data/X_test_scaled.npy",  "artifacts/data/y_test.npy"]:
    if not os.path.exists(path):
        print(f"ERROR: {path} not found — run model.py first")
        sys.exit(1)

print("Loading data...")
X_train = np.load("artifacts/data/X_train_scaled.npy")
y_train = np.load("artifacts/data/y_train.npy")
X_test  = np.load("artifacts/data/X_test_scaled.npy")
y_test  = np.load("artifacts/data/y_test.npy")
print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")

# ── Train Random Forest Classifier ────────────────────────
print("\nTraining Random Forest Classifier...")
clf = RandomForestClassifier(random_state=SEED)
clf.fit(X_train, y_train)
print("Training completed!")

# ── Make predictions ───────────────────────────────────────
y_pred = clf.predict(X_test)

# ── Evaluate ───────────────────────────────────────────────
random_forest_accuracy = accuracy_score(y_test, y_pred)
random_forest_f1       = f1_score(y_test, y_pred, average="weighted", zero_division=0)
cv_scores              = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
random_forest_cv       = cv_scores.mean()

print(f"\nAccuracy: {random_forest_accuracy:.4f}")
print(f"F1:       {random_forest_f1:.4f}")
print(f"CV:       {random_forest_cv:.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")

# ── Save model ─────────────────────────────────────────────
joblib.dump(clf, f"{artifacts_dir}/models/rf_model.pkl")
print(f"\nSaved: {artifacts_dir}/models/rf_model.pkl")

# ── Save metrics ───────────────────────────────────────────
metrics_out = {
    "timestamp": datetime.now().isoformat(),
    "model_type": "Random Forest Classifier",
    "metrics": {
        "accuracy":           float(random_forest_accuracy),
        "f1_score":           float(random_forest_f1),
        "cross_val_accuracy": float(random_forest_cv)
    }
}

with open(f"{artifacts_dir}/metrics/rf_metrics.json", "w") as f:
    json.dump(metrics_out, f, indent=4)
print(f"Saved: {artifacts_dir}/metrics/rf_metrics.json")

print("\nmodel_rf.py completed successfully!")
