import os
import sys
import json
import numpy as np
from datetime import datetime

print("=" * 50)
print("MODEL COMPARISON — ALL MODELS")
print("=" * 50)

# ── Load metrics from each model ───────────────────────────
metrics_paths = {
    "CNN":          "artifacts/metrics/test_metrics.json",
    "MLP":          "artifacts/metrics/mlp_metrics.json",
    "RandomForest": "artifacts/metrics/rf_metrics.json"
}

results = {}

for model_name, path in metrics_paths.items():
    if not os.path.exists(path):
        print(f"WARNING: {path} not found — skipping {model_name}")
        continue
    with open(path) as f:
        data = json.load(f)
    results[model_name] = data
    print(f"Loaded metrics for {model_name}")

if not results:
    print("ERROR: No model metrics found")
    sys.exit(1)

# ── Build comparison table ──────────────────────────────────
print("\n" + "=" * 60)
print("MODEL COMPARISON RESULTS")
print("=" * 60)
print(f"{'Model':<20} {'Accuracy':<12} {'F1 Score':<12} {'CV Score':<12}")
print("-" * 60)

comparison = {}

for model_name, data in results.items():
    metrics = data.get("metrics", data)
    accuracy = metrics.get("accuracy", "N/A")
    f1       = metrics.get("f1_score", "N/A")
    cv       = metrics.get("cross_val_accuracy", "N/A")

    acc_str = f"{accuracy:.4f}" if isinstance(accuracy, float) else accuracy
    f1_str  = f"{f1:.4f}"       if isinstance(f1, float)       else f1
    cv_str  = f"{cv:.4f}"       if isinstance(cv, float)       else cv

    print(f"{model_name:<20} {acc_str:<12} {f1_str:<12} {cv_str:<12}")

    comparison[model_name] = {
        "accuracy": accuracy,
        "f1_score": f1,
        "cross_val_accuracy": cv
    }

print("=" * 60)

# ── Determine best model by accuracy ──────────────────────
best_model = max(
    [(k, v["accuracy"]) for k, v in comparison.items() if isinstance(v["accuracy"], float)],
    key=lambda x: x[1]
)
print(f"\nBest model by accuracy: {best_model[0]} ({best_model[1]:.4f})")

# ── Save comparison report ─────────────────────────────────
os.makedirs("artifacts/metrics", exist_ok=True)

comparison_out = {
    "timestamp": datetime.now().isoformat(),
    "models": comparison,
    "best_model": {
        "name":     best_model[0],
        "accuracy": best_model[1]
    }
}

with open("artifacts/metrics/model_comparison.json", "w") as f:
    json.dump(comparison_out, f, indent=4)
print("Saved: artifacts/metrics/model_comparison.json")

print("\nevaluate_all.py completed successfully!")
