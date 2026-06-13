import os
import sys
import io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import json
from datetime import datetime
from io import StringIO
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
from dvclive import Live

# ── Load hyperparameters ───────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

ANN_EPOCHS    = params["model"]["epochs"]
ANN_BATCH     = params["model"]["batch_size"]
DENSE_1       = params["model"]["dense_units_1"]
DENSE_2       = params["model"]["dense_units_2"]
DENSE_3       = params["model"]["dense_units_3"]
DROPOUT_1     = params["model"]["dropout_1"]
DROPOUT_2     = params["model"]["dropout_2"]
SEED          = params["data"]["random_seed"]
TEST_SIZE     = params["data"]["test_size"]
ES_PATIENCE   = params["callbacks"]["early_stopping_patience"]
LR_PATIENCE   = params["callbacks"]["reduce_lr_patience"]
LR_FACTOR     = params["callbacks"]["reduce_lr_factor"]
LR_MIN        = params["callbacks"]["reduce_lr_min_lr"]
METRICS_PATH  = params["evaluate"]["metrics_path"]

# ── Directories ────────────────────────────────────────────
artifacts_dir = "artifacts"
os.makedirs(artifacts_dir, exist_ok=True)
os.makedirs(f"{artifacts_dir}/preprocessing", exist_ok=True)
os.makedirs(f"{artifacts_dir}/data", exist_ok=True)
os.makedirs(f"{artifacts_dir}/metrics", exist_ok=True)
os.makedirs(f"{artifacts_dir}/metadata", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("=" * 50)
print("STARTING ANN CLASSIFICATION MODEL — TRAINING")
print("=" * 50)

# ── Load data ──────────────────────────────────────────────
for path in ["train/train.csv", "test/test.csv"]:
    if not os.path.exists(path):
        print(f"ERROR: {path} not found!")
        sys.exit(1)

print("\nLoading data...")
data  = pd.read_csv("train/train.csv")
dtest = pd.read_csv("test/test.csv")
print(f"Train shape: {data.shape}  Test shape: {dtest.shape}")

# ── Missing values report ──────────────────────────────────
print(f"Missing — train: {data.isnull().any().sum()}  test: {dtest.isnull().any().sum()}")

train_test_data = [data, dtest]
for dataset in train_test_data:
    num_vars = [v for v in dataset.columns if dataset[v].dtype != 'O']
    print(f"Numerical variables: {len(num_vars)}")

# ── Drop constant columns ──────────────────────────────────
suspiciousData = [col for col in data.columns if data[col].nunique() == 1]
if suspiciousData:
    print(f"Dropping {len(suspiciousData)} constant columns")
    for dataset in train_test_data:
        dataset.drop(suspiciousData, axis=1, inplace=True)
else:
    print("No constant columns found")

# ── Encode categorical variables ───────────────────────────
cat_vars = [v for v in data.columns if data[v].dtype == 'O' and v not in ['ID', 'y']]
print(f"Categorical variables: {len(cat_vars)}")

if cat_vars:
    for var in cat_vars:
        freq = data[var].value_counts().to_dict()
        data[f"{var}_freq"]  = data[var].map(freq)
        dtest[f"{var}_freq"] = dtest[var].map(freq).fillna(0)
    data  = data.drop(cat_vars, axis=1)
    dtest = dtest.drop(cat_vars, axis=1)
    print("Categorical variables encoded")

# ── Features and target ────────────────────────────────────
if 'ID' in data.columns:
    data = data.drop("ID", axis=1)

if 'y' not in data.columns:
    print("ERROR: 'y' column not found! Columns:", data.columns.tolist())
    sys.exit(1)

X = data.drop("y", axis=1).apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.mean()).fillna(0).values
y = data["y"].values
print(f"X: {X.shape}  y: {y.shape}")

# ── Save feature column names ──────────────────────────────
feature_columns = list(data.drop("y", axis=1).columns)
with open(f"{artifacts_dir}/preprocessing/feature_columns.json", "w", encoding='utf-8') as f:
    json.dump(feature_columns, f, indent=4)
with open(f"{artifacts_dir}/feature_columns.json", "w", encoding='utf-8') as f:
    json.dump(feature_columns, f, indent=4)
print(f" Saved: feature_columns.json ({len(feature_columns)} features)")

with open(f"{artifacts_dir}/preprocessing/target_column.json", "w", encoding='utf-8') as f:
    json.dump({"target_column": "y"}, f)

# ── Train / test split ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED
)
print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")

# ── Scaling ────────────────────────────────────────────────
print("\n Creating and saving scaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print(f" Features scaled: mean≈0, std≈1")

joblib.dump(scaler, f"{artifacts_dir}/preprocessing/scaler.pkl")
print(f" Saved scaler to {artifacts_dir}/preprocessing/scaler.pkl")

# ── Save scaled data for sklearn models ───────────────────
np.save(f"{artifacts_dir}/data/X_train_scaled.npy", X_train_scaled)
np.save(f"{artifacts_dir}/data/X_test_scaled.npy",  X_test_scaled)
np.save(f"{artifacts_dir}/data/y_train.npy",         y_train)
np.save(f"{artifacts_dir}/data/y_test.npy",          y_test)
print(f" Saved scaled data to {artifacts_dir}/data/")

# ── Save split data for evaluate.py ───────────────────────
np.save(f"{artifacts_dir}/X_test_cnn.npy", X_test_scaled)
np.save(f"{artifacts_dir}/y_test.npy",     y_test)
print(" Saved: artifacts/X_test_cnn.npy, artifacts/y_test.npy")

# ── Build ANN model ────────────────────────────────────────
tf.random.set_seed(SEED)
num_features = X_train_scaled.shape[1]

model = Sequential([
    Dense(DENSE_1, activation='relu', input_shape=(num_features,)),
    Dropout(DROPOUT_1),
    Dense(DENSE_2, activation='relu'),
    Dropout(DROPOUT_2),
    Dense(DENSE_3, activation='relu'),
    Dense(7, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

stream = StringIO()
model.summary(print_fn=lambda x: stream.write(x + '\n'))
with open('model_summary.txt', 'w', encoding='utf-8') as f:
    f.write(stream.getvalue())
print("Saved: model_summary.txt")

# ── Callbacks ──────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor='val_loss', patience=ES_PATIENCE,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=LR_FACTOR,
                      patience=LR_PATIENCE, min_lr=LR_MIN, verbose=1)
]

# ── Train ──────────────────────────────────────────────────
print("\nTraining model...")
with Live(dir="dvclive", report="html") as live:
    live.log_param("ann_epochs",     ANN_EPOCHS)
    live.log_param("ann_batch_size", ANN_BATCH)
    live.log_param("dense_units",    str([DENSE_1, DENSE_2, DENSE_3]))

    history = model.fit(
        X_train_scaled, y_train,
        epochs=ANN_EPOCHS,
        batch_size=ANN_BATCH,
        validation_data=(X_test_scaled, y_test),
        callbacks=callbacks,
        verbose=1
    )

    for i in range(len(history.history['loss'])):
        live.log_metric("train_loss",     history.history['loss'][i])
        live.log_metric("val_loss",       history.history['val_loss'][i])
        live.log_metric("train_accuracy", history.history['accuracy'][i])
        live.log_metric("val_accuracy",   history.history['val_accuracy'][i])
        live.next_step()

print("Training completed!")

# ── Save model ─────────────────────────────────────────────
model.save("models/model.keras")
print(" Saved: models/model.keras")

# ── Training history plots ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history['accuracy'],     label='Train Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()

axes[1].plot(history.history['loss'],     label='Train Loss')
axes[1].plot(history.history['val_loss'], label='Validation Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()

plt.tight_layout()
plt.savefig('model_results.png',                  dpi=300, bbox_inches='tight')
plt.savefig(f'{artifacts_dir}/model_results.png', dpi=300, bbox_inches='tight')
plt.close()
print(" Saved: model_results.png")

# ── Save training history ──────────────────────────────────
history_dict = {
    "loss":         [float(v) for v in history.history['loss']],
    "val_loss":     [float(v) for v in history.history['val_loss']],
    "accuracy":     [float(v) for v in history.history['accuracy']],
    "val_accuracy": [float(v) for v in history.history['val_accuracy']],
}

with open(f"{artifacts_dir}/training_history.json", "w", encoding='utf-8') as f:
    json.dump(history_dict, f, indent=4)
with open(f"{artifacts_dir}/metrics/training_history.json", "w", encoding='utf-8') as f:
    json.dump(history_dict, f, indent=4)
print(" Saved: training_history.json")

# ── Save test metrics ──────────────────────────────────────
y_pred = model.predict(X_test_scaled, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

test_metrics = {
    'accuracy':  float(accuracy_score(y_test, y_pred_classes)),
    'timestamp': datetime.now().isoformat()
}

with open(f"{artifacts_dir}/metrics/test_metrics.json", "w", encoding='utf-8') as f:
    json.dump(test_metrics, f, indent=4)
print(" Saved: artifacts/metrics/test_metrics.json")

# ── Save data info ─────────────────────────────────────────
data_info = {
    "train_samples":             int(X_train.shape[0]),
    "test_samples":              int(X_test.shape[0]),
    "features_count":            int(X.shape[1]),
    "categorical_vars_original": len(cat_vars),
    "constant_features_dropped": len(suspiciousData),
    "num_classes":               int(len(np.unique(y)))
}
with open('data_info.json', 'w', encoding='utf-8') as f:
    json.dump(data_info, f, indent=4)
with open(f"{artifacts_dir}/metadata/data_info.json", 'w', encoding='utf-8') as f:
    json.dump(data_info, f, indent=4)
print(" Saved: data_info.json")

# ── Save model metadata ────────────────────────────────────
model_metadata = {
    'model_type': 'ANN_Classification',
    'input_shape': (num_features,),
    'num_features': len(feature_columns),
    'num_classes': int(len(np.unique(y))),
    'num_training_samples': len(y_train),
    'num_test_samples': len(y_test),
    'feature_columns_preview': feature_columns[:10],
    'target_column': 'y',
    'training_completed': datetime.now().isoformat(),
    'hyperparameters': {
        'epochs':        ANN_EPOCHS,
        'batch_size':    ANN_BATCH,
        'dense_units':   [DENSE_1, DENSE_2, DENSE_3],
        'dropout_rates': [DROPOUT_1, DROPOUT_2]
    },
    'test_performance': test_metrics
}

with open(f"{artifacts_dir}/metadata/model_info.json", "w", encoding='utf-8') as f:
    json.dump(model_metadata, f, indent=4)
print(" Saved: artifacts/metadata/model_info.json")

with open(f"{artifacts_dir}/metadata/num_features.txt", "w") as f:
    f.write(str(len(feature_columns)))

print("\n" + "="*60)
print(" TRAINING COMPLETE - ALL ARTIFACTS GENERATED")
print("="*60)
print(" model.py completed successfully!")
