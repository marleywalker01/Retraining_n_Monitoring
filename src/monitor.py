import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/monitoring.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('monitoring')

from sklearn.metrics import accuracy_score
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

def find_artifact_file(base_name):
    """Find artifact file in various possible locations"""
    possible_paths = [
        f'artifacts/data/{base_name}',
        f'artifacts/{base_name}',
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def load_all_data():
    """Load all datasets from artifacts"""
    logger.info("Loading data from artifacts...")

    X_test_path = find_artifact_file('X_test_cnn.npy')
    y_test_path = find_artifact_file('y_test.npy')

    if X_test_path is None or y_test_path is None:
        logger.error("Test data not found!")
        return None, None, None, None, None, None, None

    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    logger.info(f"Loaded test data: {X_test.shape}")

    X_train_path = find_artifact_file('X_train_cnn.npy')
    y_train_path = find_artifact_file('y_train.npy')

    X_train = None
    y_train = None
    if X_train_path and y_train_path:
        X_train = np.load(X_train_path)
        y_train = np.load(y_train_path)
        logger.info(f"Loaded training data: {X_train.shape}")

    X_new_path = find_artifact_file('X_new_cnn.npy') or find_artifact_file('X_new.npy')
    y_new_path = find_artifact_file('y_new.npy')

    X_new = None
    y_new = None
    if X_new_path and y_new_path:
        X_new = np.load(X_new_path)
        y_new = np.load(y_new_path)
        logger.info(f"Loaded new data: {X_new.shape}")
    else:
        logger.warning("No new data found, using test data as reference")
        X_new = X_test.copy()
        y_new = y_test.copy()

    model_path = 'models/model.keras'
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return None, None, None, None, None, None, None

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None, None, None, None, None, None, None

    return model, X_train, y_train, X_test, y_test, X_new, y_new

def main():
    """Main monitoring function"""

    logger.info("="*70)
    logger.info("STARTING COMPREHENSIVE MONITORING")
    logger.info("="*70)

    model, X_train, y_train, X_test, y_test, X_new, y_new = load_all_data()

    if model is None:
        logger.error("Failed to load model or data")
        return False

    # Get predictions
    logger.info("Getting model predictions...")
    try:
        y_test_pred_prob = model.predict(X_test, verbose=0)
        y_test_pred = np.argmax(y_test_pred_prob, axis=1)
        logger.info(f"Test predictions: {len(y_test_pred)} samples")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return False

    try:
        y_new_pred_prob = model.predict(X_new, verbose=0)
        y_new_pred = np.argmax(y_new_pred_prob, axis=1)
        logger.info(f"New data predictions: {len(y_new_pred)} samples")
    except Exception as e:
        logger.error(f"New data prediction failed: {e}")
        y_new_pred = y_test_pred.copy()
        y_new = y_test.copy()

    # Calculate accuracy for test data
    logger.info("Calculating test metrics...")
    test_acc = accuracy_score(y_test, y_test_pred)
    logger.info(f"Test Accuracy: {test_acc:.4f}")

    # Calculate accuracy for new data
    logger.info("Calculating new data metrics...")
    new_acc = accuracy_score(y_new, y_new_pred)
    logger.info(f"New Data Accuracy: {new_acc:.4f}")

    # Calculate performance change
    acc_drop = test_acc - new_acc
    logger.info(f"Accuracy drop: {acc_drop:.4f}")

    # Determine if retraining is needed (flag if accuracy drops by more than 5%)
    threshold = 0.05
    retrain_needed = acc_drop > threshold

    logger.info("="*70)
    if retrain_needed:
        logger.warning(f"RETRAINING NEEDED: Accuracy dropped by {acc_drop:.4f}")
    else:
        logger.info(f"NO RETRAINING NEEDED: Accuracy drop within threshold")
    logger.info("="*70)

    # Save results
    os.makedirs('artifacts/metrics', exist_ok=True)

    results = {
        'timestamp': datetime.now().isoformat(),
        'test_metrics': {
            'accuracy': float(test_acc)
        },
        'new_metrics': {
            'accuracy': float(new_acc)
        },
        'accuracy_drop': float(acc_drop),
        'threshold': threshold,
        'retrain_needed': retrain_needed
    }

    with open('artifacts/metrics/monitoring_summary.json', 'w') as f:
        json.dump(results, f, indent=2)

    with open('retrain_needed.txt', 'w') as f:
        f.write('true' if retrain_needed else 'false')

    with open('drift_status.txt', 'w') as f:
        f.write(str(retrain_needed).lower())

    logger.info("Results saved to artifacts/metrics/monitoring_summary.json")

    return retrain_needed

if __name__ == "__main__":
    try:
        retrain_needed = main()
        sys.exit(1 if retrain_needed else 0)
    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(0)
